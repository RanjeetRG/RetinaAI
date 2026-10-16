"""Train a stage classifier (M3). Run in WSL on the GPU:

    source scripts/wsl_env.sh
    python -m cnn.train --backbone efficientnetb0 --size 384 --promote

Phase 1 trains only the new head with the ImageNet backbone frozen; phase 2 fine-tunes everything with a
cosine-decayed learning rate. Augmentation runs on the CPU in the data loader (as Keras layers on the GPU it made
steps 3.5x slower in scripts/bench_cnn.py) and steps are XLA-compiled. The epoch with the best validation QWK is kept. Class weights counter the
stage imbalance (No DR is ~70% of images). Training uses mixed precision; the saved model is float32 so
the app runs it on a CPU.

Outputs: models/cnn_<backbone>_<size>.keras and .json, outputs/eval/cnn/<backbone>_<size>/history.csv;
--promote also copies the model to models/model.keras, the classifier the app uses.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime, timezone

import cv2
import keras
import numpy as np

import config
from cnn.data import SplitArrays, load_split
from cnn.metrics import quadratic_kappa
from cnn.model import BACKBONES, build_classifier, get_backbone
from utils.seed import set_seed


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Training-only augmentation of one uint8 image; returns float32 0-255.

    Fundus photos have no canonical orientation, so both flips and any rotation are valid. Brightness and contrast
    shifts mimic exposure differences between cameras and are applied inside the fundus only, so the surround stays
    black as in real inputs.
    """
    if rng.random() < 0.5:
        image = image[:, ::-1]
    if rng.random() < 0.5:
        image = image[::-1]
    size = image.shape[0]
    rotation = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-180, 180), 1.0)
    image = cv2.warpAffine(np.ascontiguousarray(image), rotation, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # fmt: skip
    out = image.astype(np.float32)
    inside = image.max(axis=-1) > 0
    mean = out[inside].mean(axis=0) if inside.any() else 0.0
    contrast = rng.uniform(0.9, 1.1)
    brightness = rng.uniform(-0.1, 0.1) * 255
    out[inside] = np.clip((out[inside] - mean) * contrast + mean + brightness, 0, 255)
    return out


class BatchLoader(keras.utils.PyDataset):
    """Batches from a uint8 array (in memory or memory-mapped), with optional per-sample weights, per-epoch
    shuffling and CPU augmentation. Augmentation is seeded per (epoch, batch), so runs are reproducible even with
    several loader threads."""

    def __init__(self, data: SplitArrays, batch_size: int, sample_weights=None, shuffle=False, seed=0,
                 augment: bool = False, **kwargs):  # fmt: skip
        super().__init__(**kwargs)
        self.data, self.batch_size, self.sample_weights, self.shuffle = data, batch_size, sample_weights, shuffle
        self.augment, self.seed, self.epoch = augment, seed, 0
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(data))
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, i):
        idx = np.sort(self.order[i * self.batch_size : (i + 1) * self.batch_size])
        images = self.data.images[idx]
        if self.augment:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            x = np.stack([augment(img, rng) for img in images])
        else:
            x = images.astype("float32")
        y = self.data.stages[idx]
        if self.sample_weights is None:
            return x, y
        return x, y, self.sample_weights[idx]

    def on_epoch_end(self) -> None:
        self.epoch += 1
        if self.shuffle:
            self.rng.shuffle(self.order)


def class_weights(stages: np.ndarray, power: float) -> np.ndarray:
    """Per-class weights (n / (k * n_c)) ** power, normalised so the average training sample weighs 1.

    power=1 is fully 'balanced'; 0.5 softens it so the rare stages are not over-predicted.
    """
    counts = np.bincount(stages, minlength=config.NUM_STAGES).astype(float)
    weights = (counts.sum() / (config.NUM_STAGES * counts)) ** power
    return weights / weights[stages].mean()


class ValidationKappa(keras.callbacks.Callback):
    """Adds val_qwk to the epoch logs and saves the classifier's weights whenever it improves.

    Must come before EarlyStopping and CSVLogger in the callback list so they see val_qwk.
    """

    def __init__(self, classifier: keras.Model, data: SplitArrays, batch_size: int, best_path):
        super().__init__()
        self.classifier, self.data, self.batch_size, self.best_path = classifier, data, batch_size, best_path
        self.best, self.best_epoch = -np.inf, -1

    def on_epoch_end(self, epoch, logs=None):
        probs = self.classifier.predict(BatchLoader(self.data, self.batch_size * 4), verbose=0)
        qwk = quadratic_kappa(self.data.stages, probs.argmax(axis=1))
        logs["val_qwk"] = qwk
        if qwk > self.best:
            self.best, self.best_epoch = qwk, epoch
            self.classifier.save_weights(self.best_path)
        print(f"epoch {epoch + 1}: val_qwk {qwk:.4f} (best {self.best:.4f} at epoch {self.best_epoch + 1})", flush=True)


def to_float32(trained: keras.Model, backbone: str, dropout: float, size: int) -> keras.Model:
    """Float32 copy of a mixed-precision classifier.

    Built with weights="imagenet" (then overwritten) because keras.applications builds a different graph
    without them: EfficientNet only adds its input-scaling Rescaling layer when ImageNet weights are requested.
    """
    final = build_classifier(backbone, dropout=dropout, weights="imagenet", input_size=size)
    final.set_weights(trained.get_weights())
    return final


def check_same_predictions(trained: keras.Model, final: keras.Model, data: SplitArrays, n: int = 64) -> None:
    """Refuse to save a converted model whose predictions differ from the trained one (beyond fp16 noise).

    Predicts in small batches: the trained model still occupies most of the GPU, and a 64-image float32 batch ran
    out of memory after 224 px training.
    """
    x = data.images[:n].astype("float32")
    a = trained.predict(x, batch_size=8, verbose=0)
    b = final.predict(x, batch_size=8, verbose=0)
    worst = float(np.abs(a - b).max())
    if worst > 0.05 or (a.argmax(1) != b.argmax(1)).mean() > 0.05:
        raise RuntimeError(f"float32 model disagrees with the trained model (max probability difference {worst:.3f})")
    print(f"float32 copy verified (max probability difference {worst:.4f})", flush=True)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    keras.mixed_precision.set_global_policy("mixed_float16")

    name = f"{args.backbone}_{args.size}"
    # Memory-map arrays larger than the default size: the 384 px training array (~12 GB) exceeds WSL's RAM
    mmap = args.size > config.CNN_SIZE
    train_data, val_data = load_split("train", args.size, mmap), load_split("validation", args.size, mmap)
    weights = class_weights(train_data.stages, args.class_weight_power)
    print("class weights:", np.round(weights, 3).tolist(), flush=True)

    model = build_classifier(args.backbone, dropout=args.dropout, input_size=args.size)
    backbone = get_backbone(model)

    config.MODELS_DIR.mkdir(exist_ok=True)
    run_dir = config.EVAL_DIR / "cnn" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.MODELS_DIR / f"cnn_{name}.best.weights.h5"
    kappa = ValidationKappa(model, val_data, args.batch_size, best_path)
    history_csv = run_dir / "history.csv"
    loader = BatchLoader(
        train_data, args.batch_size, weights[train_data.stages], shuffle=True, seed=args.seed, augment=True,
        workers=args.workers,
    )
    loss = keras.losses.SparseCategoricalCrossentropy()
    start = time.time()

    # Phase 1: new head only
    backbone.trainable = False
    # XLA-compiled steps plus CPU augmentation took 224 px fine-tuning from 323 to 69 ms/step. The first epoch of each
    # phase is slower because it includes compiling and autotuning the step.
    model.compile(optimizer=keras.optimizers.Adam(args.head_lr), loss=loss, metrics=["accuracy"], jit_compile=True)
    model.fit(
        loader, epochs=args.head_epochs, verbose=2, callbacks=[kappa, keras.callbacks.CSVLogger(history_csv)]
    )

    # Phase 2: fine-tune the whole network (BatchNorm stays frozen: the backbone is called with training=False)
    backbone.trainable = True
    schedule = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=len(loader) * args.epochs, alpha=0.01)
    model.compile(
        optimizer=keras.optimizers.AdamW(schedule, weight_decay=args.weight_decay), loss=loss, metrics=["accuracy"],
        jit_compile=True,
    )
    model.fit(
        loader,
        initial_epoch=args.head_epochs,
        epochs=args.head_epochs + args.epochs,
        verbose=2,
        callbacks=[
            kappa,
            keras.callbacks.EarlyStopping(monitor="val_qwk", mode="max", patience=args.patience),
            keras.callbacks.CSVLogger(history_csv, append=True),
        ],
    )
    minutes = (time.time() - start) / 60

    # Rebuild in float32 for CPU inference and copy the best weights across
    model.load_weights(best_path)
    keras.mixed_precision.set_global_policy("float32")
    final = to_float32(model, args.backbone, args.dropout, args.size)
    check_same_predictions(model, final, val_data)
    out_path = config.MODELS_DIR / f"cnn_{name}.keras"
    final.save(out_path)
    best_path.unlink()

    meta = {
        "backbone": args.backbone,
        "input_size": args.size,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_epoch": kappa.best_epoch + 1,
        "val_qwk": kappa.best,
        "train_minutes": round(minutes, 1),
        "train_images": len(train_data),
        "class_weights": weights.tolist(),
        "args": vars(args),
        "versions": {"keras": keras.__version__},
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    if args.promote:
        # The app loads models/model.keras (PRD)
        shutil.copy2(out_path, config.CNN_MODEL_PATH)
        shutil.copy2(out_path.with_suffix(".json"), config.CNN_MODEL_PATH.with_suffix(".json"))
    print(f"saved {out_path}: best val QWK {kappa.best:.4f} at epoch {kappa.best_epoch + 1}, {minutes:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=BACKBONES, default="efficientnetb0")
    parser.add_argument("--size", type=int, default=config.CNN_SIZE, choices=config.CNN_SIZES, help="input size in px")
    parser.add_argument("--promote", action="store_true", help="also save as models/model.keras, the app's classifier")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25, help="maximum fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="peak fine-tuning learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6, help="data-loader threads (augmentation runs on the CPU)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
