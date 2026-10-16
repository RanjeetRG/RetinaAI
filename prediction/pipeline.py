"""End-to-end analysis: grade -> simulate next stage(s) -> re-grade -> compare (M5; PRD "User workflow").

The pipeline has no side effects; prediction/runs.py saves results, the database rows and the PDF report.
Only the generators a request needs are loaded (PRD: lazy loading keeps memory low on free hosting).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import keras
import numpy as np

import config
from cnn.model import input_size, load_classifier
from gan.networks import load_generator
from preprocessing.fundus import ImageSource, cnn_input_from_gan_image, from_gan_range, load_fundus, prepare_cnn, prepare_gan, to_gan_range
from preprocessing.quality import QualityReport, check_quality
from utils.gradcam import gradcam
from utils.image_metrics import difference_map, psnr, ssim


def model_version(path: Path) -> str:
    """e.g. 'cnn_efficientnetb0@2026-10-14', from the metadata JSON saved next to each model."""
    meta_path = Path(path).with_suffix(".json")
    if not meta_path.exists():
        return Path(path).stem
    meta = json.loads(meta_path.read_text())
    if "backbone" in meta:
        name = f"cnn_{meta['backbone']}" + (f"_{meta['input_size']}" if "input_size" in meta else "")
    elif "transition" in meta:
        name = f"generator_{meta['transition']}"
    else:
        name = Path(path).stem
    return f"{name}@{meta.get('trained_at', '')[:10]}"


@dataclass
class SimulatedStage:
    from_stage: int
    to_stage: int
    image: np.ndarray  # GAN_SIZE uint8; always shown labelled "Simulated" (FR-15)
    ssim: float  # against the current retina, not the previous step
    psnr: float
    difference: np.ndarray  # per-pixel absolute difference from the current retina, 0-255
    regraded_probabilities: np.ndarray
    generator_version: str

    @property
    def regraded_stage(self) -> int:
        return int(self.regraded_probabilities.argmax())

    @property
    def regrade_matches(self) -> bool:
        return self.regraded_stage == self.to_stage


@dataclass
class Analysis:
    cached: np.ndarray  # CACHE_SIZE fundus (border cropped, square)
    cnn_input: np.ndarray  # classifier input size, enhanced, as graded
    current: np.ndarray  # GAN_SIZE, enhanced: the "current retina" shown beside every simulation
    probabilities: np.ndarray
    quality: QualityReport
    gradcam: np.ndarray | None
    model_version: str
    simulations: list[SimulatedStage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def stage(self) -> int:
        return int(self.probabilities.argmax())

    @property
    def confidence(self) -> float:
        return float(self.probabilities.max())


class Pipeline:
    def __init__(self, cnn_path: Path | None = None, max_cached_generators: int = config.MAX_STAGE):
        cnn_path = cnn_path or config.CNN_MODEL_PATH
        self.cnn = load_classifier(cnn_path)
        self.cnn_version = model_version(cnn_path)
        self.cnn_size = input_size(self.cnn)
        self.max_cached = max_cached_generators
        self._generators: OrderedDict[int, keras.Model] = OrderedDict()

    def generator(self, k: int) -> keras.Model:
        """The k -> k+1 generator, loaded on first use and kept in a small least-recently-used cache."""
        if k in self._generators:
            self._generators.move_to_end(k)
        else:
            self._generators[k] = load_generator(config.generator_path(k))
            while len(self._generators) > self.max_cached:
                self._generators.popitem(last=False)
        return self._generators[k]

    @staticmethod
    def available_steps(stage: int) -> int:
        """How many consecutive next stages can be simulated from `stage` with the generators on disk."""
        steps = 0
        while stage + steps < config.MAX_STAGE and config.generator_path(stage + steps).exists():
            steps += 1
        return steps

    def grade(self, cnn_input: np.ndarray) -> np.ndarray:
        probs = self.cnn(cnn_input[None].astype("float32"), training=False)
        return keras.ops.convert_to_numpy(probs)[0].astype(np.float64)

    def simulate(self, current: np.ndarray, image: np.ndarray, k: int) -> SimulatedStage:
        fake = keras.ops.convert_to_numpy(self.generator(k)(to_gan_range(image)[None], training=False))
        fake = from_gan_range(fake[0])
        return SimulatedStage(
            from_stage=k,
            to_stage=k + 1,
            image=fake,
            ssim=ssim(current, fake),
            psnr=psnr(current, fake),
            difference=difference_map(current, fake),
            regraded_probabilities=self.grade(cnn_input_from_gan_image(fake, self.cnn_size)),
            generator_version=model_version(config.generator_path(k)),
        )

    def analyze(self, source: ImageSource, steps: int = 1, with_gradcam: bool = True) -> Analysis:
        """Grade a fundus image and simulate up to `steps` next stages, each generator fed the previous output."""
        cached = load_fundus(source)
        cnn_input = prepare_cnn(cached, self.cnn_size)
        probabilities = self.grade(cnn_input)
        stage = int(probabilities.argmax())
        result = Analysis(
            cached=cached,
            cnn_input=cnn_input,
            current=from_gan_range(prepare_gan(cached)),
            probabilities=probabilities,
            quality=check_quality(cached),
            gradcam=gradcam(self.cnn, cnn_input, stage) if with_gradcam else None,
            model_version=self.cnn_version,
        )

        if stage == config.MAX_STAGE:  # FR-8
            result.notes.append(
                f"Stage {stage} ({config.STAGE_NAMES[stage]}) is the last stage of the scale, so there is no later "
                "stage to simulate."
            )
            return result
        wanted = min(steps, config.MAX_STAGE - stage)
        available = min(wanted, self.available_steps(stage))
        if available < wanted:
            missing = stage + available
            result.notes.append(
                f"The stage {missing} to {missing + 1} generator is not available yet, so the simulation stops "
                f"at stage {missing}."
            )
        image = result.current
        for k in range(stage, stage + available):
            sim = self.simulate(result.current, image, k)
            result.simulations.append(sim)
            image = sim.image
        return result
