"""Collect the paper's tables and figures from saved results (no TensorFlow needed).

    python -m scripts.paper_outputs

Writes outputs/eval/paper_tables.md (dataset, classifiers, generators, guidance ablation) and
outputs/eval/cnn/confusion_matrices.png (one panel per evaluated classifier, rows normalised to the true stage).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import config  # noqa: E402
from utils.visuals import GRIDLINE, INK, INK_SECONDARY, SEQUENTIAL_BLUE, SURFACE  # noqa: E402

CNN_DIR = config.EVAL_DIR / "cnn"
MODEL_LABELS = {
    "efficientnetb0_384": "EfficientNetB0, 384 px (app)",
    "efficientnetb0_224": "EfficientNetB0, 224 px",
    "resnet50_224": "ResNet50, 224 px",
    "densenet121_224": "DenseNet121, 224 px",
    "mobilenetv2_224": "MobileNetV2, 224 px",
}
SHORT_STAGES = ["0 None", "1 Mild", "2 Mod.", "3 Sev.", "4 Prolif."]


def markdown_table(df: pd.DataFrame) -> str:
    cells = df.map(lambda v: "" if pd.isna(v) else str(v))  # pandas 3 keeps NaN as missing under astype(str)
    lines = ["| " + " | ".join(cells.columns) + " |", "|" + "---|" * len(cells.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines)


def evaluated_models() -> list[str]:
    return [m for m in MODEL_LABELS if (CNN_DIR / m / "test_metrics.json").exists()]


def confusion_figure(models: list[str]) -> None:
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)
    fig, axes = plt.subplots(1, len(models), figsize=(3.9 * len(models) + 0.9, 4.3), facecolor=SURFACE, squeeze=False)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    for col, (ax, name) in enumerate(zip(axes[0], models)):
        counts = np.array(json.loads((CNN_DIR / name / "test_metrics.json").read_text())["all"]["confusion_matrix"])
        share = counts / counts.sum(axis=1, keepdims=True)
        mesh = ax.pcolormesh(share, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
        for i in range(share.shape[0]):
            for j in range(share.shape[1]):
                ink = "white" if share[i, j] > 0.5 else INK  # text colour follows the cell's darkness
                ax.text(j + 0.5, i + 0.5, f"{share[i, j]:.0%}\n{counts[i, j]}", ha="center", va="center", fontsize=7.5, color=ink)
        ax.set_xticks(np.arange(5) + 0.5, SHORT_STAGES, color=INK_SECONDARY, fontsize=7.5)
        ax.set_yticks(np.arange(5) + 0.5, SHORT_STAGES if col == 0 else [], color=INK_SECONDARY, fontsize=7.5)
        ax.invert_yaxis()
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(MODEL_LABELS[name], loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Predicted stage", color=INK_SECONDARY)
        if col == 0:
            ax.set_ylabel("True stage", color=INK_SECONDARY)
    bar = fig.colorbar(mesh, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    bar.set_label("Share of the true stage", color=INK_SECONDARY)
    bar.outline.set_edgecolor(GRIDLINE)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=7.5)
    out = CNN_DIR / "confusion_matrices.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


def classifier_tables(models: list[str]) -> str:
    summary = pd.read_csv(CNN_DIR / "summary.csv")
    summary = summary[(summary["split"] == "test") & summary["model"].isin(models)]
    overall = summary[summary["subset"] == "all"].copy()
    overall["Model"] = overall["model"].map(MODEL_LABELS)
    table = overall[["Model", "qwk", "accuracy", "macro_f1", "referable_auc", "screening_sensitivity",
                     "screening_specificity", "recall_severe", "recall_proliferative"]].round(3)  # fmt: skip
    table.columns = ["Model", "QWK", "Accuracy", "Macro F1", "Referable AUC", "Screening sens.", "Screening spec.",
                     "Recall severe", "Recall prolif."]  # fmt: skip
    table = table.sort_values("QWK", ascending=False)
    by_source = summary.pivot_table(index="model", columns="subset", values="qwk").round(3)
    by_source.insert(0, "Model", by_source.index.map(MODEL_LABELS))
    by_source = by_source[["Model", "all", "aptos", "eyepacs"]].sort_values("all", ascending=False)
    by_source.columns = ["Model", "QWK all", "QWK APTOS", "QWK EyePACS"]
    return (
        "## Classifiers: held-out test split (5,784 images)\n\n"
        "Screening sensitivity/specificity use the cut-off on P(stage >= 2) that reaches 90% sensitivity on validation.\n\n"
        + markdown_table(table)
        + "\n\n### QWK by source (camera confound check)\n\n"
        + markdown_table(by_source)
    )


def gan_tables() -> str:
    parts = []
    final = config.EVAL_DIR / "gan" / "metrics.csv"
    cols = ["transition", "judge", "method", "ssim", "fid", "kid", "stage_consistency", "mean_grade_shift",
            "global_shift", "local_change", "change_correlation"]  # fmt: skip
    if final.exists():
        parts.append("## Generators: final run\n\n" + markdown_table(pd.read_csv(final)[cols].round(3)))
    pilots = config.OUTPUTS_DIR / "gan" / "_pilots" / "comparison_1to2.csv"
    if pilots.exists():
        table = pd.read_csv(pilots).drop_duplicates(["judge", "method"])[cols].round(3)
        parts.append("## Generator ablation: guidance pilots on 1->2 (4,000 steps)\n\n" + markdown_table(table))
    return "\n\n".join(parts)


def main() -> None:
    models = evaluated_models()
    if models:
        confusion_figure(models)
    counts = pd.read_csv(config.EVAL_DIR / "data" / "class_counts.csv")
    sections = [
        "# RetinaAI: paper tables\n\nGenerated by `python -m scripts.paper_outputs` from the saved evaluation files.",
        "## Dataset (images per split, source and stage)\n\n" + markdown_table(counts),
        classifier_tables(models) if models else "",
        gan_tables(),
    ]
    out = config.EVAL_DIR / "paper_tables.md"
    out.write_text("\n\n".join(s for s in sections if s) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
