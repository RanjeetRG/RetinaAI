"""Analyse one fundus image from the command line (M5 done-check).

    python -m prediction.run path/to/fundus.jpg            # grade + simulate the next stage
    python -m prediction.run path/to/fundus.jpg --steps 4  # chain generators up to stage 4

Prints the grade and each simulated stage, saves the run to the history database and writes the PDF report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from database import db
from prediction.pipeline import Pipeline
from prediction.runs import save_run
from preprocessing.quality import validate_upload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--steps", type=int, default=1, help="how many next stages to simulate")
    parser.add_argument("--no-save", action="store_true", help="do not write the database row or the report")
    args = parser.parse_args()

    image = validate_upload(args.image.name, args.image.read_bytes())
    analysis = Pipeline().analyze(image, steps=args.steps)

    print(f"Detected stage {analysis.stage} ({config.STAGE_NAMES[analysis.stage]}), confidence {analysis.confidence:.0%}")
    for k, p in enumerate(analysis.probabilities):
        print(f"  {k} {config.STAGE_NAMES[k]:<14} {p:6.1%}")
    for warning in analysis.quality.warnings:
        print("Quality warning:", warning)
    for note in analysis.notes:
        print("Note:", note)
    for sim in analysis.simulations:
        print(
            f"Simulated stage {sim.to_stage}: SSIM {sim.ssim:.3f}, PSNR {sim.psnr:.1f} dB, "
            f"re-graded as stage {sim.regraded_stage}{'' if sim.regrade_matches else ' (does not match)'}"
        )

    if not args.no_save:
        prediction_id, report = save_run(db.connect(), analysis)
        print(f"Saved run {prediction_id}; report: {report}")
    print(config.DISCLAIMER)


if __name__ == "__main__":
    main()
