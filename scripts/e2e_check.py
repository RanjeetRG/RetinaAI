"""End-to-end check on held-out test images (M7; PRD: "20 of 20 without errors", "grade + one simulation < 10 s").

    python -m scripts.e2e_check            # 20 test images, 4 per stage, on the CPU

Runs the full app pipeline on each image (grade, Grad-CAM, one simulated stage, re-grade, PDF report) and times it.
Runs go to a temporary database and folder, so the real history is untouched. Writes outputs/eval/e2e/results.csv.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # the PRD target is a CPU-only laptop

import argparse
import tempfile
import time
from pathlib import Path

import pandas as pd

import config
from database import db
from prediction import runs
from prediction.pipeline import Pipeline
from preprocessing.quality import validate_upload

TIME_LIMIT_S = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-stage", type=int, default=4)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    test = pd.read_csv(config.MANIFEST_DIR / "test.csv")
    picks = test.groupby("stage").sample(n=args.per_stage, random_state=args.seed)
    start = time.perf_counter()
    pipeline = Pipeline()
    load_s = time.perf_counter() - start

    rows = []
    # Scratch space on D: (C: is short of space); cleaned up afterwards
    with tempfile.TemporaryDirectory(dir=config.OUTPUTS_DIR, ignore_cleanup_errors=True) as tmp:
        runs.RUNS_DIR = Path(tmp) / "runs"
        config.REPORTS_DIR = Path(tmp) / "reports"
        conn = db.connect(Path(tmp) / "e2e.db")
        try:
            run_all(pipeline, conn, picks, rows)
        finally:
            conn.close()

    report(rows, load_s)


def run_all(pipeline, conn, picks, rows) -> None:
    for i, r in enumerate(picks.itertuples()):
        row = {"path": r.path, "true_stage": r.stage}
        try:
            path = config.ROOT / r.path
            image = validate_upload(path.name, path.read_bytes())
            t0 = time.perf_counter()
            analysis = pipeline.analyze(image, steps=1)
            row["analyse_s"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            runs.save_run(conn, analysis)
            row["save_and_report_s"] = time.perf_counter() - t1
            sim = analysis.simulations[0] if analysis.simulations else None
            row.update(
                graded=analysis.stage,
                simulated_to=sim.to_stage if sim else None,
                regraded=sim.regraded_stage if sim else None,
                note="; ".join(analysis.notes),
                ok=True,
            )
        except Exception as exc:  # the check counts failures rather than stopping
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        print(f"{i + 1:2d}/{len(picks)} stage {r.stage}: {'ok' if row['ok'] else row['error']}"
              f"{'' if not row['ok'] else f'  {row["analyse_s"]:.1f} s'}", flush=True)  # fmt: skip


def report(rows, load_s: float) -> None:
    results = pd.DataFrame(rows)
    out = config.EVAL_DIR / "e2e" / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.round(3).to_csv(out, index=False)
    ok = results["ok"].sum()
    timed = results.loc[results["ok"], "analyse_s"]
    print(f"\n{ok} of {len(results)} runs without errors (PRD target: all)")
    print(f"model loading (once): {load_s:.1f} s")
    print(f"grade + one simulation: median {timed.median():.1f} s, max {timed.max():.1f} s "
          f"({'meets' if timed.max() < TIME_LIMIT_S else 'MISSES'} the {TIME_LIMIT_S:.0f} s target)")  # fmt: skip
    print("wrote", out)


if __name__ == "__main__":
    main()
