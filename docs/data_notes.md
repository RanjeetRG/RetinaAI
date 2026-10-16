# Dataset notes (M2)

How the training data was built, and what to report in the paper's dataset section. The numbers come from `python -m preprocessing.build_dataset --sources aptos eyepacs`, run on 13 October 2026 with seed 42.

## Final dataset

| Split | Source | No DR | Mild | Moderate | Severe | Proliferative | Total |
|---|---|---|---|---|---|---|---|
| train | APTOS | 1,257 | 237 | 645 | 117 | 185 | 2,441 |
| train | EyePACS | 18,051 | 1,699 | 3,714 | 609 | 484 | 24,557 |
| validation | APTOS | 269 | 51 | 138 | 25 | 39 | 522 |
| validation | EyePACS | 3,852 | 376 | 788 | 133 | 110 | 5,259 |
| test | APTOS | 270 | 50 | 138 | 25 | 40 | 523 |
| test | EyePACS | 3,878 | 363 | 777 | 130 | 113 | 5,261 |
| **total** | | **27,577** | **2,776** | **6,200** | **1,039** | **971** | **38,563** |

The machine-readable version is `outputs/eval/data/class_counts.csv`. Every collected image appears in `dataset/manifests/all_images.csv`, with its split or its reason for exclusion.

Messidor-2 (the external test set) is not built yet; the images are awaiting ADCIS approval.

## Pipeline

1. **Sources.**
   - APTOS 2019 training set: 3,662 images; no patient IDs are published, so each image is treated as its own patient.
   - EyePACS (Kaggle "diabetic-retinopathy-resized", 35,126 images, long side at most 1,024 px). The patient is the number before `_left`/`_right`.
2. **Caching.** `load_fundus` crops the black border, pads the image to a square and resizes it to 512 px; the result is stored as JPEG at quality 95.
3. **Unusable images.** 22 EyePACS images were dropped as blank, or as having almost no illuminated area.
4. **Near-duplicates.**
   - Method: a 256-bit perceptual hash of the central 70% of the green channel. Images at a Hamming distance of 20 or less are grouped.
   - Groups whose images all share one grade keep their sharpest image.
   - Groups with conflicting grades are dropped entirely.
5. **Split.**
   - 70/15/15, at the patient level, stratified by source and the patient's worst stage.
   - Both eyes of an EyePACS patient always land in the same split.
   - `tests/test_manifests.py` checks that no patient appears in two splits.

### Duplicates found

| Source | Kept one copy (removed) | Conflicting grades (all removed) |
|---|---|---|
| APTOS | 97 | 79 |
| EyePACS | 17 | 10 |

- **APTOS.** All 148 duplicate pairs are at 0–2 bits, meaning re-encoded copies of the same file. They concentrate in the rare stages: 41 of the 79 conflicting-label images are stage 3 or 4.
- **EyePACS.** Pairs are at 14–34 bits; distances above 20 come from chained groups. They are of two kinds:
  - **Re-takes of the same eye filed under different patient IDs, some with different grades.** For example, `26051_right` (grade 1) and `3754_right` (grade 2), and `17163_right` (grade 0) and `19840_right` (grade 2). Leaving these in would leak images between train and test.
  - **Featureless, washed-out or very dark images.** They have too little structure for the hash to tell them apart. A vessel-detail correlation check did not separate them from the true re-takes either.

  In total 27 EyePACS images (0.08%) were removed, mostly stage 0, so the false matches cost almost nothing.

## Quality checks (FR-2)

The thresholds in `preprocessing/quality.py` were calibrated on APTOS by looking at images at each percentile:

- **Blur:** Laplacian variance of the green channel below 6 flags 1.9% of images. The provisional threshold of 15 flagged 30%, including clearly sharp images.
- **Colour:** red/blue ratio below 0.85. Real fundus photos go down to 0.97, because some cameras give a grey-blue cast.
- **Not a fundus photo:** the four corners of the illuminated area are at least 99% lit. An ordinary photo scores about 1.0; 3 APTOS photos cropped tightly on every side also trigger it.

These checks only produce warnings in the app. In the manifests, `gan_ok` marks images with no warnings at all, and GAN training can filter on it.

## Known confound: camera and resolution versus stage in APTOS

| APTOS stage | Images ≤ 1,100 px wide | Images > 2,000 px wide |
|---|---|---|
| 0 | 65% | 34% |
| 1 | 7% | 89% |
| 2 | 7% | 83% |
| 3 | 4% | 90% |
| 4 | 6% | 81% |

Most healthy APTOS eyes were photographed with a different camera setup from the diseased ones. After resizing, the pixel count is hidden, but camera traits such as sharpness, colour cast and field shape remain. This creates two risks:

- The CNN could partly learn the camera rather than the lesions.
- A transition generator could learn to change the camera style instead of adding lesions.

Mitigations:

- EyePACS, which has no such pattern, is 90% of the training data.
- M3 reports metrics separately for APTOS and EyePACS test images.
- Messidor-2 provides an external test on unseen cameras.
- GAN outputs are checked visually for global colour or sharpness shifts.

## Known confound: field shape versus stage (both datasets)

Some cameras clip the circular field at the top and bottom. In the GAN training pool, the share of clipped fields (height below 92% of the width) is:

| Stage | Clipped field | Median height/width |
|---|---|---|
| 0 | 46.6% | 0.94 |
| 1 | 55.4% | 0.86 |
| 2 | 54.5% | 0.84 |
| 3 | 52.7% | 0.86 |
| 4 | 65.9% | 0.81 |

The first 0→1 generator exploited this: it learned to cut off the top and bottom of full-circle retinas instead of adding lesions (`outputs/gan/_discarded/0to1_v2_field_shape/`). A classifier can use the same cue.

**Fix (14 Oct 2026):**
- Every model input is cut to one standard field of view: a circle the full width of the image, clipped to a horizontal band 0.75 of its diameter tall (`config.FIELD_HEIGHT_RATIO`, `preprocessing.fundus.field_template`).
- Only 1.9% of images are clipped more than the template, so field shape no longer separates the stages.
- On full-circle images this removes the top and bottom 14% of the retina, the periphery; the optic disc and macula sit on the horizontal centre line and are kept.
- This is the same idea as the widely used Kaggle DR preprocessing that crops every image to a fixed circle.
- `tests/test_preprocessing.py` checks that a full and a clipped field of the same eye come out with the same shape.

## For the paper

- Cite APTOS 2019 (Kaggle), EyePACS (Kaggle 2015 Diabetic Retinopathy Detection) and Messidor-2. The Messidor-2 acknowledgment and its two required citations are in the README.
- Report the duplicate removal and the patient-level split. Many published APTOS results do neither, so their test scores may be inflated.
- Report the camera confound above as a limitation.
