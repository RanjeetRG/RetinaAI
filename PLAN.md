# RetinaAI — Build Plan

Started 13 October 2026 from an empty folder. The PRD lists milestones M1–M4 as done, but none of them are, so this plan covers all seven milestones from the beginning.

**Status (16 Oct 2026): v1 demo complete.** Run it with `run_app.bat` (see README).
- **Classifier:** EfficientNetB0 at 384 px. On the held-out test set (5,784 images) it reaches QWK 0.745, accuracy 80% and referable-DR AUC 0.916. At a screening cut-off chosen on validation, sensitivity is 89% with specificity 74%. Four comparison models (DenseNet121, ResNet50, MobileNetV2, EfficientNetB0 at 224 px) score lower; results are in `outputs/eval/paper_tables.md`.
- **Generators:** all four transitions are trained (classifier-guided, anti-stamp, 10,000 steps each).
- **End-to-end:** 20 of 20 held-out images ran without errors. Grading plus one simulation takes a median of 1.3 s on the CPU (PRD target: under 10 s). All 85 tests pass.
- **Not done:**
  - Messidor-2 external test (waiting for ADCIS)
  - Hugging Face deployment
  - the PRD's QWK ≥ 0.85 and macro-F1 ≥ 0.70 targets
  - the simulated lesions are subtle; see `docs/m4_limitations.md`

The notes below are the history of how it got here.

**Status (14 Oct 2026):**
- **M1 is done.** Both environments run TensorFlow 2.21 and Keras 3.15. WSL trains on the RTX 4060 (about 12 TFLOPS in float32, 28 in float16).
- **M2 is done except Messidor-2.**
  - 38,563 APTOS + EyePACS images are cached, de-duplicated and split 70/15/15 by patient.
  - The quality checks were calibrated on real images.
  - Details, counts and the APTOS camera confound are in [docs/data_notes.md](docs/data_notes.md).
  - Messidor-2 images are awaiting ADCIS approval.
- **M3 is in progress.**
  - EfficientNetB0 is trained: best validation QWK 0.705 (epoch 13 of 18, 106 minutes).
  - The first save step built EfficientNet without its ImageNet input-scaling layer. This made test scores look worse than they were (QWK 0.58) and has been fixed: training now rebuilds the model correctly and checks the saved copy predicts the same as the trained model before saving.
  - The repaired model scores test QWK 0.695 overall: 0.912 on APTOS and 0.645 on EyePACS. At a screening cut-off chosen on validation, referable sensitivity is 0.87 with specificity 0.71.
  - **Next (decided 14 Oct):**
    - After the GANs finish, fix the slow classifier training (about 3× slower than this GPU should manage).
    - Then retrain EfficientNetB0 at 384 px as the main model, since mild DR is nearly invisible at 224 px. The 224 px model stays as a comparison row.
    - This needs the input size to come from each model rather than one global `CNN_SIZE`, and the 384 px training array (about 12 GB) to be memory-mapped instead of loaded into WSL's RAM.
  - Still to train: ResNet50, DenseNet121 and MobileNetV2.
- **M4 is in progress.**
  - The code is written and tested.
  - The first 0→1 run painted blobs outside the retina from step 10,000, so it was discarded. The generator's output is now masked to the fundus.
  - The second 0→1 run finished but had learned to *clip the camera field* rather than add lesions: clipped fields are commoner at higher stages, 46.6% at stage 0 against 65.9% at stage 4. It was also discarded.
  - Every model input now gets one standard field of view (`config.FIELD_HEIGHT_RATIO = 0.75`).
  - The arrays were rebuilt and all four transitions restarted: 40,000 steps, batch 4, width 32, about 1.8 hours each.
  - **16 Oct: final settings chosen from pilots (`docs/m4_limitations.md`):** guidance 0.1, anti-stamp diversity loss 1.0, rotation augmentation, identity 5, and 20,000 steps per transition. The final run is `scripts/train_rest.sh`: DenseNet121 and MobileNetV2 baselines, then the four generators, about 6 hours.
  - **15 Oct: switched to classifier-guided generators.**
    - Plain CycleGAN came out near-identity: at step 20,000 of 1→2 the discriminators sat at the chance value 0.25.
    - The plain-CycleGAN runs are kept as the PRD's baseline: 0→1 finished (`models/baseline_cyclegan/`), 1→2 half-trained with checkpoints kept.
    - Guided training uses the frozen 224 px classifier, guidance weight 1.0 and identity weight 0.5 (`scripts/train_gans.sh`). It starts once the 224 px classifier is trained.
- **Classifier training speed is fixed (15 Oct).**
  - Augmentation moved from GPU layers to the CPU loader, and steps are XLA-compiled.
  - 224 px fine-tuning went from 323 to 69 ms per step (about 58 s per epoch instead of 280 s).
  - `scripts/train_classifiers.sh` trains the 224 px model (GAN guide) and then the 384 px model (`--promote`, the app's classifier and the independent judge of the GANs).
  - The 384 px arrays live on WSL's own disk (`~/retinagen_arrays`).
  - The classifier must also be retrained on standardised inputs; the 384 px retrain will cover this.
  - Known limitations are in [docs/m4_limitations.md](docs/m4_limitations.md).
- **M5 is mostly written.** The pipeline (grade, Grad-CAM, simulate and chain, re-grade, SSIM/PSNR/difference map), SQLite history, PDF report and `python -m prediction.run` are tested with stand-in models. They are waiting on the trained generators.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| GAN training (PRD open question 1) | **Unpaired, CycleGAN-style** training for each transition: a U-Net generator and a PatchGAN discriminator, trained with LSGAN adversarial loss, cycle-consistency loss (λ=10) and identity loss (λ=5). Only the forward generator ships, as `generator_{k}to{k+1}.keras`. | There are no follow-up photos of the same eye. Pix2Pix's L1 loss on pairs made from different eyes blurs the output or copies the other eye's anatomy. Cycle-consistency keeps the input retina's vessels and optic disc. |
| Generator upsampling | Resize then convolve (no transposed convolutions) | Avoids the checkerboard artifacts listed in the PRD's risks |
| GAN size and budget | Generator width 32 (9.8M parameters, 39 MB), batch 4, 40,000 steps for every transition | Measured on the RTX 4060: 25 images/s against 8 at the standard width 64 with batch 1. This gives 3× more training per hour, generators a quarter of the size for hosting and CPU inference, and equal budgets across transitions. |
| Classifier training | Class weights = (balanced weight)^0.5, a frozen-backbone phase then fine-tuning, mixed precision, early stopping on validation QWK | Full balanced weights over-predict the rare stages; the square root softens them |
| Datasets | APTOS 2019 and EyePACS (a pre-resized Kaggle copy) for training. Messidor-2 with Google's adjudicated grades as the external test set. | APTOS alone has only 193 stage-3 and 295 stage-4 images, too few to train a GAN |
| Python | 3.12 | Already installed on Windows and in WSL, and supported by TensorFlow. The PRD said 3.11. |
| Training hardware | This laptop's RTX 4060 (8 GB) through WSL2 Ubuntu 24.04 | TensorFlow has no GPU support on native Windows after version 2.10. WSL can already see the GPU. |
| Inference / app | Windows venv with CPU TensorFlow | Matches the PRD's requirement that inference runs on a CPU-only laptop |
| Chaining (open question 2) | Allow chaining up to stage 4 (FR-7) | Cheap to add once a single step works |
| PDF library (open question 7) | fpdf2 | Pure Python with no system dependencies |
| Grad-CAM (open question 6) | In v1 (FR-5) | Little work, and useful for the paper |
| Login (open question 4) | Single local user | The PRD lists login as a non-goal for v1 |
| Demo host (open question 5) | Hugging Face Spaces, confirmed in M7 | The free CPU tier has 16 GB RAM, enough for the CNN plus one generator |

### Image conventions (one code path for training and the app)

1. `load_fundus()` crops the black border, pads the image to a square and resizes it to 512 px. Training caches this 512 px image, and the app calls the same function.
2. CNN input: 512 → 224 px, light denoise, CLAHE, RGB values 0–255. Each backbone's own normalization is a layer **inside** the saved model, so the app and the training code cannot drift apart.
3. GAN input: 512 → 256 px, light denoise, CLAHE, scaled to [-1, 1]. The 256 px image always comes from the 512 px image, never upscaled from 224 px (a known PRD risk).

## Milestones

### M1 Environment
1. Git repo, `.gitignore`, the PRD's folder layout plus `tests/` and `scripts/`
2. `config.py` with paths, stage names, image sizes and seeds
3. Windows `.venv` (app and inference, CPU TensorFlow) with pinned `requirements.txt`
4. WSL venv (training, GPU TensorFlow) with `requirements-train.txt`

**Done when** both venvs import the same TensorFlow and Keras versions, and TensorFlow in WSL lists the RTX 4060.

### M2 Data
1. **Manual:** Kaggle API token, accept the APTOS competition rules, request Messidor-2 images (see "Manual steps")
2. `scripts/download_data.py` downloads into `dataset/raw/`
3. `preprocessing/fundus.py`: crop, pad, resize, denoise, CLAHE and normalize, used by both training and the app
4. `preprocessing/quality.py`: blank, dark, blurry and not-a-fundus checks (FR-2)
5. `preprocessing/build_dataset.py`:
   - caches every image at 512 px and drops unreadable ones
   - removes near-duplicates with a perceptual hash
   - makes a patient-level, stage-stratified 70/15/15 split (both eyes of an EyePACS patient stay in the same split)
   - keeps Messidor-2 as an external test set only
   - writes `dataset/{train,validation,test}/<stage>/` and manifest CSVs
6. `scripts/data_figures.py`: a sample grid per stage and a figure of the preprocessing steps. The class counts are in `outputs/eval/data/class_counts.csv`.

**Done when** the manifests exist, a test confirms no patient appears in two splits, and the class-count table is saved.

### M3 CNN classifier
1. `cnn/data.py`: each split preprocessed once into a uint8 array. Augmentation (flips, rotation, brightness and contrast) runs on the GPU as Keras layers, training only.
2. `cnn/model.py`: EfficientNetB0 plus the ResNet50, DenseNet121 and MobileNet baselines, all sharing one classification head
3. `cnn/train.py`:
   - two phases: first with the backbone frozen, then fine-tuned
   - class weights for the stage imbalance
   - early stopping on validation QWK
   - fixed seeds
4. `cnn/evaluate.py`: QWK, accuracy, macro precision/recall/F1, referable-DR sensitivity, per-class ROC-AUC and a confusion matrix, on the internal test split (overall and separately for APTOS and EyePACS, to expose the camera confound in `docs/data_notes.md`) and on Messidor-2
5. Train all four backbones. Save EfficientNetB0 as `models/model.keras`.

**Done when** the evaluation tables are in `outputs/eval/cnn/`. The targets are QWK ≥ 0.85, macro F1 ≥ 0.70 and referable sensitivity ≥ 0.90; any target that is missed gets the gap written down.

### M4 GAN transition generators
1. `gan/networks.py`: U-Net generator with resize-convolve upsampling and instance norm; 70×70 PatchGAN discriminator
2. `gan/cyclegan.py`: training step with the three losses and an image history pool
3. `gan/train.py --transition 0to1`:
   - the same step budget for every transition (a known PRD risk)
   - checkpoints, so a run can resume
   - a sample grid every N steps
4. Train `0to1`, `1to2`, `2to3` and `3to4`. This is the longest part of the project; plan on overnight runs.
5. `gan/generate.py`: sample grids for the paper
6. `gan/gan_metrics.py`, per transition:
   - SSIM and PSNR between input and output
   - FID and KID against real images of the target stage
   - stage consistency (the share of outputs the CNN grades as k+1)
   - baseline: the unchanged input image scored against the target stage
7. `docs/m4_limitations.md`

**Done when** all four `generator_*.keras` files load on Windows CPU and the per-transition metrics table exists.

### M5 Integration
1. `prediction/pipeline.py`:
   - loads the CNN once and lazy-loads only the generator for stage k
   - grades the image, simulates the next stage (optionally chaining up to stage 4) and re-grades the simulated image
   - computes SSIM/PSNR and a difference heatmap, plus Grad-CAM
   - returns a single result object
2. `utils/report.py`: fpdf2 PDF with the input, grade, probabilities, simulated images, metrics and disclaimer
3. `database/db.py`: SQLite tables `users`, `predictions` and `generated_images`, plus a `history` view

**Done when** `python -m prediction.run <image>` writes a PDF and the matching database rows.

### M6 Streamlit app
1. `app.py`, one page:
   - upload → validate → grade with probability chart and Grad-CAM
   - simulate, with a choice of how many steps to chain
   - compare side by side with the difference heatmap, SSIM/PSNR and re-grade
   - download the PDF
2. History tab that lists and reopens past runs (FR-14)
3. "Simulated" label on every generated image and a not-a-diagnosis disclaimer on every result (FR-15)
4. Models cached with `st.cache_resource`

**Done when** going from upload to report takes three clicks or fewer.

### M7 Test, optimize, deploy
1. pytest:
   - preprocessing gives identical results in training and the app
   - database operations
   - the pipeline runs end to end with tiny dummy models
2. End-to-end run on 20 held-out images
3. CPU timing against the 10-second target. Convert to float16 if size or speed needs it.
4. Check each dataset's license, then deploy to Hugging Face Spaces with sample images only

**Done when** 20 of 20 runs pass, the timings are recorded and the public URL works.

Each milestone saves its paper-ready tables and figures in `outputs/eval/`.

## End-of-project cleanup (after training is finished)

Kept on purpose until the project is complete, then candidates for deletion:

| Item | Size |
|---|---|
| `dataset\raw\aptos`, `dataset\raw\eyepacs` (original downloads, re-downloadable) | 9.3 GB |
| `outputs\gan\baseline_cyclegan\1to2\checkpoints` | 0.6 GB |
| `models\legacy_v0_before_standard_field` | 0.03 GB |
| `~/retinagen_arrays` inside WSL (384 px arrays) | 17 GB |
| `dataset\arrays` (224 px and GAN arrays) | 8.8 GB |
| `outputs\gan\_pilots`, `outputs\gan\_discarded` (keep whatever the paper cites) | small |

Removing files inside WSL does not shrink `D:\WSL\Ubuntu\ext4.vhdx`. Run `wsl --manage Ubuntu --set-sparse true` (after `wsl --shutdown`) so Windows reclaims the space.

## Manual steps

1. **Kaggle (blocks M2):**
   - Create an account, then Settings → API Tokens → *Generate New Token*, and save the token as plain text in `%USERPROFILE%\.kaggle\access_token` (no file extension).
   - Open the APTOS 2019 Blindness Detection competition page and accept its rules; downloads fail until you do.
2. **Messidor-2 images (needed by M3 evaluation):** request them from ADCIS through their license form. The adjudicated grades come from Kaggle.
3. **GitHub (by M7):** create an empty repo to push to.

## Changes to the PRD

- Milestone status: M1–M4 are not started.
- GAN approach: CycleGAN-style unpaired training replaces paired Pix2Pix. The generator and discriminator architectures stay the same.
- Python version: 3.12 instead of 3.11.
