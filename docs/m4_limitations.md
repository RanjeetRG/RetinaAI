# Milestone 4: transition generators, known limitations

A working list for the paper's limitations section. Each item says what was observed, what it affects and what was done about it. Results get added as each transition finishes.

## 1. Simulations are not patient-specific predictions

There are no follow-up photographs of the same eye, so each generator learns what stage k+1 looks like *in general* and how to add that to a stage-k image. Training is unpaired (CycleGAN-style): cycle-consistency keeps the input's vessels and optic disc.

- The output is a plausible next-stage appearance for this retina. It is not a forecast of this patient's disease, and it says nothing about when progression would happen.
- The app labels every generated image "SIMULATED" and shows the disclaimer (FR-15).

## 2. The generator painted outside the retina, fixed by an output mask

- **Observed (14 Oct 2026):** in the first 0→1 run, from step 10,000 onwards, the generator painted saturated blobs in the black area outside the fundus: 140–500 bright pixels in 3 of 4 validation previews, through step 21,000.
- **Not from the data:** the training images have no such marks. In the corner regions outside the disc, 0% of stage 0–3 images and 0.2% of stage-4 images contain bright pixels.
- **Likely cause:** the empty surround is an easy place for a CycleGAN generator to put content, or to hide information that helps the reverse generator satisfy the cycle loss (Chu et al., 2017, "CycleGAN, a master of steganography").
- **Fix:** a `FundusMask` layer at the end of the generator keeps generated pixels inside the input's fundus and forces the surround to black. It is part of the saved model, so training, evaluation and the app all use it.
- **Status:** the run was discarded and all four transitions were restarted with the mask. Its previews and loss history are kept in `outputs/gan/_discarded/0to1_v1_unmasked_output/`.
- **Remaining risk:** the generator could still hide such information as faint, high-frequency patterns *inside* the retina. The difference maps and the classifier re-grade are the checks for that.

## 3. The generator learned to clip the field, fixed by a standard field of view

- **Observed (14 Oct 2026):** the second 0→1 run, with the output mask, finished all 40,000 steps. Its final preview showed it cutting off the top and bottom of a full-circle retina, with almost no visible lesion change elsewhere.
- **Cause:** a field-shape confound in the data. Clipped camera fields make up 46.6% of stage-0 training images, against 55.4% at stage 1 and 65.9% at stage 4 (`docs/data_notes.md`). Producing a clipped field was an easy way to look more like stage 1.
- **Fix:** every model input, for both the generators and the classifier, is now cut to the same standard field. The run is kept as evidence in `outputs/gan/_discarded/0to1_v2_field_shape/`, generator included.
- **Status:** all four transitions were retrained on the standardised images.

## 4. Plain CycleGAN barely changed the images, so the generators are now classifier-guided

- **Observed (15 Oct 2026):** at step 20,000 the standardised-field 1→2 generator was close to an identity map.
  - Global shift was 0.3–0.7 and local change 1.5–1.8 (0–255), with no visible lesions added.
  - Both discriminators sat at the LSGAN chance value of 0.25: they could not tell real stage-2 images from lightly edited stage-1 images.
- **Cause:** lesions occupy few of the 70×70 patches a PatchGAN judges, so the adversarial signal is weak. The identity (λ=5) and cycle (λ=10) losses then make "copy the input" nearly optimal.
- **Change:**
  - A frozen 224 px EfficientNetB0 classifier adds a cross-entropy loss (λ=1) rewarding G's output for being graded stage k+1, and F's output for being graded stage k.
  - The identity weight is lowered to 0.5.
- **Safeguard against "fooling the guide":**
  - The adversarial and cycle losses stay in place.
  - Stage consistency is judged by a *different* classifier, the 384 px model.
  - Outputs are inspected visually and decomposed into global versus local change.
- **Baseline:** the plain-CycleGAN 0→1 generator (`models/baseline_cyclegan/`) is the CycleGAN baseline the PRD asks for. The half-trained 1→2 baseline was dropped to save time.

### Guidance pilots (1→2, 4,000 steps each, 16 Oct 2026)

Each pilot was scored on the 404 held-out stage-1 test images, with two judges that never guided training: the 384 px EfficientNetB0 and a 224 px ResNet50. The table is `outputs/gan/_pilots/comparison_1to2.csv`.

| Variant | SSIM | FID (no change: 24.8) | Stage 2 according to EffNet-384 / ResNet50 | Change correlation across eyes | What it does |
|---|---|---|---|---|---|
| No change | 1.00 | 24.8 | 21% / 11% | n/a | n/a |
| Guidance 1.0, identity 0.5 | 0.31 | 65.3 | 89% / n/a | n/a | A fixed "stamp" plus heavy colour distortion |
| Guidance 0.1, identity 5 | 0.96 | 30.7 | 92% / 87% | 0.20 | One red dot at the same spot in every eye |
| Guidance 0.3, identity 5 | 0.94 | 36.0 | 71% / 92% | 0.36 | Two red dots at fixed spots |
| Guidance 0.1 + diversity 1.0 + rotation | 0.96 | 35.2 | 39% / 47% | **0.13** | No fixed dots; a faint texture, no clear lesions |

**Findings:**
- **Guidance finds the cheapest change that convinces a classifier:** a haemorrhage-like red dot placed at the same position in every eye. A different-architecture judge (ResNet50) is convinced too, so the dot is a plausible DR feature rather than an invisible adversarial pattern. Its fixed position makes it a shortcut, not eye-specific progression.
- **The anti-stamp diversity loss and rotation remove the fixed dots** (lowest change correlation). At 4,000 steps the generator answers with a faint texture, which convinces the judges less and lowers realism.
- **The final generators use the anti-stamp setting**, trained for 20,000 steps per transition (`scripts/train_gans.sh`). The pilots form the ablation table.

## 5. Camera differences can masquerade as disease

In APTOS, 65% of stage-0 photographs come from a lower-resolution camera setup, against 4–7% for stages 1–4 (`docs/data_notes.md`). A generator can lower its adversarial loss by changing camera characteristics (sharpness, colour cast, field shape) instead of adding lesions.

- **Observed in the standardised-field 0→1 generator (14 Oct 2026):**
  - On its four validation previews, its main change is **global darkening of bright, yellowish photos**: about 10 units (0–255) across the whole retina, mostly red and green.
  - Darker photos are left nearly unchanged (about 1.5 units).
  - The stage pools do differ slightly in exposure and colour. Mean brightness is 112 at stage 0 against 108 at stage 1, and blue falls from 74 at stage 0 to 63 at stage 4. That is only about 0.15 of the image-to-image spread, but the generator exaggerates it on bright images.
  - `gan_metrics` now reports `global_shift` and `local_change` for every transition, so the paper can say how much of each generator's change is exposure and how much is local, lesion-like change. A pure exposure change gives local ≈ 0; a purely local change gives local ≈ 2× global.
- **Mitigations:**
  - EyePACS makes up about 90% of the training images.
  - Training uses only geometric augmentation, so colour shifts are never presented as part of a stage.
  - Mean colour and sharpness of outputs will be compared against inputs, not only lesion appearance.

## 6. Very little data at the severe end

After quality filtering, the 2→3 and 3→4 transitions have 696 stage-3 and 618 stage-4 training images. Because every transition gets the same 40,000 steps at batch 4, these generators see each target image about 230–260 times, against about 80 times for 0→1.

- The 2→3 and 3→4 generators may therefore overfit.
- Their FID is also less reliable: the reference sets are small, with 299 stage-3 and 283 stage-4 images across validation and test. KID is reported alongside FID because it is unbiased at small sample sizes.

## 7. Mild DR is hard to see at 256 px

Stage 1 is defined by microaneurysms only a few pixels across at 256 px. The 0→1 changes are expected to be subtle, and the classifier itself recognises mild DR poorly (recall 0.21 on the test set).

- So the stage-consistency check for 0→1 is limited by the classifier's own accuracy.
- Report the classifier's per-class recall next to stage consistency, so readers can separate generator failure from classifier failure.

## 8. Choices that limit comparison with other work

- **Generator width:** 32 base filters (9.8M parameters) instead of Pix2Pix's 64. This was chosen for training speed (25 against 8 images/s on the RTX 4060) and for deployment size. A width-64 ablation would show what the narrower model costs.
- **FID features:** computed with Keras' ImageNet InceptionV3. The numbers compare our own models and baselines; they cannot be compared with FIDs published using the original TF-Inception weights.
