# Training RetinaAI from scratch

You don't need this to run the app: the trained models are already in `models/`. Use this guide to rebuild the
dataset, retrain the classifier or the transition generators, or reproduce the evaluation tables.

Training was done on one laptop GPU (RTX 4060, 8 GB): about 1.5 h for the 384 px classifier and 30–60 min per
generator (10,000 steps each).

## 1. Environments

There are two environments, both on Python 3.12. They pin the same TensorFlow and Keras versions, so a model saved
in one loads in the other.

**Windows `.venv`:** the app, CPU inference, data preparation and tests.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**WSL2 Ubuntu:** GPU training. TensorFlow has had no GPU support on native Windows since version 2.10. On Linux,
use the same steps without WSL.

```bash
mkdir -p ~/venvs
curl -sSLo ~/venvs/virtualenv.pyz https://bootstrap.pypa.io/virtualenv.pyz
python3 ~/venvs/virtualenv.pyz ~/venvs/retinagen      # standalone virtualenv; no sudo needed
cd /mnt/d/path/to/RetinaAI && ~/venvs/retinagen/bin/pip install -r requirements-train.txt
```

At the start of every session:

```bash
source /mnt/d/path/to/RetinaAI/scripts/wsl_env.sh   # activates the venv, sets the CUDA library path, cd's to the repo
python -m scripts.check_gpu                          # should list your GPU
```

## 2. Data

| Dataset | Used for | Download |
|---|---|---|
| APTOS 2019 Blindness Detection | Training, validation and test | [Kaggle competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection) (accept the rules first) |
| EyePACS, pre-resized copy | Training, validation and test | [Kaggle: tanlikesmath/diabetic-retinopathy-resized](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) |
| Messidor-2 | External test only | Images: [ADCIS license form](https://www.adcis.net/en/third-party/messidor2/). Grades: [Kaggle: google-brain/messidor2-dr-grades](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) |

1. Create a Kaggle API token (Kaggle → Settings → API Tokens → *Generate New Token*).
2. Save the token as plain text in `~/.kaggle/access_token`. On Windows that is `%USERPROFILE%\.kaggle\access_token`, with no file extension.
3. Run:

```powershell
.venv\Scripts\python -m scripts.download_data       # APTOS + EyePACS into dataset/raw/
.venv\Scripts\python -m preprocessing.build_dataset # 512 px cache, de-duplication, patient-level 70/15/15 split
```

The split used for the published models is recorded in `dataset/manifests/splits.csv`. Counts, quality checks
and the APTOS camera confound are covered in [data_notes.md](data_notes.md).

**Licenses:** never commit or redistribute dataset images. Messidor-2 is free for research and education only. Any
publication that uses it must acknowledge the LaTIM laboratory and the Messidor program partners, and cite:
- Decencière et al., *Image Analysis & Stereology* 33(3):231–234, 2014. doi:10.5566/ias.1155
- Abràmoff et al., *JAMA Ophthalmology* 131(3):351–357, 2013. doi:10.1001/jamaophthalmol.2013.1743
- Krause et al., *Ophthalmology* 125(8):1264–1272, 2018, for the adjudicated grades

## 3. Classifier

Each split is preprocessed once into arrays. At 384 px the microaneurysms of mild DR stay visible.

```powershell
.venv\Scripts\python -m cnn.data                  # 224 px arrays into dataset/arrays/ (~6 GB)
```

The 384 px training array (about 12 GB) is memory-mapped during training. Build it on WSL's own disk, because
random reads through `/mnt/d` are slow:

```bash
export RETINAGEN_ARRAY_DIR=~/retinagen_arrays
python -m cnn.data --size 384 --splits train validation test
```

Train one backbone at a time (`efficientnetb0`, `resnet50`, `densenet121`, `mobilenetv2`). `--promote` also saves
the model as `models/model.keras`, the classifier the app uses.

```bash
python -m cnn.train --backbone efficientnetb0 --size 384 --promote   # the app's classifier
python -m cnn.train --backbone efficientnetb0 --size 224             # guide for the generators
python -m scripts.bench_cnn --size 384 --batch-size 16               # training-speed check
```

Long runs can be started from Windows as a detached process that logs to `outputs/<name>.log`:

```powershell
Start-Process wsl.exe -ArgumentList "-d","Ubuntu","--","bash","/mnt/d/path/to/RetinaAI/scripts/wsl_run.sh","train_efficientnetb0","python","-u","-m","cnn.train","--backbone","efficientnetb0" -WindowStyle Hidden
```

Evaluation reads each model's input size from the model. It writes metrics, per-image predictions and
`outputs/eval/cnn/summary.csv`:

```powershell
.venv\Scripts\python -m cnn.evaluate
```

## 4. Transition generators

There is one generator per transition: 0→1, 1→2, 2→3 and 3→4. They are CycleGAN-style models trained without
paired photos and guided by the frozen 224 px classifier. Training also uses an anti-stamp diversity loss and
rotation augmentation. [m4_limitations.md](m4_limitations.md) explains these choices and shows the pilot results.

```powershell
.venv\Scripts\python -m gan.data                                  # 256 px GAN arrays per split and stage
```

```bash
bash scripts/wsl_run.sh train_gans bash scripts/train_gans.sh     # all four, resumable from checkpoints
python -m gan.gan_metrics                                         # FID, KID, SSIM, stage consistency
python -m gan.generate                                            # sample grids in outputs/eval/gan/
```

## 5. Evaluation tables, end-to-end check and tests

```powershell
.venv\Scripts\python -m scripts.paper_outputs   # outputs/eval/paper_tables.md + confusion matrices
.venv\Scripts\python -m scripts.e2e_check       # 20 held-out images through the full app pipeline, timed on the CPU
.venv\Scripts\python -m pytest                  # fast tests
.venv\Scripts\python -m pytest -m slow          # only the slow tests, including the CNN training smoke test
```
