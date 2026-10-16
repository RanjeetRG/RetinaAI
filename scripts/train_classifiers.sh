#!/usr/bin/env bash
# Train the two EfficientNetB0 classifiers on the standardised images (M3), one after the other:
#   224 px - guides the GAN generators (gan.train --guide)
#   384 px - the app's classifier (--promote) and the independent judge of the generators
#     bash scripts/wsl_run.sh train_classifiers bash scripts/train_classifiers.sh
set -e
python -u -m cnn.train --backbone efficientnetb0 --size 224
RETINAGEN_ARRAY_DIR="$HOME/retinagen_arrays" python -u -m cnn.train --backbone efficientnetb0 --size 384 --batch-size 16 --promote
