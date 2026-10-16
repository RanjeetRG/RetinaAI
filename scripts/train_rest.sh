#!/usr/bin/env bash
# Final unattended training run: the two remaining baseline classifiers for the paper, then the four generators.
#     bash scripts/wsl_run.sh train_rest bash scripts/train_rest.sh
FILTER="bfc_allocator|allocation summary|cuda_malloc_async|^I0000|^W0000|^E0000|oneDNN|absl"
for backbone in densenet121 mobilenetv2; do
    if [ ! -f "models/cnn_${backbone}_224.keras" ]; then
        echo "=== classifier ${backbone} started $(date -u +%FT%TZ)"
        python -u -m cnn.train --backbone "${backbone}" --size 224 2>&1 | grep --line-buffered -v -E "${FILTER}"
    fi
done
bash scripts/train_gans.sh
echo "=== all training done $(date -u +%FT%TZ)"
