#!/usr/bin/env bash
# Train the four transition generators one after another, with the same settings and step budget (M4).
# A finished transition is skipped and an interrupted one resumes from its checkpoint, so rerun to continue.
#
#     bash scripts/wsl_run.sh train_gans bash scripts/train_gans.sh
#
# Settings chosen by the 1->2 pilots of 15-16 Oct (outputs/gan/_pilots/comparison_1to2.csv):
#   - guidance 0.1 from the frozen 224 px classifier: plain CycleGAN barely changed the images; stronger guidance
#     made one or two dots at a fixed spot in every eye
#   - anti-stamp diversity loss 1.0 + random rotation: removes the fixed-position shortcut
#   - identity weight 5 (CycleGAN default); 10,000 steps per transition, the same for all four
# The 384 px classifier (models/model.keras) and ResNet50 stay independent, to judge the results.
GUIDE="models/cnn_efficientnetb0_224.keras"
if [ ! -f "${GUIDE}" ]; then
    echo "missing ${GUIDE}: train it first (scripts/train_classifiers.sh)"
    exit 1
fi
for t in 0to1 1to2 2to3 3to4; do
    if [ -f "models/generator_${t}.keras" ]; then
        echo "${t}: already trained, skipping"
        continue
    fi
    echo "=== ${t} started $(date -u +%FT%TZ)"
    # XLA's convolution autotuner logs harmless 'ran out of memory' warnings; keep them out of the summary log
    python -u -m gan.train --transition "${t}" --steps 10000 --guide "${GUIDE}" --lambda-class 0.1 \
        --lambda-identity 5.0 --lambda-diversity 1.0 --rotate 2>&1 \
        | tee "outputs/train_gan_${t}.log" \
        | grep --line-buffered -v -E "bfc_allocator|allocation summary|cuda_malloc_async|^I0000|^W0000"
    status=${PIPESTATUS[0]}
    if [ "${status}" -ne 0 ]; then
        echo "${t} failed with exit code ${status}"
        exit "${status}"
    fi
    echo "=== ${t} finished $(date -u +%FT%TZ)"
done
