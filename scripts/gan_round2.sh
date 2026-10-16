#!/usr/bin/env bash
# Round 2 of the guidance pilots (16 Oct 2026): does an anti-stamp diversity loss plus rotation augmentation stop
# the fixed-position "dot" shortcut? Trains the pilot, trains a ResNet50 judge (a different architecture from the
# EfficientNet guide), then scores every 1->2 variant with both judges into one table:
#     outputs/gan/_pilots/comparison_1to2.csv
#     bash scripts/wsl_run.sh gan_round2 bash scripts/gan_round2.sh
set -u
T=1to2
GUIDE="models/cnn_efficientnetb0_224.keras"
P="outputs/gan/_pilots"
NAME="lc0.1_id5_div1_rot"
FILTER="bfc_allocator|allocation summary|cuda_malloc_async|^I0000|^W0000|^E0000|oneDNN|absl"

if [ ! -f "${P}/${T}_${NAME}/generator_${T}.keras" ]; then
    echo "=== train pilot ${NAME} $(date -u +%FT%TZ)"
    python -u -m gan.train --transition "${T}" --steps 4000 --guide "${GUIDE}" --lambda-class 0.1 --lambda-identity 5.0 \
        --lambda-diversity 1.0 --rotate --sample-every 1000 --checkpoint-every 4000 2>&1 | grep --line-buffered -v -E "${FILTER}"
    mkdir -p "${P}/${T}_${NAME}"
    mv "outputs/gan/${T}"/* "${P}/${T}_${NAME}/" && rmdir "outputs/gan/${T}"
    rm -rf "${P}/${T}_${NAME}/checkpoints"
    mv "models/generator_${T}.keras" "models/generator_${T}.json" "${P}/${T}_${NAME}/"
fi

if [ ! -f models/cnn_resnet50_224.keras ]; then
    echo "=== train ResNet50 judge $(date -u +%FT%TZ)"
    python -u -m cnn.train --backbone resnet50 --size 224 2>&1 | grep --line-buffered -v -E "${FILTER}"
fi

OUT="${P}/comparison_${T}.csv"
for judge in models/model.keras models/cnn_resnet50_224.keras; do
    for pilot in lc0.1_id5 lc0.3_id5 "${NAME}"; do
        echo "=== evaluate ${pilot} with $(basename "${judge}") $(date -u +%FT%TZ)"
        python -u -m gan.gan_metrics --transitions "${T}" --generator "${P}/${T}_${pilot}/generator_${T}.keras" \
            --method "${pilot}" --judge "${judge}" --out "${OUT}" 2>&1 | grep -v -E "${FILTER}" | tail -3
    done
done
echo "=== round 2 done $(date -u +%FT%TZ)"
