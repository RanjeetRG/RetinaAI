#!/usr/bin/env bash
# Short classifier-guidance pilots on one transition (default 1to2), to choose the guidance strength before the
# full four-transition run. Each pilot trains 4,000 steps, is evaluated by gan.gan_metrics (the judge is the 384 px
# classifier, models/model.keras, not the 224 px guide), and is then moved to outputs/gan/_pilots/<name>/ so the
# real run starts clean.
#     bash scripts/wsl_run.sh gan_pilots bash scripts/gan_pilots.sh
#
# A pilot whose generator already exists in models/ (trained earlier) is evaluated and archived without retraining.
set -u
T="${TRANSITION:-1to2}"
GUIDE="models/cnn_efficientnetb0_224.keras"

archive() {  # $1 = pilot name
    local dest="outputs/gan/_pilots/${T}_$1"
    mkdir -p "${dest}"
    mv "outputs/gan/${T}"/* "${dest}/" 2>/dev/null; rmdir "outputs/gan/${T}" 2>/dev/null
    rm -rf "${dest}/checkpoints"
    mv "models/generator_${T}.keras" "models/generator_${T}.json" "${dest}/"
    mv outputs/eval/gan/metrics.csv outputs/eval/gan/metrics.json "${dest}/"
    echo "=== archived ${dest}"
}

pilot() {  # $1 = name, $2 = lambda-class, $3 = lambda-identity
    if [ ! -f "models/generator_${T}.keras" ]; then
        echo "=== train $1 (lambda-class $2, lambda-identity $3) $(date -u +%FT%TZ)"
        python -u -m gan.train --transition "${T}" --steps 4000 --guide "${GUIDE}" --lambda-class "$2" \
            --lambda-identity "$3" --sample-every 1000 --checkpoint-every 4000 2>&1 \
            | grep --line-buffered -v -E "bfc_allocator|allocation summary|cuda_malloc_async|^I0000|^W0000"
    fi
    echo "=== evaluate $1 $(date -u +%FT%TZ)"
    python -u -m gan.gan_metrics --transitions "${T}" 2>&1 | grep -v -E "^I0000|^W0000|^E0000|oneDNN|absl"
    archive "$1"
}

pilot lc1.0_id0.5 1.0 0.5
pilot lc0.1_id5 0.1 5.0
pilot lc0.3_id5 0.3 5.0
echo "=== pilots done $(date -u +%FT%TZ)"
