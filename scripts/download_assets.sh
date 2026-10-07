#!/usr/bin/env bash
# Download the two pre-trained assets: glove.6B (100d is extracted) and
# bert-base-uncased.
set -euo pipefail

HW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HW_DIR"

if [[ -n "${USE_PROXY:-1}" && -z "${http_proxy:-}" ]]; then
    export http_proxy=http://star-proxy.oa.com:3128
    export https_proxy=http://star-proxy.oa.com:3128
fi

mkdir -p data/glove

# The official Stanford host (http://nlp.stanford.edu/data/glove.6B.zip) is the
# canonical URL from the assignment, but it served ~70 KB/s from this cluster
# (3h for 822 MB). stanfordnlp/glove on HuggingFace mirrors the identical file
# at ~9 MB/s, so we pull from there and fall back to Stanford.
GLOVE_HF="https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_STANFORD="http://nlp.stanford.edu/data/glove.6B.zip"

if [[ ! -f data/glove/glove.6B.100d.txt ]]; then
    if [[ ! -f data/glove/glove.6B.zip ]]; then
        echo "==> downloading glove.6B.zip (822 MB)"
        curl -sSL -o data/glove/glove.6B.zip "$GLOVE_HF" \
            || curl -sSL -o data/glove/glove.6B.zip "$GLOVE_STANFORD"
    fi
    echo "==> extracting glove.6B.100d.txt"
    .venv/bin/python -c "
import zipfile
zipfile.ZipFile('data/glove/glove.6B.zip').extract('glove.6B.100d.txt', 'data/glove')
"
else
    echo "==> glove.6B.100d.txt already present"
fi
wc -l data/glove/glove.6B.100d.txt

echo "==> downloading google-bert/bert-base-uncased"
HF_HOME="$HW_DIR/data/hf_cache" .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("google-bert/bert-base-uncased",
                      allow_patterns=["*.json", "*.txt", "*.safetensors"])
print("  cached at", p)
PY

echo "==> assets ready"
