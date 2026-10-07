#!/usr/bin/env bash
# Create the virtualenv and install the two packages the assignment needs on top
# of an existing torch / transformers / scikit-learn install.
#
# If you are NOT on the machine this was developed on, point BASE_PYTHON at any
# python >= 3.9 that has torch, transformers and scikit-learn available, or drop
# --system-site-packages below and let pip install everything.
set -euo pipefail

HW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HW_DIR"

# A python whose sqlite3 module actually loads -- nltk imports sqlite3 at import
# time, and some conda envs on this cluster ship a broken _sqlite3.so.
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/envs/torch-base/bin/python}"

if [[ -n "${USE_PROXY:-1}" && -z "${http_proxy:-}" ]]; then
    export http_proxy=http://star-proxy.oa.com:3128
    export https_proxy=http://star-proxy.oa.com:3128
fi

echo "==> creating venv with $BASE_PYTHON"
"$BASE_PYTHON" -m venv --system-site-packages .venv

echo "==> installing gensim + nltk"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q gensim nltk

echo "==> fetching nltk punkt tokenizer into data/nltk_data"
mkdir -p data/nltk_data/tokenizers
for pkg in punkt punkt_tab; do
    if [[ ! -d "data/nltk_data/tokenizers/$pkg" ]]; then
        curl -sSL -o "data/nltk_data/tokenizers/$pkg.zip" \
            "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/$pkg.zip"
        .venv/bin/python -c "import zipfile; zipfile.ZipFile('data/nltk_data/tokenizers/$pkg.zip').extractall('data/nltk_data/tokenizers')"
        echo "    $pkg ready"
    else
        echo "    $pkg already present"
    fi
done

echo "==> verifying"
NLTK_DATA="$HW_DIR/data/nltk_data" .venv/bin/python - <<'PY'
import gensim, nltk, sklearn, torch, transformers
from nltk import word_tokenize
print("  gensim", gensim.__version__, "| nltk", nltk.__version__)
print("  sklearn", sklearn.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__)
print("  tokenizer:", word_tokenize("A quick check of the U.S. tokenizer."))
print("  cuda:", torch.cuda.is_available())
PY

echo "==> setup complete"
