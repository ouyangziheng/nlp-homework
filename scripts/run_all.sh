#!/usr/bin/env bash
# Run the three tasks, then the comparative analysis, figures and the report PDF.
# Assumes scripts/setup.sh and scripts/download_assets.sh have already run.
set -euo pipefail

HW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HW_DIR"

export PYTHONPATH="$HW_DIR/src"
export NLTK_DATA="$HW_DIR/data/nltk_data"
export HF_HOME="$HW_DIR/data/hf_cache"
export HF_HUB_OFFLINE=1          # assets are already cached; do not hit the network
export TOKENIZERS_PARALLELISM=false
# gensim hashes words to assign their initial vectors, so a stable hash seed is
# needed alongside workers=1 for Word2Vec to be exactly reproducible.
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY=.venv/bin/python
mkdir -p logs results

echo "############ Task 1: Bag of Words ############"
$PY src/task1_bow.py 2>&1 | tee logs/task1.log

echo
echo "############ Task 2: Word2Vec / GloVe ############"
$PY src/task2_word2vec.py 2>&1 | tee logs/task2.log

echo
echo "############ Task 3: BERT fine-tuning ############"
$PY src/task3_bert.py 2>&1 | tee logs/task3.log

echo
echo "############ Comparative analysis ############"
$PY src/analysis.py 2>&1 | tee logs/analysis.log

echo
echo "############ Figures ############"
$PY src/make_figures_cvpr.py 2>&1 | tee logs/figures.log

echo
echo "############ Report PDF ############"
if command -v xelatex >/dev/null; then
    cd report_cvpr
    # xelatex is required: the report is in Chinese and uses xeCJK.
    xelatex -interaction=nonstopmode main.tex > ../logs/cvpr_latex.log 2>&1 || true
    bibtex main                               >> ../logs/cvpr_latex.log 2>&1 || true
    xelatex -interaction=nonstopmode main.tex >> ../logs/cvpr_latex.log 2>&1 || true
    xelatex -interaction=nonstopmode main.tex >> ../logs/cvpr_latex.log 2>&1 || true
    cd ..
    if [[ -f report_cvpr/main.pdf ]]; then
        echo "  wrote report_cvpr/main.pdf"
    else
        echo "  PDF build failed; see logs/cvpr_latex.log"
    fi
else
    echo "  xelatex not found; skipping PDF (report_cvpr/main.tex is still there)"
fi

echo
echo "############ Done ############"
echo "Results:  results/*.json"
echo "Report:   report_cvpr/main.pdf"
