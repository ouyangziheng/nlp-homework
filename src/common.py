"""Shared utilities: data loading, the canonical split, tokenization cache, metrics.

Every task imports the split from here so that Task 1 / 2 / 3 are trained and
evaluated on exactly the same documents, which is what makes the numbers in the
report comparable.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

HW_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = HW_ROOT / "data"
RESULTS_DIR = HW_ROOT / "results"
CACHE_DIR = DATA_DIR / "cache"

NYT_CSV = HW_ROOT / "HW-1" / "nyt.csv"
AG_CSV = HW_ROOT / "HW-1" / "ag.csv"
GLOVE_TXT = DATA_DIR / "glove" / "glove.6B.100d.txt"

# nltk needs to find the punkt model we vendored into data/nltk_data.
os.environ.setdefault("NLTK_DATA", str(DATA_DIR / "nltk_data"))

SEED = 42
EMBED_DIM = 100
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train / validation / test, per the assignment


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Data loading and the canonical split
# --------------------------------------------------------------------------- #
def load_nyt() -> pd.DataFrame:
    df = pd.read_csv(NYT_CSV)
    if list(df.columns) != ["text", "label"]:
        raise ValueError(f"nyt.csv should have columns [text, label], got {list(df.columns)}")
    df["text"] = df["text"].astype(str)
    return df


def load_ag() -> pd.DataFrame:
    df = pd.read_csv(AG_CSV)
    df["text"] = df["text"].astype(str)
    return df


def split_nyt(df: pd.DataFrame, seed: int = SEED) -> Dict[str, pd.DataFrame]:
    """Shuffle the whole dataset once, then cut 80/10/10.

    The assignment asks for a random shuffle (not a stratified split), so that is
    what we do; `split_report` below records the resulting class balance per split
    so the report can show the classes stayed proportionally represented.
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    return {
        "train": shuffled.iloc[:n_train].reset_index(drop=True),
        "val": shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True),
        "test": shuffled.iloc[n_train + n_val :].reset_index(drop=True),
    }


def get_splits() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Return the canonical split plus the sorted label vocabulary."""
    set_seed()
    df = load_nyt()
    splits = split_nyt(df)
    labels = sorted(df["label"].unique())
    return splits, labels


def encode_labels(series: pd.Series, labels: Sequence[str]) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(labels)}
    return series.map(idx).to_numpy()


def split_report(splits: Dict[str, pd.DataFrame], labels: Sequence[str]) -> dict:
    out = {"labels": list(labels), "sizes": {}, "distribution": {}}
    for name, part in splits.items():
        out["sizes"][name] = len(part)
        counts = part["label"].value_counts()
        out["distribution"][name] = {lab: int(counts.get(lab, 0)) for lab in labels}
    return out


# --------------------------------------------------------------------------- #
# Tokenization (nltk word_tokenize, cached because it dominates runtime)
# --------------------------------------------------------------------------- #
def _tokenize_corpus(texts: Sequence[str]) -> List[List[str]]:
    from nltk import word_tokenize

    out = []
    for i, t in enumerate(texts):
        out.append(word_tokenize(t.lower()))
        if (i + 1) % 20000 == 0:
            log(f"  tokenized {i + 1}/{len(texts)}")
    return out


def tokenized_nyt() -> List[List[str]]:
    """Tokens for every NYT document, in the original csv row order."""
    cache = CACHE_DIR / "nyt_tokens.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log("tokenizing NYT with nltk.word_tokenize (cached afterwards)")
    toks = _tokenize_corpus(load_nyt()["text"].tolist())
    with cache.open("wb") as fh:
        pickle.dump(toks, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return toks


def tokenized_ag() -> List[List[str]]:
    cache = CACHE_DIR / "ag_tokens.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log("tokenizing AG News with nltk.word_tokenize (cached afterwards)")
    toks = _tokenize_corpus(load_ag()["text"].tolist())
    with cache.open("wb") as fh:
        pickle.dump(toks, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return toks


def tokens_for_splits() -> Dict[str, List[List[str]]]:
    """Tokens aligned to the canonical split.

    `split_nyt` shuffles with a fixed seed and keeps the original index available
    via `sample`, but we reset it, so we re-derive the permutation here with the
    same seed to map rows back onto the cached token list.
    """
    set_seed()
    df = load_nyt()
    all_tokens = tokenized_nyt()
    order = df.sample(frac=1.0, random_state=SEED).index.to_numpy()
    n = len(df)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    bounds = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    return {name: [all_tokens[i] for i in idx] for name, idx in bounds.items()}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_f1": {
            lab: float(v)
            for lab, v in zip(labels, f1_score(y_true, y_pred, average=None, labels=range(len(labels))))
        },
        "report": classification_report(
            y_true, y_pred, target_names=list(labels), digits=4, zero_division=0
        ),
    }


def save_result(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log(f"saved {path}")
    return path
