"""Task 2 — averaged 100-d word embeddings + Logistic Regression.

Three sources of 100-dimensional word vectors:

  1. glove   — pre-trained glove.6B.100d
  2. ag      — Word2Vec trained by us on the 90k AG News texts
  3. nyt     — Word2Vec trained by us on the NYT texts

A document is represented as the mean of the vectors of its in-vocabulary
words:  e(d) = (1/n) * sum_i e(w_i).  Documents with no in-vocabulary word fall
back to the zero vector (counted and reported, since it is a real failure mode
of this representation rather than something to hide).

To avoid train/test leakage the NYT Word2Vec model is trained on the *training
split only*, not the full NYT corpus — the test documents never contribute to
the vectors. Features are standardized before the Logistic Regression because
averaged embeddings are dense and unscaled, which otherwise makes lbfgs
converge slowly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C


# --------------------------------------------------------------------------- #
# Embedding sources
# --------------------------------------------------------------------------- #
def load_glove(path: Path = C.GLOVE_TXT) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: bash scripts/download_assets.sh (downloads glove.6B.zip)"
        )
    C.log(f"loading GloVe from {path}")
    vectors: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    C.log(f"  {len(vectors)} GloVe vectors, dim={len(next(iter(vectors.values())))}")
    return vectors


def train_word2vec(
    corpus: List[List[str]], name: str, dim: int, window: int, min_count: int, epochs: int, workers: int
) -> Dict[str, np.ndarray]:
    """Train (or load a cached) gensim Word2Vec model and return its vectors."""
    from gensim.models import Word2Vec

    model_path = C.DATA_DIR / "w2v" / f"w2v_{name}_{dim}d.model"
    if model_path.exists():
        C.log(f"loading cached Word2Vec: {model_path}")
        model = Word2Vec.load(str(model_path))
    else:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        C.log(
            f"training Word2Vec on {name}: {len(corpus)} docs, "
            f"{sum(len(d) for d in corpus)} tokens, dim={dim}, window={window}, "
            f"min_count={min_count}, epochs={epochs}"
        )
        model = Word2Vec(
            sentences=corpus,
            vector_size=dim,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=workers,
            sg=1,  # skip-gram: better for the smaller NYT corpus
            seed=C.SEED,
        )
        model.save(str(model_path))
        C.log(f"  saved {model_path}")
    kv = model.wv
    C.log(f"  Word2Vec[{name}] vocabulary: {len(kv.index_to_key)} words")
    return {w: kv[w] for w in kv.index_to_key}


# --------------------------------------------------------------------------- #
# Document representation
# --------------------------------------------------------------------------- #
def average_embed(
    docs: Sequence[Sequence[str]], vectors: Dict[str, np.ndarray], dim: int
) -> tuple[np.ndarray, int]:
    X = np.zeros((len(docs), dim), dtype=np.float32)
    empty = 0
    for i, toks in enumerate(docs):
        hits = [vectors[t] for t in toks if t in vectors]
        if hits:
            X[i] = np.mean(hits, axis=0)
        else:
            empty += 1
    return X, empty


def coverage(docs: Sequence[Sequence[str]], vectors: Dict[str, np.ndarray]) -> float:
    total = hit = 0
    for toks in docs:
        total += len(toks)
        hit += sum(1 for t in toks if t in vectors)
    return hit / max(total, 1)


def run_one(
    name: str,
    vectors: Dict[str, np.ndarray],
    tokens: Dict[str, List[List[str]]],
    y: Dict[str, np.ndarray],
    labels: Sequence[str],
    dim: int,
    max_iter: int,
    C_param: float,
) -> dict:
    C.log(f"=== Embedding[{name}] ===")
    X, empties = {}, {}
    for split in ("train", "val", "test"):
        X[split], empties[split] = average_embed(tokens[split], vectors, dim)
    cov = coverage(tokens["test"], vectors)
    C.log(f"  token coverage on test: {cov:.4f}; all-OOV docs: {empties}")

    clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=max_iter, C=C_param, n_jobs=-1)
    )
    clf.fit(X["train"], y["train"])

    val = C.evaluate(y["val"], clf.predict(X["val"]), labels)
    test = C.evaluate(y["test"], clf.predict(X["test"]), labels)
    C.log(f"  val  acc={val['accuracy']:.4f} macro-F1={val['macro_f1']:.4f}")
    C.log(f"  TEST acc={test['accuracy']:.4f} macro-F1={test['macro_f1']:.4f}")
    print(test["report"])

    return {
        "method": f"embed_{name}",
        "embedding_vocab": len(vectors),
        "dim": dim,
        "test_token_coverage": cov,
        "all_oov_docs": empties,
        "classifier": {
            "type": "StandardScaler + LogisticRegression",
            "max_iter": max_iter,
            "C": C_param,
        },
        "val": {k: v for k, v in val.items() if k != "report"},
        "test": {k: v for k, v in test.items() if k != "report"},
        "test_report": test["report"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=C.EMBED_DIM)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    # workers=1 is required for a bit-reproducible Word2Vec: with several threads
    # gensim applies updates in nondeterministic order, which moves Macro-F1 by
    # ~0.002-0.007 between runs even with a fixed seed. Single-threaded costs a
    # few minutes here and makes the reported numbers exactly reproducible.
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--sources", default="glove,ag,nyt")
    args = ap.parse_args()

    C.set_seed()
    splits, labels = C.get_splits()
    tokens = C.tokens_for_splits()
    y = {k: C.encode_labels(v["label"], labels) for k, v in splits.items()}
    C.log(f"split sizes: { {k: len(v) for k, v in splits.items()} }")

    results = {}
    for src in args.sources.split(","):
        if src == "glove":
            vectors = load_glove()
        elif src == "ag":
            vectors = train_word2vec(
                C.tokenized_ag(), "ag", args.dim, args.window, args.min_count, args.epochs, args.workers
            )
        elif src == "nyt":
            # Training split only -- the test documents must not shape the vectors.
            vectors = train_word2vec(
                tokens["train"], "nyt_trainonly", args.dim, args.window, args.min_count, args.epochs, args.workers
            )
        else:
            raise ValueError(f"unknown source: {src}")
        results[src] = run_one(
            src, vectors, tokens, y, labels, args.dim, args.max_iter, args.C
        )
        del vectors

    C.save_result(
        "task2_word2vec",
        {
            "task": "Task 2 - Averaged 100-d word embeddings + Logistic Regression",
            "config": vars(args),
            "split": C.split_report(splits, labels),
            "results": results,
        },
    )

    print("\n===== Task 2 summary (NYT test set) =====")
    print(f"{'embedding':<22}{'Accuracy':>10}{'Macro-F1':>11}{'coverage':>10}")
    for src, r in results.items():
        print(
            f"{src:<22}{r['test']['accuracy']:>10.4f}{r['test']['macro_f1']:>11.4f}"
            f"{r['test_token_coverage']:>10.4f}"
        )


if __name__ == "__main__":
    main()
