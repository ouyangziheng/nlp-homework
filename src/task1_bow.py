"""Task 1 — Bag-of-Words document representations + Logistic Regression.

Three representations, all built over a vocabulary fitted on the training split
only (fitting the vocabulary on test data would leak information):

  1. binary  — x_i = 1 if word i occurs in the document, else 0
  2. freq    — x_i = raw count of word i in the document
  3. tfidf   — tf-idf weighted counts

The assignment text asks for three BoW methods but spells out only the first
two; tf-idf is the standard third variant, so that is what we use. See the
report for the note on this.

Tokenization is nltk.word_tokenize (lowercased), shared with Task 2 via the
cache in common.py, so the vocabulary is identical across representations.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import common as C

# Vectorizers receive pre-tokenized documents, so the analyzer is the identity.
IDENTITY = dict(analyzer=lambda toks: toks, lowercase=False, token_pattern=None)


def build_vectorizer(kind: str):
    if kind == "binary":
        return CountVectorizer(binary=True, **IDENTITY)
    if kind == "freq":
        return CountVectorizer(binary=False, **IDENTITY)
    if kind == "tfidf":
        return TfidfVectorizer(**IDENTITY)
    raise ValueError(f"unknown BoW kind: {kind}")


def run_one(kind: str, tokens, y, labels, max_iter: int, C_param: float) -> dict:
    C.log(f"=== BoW[{kind}] ===")
    vec = build_vectorizer(kind)
    X_train = vec.fit_transform(tokens["train"])
    X_val = vec.transform(tokens["val"])
    X_test = vec.transform(tokens["test"])
    C.log(f"  vocabulary |V| = {len(vec.vocabulary_)}")
    C.log(f"  train matrix {X_train.shape}, density {X_train.nnz / np.prod(X_train.shape):.5f}")

    clf = LogisticRegression(max_iter=max_iter, C=C_param, n_jobs=-1)
    clf.fit(X_train, y["train"])

    val = C.evaluate(y["val"], clf.predict(X_val), labels)
    test = C.evaluate(y["test"], clf.predict(X_test), labels)
    C.log(f"  val  acc={val['accuracy']:.4f} macro-F1={val['macro_f1']:.4f}")
    C.log(f"  TEST acc={test['accuracy']:.4f} macro-F1={test['macro_f1']:.4f}")
    print(test["report"])

    return {
        "method": f"bow_{kind}",
        "vocab_size": len(vec.vocabulary_),
        "train_shape": list(X_train.shape),
        "classifier": {"type": "LogisticRegression", "max_iter": max_iter, "C": C_param},
        "val": {k: v for k, v in val.items() if k != "report"},
        "test": {k: v for k, v in test.items() if k != "report"},
        "test_report": test["report"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--kinds", default="binary,freq,tfidf")
    args = ap.parse_args()

    C.set_seed()
    splits, labels = C.get_splits()
    tokens = C.tokens_for_splits()
    y = {k: C.encode_labels(v["label"], labels) for k, v in splits.items()}

    C.log(f"split sizes: { {k: len(v) for k, v in splits.items()} }")
    C.log(f"labels: {labels}")

    results = {}
    for kind in args.kinds.split(","):
        results[kind] = run_one(kind, tokens, y, labels, args.max_iter, args.C)

    C.save_result(
        "task1_bow",
        {
            "task": "Task 1 - Bag of Words + Logistic Regression",
            "split": C.split_report(splits, labels),
            "results": results,
        },
    )

    print("\n===== Task 1 summary (NYT test set) =====")
    print(f"{'representation':<16}{'Accuracy':>10}{'Macro-F1':>11}{'|V|':>9}")
    for kind, r in results.items():
        print(f"{kind:<16}{r['test']['accuracy']:>10.4f}{r['test']['macro_f1']:>11.4f}{r['vocab_size']:>9}")


if __name__ == "__main__":
    main()
