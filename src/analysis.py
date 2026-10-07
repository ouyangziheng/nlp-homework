"""Aggregate all task results and run the truncation control experiment.

BERT scores lowest of all seven configurations, which looks surprising until you
notice that the assignment fixes max_length=64 while NYT documents have a median
length of ~670 words. BERT therefore classifies roughly the first 9% of each
article, whereas every BoW / embedding model reads the whole thing.

To show that this is the cause rather than a property of BERT, we re-run the
best BoW representation and the best embedding representation on documents
truncated to their first 64 word tokens -- the same budget BERT gets. If the
truncated baselines fall to BERT's level or below, the gap is explained by the
input budget, not by the model.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C
from task2_word2vec import average_embed, load_glove


def truncation_stats(splits) -> dict:
    """How much of each document survives a 64-word-piece budget?"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    texts = splits["test"]["text"].tolist()
    full = [len(tok(t, truncation=False)["input_ids"]) for t in texts]
    full = np.array(full)
    kept = np.minimum(full, 64) / full
    return {
        "median_wordpieces_full_doc": float(np.median(full)),
        "mean_wordpieces_full_doc": float(full.mean()),
        "p95_wordpieces_full_doc": float(np.percentile(full, 95)),
        "docs_longer_than_64": int((full > 64).sum()),
        "docs_total": int(len(full)),
        "frac_docs_truncated": float((full > 64).mean()),
        "mean_fraction_of_doc_seen": float(kept.mean()),
        "median_fraction_of_doc_seen": float(np.median(kept)),
    }


def control_truncated(tokens, y, labels, limit: int = 64) -> dict:
    """Best BoW and best embedding, restricted to the first `limit` tokens."""
    cut = {k: [d[:limit] for d in v] for k, v in tokens.items()}
    out = {}

    C.log(f"control: BoW-binary on first {limit} tokens")
    vec = CountVectorizer(binary=True, analyzer=lambda t: t, lowercase=False, token_pattern=None)
    Xtr = vec.fit_transform(cut["train"])
    clf = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xtr, y["train"])
    res = C.evaluate(y["test"], clf.predict(vec.transform(cut["test"])), labels)
    C.log(f"  acc={res['accuracy']:.4f} macro-F1={res['macro_f1']:.4f}")
    out["bow_binary_trunc64"] = {k: v for k, v in res.items() if k != "report"}

    C.log(f"control: GloVe average on first {limit} tokens")
    glove = load_glove()
    Xtr, _ = average_embed(cut["train"], glove, C.EMBED_DIM)
    Xte, _ = average_embed(cut["test"], glove, C.EMBED_DIM)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, n_jobs=-1))
    pipe.fit(Xtr, y["train"])
    res = C.evaluate(y["test"], pipe.predict(Xte), labels)
    C.log(f"  acc={res['accuracy']:.4f} macro-F1={res['macro_f1']:.4f}")
    out["glove_trunc64"] = {k: v for k, v in res.items() if k != "report"}
    return out


NAMES = {
    "bow_binary": ("Task 1", "Binary Bag of Words"),
    "bow_freq": ("Task 1", "Word Frequency"),
    "bow_tfidf": ("Task 1", "TF-IDF"),
    "embed_glove": ("Task 2", "Pre-trained GloVe (100d)"),
    "embed_ag": ("Task 2", "Word2Vec on AG News (100d)"),
    "embed_nyt": ("Task 2", "Word2Vec on NYT (100d)"),
    "bert": ("Task 3", "BERT-base-uncased (len 64, 3 ep)"),
}


def main() -> None:
    C.set_seed()
    splits, labels = C.get_splits()
    tokens = C.tokens_for_splits()
    y = {k: C.encode_labels(v["label"], labels) for k, v in splits.items()}

    t1 = json.loads((C.RESULTS_DIR / "task1_bow.json").read_text())
    t2 = json.loads((C.RESULTS_DIR / "task2_word2vec.json").read_text())
    t3 = json.loads((C.RESULTS_DIR / "task3_bert.json").read_text())

    rows = []
    for kind, r in t1["results"].items():
        rows.append((f"bow_{kind}", r["test"]["accuracy"], r["test"]["macro_f1"], r["test"]["per_class_f1"]))
    for src, r in t2["results"].items():
        rows.append((f"embed_{src}", r["test"]["accuracy"], r["test"]["macro_f1"], r["test"]["per_class_f1"]))
    rows.append(("bert", t3["test"]["accuracy"], t3["test"]["macro_f1"], t3["test"]["per_class_f1"]))

    trunc = truncation_stats(splits)
    C.log(f"truncation: {json.dumps(trunc, indent=2)}")
    control = control_truncated(tokens, y, labels)

    payload = {
        "split": C.split_report(splits, labels),
        "comparison": [
            {"key": k, "task": NAMES[k][0], "name": NAMES[k][1], "accuracy": a, "macro_f1": f, "per_class_f1": p}
            for k, a, f, p in rows
        ],
        "truncation_analysis": trunc,
        "truncation_control": control,
    }
    C.save_result("analysis", payload)

    print("\n================ All results (NYT test set) ================")
    print(f"{'Task':<9}{'Representation':<36}{'Accuracy':>10}{'Macro-F1':>11}")
    print("-" * 66)
    for k, a, f, _ in rows:
        print(f"{NAMES[k][0]:<9}{NAMES[k][1]:<36}{a:>10.4f}{f:>11.4f}")

    print("\n================ Per-class F1 ================")
    print(f"{'Representation':<36}" + "".join(f"{lab:>12}" for lab in labels))
    print("-" * (36 + 12 * len(labels)))
    for k, _, _, p in rows:
        print(f"{NAMES[k][1]:<36}" + "".join(f"{p[lab]:>12.4f}" for lab in labels))

    print("\n================ Truncation control (64-token budget) ================")
    print(f"BERT (64 word-pieces)              acc={t3['test']['accuracy']:.4f}  macro-F1={t3['test']['macro_f1']:.4f}")
    for k, v in control.items():
        print(f"{k:<35}acc={v['accuracy']:.4f}  macro-F1={v['macro_f1']:.4f}")
    print(
        f"\nBERT sees a median of {trunc['median_fraction_of_doc_seen']*100:.1f}% of each test document; "
        f"{trunc['frac_docs_truncated']*100:.1f}% of documents are truncated."
    )


if __name__ == "__main__":
    main()
