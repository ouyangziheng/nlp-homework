"""Task 3 — fine-tune bert-base-uncased for NYT topic classification.

Assignment-mandated settings: max_length = 64, 3 epochs. Everything else
(batch size, learning rate, warmup) follows the standard BERT fine-tuning
recipe from Devlin et al.

A plain PyTorch training loop is used rather than transformers.Trainer, so the
procedure is explicit and does not depend on the Trainer API, which has moved
between transformers versions.

Note on max_length=64: NYT documents have a median length of ~670 words, so 64
word-piece tokens covers only the opening sentence or two of each article. This
is the required setting; the report discusses what it costs.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import common as C

MODEL_NAME = "google-bert/bert-base-uncased"


class NYTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-length", type=int, default=64, help="assignment requires 64")
    ap.add_argument("--epochs", type=int, default=3, help="assignment requires 3")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    C.set_seed()
    splits, labels = C.get_splits()
    y = {k: C.encode_labels(v["label"], labels) for k, v in splits.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C.log(f"device={device}, labels={labels}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels)
    ).to(device)

    datasets = {
        name: NYTDataset(part["text"].tolist(), y[name], tokenizer, args.max_length)
        for name, part in splits.items()
    }
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(datasets["val"], batch_size=args.eval_batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=args.eval_batch_size)

    # No weight decay on bias / LayerNorm, per the original BERT recipe.
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if any(s in n for s in ("bias", "LayerNorm.weight")) else decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    C.log(
        f"training: {len(datasets['train'])} examples, {len(train_loader)} steps/epoch, "
        f"{total_steps} total steps, max_length={args.max_length}, epochs={args.epochs}"
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 100 == 0:
                C.log(f"  epoch {epoch} step {step}/{len(train_loader)} loss={running / step:.4f}")
        val = C.evaluate(y["val"], predict(model, val_loader, device), labels)
        C.log(
            f"epoch {epoch}: train_loss={running / len(train_loader):.4f} "
            f"val_acc={val['accuracy']:.4f} val_macro_f1={val['macro_f1']:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(train_loader),
                "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
            }
        )

    # Report the model after the full 3 epochs, as the assignment specifies.
    test = C.evaluate(y["test"], predict(model, test_loader, device), labels)
    C.log(f"TEST acc={test['accuracy']:.4f} macro-F1={test['macro_f1']:.4f}")
    print(test["report"])

    C.save_result(
        "task3_bert",
        {
            "task": "Task 3 - BERT-base-uncased fine-tuning",
            "model": MODEL_NAME,
            "config": vars(args),
            "split": C.split_report(splits, labels),
            "history": history,
            "val": {k: v for k, v in history[-1].items()},
            "test": {k: v for k, v in test.items() if k != "report"},
            "test_report": test["report"],
        },
    )

    print("\n===== Task 3 summary (NYT test set) =====")
    print(f"BERT-base-uncased  Accuracy={test['accuracy']:.4f}  Macro-F1={test['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
