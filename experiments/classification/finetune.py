"""Fine-tuning baseline: BERT-based sequence classifier via HuggingFace Transformers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FINETUNE_MODEL     = "bert-base-uncased"
MAX_LENGTH         = 128
TRAIN_BATCH_SIZE   = 32
EVAL_BATCH_SIZE    = 64
LEARNING_RATE      = 2e-5
NUM_EPOCHS         = 10          # generous ceiling; early stopping handles the rest
WARMUP_RATIO       = 0.1
EARLY_STOPPING     = 3           # patience in epochs


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class _TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float(accuracy_score(labels, preds))}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def finetune_and_evaluate(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    num_labels: int,
    output_dir: Path | None = None,
    model_name: str = FINETUNE_MODEL,
) -> tuple[float, object, object]:
    """Fine-tune a pretrained classifier and return (accuracy, model, tokenizer).

    Loads fresh pretrained weights each call so learning-curve runs are independent.
    Uses early stopping (patience=EARLY_STOPPING epochs) on eval accuracy.

    Args:
        train_texts:  Training documents.
        train_labels: Integer class labels for training documents.
        test_texts:   Test documents (held-out, never seen during training).
        test_labels:  Integer class labels for test documents.
        num_labels:   Number of output classes.
        output_dir:   Where to write checkpoints. If None, uses a temp location.
        model_name:   HuggingFace model ID.

    Returns:
        (accuracy_on_test, trained_model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_enc  = tokenizer(test_texts,  truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = _TextDataset(train_enc, train_labels)
    test_dataset  = _TextDataset(test_enc,  test_labels)

    ckpt_dir = str(output_dir) if output_dir else "/tmp/finetune_ckpt"

    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        logging_steps=max(1, len(train_dataset) // TRAIN_BATCH_SIZE),
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING)],
    )

    trainer.train()
    results     = trainer.evaluate()
    log_history = trainer.state.log_history
    return results["eval_accuracy"], model, tokenizer, log_history


def save_loss_curves(log_history: list[dict], output_path: Path, title: str = ""):
    """Plot train and eval loss per epoch and save to output_path."""
    # Average training loss over steps within each epoch
    train_by_epoch: dict[float, list[float]] = {}
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            epoch = round(entry["epoch"], 6)
            train_by_epoch.setdefault(epoch, []).append(entry["loss"])

    eval_entries = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

    if not train_by_epoch and not eval_entries:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if train_by_epoch:
        epochs = sorted(train_by_epoch)
        losses = [np.mean(train_by_epoch[e]) for e in epochs]
        ax.plot(epochs, losses, color="steelblue", marker=".", label="train loss")

    if eval_entries:
        epochs, losses = zip(*eval_entries)
        ax.plot(epochs, losses, color="tomato", marker="o", label="eval loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title or "Fine-tuning loss curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_finetuned(model, tokenizer, output_dir: Path):
    """Save a fine-tuned model and tokenizer for later reuse."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
