"""Fine-tuning baselines for text classification via HuggingFace Transformers + PEFT.

Supports both encoder models (BERT-style) and decoder-only causal LMs (Qwen,
Llama, Gemma, Phi …).  Decoder models are fine-tuned on the last non-padding
token's representation, using LoRA for models where full fine-tuning would be
impractical at small training sizes.

Gated models (Llama, Gemma) require a HuggingFace account, access approval,
and `huggingface-cli login` before use.  Uncomment them in FINETUNE_MODELS
once access is granted.
"""

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
# Model catalogue
# ---------------------------------------------------------------------------
# Each entry describes one fine-tuning baseline.  Add or remove models here;
# experiment_classification.py imports this list and sweeps over all entries.
#
# Fields:
#   name        — HuggingFace model ID
#   decoder     — True for causal / decoder-only LMs (needs left-padding)
#   max_length  — token budget per document
#   batch_size  — per-device training batch size (reduce if OOM)
#   use_lora    — wrap with LoRA adapters instead of full fine-tuning
#   lora_r      — LoRA rank (ignored when use_lora=False)
#   lora_alpha  — LoRA scaling factor (ignored when use_lora=False)

FINETUNE_MODELS = [
    # --- Encoder (BERT-style) — full fine-tuning, fast ---
    {
        "name": "bert-base-uncased",
        "decoder": False, "max_length": 128,
        "batch_size": 32, "use_lora": False,
    },

    # --- Qwen 2.5 (decoder-only, openly accessible) ---
    {
        "name": "Qwen/Qwen2.5-0.5B",
        "decoder": True, "max_length": 256,
        "batch_size": 16, "use_lora": False,  # 0.5B fits in full fine-tune
    },
    {
        "name": "Qwen/Qwen2.5-1.5B",
        "decoder": True, "max_length": 256,
        "batch_size": 8, "use_lora": True, "lora_r": 16, "lora_alpha": 32,
    },

    # --- Llama 3.2 (gated — requires HF access + huggingface-cli login) ---
    # {
    #     "name": "meta-llama/Llama-3.2-1B",
    #     "decoder": True, "max_length": 256,
    #     "batch_size": 8, "use_lora": True, "lora_r": 16, "lora_alpha": 32,
    # },
    # {
    #     "name": "meta-llama/Llama-3.2-3B",
    #     "decoder": True, "max_length": 256,
    #     "batch_size": 4, "use_lora": True, "lora_r": 16, "lora_alpha": 32,
    # },

    # --- Gemma 2 (gated — requires HF access + huggingface-cli login) ---
    # {
    #     "name": "google/gemma-2-2b",
    #     "decoder": True, "max_length": 256,
    #     "batch_size": 4, "use_lora": True, "lora_r": 16, "lora_alpha": 32,
    # },
]

# Shared training hyper-parameters (applies to all models)
NUM_EPOCHS      = 10   # generous ceiling; early stopping handles the rest
WARMUP_RATIO    = 0.1
LEARNING_RATE   = 2e-5
EARLY_STOPPING  = 3    # patience in eval-accuracy epochs

# Keep FINETUNE_MODEL for backward compatibility with code that imports it.
FINETUNE_MODEL = FINETUNE_MODELS[0]["name"]


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
    return {"accuracy": float(accuracy_score(labels, np.argmax(logits, axis=-1)))}


# ---------------------------------------------------------------------------
# Model + tokenizer setup
# ---------------------------------------------------------------------------

def _load_and_evaluate(
    save_dir: Path,
    test_texts: list[str],
    test_labels: list[int],
    num_labels: int,
    model_cfg: dict,
) -> tuple[float, object, object]:
    """Load a previously saved model and evaluate it on the test set."""
    is_dec    = model_cfg.get("decoder", False)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    if is_dec and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        save_dir,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16 if is_dec else torch.float32,
        ignore_mismatched_sizes=True,
    )
    max_length = model_cfg.get("max_length", 128)
    test_enc   = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)
    args       = TrainingArguments(output_dir="/tmp/eval_only", report_to="none",
                                   per_device_eval_batch_size=64, bf16=is_dec)
    trainer    = Trainer(model=model, args=args, compute_metrics=_compute_metrics)
    results    = trainer.evaluate(_TextDataset(test_enc, test_labels))
    return results["eval_accuracy"], model, tokenizer


def _setup(model_cfg: dict, num_labels: int):
    """Load tokenizer and model, applying LoRA if requested.

    Decoder-only models require left-padding so that the classification head
    (which reads the last non-padding token) sees the end of the sequence.
    Models without a pad token (most causal LMs) use the EOS token as pad.
    """
    name     = model_cfg["name"]
    is_dec   = model_cfg.get("decoder", False)
    use_lora = model_cfg.get("use_lora", False)

    tokenizer = AutoTokenizer.from_pretrained(name)
    if is_dec:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16 if is_dec else torch.float32,
        ignore_mismatched_sizes=True,
    )
    if is_dec and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    if use_lora:
        from peft import get_peft_model, LoraConfig, TaskType
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=model_cfg.get("lora_r", 16),
            lora_alpha=model_cfg.get("lora_alpha", 32),
            target_modules="all-linear",
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_cfg)

    return model, tokenizer


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
    model_cfg: dict | None = None,
    save_dir: Path | None = None,
) -> tuple[float, object, object, list[dict]]:
    """Fine-tune a classifier and return (accuracy, model, tokenizer, log_history).

    If save_dir already contains a trained model (detected by the presence of
    config.json), training is skipped and the saved model is evaluated instead.
    After a fresh training run the model is saved to save_dir automatically.

    output_dir is used for Trainer checkpoints during training (temporary).
    save_dir is the persistent location for the final best model.

    Args:
        model_cfg: Entry from FINETUNE_MODELS.  Defaults to the first entry
                   (bert-base-uncased) for backward compatibility.
        save_dir:  If provided, the trained model is saved here and restored on
                   subsequent calls, avoiding redundant training.
    """
    if model_cfg is None:
        model_cfg = FINETUNE_MODELS[0]

    # Restore from disk if already trained.
    if save_dir is not None and (save_dir / "config.json").exists():
        acc, model, tokenizer = _load_and_evaluate(
            save_dir, test_texts, test_labels, num_labels, model_cfg
        )
        return acc, model, tokenizer, []   # empty log — no training ran

    model, tokenizer = _setup(model_cfg, num_labels)

    max_length = model_cfg.get("max_length", 128)
    train_enc  = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_enc   = tokenizer(test_texts,  truncation=True, padding=True, max_length=max_length)

    train_dataset = _TextDataset(train_enc, train_labels)
    test_dataset  = _TextDataset(test_enc,  test_labels)

    ckpt_dir = str(output_dir) if output_dir else "/tmp/finetune_ckpt"

    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=model_cfg.get("batch_size", 32),
        per_device_eval_batch_size=64,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        logging_steps=max(1, len(train_dataset) // model_cfg.get("batch_size", 32)),
        bf16=model_cfg.get("decoder", False),  # bfloat16 for causal LMs
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

    # Persist the best model so future runs can skip training.
    if save_dir is not None:
        save_finetuned(model, tokenizer, save_dir)

    return results["eval_accuracy"], model, tokenizer, log_history


def save_loss_curves(log_history: list[dict], output_path: Path, title: str = ""):
    """Plot train and eval loss per epoch and save to output_path."""
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
    """Save a fine-tuned model and tokenizer in HuggingFace native format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
