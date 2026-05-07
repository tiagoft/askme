"""
Experiment: NLI Feature Matrix vs TF-IDF Baseline Classification

For each train/test feature matrix pair produced by experiment_question_diversity.py:
  1. Loads train NLI features (built from the dataset's train split).
  2. Loads test NLI features (built from the dataset's test split).
  3. Trains a Random Forest on train NLI features, evaluates on test NLI features.
  4. Trains a Random Forest on TF-IDF features as baseline (same train/test split).
  5. Plots learning curves (accuracy vs. training set size) for both feature sets.

Results are printed as a table and saved to OUTPUT_CSV.

Usage:
    python examples/experiment_classification.py
"""

import csv
import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from finetune import finetune_and_evaluate, save_finetuned, save_loss_curves, FINETUNE_MODEL

RESULTS_DIR = Path("/mnt/data3/askme_classification_results")
RESULTS_DIR.mkdir(exist_ok=True)
FEATURE_DIR = RESULTS_DIR
OUTPUT_CSV  = RESULTS_DIR / "experiment_classification.csv"

# Must match the dataset configs used in experiment_question_diversity.py
DATASETS = {
    "ag_news":       {"hf_name": "ag_news",              "train_split": "train", "test_split": "test", "text_field": "text"},
    "20_newsgroups": {"hf_name": "SetFit/20_newsgroups", "train_split": "train", "test_split": "test", "text_field": "text"},
    "wikipedia":     {"hf_name": "json",                 "train_split": "train", "test_split": "test", "text_field": "text",
                      "data_files": {"train": "/mnt/data3/nlp_datasets/wikipedia/train.metadata.jsonl",
                                     "test":  "/mnt/data3/nlp_datasets/wikipedia/test.metadata.jsonl"}},
    "bills":         {"hf_name": "json",                 "train_split": "train", "test_split": "test", "text_field": "summary",
                      "data_files": {"train": "/mnt/data3/nlp_datasets/bills/train.metadata.jsonl",
                                     "test":  "/mnt/data3/nlp_datasets/bills/test.metadata.jsonl"}},
    "rotten_tomatoes": {"hf_name": "rotten_tomatoes",   "train_split": "train", "test_split": "test", "text_field": "text"},
}

RANDOM_SEED        = 42
N_TREES            = 100
TFIDF_MAX_FEATURES = 5000
TRAIN_FRACS = [0.001, 0.01, 0.1, 1.00]
USE_FINETUNE       = True   # set False to skip the (slow) fine-tuning baseline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_feature_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Return (train_parquet, test_parquet) pairs found in directory."""
    pairs = []
    for train_path in sorted(directory.glob("features_train_*.parquet")):
        test_path = directory / train_path.name.replace("features_train_", "features_test_", 1)
        if test_path.exists():
            pairs.append((train_path, test_path))
        else:
            print(f"  Warning: no test matrix for {train_path.name}, skipping.")
    return pairs


def parse_filename(path: Path) -> tuple[str, str, int, int] | None:
    """Return (dataset_name, model_name, n_sample, n_docs) from a features_train_*.csv path."""
    rest = path.stem[len("features_train_"):]      # e.g. "ag_news_qwen3:14b_14_120000"
    for name in DATASETS:
        if rest.startswith(name + "_"):
            remainder = rest[len(name) + 1:]       # e.g. "qwen3:14b_14_120000"
            model_sample, _, n_docs_part   = remainder.rpartition("_")
            model_part,   _, n_sample_part = model_sample.rpartition("_")
            if n_docs_part.isdigit() and n_sample_part.isdigit() and model_part:
                return name, model_part, int(n_sample_part), int(n_docs_part)
    return None


def load_feature_matrix(path: Path) -> tuple[np.ndarray, list[int], list[int]]:
    """Return (X, labels, orig_indices) where orig_indices are original dataset row indices."""
    df = pd.read_parquet(path)
    labels      = df["label"].tolist()
    # orig_idx was added to fix the shuffle→original index alignment bug.
    # Fall back to doc_index for legacy parquets that predate this fix.
    orig_indices = df["orig_idx"].tolist() if "orig_idx" in df.columns else df["doc_index"].tolist()
    feature_cols = [c for c in df.columns if c not in ("label", "doc_index", "orig_idx")]
    X = df[feature_cols].to_numpy(dtype=float)
    return X, labels, orig_indices


def load_texts(dataset_name: str, split: str, indices: list[int]) -> list[str]:
    """Load texts at specific original dataset indices."""
    cfg = DATASETS[dataset_name]
    if "data_files" in cfg:
        ds = load_dataset("json", data_files=[cfg["data_files"][split]], split="train")
    else:
        ds = load_dataset(cfg["hf_name"], split=split)
    return [ds[i][cfg["text_field"]] for i in indices]


def run_rf(X_train, X_test, y_train, y_test) -> tuple[float, RandomForestClassifier]:
    clf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    return float(accuracy_score(y_test, clf.predict(X_test))), clf


def compute_learning_curves(
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_texts: list[str],
    test_texts: list[str],
    y_train: list[int],
    y_test: list[int],
    num_labels: int,
    finetune_ckpt_dir: Path | None = None,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Return (train_sizes, nli_accuracies, tfidf_accuracies, finetune_accuracies).

    Subsamples the training set at each fraction; the test set is always fixed.
    Fine-tuning loads fresh pretrained weights at every fraction so runs are independent.
    """
    sizes, accs_nli, accs_tfidf, accs_ft = [], [], [], []

    for frac in TRAIN_FRACS:
        n_sub = max(num_labels, int(frac * len(y_train)))
        if n_sub >= len(y_train):
            sub_idx = np.arange(len(y_train))
        else:
            sub_idx, _ = train_test_split(
                np.arange(len(y_train)),
                train_size=n_sub,
                random_state=RANDOM_SEED,
                stratify=y_train,
            )

        y_sub        = [y_train[i] for i in sub_idx]
        train_sub    = [train_texts[i] for i in sub_idx]
        n            = len(sub_idx)

        acc_nli, _ = run_rf(X_train[sub_idx], X_test, y_sub, y_test)

        tfidf        = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        X_tr_tf      = tfidf.fit_transform(train_sub)
        X_te_tf      = tfidf.transform(test_texts)
        acc_tfidf, _ = run_rf(X_tr_tf, X_te_tf, y_sub, y_test)

        if USE_FINETUNE:
            ckpt = (finetune_ckpt_dir / f"n{n}") if finetune_ckpt_dir else None
            acc_ft, _, _, ft_log = finetune_and_evaluate(
                train_sub, y_sub, test_texts, y_test,
                num_labels=num_labels, output_dir=ckpt,
            )
            if ckpt:
                save_loss_curves(
                    ft_log,
                    ckpt.parent / f"loss_curves_n{n}.png",
                    title=f"Fine-tune loss — n_train={n}",
                )
        else:
            acc_ft = float("nan")

        sizes.append(n)
        accs_nli.append(acc_nli)
        accs_tfidf.append(acc_tfidf)
        accs_ft.append(acc_ft)

    return sizes, accs_nli, accs_tfidf, accs_ft


def save_learning_curve_plot(all_curves: list[dict], output_path: Path):
    """Plot all NLI curves in colour (solid), one TF-IDF curve (black dashed),
    and one fine-tuning curve (grey dotted) if available."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, curve in enumerate(all_curves):
        color = colors[i % len(colors)]
        ax.plot(curve["sizes"], curve["accs_nli"], color=color, linestyle="-", marker="o",
                label=f"{curve['label']}  NLI")

    ax.plot(all_curves[0]["sizes"], all_curves[0]["accs_tfidf"],
            color="black", linestyle="--", marker="s", label="TF-IDF")

    ft = all_curves[0].get("accs_ft", [])
    if ft and not all(np.isnan(v) for v in ft):
        ax.plot(all_curves[0]["sizes"], ft,
                color="grey", linestyle=":", marker="^", label=f"Fine-tune ({FINETUNE_MODEL})")

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning curves: NLI features (—) vs TF-IDF (--) vs Fine-tuning (···)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

COLS  = ["dataset", "model", "n_sample", "n_docs", "n_questions", "n_train", "n_test", "acc_nli", "acc_tfidf", "acc_ft", "delta"]
COL_W = [12, 16, 8, 8, 11, 8, 7, 9, 10, 8, 8]


def _header():
    print()
    print("  ".join(c.ljust(w) for c, w in zip(COLS, COL_W)))
    print("  ".join("-" * w for w in COL_W))


def _row(rec: dict):
    print("  ".join(str(rec.get(c, "")).ljust(w) for c, w in zip(COLS, COL_W)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment():
    pairs = find_feature_pairs(FEATURE_DIR)
    if not pairs:
        print("No feature matrix pairs found in", FEATURE_DIR)
        print("Run experiment_question_diversity.py first.")
        return

    print("=" * 70)
    print("Classification Experiment: NLI features vs TF-IDF baseline")
    print("=" * 70)
    print(f"Found {len(pairs)} train/test feature matrix pair(s).")

    results   = []
    all_curves = []
    _header()

    for train_path, test_path in pairs:
        parsed = parse_filename(train_path)
        if parsed is None:
            print(f"  Skipping {train_path.name} — cannot parse filename.")
            continue

        dataset_name, model_name, n_sample, n_docs = parsed
        if dataset_name not in DATASETS:
            print(f"  Skipping {train_path.name} — unknown dataset '{dataset_name}'.")
            continue

        X_train, y_train, train_idx = load_feature_matrix(train_path)
        X_test,  y_test,  test_idx  = load_feature_matrix(test_path)
        n_questions = X_train.shape[1] // 3

        train_texts = load_texts(dataset_name, DATASETS[dataset_name]["train_split"], train_idx)
        test_texts  = load_texts(dataset_name, DATASETS[dataset_name]["test_split"],  test_idx)

        # Full-training-set accuracy
        acc_nli,   rf_nli   = run_rf(X_train, X_test, y_train, y_test)
        tfidf     = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        X_tr_tf   = tfidf.fit_transform(train_texts)
        X_te_tf   = tfidf.transform(test_texts)
        acc_tfidf, rf_tfidf = run_rf(X_tr_tf, X_te_tf, y_train, y_test)

        num_labels = len(set(y_train))
        stem       = f"{dataset_name}_{model_name}_{n_sample}_{n_docs}"

        # Save RF models
        joblib.dump({"rf": rf_nli},                       RESULTS_DIR / f"model_nli_{stem}.joblib")
        joblib.dump({"rf": rf_tfidf, "vectorizer": tfidf}, RESULTS_DIR / f"model_tfidf_{stem}.joblib")

        # Full-set fine-tuning
        if USE_FINETUNE:
            ft_dir = RESULTS_DIR / f"model_finetune_{stem}"
            acc_ft_full, ft_model, ft_tokenizer, ft_log = finetune_and_evaluate(
                train_texts, y_train, test_texts, y_test,
                num_labels=num_labels, output_dir=ft_dir / "ckpt",
            )
            save_finetuned(ft_model, ft_tokenizer, ft_dir)
            save_loss_curves(
                ft_log,
                RESULTS_DIR / f"loss_curves_finetune_{stem}.png",
                title=f"Fine-tune loss — {dataset_name} / {model_name} / n_docs={n_docs}",
            )
        else:
            acc_ft_full = float("nan")

        rec = {
            "dataset":     dataset_name,
            "model":       model_name,
            "n_sample":    n_sample,
            "n_docs":      n_docs,
            "n_questions": n_questions,
            "n_train":     len(y_train),
            "n_test":      len(y_test),
            "acc_nli":     round(acc_nli,     4),
            "acc_tfidf":   round(acc_tfidf,   4),
            "acc_ft":      round(acc_ft_full, 4) if USE_FINETUNE else "",
            "delta":       round(acc_nli - acc_tfidf, 4),
        }
        results.append(rec)
        _row(rec)

        print(f"  Computing learning curves ({len(TRAIN_FRACS)} points)…", flush=True)
        sizes, accs_nli, accs_tfidf, accs_ft = compute_learning_curves(
            X_train, X_test, train_texts, test_texts, y_train, y_test,
            num_labels=num_labels,
            finetune_ckpt_dir=RESULTS_DIR / f"lc_finetune_{stem}",
        )
        all_curves.append({
            "sizes":      sizes,
            "accs_nli":   accs_nli,
            "accs_tfidf": accs_tfidf,
            "accs_ft":    accs_ft,
            "label":      f"{model_name} / n={n_sample}",
        })

    if all_curves:
        plot_path = FEATURE_DIR / "learning_curves.png"
        save_learning_curve_plot(all_curves, plot_path)
        print(f"Learning curves saved to {plot_path.name}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_experiment()
