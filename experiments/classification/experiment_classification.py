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

from finetune import finetune_and_evaluate, save_finetuned, save_loss_curves, FINETUNE_MODELS

RESULTS_DIR = Path("/mnt/data3/askme_classification_results")
RESULTS_DIR.mkdir(exist_ok=True)
FEATURE_DIR = RESULTS_DIR
OUTPUT_CSV        = RESULTS_DIR / "experiment_classification.csv"
OUTPUT_CURVES_CSV = RESULTS_DIR / "learning_curves.csv"

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
USE_FINETUNE       = True   # set False to skip all fine-tuning baselines

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


def _random_features(n_docs: int, n_questions: int, seed: int) -> np.ndarray:
    """Generate a random NLI feature matrix of shape (n_docs, 3 * n_questions).

    Each question contributes three values (ent, neu, con) drawn from a
    symmetric Dirichlet distribution (α=[1,1,1]), so they sum to 1 per
    question — matching the structure of the real NLI feature matrices.
    This simulates the output of asking random, collection-unrelated questions.
    """
    rng = np.random.default_rng(seed)
    # Dirichlet(1,1,1) = uniform on the 2-simplex
    triples = rng.dirichlet([1, 1, 1], size=(n_docs, n_questions))
    return triples.reshape(n_docs, 3 * n_questions)


def compute_learning_curves(
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_texts: list[str],
    test_texts: list[str],
    y_train: list[int],
    y_test: list[int],
    num_labels: int,
    finetune_ckpt_dir: Path | None = None,
) -> tuple[list[int], list[float], list[float], list[float], dict[str, list[float]]]:
    """Return (train_sizes, nli_accs, tfidf_accs, random_accs, ft_accs_by_model).

    random_accs: RF trained on Dirichlet-random features of the same shape as
    the NLI matrix — the "random questions" null baseline.
    ft_accs_by_model: model name → list of test accuracies per training fraction.
    """
    sizes, accs_nli, accs_tfidf, accs_random = [], [], [], []
    ft_accs: dict[str, list[float]] = {m["name"]: [] for m in FINETUNE_MODELS} if USE_FINETUNE else {}
    n_questions = X_train.shape[1] // 3

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

        y_sub     = [y_train[i] for i in sub_idx]
        train_sub = [train_texts[i] for i in sub_idx]
        n         = len(sub_idx)

        acc_nli, _ = run_rf(X_train[sub_idx], X_test, y_sub, y_test)

        tfidf        = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        X_tr_tf      = tfidf.fit_transform(train_sub)
        X_te_tf      = tfidf.transform(test_texts)
        acc_tfidf, _ = run_rf(X_tr_tf, X_te_tf, y_sub, y_test)

        # Random-questions baseline: Dirichlet-random features, same shape as NLI matrix.
        X_rand_tr = _random_features(n, n_questions, seed=RANDOM_SEED)
        X_rand_te = _random_features(len(y_test), n_questions, seed=RANDOM_SEED + 1)
        acc_random, _ = run_rf(X_rand_tr, X_rand_te, y_sub, y_test)

        if USE_FINETUNE:
            for model_cfg in FINETUNE_MODELS:
                ft_name  = model_cfg["name"].replace("/", "_")
                save_dir = (finetune_ckpt_dir / ft_name / f"n{n}") \
                           if finetune_ckpt_dir else None
                ckpt_dir = (save_dir / "ckpt") if save_dir else None
                acc_ft, _, _, ft_log = finetune_and_evaluate(
                    train_sub, y_sub, test_texts, y_test,
                    num_labels=num_labels, output_dir=ckpt_dir,
                    model_cfg=model_cfg, save_dir=save_dir,
                )
                ft_accs[model_cfg["name"]].append(acc_ft)
                if save_dir and ft_log:   # ft_log is empty when restored from disk
                    save_loss_curves(
                        ft_log,
                        save_dir.parent / f"loss_curves_n{n}.png",
                        title=f"{model_cfg['name']} loss — n_train={n}",
                    )

        sizes.append(n)
        accs_nli.append(acc_nli)
        accs_tfidf.append(acc_tfidf)
        accs_random.append(acc_random)

    return sizes, accs_nli, accs_tfidf, accs_random, ft_accs


def save_learning_curve_plot(all_curves: list[dict], output_path: Path):
    """Plot learning curves for all methods.

    - Solid coloured lines: NLI features, one colour per LLM question generator.
    - Black dashed: TF-IDF (model-independent, drawn once from the first curve).
    - Grey shades dotted: one curve per fine-tuning model.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Grey shades for fine-tuning models (light → dark)
    ft_models = list(all_curves[0].get("accs_ft", {}).keys())
    grey_shades = [str(0.65 - 0.3 * i / max(len(ft_models) - 1, 1)) for i in range(len(ft_models))]
    ft_markers  = ["^", "D", "v", "P", "X"]

    for i, curve in enumerate(all_curves):
        color = colors[i % len(colors)]
        ax.plot(curve["sizes"], curve["accs_nli"], color=color, linestyle="-", marker="o",
                label=f"{curve['label']}  NLI")

    ax.plot(all_curves[0]["sizes"], all_curves[0]["accs_tfidf"],
            color="black", linestyle="--", marker="s", label="TF-IDF")

    ax.plot(all_curves[0]["sizes"], all_curves[0]["accs_random"],
            color="red", linestyle=":", marker="x", label="Random questions")

    # One grey curve per fine-tuning model (drawn from first curve only;
    # they share the same train/test split so adding more curves would overlap).
    for j, (model_name, accs) in enumerate(all_curves[0].get("accs_ft", {}).items()):
        if accs and not all(np.isnan(v) for v in accs):
            short = model_name.split("/")[-1]  # e.g. "Qwen2.5-0.5B"
            ax.plot(all_curves[0]["sizes"], accs,
                    color=grey_shades[j % len(grey_shades)],
                    linestyle=":", marker=ft_markers[j % len(ft_markers)],
                    label=f"Fine-tune {short}")

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

_FT_COLS = [f"acc_ft_{m['name'].replace('/','_')}" for m in FINETUNE_MODELS]
COLS  = ["dataset", "model", "n_sample", "n_docs", "n_questions", "n_train", "n_test",
         "acc_nli", "acc_tfidf", "acc_random", "delta"] + _FT_COLS
COL_W = [12, 16, 8, 8, 11, 8, 7, 9, 10, 10, 8] + [10] * len(_FT_COLS)


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

        # Full-training-set fine-tuning for each model in FINETUNE_MODELS.
        # Results go into the summary CSV; models are saved for later analysis.
        ft_full_accs = {}
        if USE_FINETUNE:
            for model_cfg in FINETUNE_MODELS:
                ft_name = model_cfg["name"]
                ft_stem = ft_name.replace("/", "_")
                ft_dir  = RESULTS_DIR / f"model_finetune_{stem}_{ft_stem}"
                acc_ft_full, ft_model, ft_tok, ft_log = finetune_and_evaluate(
                    train_texts, y_train, test_texts, y_test,
                    num_labels=num_labels, output_dir=ft_dir / "ckpt",
                    model_cfg=model_cfg, save_dir=ft_dir,
                )
                ft_full_accs[ft_name] = acc_ft_full
                save_loss_curves(
                    ft_log,
                    RESULTS_DIR / f"loss_curves_finetune_{stem}_{ft_stem}.png",
                    title=f"{ft_name} — {dataset_name} / {model_name} / n_docs={n_docs}",
                )

        # Build summary record — one column per fine-tuning model.
        # Full-training-set random baseline
        X_rand_tr_full = _random_features(len(y_train), n_questions, seed=RANDOM_SEED)
        X_rand_te_full = _random_features(len(y_test),  n_questions, seed=RANDOM_SEED + 1)
        acc_random_full, _ = run_rf(X_rand_tr_full, X_rand_te_full, y_train, y_test)

        rec = {
            "dataset":     dataset_name,
            "model":       model_name,
            "n_sample":    n_sample,
            "n_docs":      n_docs,
            "n_questions": n_questions,
            "n_train":     len(y_train),
            "n_test":      len(y_test),
            "acc_nli":     round(acc_nli,          4),
            "acc_tfidf":   round(acc_tfidf,         4),
            "acc_random":  round(acc_random_full,   4),
            "delta":       round(acc_nli - acc_tfidf, 4),
            **{f"acc_ft_{m['name'].replace('/','_')}": round(ft_full_accs.get(m["name"], float("nan")), 4)
               for m in FINETUNE_MODELS},
        }
        results.append(rec)
        _row(rec)

        print(f"  Computing learning curves ({len(TRAIN_FRACS)} points × {len(FINETUNE_MODELS)} ft models)…",
              flush=True)
        sizes, accs_nli, accs_tfidf, accs_random, ft_accs = compute_learning_curves(
            X_train, X_test, train_texts, test_texts, y_train, y_test,
            num_labels=num_labels,
            finetune_ckpt_dir=RESULTS_DIR / f"lc_finetune_{stem}",
        )
        all_curves.append({
            "sizes":       sizes,
            "accs_nli":    accs_nli,
            "accs_tfidf":  accs_tfidf,
            "accs_random": accs_random,
            "accs_ft":     ft_accs,
            "label":       f"{model_name} / n={n_sample}",
            # metadata for CSV export
            "dataset":     dataset_name,
            "model":       model_name,
            "n_sample":    n_sample,
            "n_docs":      n_docs,
        })

    if all_curves:
        plot_path = FEATURE_DIR / "learning_curves.png"
        save_learning_curve_plot(all_curves, plot_path)
        print(f"Learning curves saved to {plot_path.name}")

        # Save learning curve data so plots can be regenerated without re-running.
        # One row per (feature_matrix, training_fraction).
        _ft_lc_cols = [f"acc_ft_{m['name'].replace('/','_')}" for m in FINETUNE_MODELS]
        lc_fieldnames = ["dataset", "model", "n_sample", "n_docs",
                         "n_train", "acc_nli", "acc_tfidf", "acc_random"] + _ft_lc_cols
        lc_rows = []
        for curve in all_curves:
            for i, n in enumerate(curve["sizes"]):
                row = {
                    "dataset":    curve["dataset"],
                    "model":      curve["model"],
                    "n_sample":   curve["n_sample"],
                    "n_docs":     curve["n_docs"],
                    "n_train":    n,
                    "acc_nli":    round(curve["accs_nli"][i],    4),
                    "acc_tfidf":  round(curve["accs_tfidf"][i],  4),
                    "acc_random": round(curve["accs_random"][i], 4),
                    **{f"acc_ft_{m['name'].replace('/','_')}":
                       round(curve["accs_ft"].get(m["name"], [float("nan")] * len(curve["sizes"]))[i], 4)
                       for m in FINETUNE_MODELS},
                }
                lc_rows.append(row)
        with open(OUTPUT_CURVES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=lc_fieldnames)
            writer.writeheader()
            writer.writerows(lc_rows)
        print(f"Learning curve data saved to {OUTPUT_CURVES_CSV.name}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_experiment()
