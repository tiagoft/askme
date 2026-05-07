#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Full experimental pipeline for the paper:
#   "Zero-Shot Feature Extraction via LLM-Generated Questions
#    for Low-Resource Text Classification"
#
# Usage:
#   bash experiments/classification/run_all.sh
#   bash experiments/classification/run_all.sh --skip-diversity   # if parquets exist
#   bash experiments/classification/run_all.sh --skip-finetune    # RF only, fast
#
# All outputs go to /mnt/data3/askme_classification_results/
# Runtime estimate (single GPU, llama3.1:8b only):
#   Step 1 (diversity):    ~8–12 h  (NLI on 120k docs × 4 fracs × 4 datasets)
#   Step 2 (classification): ~2–4 h  (RF + BERT; more with larger ft models)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCRIPT_DIR="experiments/classification"
PYTHON="$(which python3)"

# Parse optional flags
SKIP_DIVERSITY=0
SKIP_FINETUNE=0
for arg in "$@"; do
    case $arg in
        --skip-diversity) SKIP_DIVERSITY=1 ;;
        --skip-finetune)  SKIP_FINETUNE=1  ;;
    esac
done

echo "============================================================"
echo "  Classification experiment pipeline"
echo "  $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# STEP 1 — Question generation + NLI feature matrices
# -----------------------------------------------------------------------------
# For each (dataset, LLM model, pool size):
#   1. Samples n_sample=14 diverse documents via FAISS k-means
#   2. Prompts the LLM to generate N_QUESTIONS=20 yes/no questions
#   3. Scores every train document with NLI → features_train_*.parquet
#   4. Scores every test  document with NLI → features_test_*.parquet
#   5. Measures question diversity (lexical/semantic/logical/functional NMI)
#      and computes a random-questions null reference (fun_random column)
#
# Pool sizes are TRAIN_FRACS × total_train_docs — same fractions used in Step 2,
# enabling direct comparison between pool size and classifier training size.
#
# Expected outputs:
#   results/features_train_{dataset}_{model}_14_{n_docs}.parquet
#   results/features_test_{dataset}_{model}_14_{n_docs}.parquet
#   results/experiment_question_diversity.csv   (diversity + null reference)
#   results/experiment_question_diversity.log   (full log with questions)
# -----------------------------------------------------------------------------

if [ "$SKIP_DIVERSITY" -eq 0 ]; then
    echo ""
    echo "--- Step 1: Question diversity + NLI feature matrices ---"
    echo "    Models   : llama3.1:8b (edit OLLAMA_MODELS in the script to add more)"
    echo "    Datasets : ag_news, 20_newsgroups, wikipedia, bills"
    echo "    Pool fracs: 0.001, 0.01, 0.1, 1.0 of each train split"
    echo ""

    # --- 1a: Main sweep (default models and datasets as configured) ----------
    # To sweep additional models, edit OLLAMA_MODELS in experiment_question_diversity.py
    # or run multiple times with different MODEL overrides via env var (future work).
    echo "[1a] Main sweep — llama3.1:8b, all datasets"
    "$PYTHON" "$SCRIPT_DIR/experiment_question_diversity.py"

    # --- 1b: Additional LLM models (uncomment as needed) --------------------
    # Each additional model run appends new parquet pairs that Step 2 picks up
    # automatically.  The diversity CSV is overwritten per run, so run one model
    # at a time or aggregate manually.
    #
    # echo "[1b] qwen3:8b"
    # sed -i 's/^OLLAMA_MODELS = .*/OLLAMA_MODELS = ["qwen3:8b"]/' ...
    # (easier: edit OLLAMA_MODELS in the .py file directly)

else
    echo ""
    echo "[Step 1 skipped — using existing parquet files in results/]"
fi

# -----------------------------------------------------------------------------
# STEP 2 — Classification: NLI features vs baselines + learning curves
# -----------------------------------------------------------------------------
# Auto-discovers all features_train_*.parquet / features_test_*.parquet pairs
# and for each pair trains three classifiers across TRAIN_FRACS subsamples:
#
#   NLI + RF      Our method.  Compact (3·N_QUESTIONS features), zero-shot
#                 w.r.t. labels.  Expected to start high (~80%) even with
#                 very few labeled examples and plateau quickly.
#
#   TF-IDF + RF   Bag-of-words baseline.  Starts low (~50%) at small sizes,
#                 converges slowly.  Model-agnostic (one curve for all LLMs).
#
#   Random + RF   Null baseline: Dirichlet-random feature matrix, same shape
#                 as the NLI matrix.  Expected to perform near chance level.
#                 Shows that the NLI scoring, not just the dimensionality,
#                 carries the useful signal.
#
#   BERT fine-tune  Strong supervised baseline (bert-base-uncased + early
#                 stopping).  Expected to fail at < 100 labeled examples and
#                 converge slowly due to overfitting on small training sets.
#
#   Qwen2.5-0.5B  Modern small LM fine-tuning (full, ~500 M params).
#   Qwen2.5-1.5B  Modern medium LM fine-tuning (LoRA rank=16, ~1.5 B params).
#   [Llama/Gemma entries are commented out in finetune.py; enable after
#    obtaining gated-repo access and running huggingface-cli login]
#
# Expected outputs:
#   results/experiment_classification.csv       (full-data accuracy per method)
#   results/learning_curves.png                 (MAIN FIGURE of the paper)
#   results/loss_curves_finetune_*.png          (train/eval loss per ft model)
#   results/lc_finetune_*/loss_curves_n*.png    (per-fraction loss curves)
#   results/model_nli_*.joblib                  (saved RF on NLI features)
#   results/model_tfidf_*.joblib                (saved RF + TF-IDF vectorizer)
#   results/model_finetune_*/                   (saved HuggingFace model dirs)
# -----------------------------------------------------------------------------

echo ""
echo "--- Step 2: Classification + learning curves ---"

if [ "$SKIP_FINETUNE" -eq 1 ]; then
    echo "    Fine-tuning skipped (--skip-finetune).  RF only."
    # Temporarily disable fine-tuning by patching the flag inline
    "$PYTHON" - <<'EOF'
import sys
sys.path.insert(0, "experiments/classification")
import experiment_classification as ec
ec.USE_FINETUNE = False
ec.run_experiment()
EOF
else
    echo "    Fine-tuning: BERT, Qwen2.5-0.5B, Qwen2.5-1.5B"
    "$PYTHON" "$SCRIPT_DIR/experiment_classification.py"
fi

# -----------------------------------------------------------------------------
# DONE
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  All experiments complete.  $(date)"
echo "  Results in: /mnt/data3/askme_classification_results/"
echo ""
echo "  Key outputs:"
echo "    learning_curves.png          ← main figure"
echo "    experiment_classification.csv ← accuracy table"
echo "    experiment_question_diversity.csv ← diversity + null reference"
echo "============================================================"
