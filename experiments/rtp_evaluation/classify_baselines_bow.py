# Tree-based text classification on AG News and 20 Newsgroups using Bag-of-Words (scikit-learn only)

from os import pipe
from sklearn.datasets import fetch_20newsgroups, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from sklearn.metrics import homogeneity_score
from collections import Counter, defaultdict
from sklearn.metrics import adjusted_rand_score
from askme.utils.chunking import TextEmbeddingWithChunker
from pathlib import Path

from load_dataset import load_dataset_sample

def get_leaf_ids(pipeline, X):
    """
    pipeline: sklearn Pipeline with final step named 'clf'
    X: raw text input (before vectorization)
    """
    # Transform text to BoW
    # if pipeline has a named step "bow"
    if "bow" in pipeline.named_steps:
        X_vec = pipeline.named_steps["bow"].transform(X)
    else:
        X_vec = X

    # Get leaf indices for each sample
    leaf_ids = pipeline.named_steps["clf"].apply(X_vec)
    #print(f"Unique leaf IDs: {np.unique(leaf_ids)}")
    return leaf_ids


def make_vec_pipeline(
        *,
        max_depth=6,
        min_samples_leaf=200,
        random_state=42,
):
    return Pipeline(steps=[
        (
            "clf",
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            ),
        ),
    ])


def make_bow_tree_pipeline(
        *,
        max_features=50000,
        ngram_range=(1, 1),
        max_depth=6,
        min_samples_leaf=200,
        min_samples_split=0.1,
        random_state=42,
):
    return Pipeline(steps=[
        ("bow",
         CountVectorizer(max_features=max_features, ngram_range=ngram_range)),
        (
            "clf",
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=random_state,
                ),
        ),
    ])


def get_embeddings_with_chunker(
    text_collection,
    embedding_model_name='sentence-transformers/paraphrase-albert-small-v2',
    chunk_size=200,
    overlap=50,
    cache_dir='~/.askme_cache',
    device='cpu',
    verbose=True,
):
    embedding_model = TextEmbeddingWithChunker(
        model_name=embedding_model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        device=device,
    )

    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_model.load_cache(str(cache_dir / 'embedding_cache.pkl'))
    if verbose:
        print(
            f"Loaded embedding cache from {cache_dir / 'embedding_cache.pkl'}")
        print(f"Cache contiains {len(embedding_model.cache)} entries.")

    embeddings = []
    for text in text_collection:
        embedding = embedding_model(text)
        embeddings.append(embedding)
    embeddings = np.vstack(embeddings)

    return embeddings


def evaluate(name, y_true, y_pred, target_names=None):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0,
        ))


def weighted_leaf_purity(y_true, leaf_ids):
    leaf_to_labels = defaultdict(list)

    for y, leaf in zip(y_true, leaf_ids):
        leaf_to_labels[leaf].append(y)

    total = len(y_true)
    weighted_sum = 0.0

    for labels in leaf_to_labels.values():
        counts = Counter(labels)
        purity = max(counts.values()) / len(labels)
        weighted_sum += purity * (len(labels) / total)

    return weighted_sum


def leaf_evaluation(X_test, y_test, pipe):
    leaf_ids = get_leaf_ids(pipe, X_test)
    nmi = normalized_mutual_info_score(y_test, leaf_ids)
    print(f"NMI between true labels and leaf assignments: {nmi:.4f}")
    homogeneity = homogeneity_score(y_test, leaf_ids)
    print(f"Global leaf homogeneity: {homogeneity:.4f}")
    w_purity = weighted_leaf_purity(y_test, leaf_ids)
    print(f"Weighted leaf purity: {w_purity:.4f}")
    ari = adjusted_rand_score(y_test, leaf_ids)
    print(
        f"Adjusted Rand Index (ARI) between true labels and leaf assignments: {ari:.4f}"
    )


def run_20newsgroups():
    # Using official sklearn loader (has a predefined train/test split)  :contentReference[oaicite:0]{index=0}
    train = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    test = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    pipe = make_bow_tree_pipeline(max_features=50000, max_depth=6)
    pipe.fit(train.data, train.target)
    
    print("Evaluate on test set...")
    pred = pipe.predict(test.data)
    evaluate("20 Newsgroups (BoW + DecisionTree)",
             test.target,
             pred,
             target_names=train.target_names)
    leaf_evaluation(test.data, test.target, pipe)
    
    print("Evaluate on discovery set...")
    pred = pipe.predict(train.data)
    evaluate("20 Newsgroups (BoW + DecisionTree)",
             train.target,
             pred,
             target_names=train.target_names)
    leaf_evaluation(train.data, train.target, pipe)



def run_20newsgroups_embeddings():
    # Using official sklearn loader (has a predefined train/test split)  :contentReference[oaicite:0]{index=0}
    train = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    
    test = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    pipe = make_bow_tree_pipeline(max_features=80000, max_depth=70)
    pipe.fit(train.data, train.target)
    
    print("Getting embeddings for test set...")
    pred = pipe.predict(test.data)

    evaluate("20 Newsgroups (BoW + DecisionTree)",
             test.target,
             pred,
             target_names=train.target_names)
    leaf_evaluation(test.data, test.target, pipe)


def run_agnews():
    # AG News is not bundled in sklearn; fetch via OpenML using sklearn.datasets.fetch_openml  :contentReference[oaicite:1]{index=1}
    # Note: dataset naming/version can vary on OpenML; if this fails, search OpenML for the correct name/data_id.
    from datasets import load_dataset
    agnews = load_dataset('ag_news')

    X = list(agnews['train']['text']) + list(agnews['test']['text'])
    y = list(agnews['train']['label']) + list(agnews['test']['label'])

    # Handle typical OpenML return shapes/types.
    # Many OpenML text datasets store the text in a single column.
    if hasattr(X, "shape") and len(getattr(X, "shape",
                                           ())) == 2 and X.shape[1] == 1:
        X_text = [str(v[0]) for v in X]
    else:
        # If it's already a 1D array-like of strings, or multiple columns, stringify rows safely.
        try:
            X_text = [str(v) for v in X]
        except Exception:
            X_text = [str(row) for row in X]

    # Targets from OpenML are often strings; keep them as-is (classification_report will handle them).
    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None,
    )

    pipe = make_bow_tree_pipeline(max_features=50000, max_depth=6)
    pipe.fit(X_train, y_train)
    
    
    print("Evaluate on train set...")
    pred = pipe.predict(X_test)

    # target_names: sort unique labels for stable reporting
    target_names = sorted({str(v) for v in y})
    evaluate("AG News (OpenML) (BoW + DecisionTree)",
             y_test,
             pred,
             target_names=target_names)
    leaf_evaluation(X_test, y_test, pipe)


    print("Evaluate on discovery set...")
    pred = pipe.predict(X_train)

    # target_names: sort unique labels for stable reporting
    target_names = sorted({str(v) for v in y})
    evaluate("AG News (OpenML) (BoW + DecisionTree)",
             y_train,
             pred,
             target_names=target_names)
    leaf_evaluation(X_train, y_train, pipe)
    
    
if __name__ == "__main__":
    # Practical note:
    # Decision trees on high-dimensional BoW can be slow/large; use max_depth/min_samples_leaf/max_features to control.
    run_20newsgroups()
    run_agnews()
