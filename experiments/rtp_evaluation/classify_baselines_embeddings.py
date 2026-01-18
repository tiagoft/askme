# Tree-based text classification on AG News and 20 Newsgroups using Bag-of-Words (scikit-learn only)

from os import pipe
from sklearn.datasets import fetch_20newsgroups, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from sklearn.metrics import homogeneity_score
from collections import Counter, defaultdict
from sklearn.metrics import adjusted_rand_score
from askme.utils.chunking import TextEmbeddingWithChunker
from pathlib import Path
from tqdm import tqdm
import time

from load_dataset import load_dataset_sample


def get_leaf_ids(pipeline, X):
    """
    pipeline: sklearn Pipeline with final step named 'clf'
    X: raw text input (before vectorization)
    """

    # Get leaf indices for each sample
    leaf_ids = pipeline.named_steps["clf"].apply(X)
    return leaf_ids


def make_pipeline(
    *,
    max_depth=6,
    min_samples_leaf=200,
    min_samples_split=0.1,
    random_state=42,
):
    return Pipeline(steps=[
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
    dataset_name: str = 'fancyzhx/ag_news',
    embedding_model_name='sentence-transformers/paraphrase-albert-small-v2',
    chunk_size=200,
    overlap=50,
    cache_dir='~/.askme_cache',
    device='cpu',
    verbose=True,
    split='train',
):
    texts, labels = load_dataset_sample(n_samples=None,
                                        seed=42,
                                        dataset_name=dataset_name,
                                        split=split)
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
        print(f"Cache contains {len(embedding_model.cache)} entries.")

    embeddings = []
    for text in tqdm(texts):
        embedding = embedding_model(text)
        embeddings.append(embedding)
    embeddings = np.vstack(embeddings)

    return embeddings, labels


def evaluate(y_true, y_pred, target_names=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {acc:.4f}\nF1 (macro): {f1:.4f}")


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

def run_test_knn(embeddings, labels, embeddings_test=None, labels_test=None):
    from sklearn.neighbors import KNeighborsClassifier

    pipe = Pipeline(steps=[
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                algorithm='auto',
            ),
        ),
    ])
    print("Training KNN classifier...")
    t0 = time.time()
    pipe.fit(embeddings, labels)
    t1 = time.time()
    print(f"Training completed in {t1 - t0:.2f} seconds.")

    print("Evaluate on discovery set...")
    pred = pipe.predict(embeddings_test)
    evaluate(
        labels_test,
        pred,
    )


def run_test_tree(embeddings, labels, embeddings_test=None, labels_test=None):
    pipe = make_pipeline(max_depth=6)
    print("Training decision tree...")
    t0 = time.time()
    pipe.fit(embeddings, labels)
    t1 = time.time()
    print(f"Training completed in {t1 - t0:.2f} seconds.")

    print("Evaluate on discovery set...")
    pred = pipe.predict(embeddings_test)
    evaluate(
        labels_test,
        pred,
    )
    

def run_test(embeddings, labels):
    pipe = make_pipeline(max_depth=6)
    print("Training decision tree...")
    t0 = time.time()
    pipe.fit(embeddings, labels)
    t1 = time.time()
    print(f"Training completed in {t1 - t0:.2f} seconds.")

    print("Evaluate on discovery set...")
    pred = pipe.predict(embeddings)
    evaluate(
        labels,
        pred,
    )
    leaf_evaluation(embeddings, labels, pipe)


if __name__ == "__main__":
    dataset_name = 'fancyzhx/ag_news'
    dataset_name = 'SetFit/20_newsgroups'
    embeddings, labels = get_embeddings_with_chunker(
        dataset_name=dataset_name,
        embedding_model_name='sentence-transformers/paraphrase-albert-small-v2',
        chunk_size=150,
        overlap=50,
        cache_dir='~/.askme_cache',
        device='cpu',
        verbose=True,
    )
    
    embeddings_test, labels_test = get_embeddings_with_chunker(
        dataset_name=dataset_name,
        embedding_model_name='sentence-transformers/paraphrase-albert-small-v2',
        chunk_size=150,
        overlap=50,
        cache_dir='~/.askme_cache',
        device='cpu',
        verbose=True,
        split='test',
    )
    
        # Make a figure with t-SNE of the embeddings colored by labels
        
    # import matplotlib.pyplot as plt
    # from sklearn.manifold import TSNE
    # embeddings_sample = embeddings[:5000]
    # labels_sample = labels[:5000]
    # tsne = TSNE(n_components=2, random_state=42)
    # embeddings_2d = tsne.fit_transform(embeddings_sample)
    # plt.figure(figsize=(10, 10))
    # scatter = plt.scatter(embeddings_2d[:, 0],
    #                         embeddings_2d[:, 1],
    #                         c=labels_sample,
    #                         cmap='tab10',
    #                         alpha=0.7)
    # plt.colorbar(scatter)
    # plt.title("t-SNE visualization of embeddings colored by labels")
    # plt.savefig("tsne_embeddings.png")
    # Downsample embeddings to 1k elements
    embeddings = embeddings[:10000]
    labels = labels[:10000]
    embeddings_test = embeddings_test[:2000]
    labels_test = labels_test[:2000]
    run_test_knn(embeddings, labels, embeddings_test, labels_test   )
    run_test_tree(embeddings, labels, embeddings_test, labels_test)
    #Practical note:
    #Decision trees on high-dimensional BoW can be slow/large; use max_depth/min_samples_leaf/max_features to control.
    #run_20newsgroups()
    #run_agnews()
