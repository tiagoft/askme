import askme.rtp as rtp
from askme.rtp.make_collection_index import make_faiss_index
from askme.rtp.label_propagation import propagate_labels, make_knn_graph, sparse_affinity
from askme.utils import TextEmbeddingWithChunker, kmeans_with_faiss, chunk_text
from askme.makequestions import api, makequestion
from askme.askquestions import check_entailment, models

import faiss
from tqdm import tqdm
import numpy as np

small_text_collection = [
    "The cat sat on the mat.",
    "The cat in in the box." ,
    "The dog barked loudly.",
    "I like cats",
    "I like dogs",
    "The dog is in the yard.",
    "Birds can fly high in the sky.",
    "Fish swim in the ocean.",
    "Elephants are the largest land animals.",
    "Lions are known as the kings of the jungle.",
]

def main():
    # Step one: vectorize documents
    print("Vectorizing documents...")
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_name = 'sentence-transformers/paraphrase-albert-small-v2'
    embedding_model = TextEmbeddingWithChunker(
        model_name=model_name,
        chunk_size=50,
        overlap=10,
        device='cpu',
    )
    print("Making FAISS index...")
    faiss_index, X = make_faiss_index(
        text_collection=small_text_collection,
        embedding_model=embedding_model,
        dimension=embedding_model.model.get_sentence_embedding_dimension(),
        use_gpu=False,
        return_embeddings=True,
    )
    
    print("Getting medoids...")
    # Step two: get some medoids via k-means
    n_clusters = 4
    medoid_indices = kmeans_with_faiss(
        faiss_index=faiss_index,
        X=X,
        n_clusters=n_clusters,)
    
    medoids = []
    print("Medoids:")
    print(medoid_indices)
    for medoid_index in medoid_indices:
        medoids.append(small_text_collection[medoid_index])
    print(medoids)
    
    # Step three: use the medoids to make a question
    print("Ok. Using the API to find a hypothesis...")
    model = api.make_model("gpt-4o-mini")
    response = makequestion.make_a_question_about_collection(
        collection=medoids,
        model=model,
        retries=5,
    )
    hypothesis = response.output.hypothesis
    print("Generated hypothesis:")
    print(hypothesis)
    
    usage = response.usage()
    print("Usage:")
    print(usage)
    
    # Step 4: use entailment to answer the question for N documents
    n_documents_to_answer = 6
    print(f"Answering the question for {n_documents_to_answer} documents...")
    nli_model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    model, tokenizer = models.make_nli_model(model_name=nli_model_name)
    
    doc_indices = kmeans_with_faiss(
        faiss_index=faiss_index,
        X=X,
        n_clusters=n_documents_to_answer,
    )
    
    answers = -np.ones((len(small_text_collection),), dtype=object)
    for doc_index in tqdm(doc_indices):
        document = small_text_collection[doc_index]
        chunked_text = chunk_text(document, chunk_size=200, overlap=20)
        pooled_results = check_entailment.pool_nli_scores(
            check_fn=check_entailment.check_entailment_nli,
            premise=document,
            hypothesis=hypothesis,
            chunk_size=200,
            overlap=20,
            model=model,
            tokenizer=tokenizer,
            device='cpu',
        )
        entails, entailment_score, contradiction_score, P_entailment = pooled_results
        if entails:
            answers[doc_index] = 1
        else:
            answers[doc_index] = 0
    
    # Step 5: propagate labels
    print("Propagating labels...")
    indices, distances = make_knn_graph(np.array(X).astype('float32'), faiss_index, n_neighbors=2)
    W = sparse_affinity(indices, distances, sigma=1.0)
    propagated_labels = propagate_labels(W, answers, alpha=0.99, max_iter=100, tol=1e-3)
    
    print("Propagated labels:")
    print(propagated_labels)
    print("Original labels:")
    print(answers)

    
if __name__ == "__main__":
    main()