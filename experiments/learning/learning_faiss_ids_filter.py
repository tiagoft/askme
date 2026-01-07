import time
from tqdm import tqdm
import faiss
import numpy as np


def main():
    # Generate random data
    n_samples = 1e6
    n_features = 1024
    X = np.random.random((int(n_samples), int(n_features))).astype('float32')
    print(f"Data shape: {X.shape}")

    t0 = time.time()

    # Step 2: Create IVF index
    index = faiss.IndexFlatL2(n_features)

    # Step 3: Add data to the index
    for i in tqdm(range(int(n_samples))):
        index.add(X[i:i + 1])
        
    t1 = time.time()
    
    print(f"Index total number of vectors: {index.ntotal}")
    print(f"Indexing time: {t1 - t0} seconds")
    print("Index:", index)

if __name__ == "__main__":
    main()