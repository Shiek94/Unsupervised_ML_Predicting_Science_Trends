# quick_scout_k.py (drop-in helper)
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

OUT = Path("emb_out")
Z = np.load(OUT/"embeddings.npy", mmap_mode="r")
Z = normalize(Z, copy=False)

rng = np.random.RandomState(0)
n = min(300_000, len(Z))
idx = rng.choice(len(Z), size=n, replace=False)
Zs = Z[idx]

for K in [70, 80, 90, 100, 120, 140, 160]:
    km = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=50_000, n_init=10)
    labels = km.fit_predict(Zs)
    sil = silhouette_score(Zs, labels, metric="cosine", sample_size=50_000, random_state=0)
    print(f"K={K:<3}  scout silhouette={sil:.4f}")
