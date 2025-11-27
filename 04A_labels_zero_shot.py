# Map clusters to human labels via zero-shot in the same TF-IDF, SVD space
# We decided to use 04C_labels_from_categories.py in our main analysis, but zero-shot can be a good alternative.

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

OUT = Path("emb_out")
Z = np.load(OUT / "embeddings.npy")
meta = pd.read_parquet(OUT / "doc_meta_with_clusters.parquet")
tfidf = load(OUT / "tfidf.joblib")
svd   = load(OUT / "svd.joblib")

# Candidate topic labels
CANDIDATES = [
    # methods
    "machine learning", "deep learning", "neural networks",
    "reinforcement learning", "computer vision", "natural language processing",
    "optimization", "statistics", "graph theory", "numerical analysis",
    # physics
    "quantum mechanics", "quantum information", "condensed matter",
    "particle physics", "quantum field theory", "string theory",
    "general relativity", "cosmology", "astrophysics", "high energy physics",
    # math
    "algebraic geometry", "number theory", "probability", "topology",
    "differential geometry", "operator theory", "partial differential equations",
    # cs
    "databases", "computer networks", "cybersecurity", "distributed systems",
    "theoretical computer science", "formal languages", "cryptography",
    # other arXiv areas
    "bioinformatics", "computational biology", "econometrics", "finance",
    "materials science", "fluid dynamics", "control theory",
]

# 2) Embed candidates in THE SAME space (tfidf -> svd)
cand_tfidf = tfidf.transform(CANDIDATES)
cand_emb = svd.transform(cand_tfidf)
cand_emb = normalize(cand_emb)                   # cosine-friendly

# 3) Cluster centroids in embedding space
Z_norm = normalize(Z)
labels = meta["cluster"].to_numpy()
K = labels.max() + 1

centroids = np.zeros((K, Z.shape[1]), dtype=np.float32)
for k in range(K):
    idx = np.where(labels == k)[0]
    centroids[k] = Z_norm[idx].mean(axis=0)

# 4) Similarity centroid <-> candidate labels
S = cosine_similarity(centroids, cand_emb)      # (K, n_candidates)
top1 = S.argmax(axis=1)
top3 = np.argsort(-S, axis=1)[:, :3]

# 5) Build a mapping: cluster -> name
rows = []
for k in range(K):
    name1 = CANDIDATES[top1[k]]
    name3 = ", ".join(CANDIDATES[i] for i in top3[k])
    rows.append({"cluster": k, "label_auto": name1, "label_alt": name3})
labels_df = pd.DataFrame(rows)

# Save and also join into meta for later plots
labels_df.to_csv(OUT / "cluster_labels_zero_shot.csv", index=False)
meta2 = meta.merge(labels_df, on="cluster", how="left")
meta2.to_parquet(OUT / "doc_meta_with_labels.parquet", index=False)

print("Saved:")
print(" - emb_out/cluster_labels_zero_shot.csv")
print(" - emb_out/doc_meta_with_labels.parquet")
