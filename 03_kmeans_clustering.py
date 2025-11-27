# Sweep K with spherical MiniBatchKMeans, finalize with full KMeans at best K
# save clusters, top terms, yearly share

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

# Load data
OUT = Path("emb_out")
Z = np.load(OUT / "embeddings.npy", mmap_mode="r")
meta = pd.read_parquet(OUT / "doc_meta.parquet")
tfidf = load(OUT / "tfidf.joblib")
svd   = load(OUT / "svd.joblib")

# Spherical k-means (cosine via unit-length rows)
Z = normalize(Z, norm="l2", copy=False)

def try_k(K: int):
    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=0,
        batch_size=50_000,
        n_init=10,
        max_no_improvement=50,
        reassignment_ratio=0.01,
        verbose=0
    )

    labels = km.fit_predict(Z)

    # sample for silhouette
    rng  = np.random.RandomState(0)
    samp = min(100_000, len(Z))
    idx  = rng.choice(len(Z), size=samp, replace=False)
    sil  = silhouette_score(Z[idx], labels[idx], metric="cosine")

    print(f"K={K:<4}  sil(cos)={sil:0.4f}")
    return sil, labels, km

# Scouting for the best value of K
K_GRID = [40, 50, 60, 80]
best = None
for K in K_GRID:
    sil, labels, km = try_k(K)
    cand = (sil, K, labels, km)
    if best is None or sil > best[0]:
        best = cand

sil_scout, K_best, labels_scout, km_scout = best
print(f"\nChosen K={K_best} (scout sil≈{sil_scout:0.4f})")

# Finalize with full KMeans
km_full = KMeans(
    n_clusters=K_best,
    n_init=10,
    random_state=0
)

labels  = km_full.fit_predict(Z)

# recompute silhouette on a sample
rng  = np.random.RandomState(0)
samp = min(100_000, len(Z))
idx  = rng.choice(len(Z), size=samp, replace=False)
sil_final = silhouette_score(Z[idx], labels[idx], metric="cosine")
print(f"Final KMeans @K={K_best}: sil(cos, sample)={sil_final:0.4f}")

# Save clusters
meta = meta.copy()
meta["cluster"] = labels.astype(int)
meta.to_parquet(OUT / "doc_meta_with_clusters.parquet", index=False)

# Top terms per cluster
feature_names = np.array(tfidf.get_feature_names_out(), dtype=object)

# Map cluster centers in SVD space back to TF-IDF space
C_tfidf = km_full.cluster_centers_ @ svd.components_
topn = 12
rows = []
for k in range(K_best):
    idx = np.argsort(C_tfidf[k])[-topn:][::-1]
    rows.append({"cluster": k, "top_terms": ", ".join(feature_names[idx])})
pd.DataFrame(rows).to_csv(OUT / "cluster_top_terms.csv", index=False)

# Yearly trend (share per year)
trend = (meta.groupby(["update_year","cluster"]).size()
           .groupby(level=0).apply(lambda s: s / s.sum())
           .unstack(fill_value=0).sort_index())
trend.to_csv(OUT / "cluster_share_by_year.csv")

print("\nCluster sizes (top 10):")
print(meta["cluster"].value_counts().head(10))
print("\nSaved:")
print(" - emb_out/doc_meta_with_clusters.parquet")
print(" - emb_out/cluster_top_terms.csv")
print(" - emb_out/cluster_share_by_year.csv")
