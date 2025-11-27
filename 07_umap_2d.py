# Create a UMAP 2D scatter plot of document embeddings colored by cluster with a subset of the documents.

import numpy as np
import pandas as pd
from pathlib import Path
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("emb_out")

# Config
SAMPLE_N       = 200_000
RANDOM_STATE   = 0
N_NEIGHBORS    = 30
MIN_DIST       = 0.05
METRIC         = "cosine"
LABEL_SOURCE   = "categories"      # "categories" | "zero_shot"
LEGEND_TOP_K   = 12                # legend shows top-K the largest clusters for readability

# Load data
Z = np.load(OUT / "embeddings.npy", mmap_mode="r")
meta = pd.read_parquet(OUT / "doc_meta_with_clusters.parquet")
assert len(meta) == Z.shape[0], (len(meta), Z.shape[0])
labels = meta["cluster"].astype(int).to_numpy()

# Label maps
label_map = None
if LABEL_SOURCE == "categories":
    # Requires you ran 04C_labels_from_categories.py
    lab = (pd.read_csv(OUT / "cluster_labels_from_categories.csv")
             .assign(cluster=lambda d: d["cluster"].astype(int))
             .set_index("cluster"))
    label_map = lab["label_auto"].to_dict()
elif LABEL_SOURCE == "zero_shot":
    # Requires you ran 04A_labels_zero_shot.py
    lab = (pd.read_csv(OUT / "cluster_labels_zero_shot.csv")
             .assign(cluster=lambda d: d["cluster"].astype(int))
             .set_index("cluster"))
    label_map = lab["label_auto"].to_dict()
else:
    label_map = {}

# Sampling
rng = np.random.RandomState(RANDOM_STATE)
n = len(Z)
sample = min(SAMPLE_N, n)
idx = rng.choice(n, size=sample, replace=False)
Zs = Z[idx]
ls = labels[idx]
n_clusters = int(np.unique(ls).size)

# UMAP
u = umap.UMAP(
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    random_state=RANDOM_STATE,
    metric=METRIC,
    verbose=True,
)
xy = u.fit_transform(Zs)

# Colors & legend (top-K by size)
sizes = pd.Series(ls).value_counts().sort_values(ascending=False)
top_clusters = sizes.head(LEGEND_TOP_K).index.tolist()

# Color palette: cycle tab20 if more than 20 clusters
cmap = matplotlib.colormaps.get_cmap("tab20")
colors = np.array([cmap(i % 20) for i in range(n_clusters)])

# Map cluster id -> color
uniq = np.unique(labels)
id2pos = {cid: i for i, cid in enumerate(uniq)}
point_colors = np.array([colors[id2pos[c]] for c in ls])

# Plot scatter
fig, ax = plt.subplots(figsize=(10.5, 9))
ax.scatter(xy[:, 0], xy[:, 1], s=1, c=point_colors, alpha=0.7, linewidth=0)
ax.set_title(f"UMAP of {sample:,} docs — {n_clusters} clusters", pad=8)
ax.axis("off")

# Legend: only top-K clusters
from matplotlib.lines import Line2D
handles, labels_txt = [], []
for cid in top_clusters:
    name = label_map.get(int(cid), str(cid))
    col = colors[id2pos[cid]]
    handles.append(Line2D([0], [0], color=col, lw=3))
    labels_txt.append(f"{cid}: {name}")

leg = ax.legend(handles, labels_txt, loc="center left", bbox_to_anchor=(1.02, 0.5),
                frameon=False, title="Top clusters")
if leg:
    for line in leg.get_lines():
        line.set_linewidth(3)

plt.tight_layout()
fig.savefig(OUT / "umap_2d.png", dpi=200, bbox_inches="tight")
print("Saved:", OUT / "umap_2d.png")
