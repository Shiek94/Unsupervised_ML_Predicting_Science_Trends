# Build cluster share per year CSV file

import os
os.environ["MPLBACKEND"] = "Agg"

import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("emb_out")

# Load yearly share table
trend = pd.read_csv(OUT / "cluster_share_by_year.csv", index_col=0)

# Keep only columns that look like integer cluster ids (e.g. "0","12")
def parse_cluster_col(name: str):
    m = re.fullmatch(r"\s*(\d+)\s*", str(name))
    return int(m.group(1)) if m else None

col_map = {c: parse_cluster_col(c) for c in trend.columns}
keep_cols = [c for c, k in col_map.items() if k is not None]

# Subset & reindex columns by numeric cluster id
trend = trend[keep_cols].copy()
trend.columns = [col_map[c] for c in keep_cols]
trend = trend.sort_index(axis=1)

# Short labels for legend
terms = pd.read_csv(OUT / "cluster_top_terms.csv")
terms["label"] = terms["top_terms"].str.split(", ").str[:3].str.join(", ")
label_map = dict(zip(terms["cluster"], terms["label"]))

# Pick top 8 clusters by total share
top_ids = trend.sum().sort_values(ascending=False).head(8).index.tolist()

ax = trend[top_ids].plot(figsize=(11, 7), linewidth=2)
ax.set_ylabel("Share of papers")
ax.set_xlabel("Year")
ax.set_title("Cluster share by year (top 8)")

legend_labels = [f"{cid}: {label_map.get(int(cid), '')}" for cid in top_ids]
ax.legend(legend_labels, fontsize=8, ncol=2)

# Finalize & save plot
plt.tight_layout()
out_file = OUT / "yearly_trends_top8.png"
plt.savefig(out_file, dpi=200)
print(f"Saved plot -> {out_file.resolve()}")
