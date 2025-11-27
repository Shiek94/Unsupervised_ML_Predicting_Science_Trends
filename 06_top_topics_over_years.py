# Line chart of top-N topics over time, with selectable ranking mode:
#   "overall" -> top by mean share across all years
#   "last"    -> top by *last* year share
#   "last3"   -> top by sum of shares over the last 3 years in the data
#   (we ran with the "last3" option, as it represents recent trends the best)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("emb_out")

# Config
TREND_CSV      = OUT / "cluster_share_by_year.csv"
LABELS_SOURCE  = "categories"   # "categories" | "zero_shot"
N_TOP          = 10
TOP_MODE       = "last3"        # "overall" | "last" | "last3"

FIG_PATH       = OUT / "top_topics_over_years_named.png"

# Load trend
trend = pd.read_csv(TREND_CSV, index_col=0)

# Keep ONLY columns that look like integer cluster ids
numeric_cols = [c for c in trend.columns if str(c).strip().isdigit()]
dropped = sorted(set(trend.columns) - set(numeric_cols))
if dropped:
    print("Ignoring non-cluster columns:", dropped)
trend = trend[numeric_cols]

trend.index = trend.index.astype(int)
trend = trend.sort_index()

# Choose top-N clusters according to mode
if TOP_MODE == "overall":
    scores = trend.mean(axis=0)
elif TOP_MODE == "last":
    last_year = trend.index.max()
    scores = trend.loc[last_year]
elif TOP_MODE == "last3":
    # Sum of shares over the last 3 *available* years in the data
    years = trend.index.to_list()
    last3 = years[-3:] if len(years) >= 3 else years
    scores = trend.loc[last3].sum(axis=0)
else:
    raise ValueError("TOP_MODE must be one of: overall | last | last3")

top_cols = list(scores.sort_values(ascending=False).head(N_TOP).index)

# String<->int id maps
cols_as_int = {c: int(c) for c in top_cols}
cols_as_str = {v: k for k, v in cols_as_int.items()}
top_ids = list(cols_as_int.values())

plot_df = trend[[cols_as_str[k] for k in top_ids]]

# Load labels
if LABELS_SOURCE == "categories":
    labs = (pd.read_csv(OUT / "cluster_labels_from_categories.csv")
              .assign(cluster=lambda d: d["cluster"].astype(int))
              .set_index("cluster"))
elif LABELS_SOURCE == "zero_shot":
    labs = (pd.read_csv(OUT / "cluster_labels_zero_shot.csv")
              .assign(cluster=lambda d: d["cluster"].astype(int))
              .set_index("cluster"))
else:
    labs = pd.DataFrame({"label_auto": []})

def short_label(k: int) -> str:
    base = str(k)
    if k in labs.index and isinstance(labs.at[k, "label_auto"], str):
        # Only the first element if it's a comma-separated list
        base = labs.at[k, "label_auto"].split(",")[0].strip()
    return f"{k}: {base}"

# Plot
year_min, year_max = plot_df.index.min(), plot_df.index.max()
plt.figure(figsize=(11.5, 7.2))
ax = plt.gca()

colors = plt.cm.tab10(np.linspace(0, 1, len(top_ids)))
for k, color in zip(top_ids, colors):
    col = cols_as_str[k]
    ax.plot(plot_df.index.values, plot_df[col].values, lw=2.2, color=color, label=short_label(k))

# Dynamic title
mode_desc = {"overall": "by average share", "last": f"in {year_max}", "last3": f"in last 3 years ({', '.join(map(str, plot_df.index.to_list()[-3:]))})"}
ax.set_title(f"Top {N_TOP} topics {mode_desc[TOP_MODE]} — {year_min}–{year_max}", pad=12)
ax.set_xlabel("Year")
ax.set_ylabel("Share of papers")
ax.set_ylim(0, plot_df.max().max() * 1.08)
ax.grid(True, alpha=0.25)
leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
for line in getattr(leg, "get_lines", lambda: [])():
    line.set_linewidth(3)

# Finalize & save plot
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=180, bbox_inches="tight")
print(f"Saved figure -> {FIG_PATH}")
