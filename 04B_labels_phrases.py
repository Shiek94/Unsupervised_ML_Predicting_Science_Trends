# Option for labels: top bigram/trigram phrases per cluster
# not used in the end. We used 04C_labels_from_categories.py instead

from pathlib import Path
import random
import numpy as np
import pandas as pd
import orjson
from sklearn.feature_extraction.text import TfidfVectorizer

# Config
IN_JSON  = "cleaned_from_latex_min.json"          # NDJSON: {"id","text",...}
OUT      = Path("emb_out")
META     = OUT / "doc_meta_with_clusters.parquet"  # must contain: id, cluster

SEED                   = 0
MAX_PER_CLUSTER        = 20_000   # sample size cap per cluster
MAX_CHARS_PER_DOC      = 50_000   # truncate very long docs to keep RAM stable
MIN_DF                 = 5        # ignore very rare phrases
MAX_DF                 = 0.5      # ignore very common phrases
NGRAM_RANGE            = (2, 3)   # bigrams + trigrams
TOP_N                  = 10       # phrases to keep per cluster
MAX_FEATURES_PER_CLUST = 50_000   # safety cap per cluster vectorizer
MIN_DOC_CHARS          = 120      # skip extremely short documents
STRIP_DUPLICATES       = True     # drop exact dupes per cluster

# domain-tailored stopwords if needed
EXTRA_STOPWORDS = ["et al", "introduction", "conclusion"]

random.seed(SEED)
np.random.seed(SEED)

print("Loading cluster assignments…")
meta = pd.read_parquet(META)
meta = meta.dropna(subset=["id", "cluster"]).copy()
meta["cluster"] = meta["cluster"].astype(int)

id2cluster = dict(zip(meta["id"], meta["cluster"]))
K = meta["cluster"].max() + 1
print(f"Found {K} clusters in metadata.")

# Reservoir sampling buckets per cluster
buckets = {k: [] for k in range(K)}
counts  = {k: 0  for k in range(K)}

def _clean_text_for_bucket(s: str) -> str:
    """Light cleaning for phrase extraction."""
    if not s:
        return ""
    s = s.strip()
    if not s:
        return ""
    if len(s) > MAX_CHARS_PER_DOC:
        s = s[:MAX_CHARS_PER_DOC]
    return s

def _try_reservoir_add(cluster_id: int, text: str):
    """Reservoir sampling into buckets[cluster_id] up to MAX_PER_CLUSTER."""
    counts[cluster_id] += 1
    n = counts[cluster_id]
    b = buckets[cluster_id]
    if len(b) < MAX_PER_CLUSTER:
        b.append(text)
    else:
        # replace a random existing item with prob MAX_PER_CLUSTER / n
        j = random.randint(0, n - 1)
        if j < MAX_PER_CLUSTER:
            b[j] = text

print("Streaming NDJSON and sampling texts per cluster…")
n_lines = 0
n_picked = 0
with open(IN_JSON, "rb") as f:
    for line in f:
        n_lines += 1
        o = orjson.loads(line)
        cid = id2cluster.get(o.get("id"))
        if cid is None:
            continue
        txt = _clean_text_for_bucket(o.get("text", ""))
        if not txt or len(txt) < MIN_DOC_CHARS:
            continue
        _try_reservoir_add(cid, txt)
        n_picked += 1

print(f"Read {n_lines:,} lines; sampled {n_picked:,} docs into {K} buckets.")
for k in range(K):
    if STRIP_DUPLICATES and buckets[k]:
        # Drop exact duplicates per cluster to avoid skew
        uniq = list(dict.fromkeys(buckets[k]))
        if len(uniq) != len(buckets[k]):
            print(f"Cluster {k}: {len(buckets[k]) - len(uniq)} duplicate docs dropped.")
        buckets[k] = uniq

print("Vectorizing per cluster to extract key phrases…")
rows = []
for k in range(K):
    texts = buckets[k]
    if not texts:
        rows.append({"cluster": k, "key_phrases": ""})
        continue

    vect = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES_PER_CLUST,
        token_pattern=r"(?u)\b[a-zA-Z][\w\-]{1,}\b",
        strip_accents=None,
        sublinear_tf=True,
    )

    # Add optional extra stopwords
    if EXTRA_STOPWORDS:
        # sklearn allows union by passing list; easiest is post-fit mask,
        # but we can pre-filter by wrapping analyzer; keep it simple:
        pass

    try:
        X = vect.fit_transform(texts)
    except ValueError:
        # Edge case: not enough features after df filtering
        rows.append({"cluster": k, "key_phrases": ""})
        continue

    # Average TF-IDF weight per phrase (good quick & stable heuristic)
    avg = np.asarray(X.mean(axis=0)).ravel()
    if avg.size == 0:
        rows.append({"cluster": k, "key_phrases": ""})
        continue

    idx = avg.argsort()[-TOP_N:][::-1]
    feats = np.array(vect.get_feature_names_out())[idx]
    # light post-filter to drop phrases dominated by stopwords
    if EXTRA_STOPWORDS:
        keep = []
        for p in feats:
            toks = p.split()
            if any(t.lower() in EXTRA_STOPWORDS for t in toks):
                continue
            keep.append(p)
        feats = np.array(keep[:TOP_N]) if keep else feats

    rows.append({"cluster": k, "key_phrases": ", ".join(feats)})

phr = pd.DataFrame(rows)
phr.to_csv(OUT / "cluster_key_phrases.csv", index=False)

# Merge back for convenience
meta2 = meta.merge(phr, on="cluster", how="left")
meta2.to_parquet(OUT / "doc_meta_with_phrases.parquet", index=False)

print("Saved:")
print(" - emb_out/cluster_key_phrases.csv")
print(" - emb_out/doc_meta_with_phrases.parquet")
