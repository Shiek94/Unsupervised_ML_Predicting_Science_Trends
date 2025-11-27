# TF-IDF + SVD embeddings for documents with whitening

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text as sk_text
from joblib import dump

# NDJSON with: id, text, categories, update_year
IN_PATH  = "cleaned_from_latex_min.json"
OUT_DIR  = Path("emb_out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# TF-IDF config
MAX_FEATURES = 200_000
MIN_DF       = 5
MAX_DF       = 0.80
NGRAM_RANGE  = (1, 2)
DTYPE        = np.float32
STRIP_ACC    = "unicode"

# Small custom stop-list that often pollutes science corpora
EXTRA_STOP = {
    "paper","result","results","approach","approaches","method","methods",
    "show","shows","model","models","problem","problems","note","work",
    "study","studies","propose","proposed","provide","provided","using",
    "use","used","new","based","present","presented","analysis","data"
}
STOPWORDS = sk_text.ENGLISH_STOP_WORDS.union(EXTRA_STOP)
STOPWORDS = list(STOPWORDS)

# SVD config
N_COMPONENTS = 300
RANDOM_STATE = 0
N_ITER       = 7

# Load data into pandas DataFrame
print("Loading data (this is the heaviest step)...")
df = pd.read_json(IN_PATH, lines=True)

# Drop empty & very short texts
df = df[df["text"].fillna("").str.strip().astype(bool)].reset_index(drop=True)
df = df[df["text"].str.split().str.len() >= 20].reset_index(drop=True)

texts = df["text"].astype(str).tolist()
print(f"Docs: {len(texts):,}")

# Build a TF-IDF object & transform texts into sparse matrix
print("Vectorizing (TF-IDF)…")
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words=STOPWORDS,
    min_df=MIN_DF,
    max_df=MAX_DF,
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    dtype=DTYPE,
    sublinear_tf=True,
    smooth_idf=True,
    strip_accents=STRIP_ACC,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,}\b",
)
X = tfidf.fit_transform(texts)
print(f"TF-IDF shape: {X.shape}, nnz: {X.nnz:,}")

# Apply TruncatedSVD to reduce dimensionality
print("Reducing (TruncatedSVD)…")
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE, n_iter=N_ITER)
Z = svd.fit_transform(X).astype(np.float32)  # (n_docs, n_components)

# whiten components to equalize scales (improves clustering)
sing = svd.singular_values_.astype(np.float32)
Z /= sing

print(f"Embeddings shape: {Z.shape}")

# Save outputs
np.save(OUT_DIR / "embeddings.npy", Z)
dump(tfidf, OUT_DIR / "tfidf.joblib")
dump(svd,   OUT_DIR / "svd.joblib")
df[["id", "categories", "update_year"]].to_parquet(OUT_DIR / "doc_meta.parquet", index=False)

# Calculate explained variance
explained = svd.explained_variance_ratio_.sum()
print(f"Explained variance (SVD): {explained:.3f}")
print("Done. Files in:", OUT_DIR.resolve())
