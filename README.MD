# Clustering 1.8M arXiv Papers into Interpretable Topics

A reproducible, lightweight NLP pipeline that cleans arXiv metadata, builds TF-IDF→SVD embeddings,
clusters with (mini-batch) k-means, assigns human-readable topic labels, and visualizes 2D maps + yearly trend lines.

## What you get
- Cleaned corpus (title+abstract), doc metadata (id, categories, year)
- Dense embeddings (TF-IDF → Truncated SVD + whitening)
- Clusters with labels (zero-shot phrases and/or arXiv categories)
- Figures: UMAP topic map, Top-N topics over time

## Software requirements
- Python ≥ 3.10
- pip/venv (or conda)

## Recommended packages:
- numpy, pandas, scikit-learn, umap-learn, matplotlib, joblib
- pyarrow (Parquet I/O), orjson (fast NDJSON)

## Data
- Place your NDJSON from https://www.kaggle.com/datasets/Cornell-University/arxiv in your project root folder as `arxiv_metadata_science_trends.json`

## Quickstart
1) Clean & slim input (keeps id, categories, update_year, text)
`python 01_clean_data.py` --in cleaned_from_latex_min.json --out cleaned_min.json
2) Embeddings: TF-IDF → SVD (+whitening); saves embeddings.npy, tfidf.joblib, svd.joblib, doc_meta.parquet
`python 02_tfidf_svd.py` --in cleaned_min.json --out_dir embeddings/
3) Full clustering sweep and save best K; also writes cluster_top_terms.csv and cluster_share_by_year.csv
`python 03_kmeans_clustering.py`
4) Labels from arXiv categories (dominant category per cluster)
`python 04C_labels_from_categories.py` (04A and 04B are alternative label methods)
5) Yearly share table as csv file and figure
`python 05_yearly_share.py`
6) Top-N topics over years figure (choose how Top-N are picked; see flags below)
`python 06_top_topics_over_years.py`
7) UMAP 2D scatter plot (colors by cluster, legend via chosen label source)
`python 07_umap_2d.py`

## Saving models and outputs
- All models and outputs are saved in the `emb_out/` folder by default.
- You can manually save results of a run via `archive_run.py` to `outputs/` with timestamp.

## Outputs
- emb_out/embeddings.npy — (n_docs × d) SVD embeddings (whitened)
- emb_out/doc_meta.parquet — id, categories, update_year
- emb_out/doc_meta_with_clusters.parquet — + cluster
- emb_out/cluster_top_terms.csv — centroid top TF-IDF terms (for sanity check)
- emb_out/cluster_share_by_year.csv — yearly topic shares (rows sum to 1)
- emb_out/cluster_labels_zero_shot.csv / cluster_key_phrases.csv / cluster_labels_from_categories.csv

## Figures
- emb_out/umap_2d.png
- emb_out/top10_topics_over_years.png
