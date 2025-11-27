# Create human-readable cluster names from arXiv category codes
# arXiv categories were pulled directly from https://arxiv.org/category_taxonomy and mapped to category codes

from pathlib import Path
import re
import pandas as pd

# Load metadata with clusters
IN_META = Path("emb_out/doc_meta_with_clusters.parquet")
OUT_DIR = Path("emb_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# arXiv code translations via a dictionary

NAME_MAP = {
    # Computer Science
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language (NLP)",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.OH": "Other Computer Science",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Symbolic Computation",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",

    # Economics
    "econ.EM": "Econometrics",
    "econ.GN": "General Economics",
    "econ.TH": "Theoretical Economics",

    # Electrical Engineering & Systems Science
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control",

    # Mathematics
    "math.AC": "Commutative Algebra",
    "math.AG": "Algebraic Geometry",
    "math.AP": "Analysis of PDEs",
    "math.AT": "Algebraic Topology",
    "math.CA": "Classical Analysis and ODEs",
    "math.CO": "Combinatorics",
    "math.CT": "Category Theory",
    "math.CV": "Complex Variables",
    "math.DG": "Differential Geometry",
    "math.DS": "Dynamical Systems",
    "math.FA": "Functional Analysis",
    "math.GM": "General Mathematics",
    "math.GN": "General Topology",
    "math.GR": "Group Theory",
    "math.GT": "Geometric Topology",
    "math.HO": "History and Overview",
    "math.IT": "Information Theory",
    "math.KT": "K-Theory and Homology",
    "math.LO": "Logic",
    "math.MG": "Metric Geometry",
    "math.MP": "Mathematical Physics",
    "math.NA": "Numerical Analysis",
    "math.NT": "Number Theory",
    "math.OA": "Operator Algebras",
    "math.OC": "Optimization and Control",
    "math.PR": "Probability",
    "math.QA": "Quantum Algebra",
    "math.RA": "Rings and Algebras",
    "math.RT": "Representation Theory",
    "math.SG": "Symplectic Geometry",
    "math.SP": "Spectral Theory",
    "math.ST": "Statistics Theory",

    # Astrophysics
    "astro-ph": "Astrophysics",
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
    "astro-ph.EP": "Earth and Planetary Astrophysics",
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.HE": "High Energy Astrophysical Phenomena",
    "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
    "astro-ph.SR": "Solar and Stellar Astrophysics",

    # Condensed Matter
    "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.other": "Other Condensed Matter",
    "cond-mat.quant-gas": "Quantum Gases",
    "cond-mat.soft": "Soft Condensed Matter",
    "cond-mat.stat-mech": "Statistical Mechanics",
    "cond-mat.str-el": "Strongly Correlated Electrons",
    "cond-mat.supr-con": "Superconductivity",

    # GR & Quantum Cosmology
    "gr-qc": "General Relativity and Quantum Cosmology",

    # High Energy Physics
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",

    # Mathematical Physics
    "math-ph": "Mathematical Physics",

    # Nonlinear Sciences
    "nlin.AO": "Adaptation and Self-Organizing Systems",
    "nlin.CD": "Chaotic Dynamics",
    "nlin.CG": "Cellular Automata and Lattice Gases",
    "nlin.PS": "Pattern Formation and Solitons",
    "nlin.SI": "Exactly Solvable and Integrable Systems",

    # Nuclear Physics
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",

    # Physics (broad)
    "physics.acc-ph": "Accelerator Physics",
    "physics.ao-ph": "Atmospheric and Oceanic Physics",
    "physics.app-ph": "Applied Physics",
    "physics.atm-clus": "Atomic and Molecular Clusters",
    "physics.atom-ph": "Atomic Physics",
    "physics.bio-ph": "Biological Physics",
    "physics.chem-ph": "Chemical Physics",
    "physics.class-ph": "Classical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis, Statistics and Probability",
    "physics.ed-ph": "Physics Education",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.gen-ph": "General Physics",
    "physics.geo-ph": "Geophysics",
    "physics.hist-ph": "History and Philosophy of Physics",
    "physics.ins-det": "Instrumentation and Detectors",
    "physics.med-ph": "Medical Physics",
    "physics.optics": "Optics",
    "physics.plasm-ph": "Plasma Physics",
    "physics.pop-ph": "Popular Physics",
    "physics.soc-ph": "Physics and Society",
    "physics.space-ph": "Space Physics",

    # Quantum Physics
    "quant-ph": "Quantum Physics",

    # Quantitative Biology
    "q-bio.BM": "Biomolecules",
    "q-bio.CB": "Cell Behavior",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.OT": "Other Quantitative Biology",
    "q-bio.PE": "Populations and Evolution",
    "q-bio.QM": "Quantitative Methods",
    "q-bio.SC": "Subcellular Processes",
    "q-bio.TO": "Tissues and Organs",

    # Quantitative Finance
    "q-fin.CP": "Computational Finance",
    "q-fin.EC": "Economics",
    "q-fin.GN": "General Finance",
    "q-fin.MF": "Mathematical Finance",
    "q-fin.PM": "Portfolio Management",
    "q-fin.PR": "Pricing of Securities",
    "q-fin.RM": "Risk Management",
    "q-fin.ST": "Statistical Finance",
    "q-fin.TR": "Trading and Market Microstructure",

    # Statistics
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.ML": "Machine Learning",
    "stat.OT": "Other Statistics",
    "stat.TH": "Statistics Theory",
}



CAT_TOKEN_RE = re.compile(r"[A-Za-z\-]+(?:\.[A-Za-z\-]+)?")

def parse_codes(cat_field: str) -> list[str]:
    """Extract arXiv category codes from a category string."""
    if not isinstance(cat_field, str):
        return []
    # categories often space-separated; we also accept commas/semicolons
    return CAT_TOKEN_RE.findall(cat_field)

def pretty_join(items: list[str], max_len: int = 64) -> str:
    """Join items with comma, truncate to max_len chars."""
    s = ", ".join(items)
    return s if len(s) <= max_len else s[:max_len - 1] + "…"

# Load metadata
print("Loading metadata with clusters…")
meta = pd.read_parquet(IN_META)

if "cluster" not in meta.columns:
    raise ValueError("Input parquet must contain a 'cluster' column.")


# Explode categories -> counts
print("Parsing category codes…")
meta = meta.copy()
meta["codes"] = meta["categories"].apply(parse_codes)

exploded = meta[["cluster", "codes"]].explode("codes").dropna()
exploded["codes"] = exploded["codes"].astype(str)

print("Counting codes per cluster…")
counts = (exploded
          .groupby(["cluster", "codes"])
          .size()
          .rename("n")
          .reset_index())

# Also compute per-cluster totals to get shares
cluster_totals = counts.groupby("cluster")["n"].sum().rename("cluster_total")
counts = counts.merge(cluster_totals, on="cluster", how="left")
counts["share"] = counts["n"] / counts["cluster_total"]

# Save raw counts
counts.sort_values(["cluster", "n"], ascending=[True, False]).to_csv(
    OUT_DIR / "cluster_category_counts.csv", index=False
)


# Build labels per cluster
TOP_N = 3                 # how many category codes to show in label
MIN_SHARE = 0.05          # only keep codes that contribute at least 5% of a cluster

rows = []
for k, sub in counts.groupby("cluster"):
    sub = sub.sort_values("n", ascending=False)

    # Filter by share then take TOP_N
    sub_filt = sub[sub["share"] >= MIN_SHARE].head(TOP_N)
    if sub_filt.empty:
        sub_filt = sub.head(TOP_N)

    codes = sub_filt["codes"].tolist()
    names = [NAME_MAP.get(c, c) for c in codes]

    # Short label: join human names (or codes if unknown)
    # Longer label: include both code and name
    label_auto = pretty_join(names, max_len=80)
    label_alt = pretty_join([f"{c} — {NAME_MAP.get(c, c)}" for c in codes], max_len=120)

    rows.append({
        "cluster": int(k),
        "label_auto": label_auto,
        "label_alt": label_alt,
        "top_codes": ", ".join(codes),
        "top_names": ", ".join(names),
    })

# Save labels
labels = pd.DataFrame(rows).sort_values("cluster")
labels.to_csv(OUT_DIR / "cluster_labels_from_categories.csv", index=False)


# Join back to meta for plotting
meta_out = meta.merge(labels[["cluster", "label_auto", "label_alt"]],
                      on="cluster", how="left")
meta_out.to_parquet(OUT_DIR / "doc_meta_with_cat_labels.parquet", index=False)

print("Saved:")
print(" - emb_out/cluster_category_counts.csv")
print(" - emb_out/cluster_labels_from_categories.csv")
print(" - emb_out/doc_meta_with_cat_labels.parquet")
