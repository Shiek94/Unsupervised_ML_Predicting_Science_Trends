# Stream and clean NDJSON from LaTeX commands
# Build text from abstracts and titles // get update_year from update_date // keep id and categories

import orjson
import datetime
import re

in_path = "arxiv_metadata_science_trends.json"
out_path = "cleaned_from_latex_min.json"

keep = ("id", "title", "abstract", "categories", "update_date")

# Replace common LaTeX operators/relations with readable tokens
OP_MAP = [
    (r"\\geq?",  " >="),
    (r"\\leq?",  " <="),
    (r"\\approx"," ~ "),
    (r"\\sim",   " ~ "),
    (r"\\pm",    " +/- "),
    (r"\\times", " x "),
    (r"\\cdot",  " * "),
    (r"\\infty", " infinity "),
    (r"\\to",    " to "),
    (r"\\mapsto"," -> "),
]

# Convert commands to words (\alpha -> alpha, \Lambda -> Lambda, \ell -> ell, etc.)
CMD = re.compile(r"\\([a-zA-Z]+)")

# Remove braces/sub/superscripts but keep content
remove_bracers_and_subscript = re.compile(r"[{}_^]")

# Remove style-only words left after command removal
STYLE_WORDS = re.compile(r"\b(mathbb|mathrm|mathbf|mathcal|operatorname|rm)\b", re.IGNORECASE)

# Normalize $...$, \( ... \), \[ ... \] by keeping their cleaned inner content
remove_latex_inline_math  = re.compile(r"\${1,2}([^$]+)\${1,2}")
remove_latex_parentheses  = re.compile(r"\\\(([^)]+)\\\)")
remove_latex_brackets     = re.compile(r"\\\[(.*?)\\")

def _op_replace(s: str) -> str:
    for pat, repl in OP_MAP:
        s = re.sub(pat, repl, s)
    return s

def _normalize_math_blocks(text: str) -> str:
    def _repl(m):
        inner = _op_replace(m.group(1))
        inner = CMD.sub(r"\1", inner)
        inner = remove_bracers_and_subscript.sub(" ", inner)
        inner = STYLE_WORDS.sub(" ", inner)
        return " " + " ".join(inner.split()) + " "
    text = remove_latex_inline_math.sub(_repl, text)
    text = remove_latex_parentheses.sub(_repl, text)
    text = remove_latex_brackets.sub(_repl, text)
    return text

# Fix "x 10 (-22)" -> "e-22"
SCI_FIX = re.compile(r"\b[x×]\s*10\s*\(\s*(-?\d+)\s*\)")

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = _op_replace(s)
    s = _normalize_math_blocks(s)
    s = CMD.sub(r"\1", s)
    s = STYLE_WORDS.sub(" ", s)
    s = remove_bracers_and_subscript.sub(" ", s)
    s = SCI_FIX.sub(r"e\1", s)
    s = " ".join(s.split())
    return s

with open(in_path, "rb") as f_in, open(out_path, "wb") as f_out:
    for line in f_in:
        o = orjson.loads(line)

        # pull the necessary fields
        doc_id     = o.get("id")
        categories = o.get("categories")
        update_date = o.get("update_date")

        # clean title/abstract and build combined text (title weighted 2×)
        title    = clean_text((o.get("title") or "").strip())
        abstract = clean_text((o.get("abstract") or "").strip())
        text     = " ".join(f"{title}. {title}. {abstract}".split())

        # year only
        update_year = None
        if update_date:
            date_obj = datetime.datetime.strptime(update_date, "%Y-%m-%d")
            update_year = date_obj.year

        # rewrite the cleaned doc
        rec = {
            "id": doc_id,
            "categories": categories,
            "text": text,
            "update_year": update_year,
        }
        f_out.write(orjson.dumps(rec, option=orjson.OPT_APPEND_NEWLINE))
