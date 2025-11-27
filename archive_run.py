# archive_run.py
# Copy artifacts and figures from emb_out/ into outputs/YYYYMMDD_HHMMSS_run/
# Safe to run multiple times; it never deletes anything.

from pathlib import Path
from datetime import datetime
import shutil
import glob

ROOT = Path(".")
SRC  = ROOT / "emb_out"
DST  = ROOT / "outputs" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_run")

# What to copy (extend if you add new artifact types)
PATTERNS = [
    "*.npy", "*.joblib", "*.parquet", "*.csv", "*.json",
    "*.png", "*.jpg", "*.jpeg", "*.pdf",
    # add model files if any: "*.pt", "*.bin", etc.
]

def main():
    if not SRC.exists():
        raise SystemExit(f"Source folder missing: {SRC.resolve()}")
    DST.mkdir(parents=True, exist_ok=True)

    copied = 0
    for pat in PATTERNS:
        for src_path in glob.glob(str(SRC / pat)):
            src = Path(src_path)
            dst = DST / src.name
            shutil.copy2(src, dst)
            copied += 1

    print(f"Archived {copied} files to {DST.resolve()}")
    print("You can now safely clean temporary clutter in your working folder.")

if __name__ == "__main__":
    main()
