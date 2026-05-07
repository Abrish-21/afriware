"""
load_datasets.py
Downloads and caches the datasets used in AfriAware.
Run directly:  python src/data/load_datasets.py
"""

import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─── AfriHate (Amharic subset) ──────────────────────────────────────────────

def load_afrihate_amharic() -> pd.DataFrame:
    """
    Loads the Amharic subset of AfriHate from HuggingFace.
    Labels: hate | abusive | normal
    Columns: id, text, label
    """
    print("Loading AfriHate (amh) from HuggingFace...")
    ds = load_dataset("afrihate/afrihate", "amh", trust_remote_code=True)

    frames = []
    for split_name, split in ds.items():
        df = split.to_pandas()
        df["split"] = split_name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Normalise column names
    if "tweet" in df.columns:
        df = df.rename(columns={"tweet": "text"})

    out_path = RAW_DIR / "afrihate_amharic.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ AfriHate saved → {out_path}  ({len(df)} rows)")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}\n")
    return df


# ─── RANLP Amharic Hate Speech ───────────────────────────────────────────────

def load_ranlp_amharic() -> pd.DataFrame:
    """
    Loads the RANLP 2023 Amharic hate speech dataset.
    Labels: Hate | Offensive | Normal   (15,100 tweets)
    """
    print("Loading RANLP Amharic hate speech from HuggingFace...")
    ds = load_dataset("uhhlt/amharichatespeechranlp", trust_remote_code=True)

    frames = []
    for split_name, split in ds.items():
        df = split.to_pandas()
        df["split"] = split_name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Standardise to lowercase labels for consistency
    if "label" in df.columns:
        df["label"] = df["label"].str.lower()
    if "text" not in df.columns and "tweet" in df.columns:
        df = df.rename(columns={"tweet": "text"})

    out_path = RAW_DIR / "ranlp_amharic.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ RANLP saved → {out_path}  ({len(df)} rows)")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}\n")
    return df


# ETH_FAKE dataset support removed (dataset unavailable upstream).


# ─── Dataset summary ────────────────────────────────────────────────────────

def print_summary(datasets: dict):
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    for name, df in datasets.items():
        if df is None or df.empty:
            print(f"\n{name}: NOT LOADED")
            continue
        print(f"\n{name}")
        print(f"  Rows   : {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        if "label" in df.columns:
            for label, count in df["label"].value_counts().items():
                pct = count / len(df) * 100
                print(f"    {label:<12}: {count:>5} ({pct:.1f}%)")
        if "split" in df.columns:
            print(f"  Splits : {df['split'].value_counts().to_dict()}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loaded = {}

    loaded["AfriHate (amh)"] = load_afrihate_amharic()
    loaded["RANLP (amh)"]    = load_ranlp_amharic()
    # ETH_FAKE removed: do not attempt to load

    print_summary(loaded)

    # Save a manifest
    manifest = {
        name: {"rows": len(df), "columns": list(df.columns)}
        for name, df in loaded.items()
        if df is not None and not df.empty
    }
    manifest_path = RAW_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✓ Manifest saved → {manifest_path}")
