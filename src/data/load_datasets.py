"""
load_datasets.py
Downloads and caches all three datasets used in AfriAware.
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


# ─── ETH_FAKE (Amharic Fake News) ───────────────────────────────────────────

def load_eth_fake(csv_path: str = None) -> pd.DataFrame:
    """
    Loads the ETH_FAKE dataset.

    If csv_path is provided, loads from that local file.
    Otherwise attempts to load from HuggingFace (community mirror).

    Dataset: 3,417 real + 3,417 fake Amharic news articles
    Labels: real | fake
    """
    if csv_path and Path(csv_path).exists():
        print(f"Loading ETH_FAKE from local file: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("Loading ETH_FAKE from HuggingFace (community mirror)...")
        try:
            ds = load_dataset("Fanpoliti/ETH_FAKE", trust_remote_code=True)
            frames = []
            for split_name, split in ds.items():
                d = split.to_pandas()
                d["split"] = split_name
                frames.append(d)
            df = pd.concat(frames, ignore_index=True)
        except Exception:
            print(
                "  ⚠ ETH_FAKE not found on HuggingFace.\n"
                "  Download manually from:\n"
                "    https://github.com/Fanpoliti/ETH_FAKE\n"
                "  Then run:  load_eth_fake(csv_path='data/raw/ETH_FAKE.csv')"
            )
            return pd.DataFrame()

    # Normalise column names
    col_map = {}
    for c in df.columns:
        if c.lower() in ("content", "article", "body", "news"):
            col_map[c] = "text"
        if c.lower() in ("class", "category", "label_str"):
            col_map[c] = "label"
    df = df.rename(columns=col_map)

    if "label" in df.columns:
        df["label"] = df["label"].str.lower().str.strip()

    out_path = RAW_DIR / "eth_fake.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ ETH_FAKE saved → {out_path}  ({len(df)} rows)")
    if "label" in df.columns:
        print(f"  Label distribution:\n{df['label'].value_counts().to_string()}\n")
    return df


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
    loaded["ETH_FAKE"]       = load_eth_fake()

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
