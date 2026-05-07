"""
preprocess.py
Amharic-specific text normalization and dataset splitting for AfriAware.
"""

import re
import unicodedata
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─── Amharic normalisation ───────────────────────────────────────────────────

# Ethiopic Unicode block: U+1200–U+137F (Amharic lives here)
# Some characters have equivalent forms that should be unified.
AMHARIC_EQUIV = {
    # Equivalent Ethiopic characters (different codepoints, same sound)
    "\u1200": "\u1200",  # ሀ → ሀ  (keep as reference)
    "\u1210": "\u1210",  # ሐ → ሐ
    "\u1215": "\u1200",  # ኅ → ሀ  (rare equivalent)
    "\u1206": "\u1200",  # ሆ stays (but ሗ → ሆ in some normalisations)
    # Doubled vowels
    "\u12A0": "\u12A0",  # አ  standard
    "\u1A00": "\u12A0",  # rare variant
}

URL_RE    = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")      # keep text, remove #
EMOJI_RE  = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
MULTI_SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]{2,}")  # collapse repeated punctuation


def normalize_amharic(text: str) -> str:
    """
    Light normalisation suited for fine-tuning multilingual BERT.
    Keeps Ethiopic script intact; strips noise only.

    Steps:
    1. Unicode NFC normalisation
    2. Remove URLs
    3. Remove @mentions (keep hashtag text)
    4. Remove emojis
    5. Collapse repeated punctuation
    6. Collapse whitespace
    7. Strip
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)   # "#ሰላም" → " ሰላም "
    text = EMOJI_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def apply_equivalences(text: str) -> str:
    """
    Optional: unify known Ethiopic character equivalences.
    Use only if your tokenizer doesn't handle them natively.
    """
    return "".join(AMHARIC_EQUIV.get(c, c) for c in text)


# ─── Label mapping ───────────────────────────────────────────────────────────

HATE_SPEECH_LABEL_MAP = {
    # AfriHate
    "hate":     0,
    "abusive":  1,
    "normal":   2,
    # RANLP (mapped to same ids)
    "offensive": 1,  # treat offensive ≈ abusive
}

FAKE_NEWS_LABEL_MAP = {
    "fake": 0,
    "real": 1,
}


def encode_labels(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Adds a 'label_id' column with integer labels.
    task: 'hate_speech' | 'fake_news'
    """
    mapping = HATE_SPEECH_LABEL_MAP if task == "hate_speech" else FAKE_NEWS_LABEL_MAP
    df = df.copy()
    df["label_id"] = df["label"].str.lower().str.strip().map(mapping)

    unknown = df["label_id"].isna().sum()
    if unknown > 0:
        print(f"  ⚠ {unknown} rows have unknown labels — dropping them.")
        df = df.dropna(subset=["label_id"])

    df["label_id"] = df["label_id"].astype(int)
    return df


# ─── Full preprocessing pipeline ────────────────────────────────────────────

def preprocess_dataset(
    df: pd.DataFrame,
    task: str,
    text_col: str = "text",
    label_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Full preprocessing pipeline for a single dataset.

    Returns a dict with 'train', 'val', 'test' DataFrames and
    'label_map' for reference.

    Args:
        df         : Raw DataFrame (must have text_col and label_col)
        task       : 'hate_speech' | 'fake_news'
        text_col   : Column with raw text
        label_col  : Column with string labels
        test_size  : Fraction for test split
        val_size   : Fraction of remaining data for validation
        random_state: Reproducibility seed
    """
    df = df.copy()

    # 1. Rename columns to standard names
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]].dropna()

    # 2. Normalise text
    print("  Normalising text...")
    df["text"] = df["text"].map(normalize_amharic)
    df = df[df["text"].str.len() > 2]  # drop near-empty

    # 3. Encode labels
    df = encode_labels(df, task)

    # 4. Use pre-existing splits if available, else create them
    if "split" in df.columns:
        train = df[df["split"] == "train"].drop(columns=["split"], errors="ignore")
        test  = df[df["split"] == "test"].drop(columns=["split"], errors="ignore")
        val   = df[df["split"] == "validation"].drop(columns=["split"], errors="ignore")
        if val.empty:
            # create val from train
            train, val = train_test_split(
                train, test_size=val_size, random_state=random_state, stratify=train["label_id"]
            )
    else:
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df["label_id"]
        )
        train, val = train_test_split(
            train_val,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=train_val["label_id"],
        )

    label_map = (
        HATE_SPEECH_LABEL_MAP if task == "hate_speech" else FAKE_NEWS_LABEL_MAP
    )
    id2label  = {v: k for k, v in label_map.items()}

    splits = {"train": train, "val": val, "test": test}
    for name, split in splits.items():
        print(f"  {name:<6}: {len(split):>5} rows | "
              + " | ".join(f"{id2label.get(lid, lid)}: {cnt}"
                           for lid, cnt in split["label_id"].value_counts().sort_index().items()))

    return {"train": train, "val": val, "test": test, "label_map": label_map, "id2label": id2label}


def save_splits(splits: dict, prefix: str):
    """Saves train/val/test CSVs to data/processed/."""
    for split_name in ("train", "val", "test"):
        path = PROCESSED_DIR / f"{prefix}_{split_name}.csv"
        splits[split_name].to_csv(path, index=False)
        print(f"  Saved → {path}")


# ─── Main (run as script) ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Hate speech: AfriHate Amharic
    afrihate_path = RAW_DIR / "afrihate_amharic.csv"
    if afrihate_path.exists():
        print("\n--- Preprocessing AfriHate (amh) ---")
        df = pd.read_csv(afrihate_path)
        splits = preprocess_dataset(df, task="hate_speech")
        save_splits(splits, prefix="afrihate_amh")

    # Hate speech: RANLP (larger, can be used for pre-training)
    ranlp_path = RAW_DIR / "ranlp_amharic.csv"
    if ranlp_path.exists():
        print("\n--- Preprocessing RANLP (amh) ---")
        df = pd.read_csv(ranlp_path)
        splits = preprocess_dataset(df, task="hate_speech")
        save_splits(splits, prefix="ranlp_amh")

    # Note: ETH_FAKE processing removed (dataset unavailable)

    print("\n✓ Preprocessing complete.")
