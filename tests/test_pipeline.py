"""
test_pipeline.py
Basic unit tests for AfriAware preprocessing and utilities.
Run: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
from src.data.preprocess import normalize_amharic, encode_labels, preprocess_dataset


# ─── Normalization tests ──────────────────────────────────────────────────────

class TestNormalizeAmharic:
    def test_removes_url(self):
        assert "https://example.com" not in normalize_amharic("ሰላም https://example.com ዓለም")

    def test_removes_mention(self):
        assert "@user" not in normalize_amharic("@user ጥሩ ጽሑፍ")

    def test_keeps_hashtag_text(self):
        result = normalize_amharic("#ኢትዮጵያ ጥሩ ሀገር")
        assert "ኢትዮጵያ" in result
        assert "#" not in result

    def test_removes_emoji(self):
        result = normalize_amharic("ሰላም 😊 ዓለም")
        assert "😊" not in result
        assert "ሰላም" in result

    def test_collapses_whitespace(self):
        result = normalize_amharic("ሰላም    ዓለም")
        assert "  " not in result

    def test_handles_none(self):
        assert normalize_amharic(None) == ""

    def test_handles_empty(self):
        assert normalize_amharic("") == ""

    def test_preserves_amharic(self):
        text = "ኢትዮጵያ ሀገራችን ናት"
        result = normalize_amharic(text)
        assert "ኢትዮጵያ" in result
        assert "ሀገራችን" in result


# ─── Label encoding tests ─────────────────────────────────────────────────────

class TestEncodeLabels:
    def make_df(self, labels):
        return pd.DataFrame({"text": ["sample"] * len(labels), "label": labels})

    def test_hate_speech_labels(self):
        df = self.make_df(["hate", "abusive", "normal"])
        result = encode_labels(df, "hate_speech")
        assert list(result["label_id"]) == [0, 1, 2]

    def test_fake_news_labels(self):
        df = self.make_df(["fake", "real"])
        result = encode_labels(df, "fake_news")
        assert list(result["label_id"]) == [0, 1]

    def test_drops_unknown_labels(self):
        df = self.make_df(["hate", "UNKNOWN", "normal"])
        result = encode_labels(df, "hate_speech")
        assert len(result) == 2

    def test_case_insensitive(self):
        df = self.make_df(["HATE", "Abusive", "Normal"])
        result = encode_labels(df, "hate_speech")
        assert list(result["label_id"]) == [0, 1, 2]


# ─── Preprocessing pipeline tests ────────────────────────────────────────────

class TestPreprocessDataset:
    @pytest.fixture
    def hate_df(self):
        return pd.DataFrame({
            "text":  ["ሰላም ሰው", "ጥሩ ጽሑፍ", "ሌላ ጽሑፍ"] * 20,
            "label": ["normal",  "hate",    "abusive"] * 20,
        })

    def test_returns_three_splits(self, hate_df):
        splits = preprocess_dataset(hate_df, task="hate_speech")
        assert "train" in splits
        assert "val"   in splits
        assert "test"  in splits

    def test_no_overlap_between_splits(self, hate_df):
        splits = preprocess_dataset(hate_df, task="hate_speech")
        train_texts = set(splits["train"]["text"].tolist())
        test_texts  = set(splits["test"]["text"].tolist())
        # Note: identical text strings can appear in multiple splits (not unique)
        # so we check indices instead
        train_idx = set(splits["train"].index)
        test_idx  = set(splits["test"].index)
        assert train_idx.isdisjoint(test_idx)

    def test_label_ids_present(self, hate_df):
        splits = preprocess_dataset(hate_df, task="hate_speech")
        assert "label_id" in splits["train"].columns
        assert splits["train"]["label_id"].notna().all()

    def test_correct_task_labels(self, hate_df):
        splits = preprocess_dataset(hate_df, task="hate_speech")
        all_ids = set(splits["train"]["label_id"].unique())
        assert all_ids.issubset({0, 1, 2})


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
