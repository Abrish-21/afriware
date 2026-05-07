
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import evaluate



ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR    = ROOT / "models" / "hate_speech"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "base_model":   "Davlan/bert-base-multilingual-cased-finetuned-amharic",
    "dataset_prefix": "afrihate_amh",       # switch to ranlp_amh for 15k dataset
    "num_labels":   3,
    "id2label":     {0: "hate", 1: "abusive", 2: "normal"},
    "label2id":     {"hate": 0, "abusive": 1, "normal": 2},
    "max_length":   128,
    "batch_size":   16,
    "num_epochs":   5,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "seed":         42,
}


# ─── Dataset class ───────────────────────────────────────────────────────────

class AmharicDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(df["label_id"].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ─── Metrics ─────────────────────────────────────────────────────────────────

_f1_metric  = evaluate.load("f1")
_acc_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    f1  = _f1_metric.compute(predictions=preds, references=labels, average="macro")
    acc = _acc_metric.compute(predictions=preds, references=labels)

    return {"f1_macro": f1["f1"], "accuracy": acc["accuracy"]}


# ─── Trainer setup ───────────────────────────────────────────────────────────

def build_trainer(config: dict) -> Trainer:
    prefix = config["dataset_prefix"]

    train_df = pd.read_csv(PROCESSED_DIR / f"{prefix}_train.csv")
    val_df   = pd.read_csv(PROCESSED_DIR / f"{prefix}_val.csv")

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

    train_ds = AmharicDataset(train_df, tokenizer, config["max_length"])
    val_ds   = AmharicDataset(val_df,   tokenizer, config["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        num_labels=config["num_labels"],
        id2label=config["id2label"],
        label2id=config["label2id"],
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=config["seed"],
        logging_steps=50,
        report_to="none",   # set to "wandb" if you want experiment tracking
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    return trainer, tokenizer


# ─── Train ───────────────────────────────────────────────────────────────────

def train(config: dict = None):
    config = config or DEFAULT_CONFIG
    print(f"\n{'='*55}")
    print("AfriAware — Hate Speech Model Training")
    print(f"{'='*55}")
    print(f"Base model : {config['base_model']}")
    print(f"Dataset    : {config['dataset_prefix']}")
    print(f"Epochs     : {config['num_epochs']}")
    print(f"Batch size : {config['batch_size']}")

    trainer, tokenizer = build_trainer(config)
    trainer.train()

    # Save final model
    model_path = OUTPUT_DIR / "final"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    # Save config
    with open(model_path / "afriaware_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Evaluate on test set
    prefix   = config["dataset_prefix"]
    test_df  = pd.read_csv(PROCESSED_DIR / f"{prefix}_test.csv")
    test_ds  = AmharicDataset(test_df, tokenizer, config["max_length"])
    results  = trainer.evaluate(test_ds)

    print(f"\n{'='*55}")
    print("TEST SET RESULTS")
    print(f"{'='*55}")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    with open(OUTPUT_DIR / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Model saved to {model_path}")
    return trainer


# ─── Predict ─────────────────────────────────────────────────────────────────

def predict(text: str, model_path: str = None, config: dict = None) -> dict:
    """
    Runs inference on a single Amharic text.
    Returns: {"label": str, "confidence": float, "scores": dict}
    """
    config     = config or DEFAULT_CONFIG
    model_path = model_path or str(OUTPUT_DIR / "final")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=config["max_length"],
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs    = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id  = int(torch.argmax(logits, dim=-1))
    id2label = config["id2label"]

    return {
        "label":      id2label[pred_id],
        "confidence": round(probs[pred_id], 4),
        "scores":     {id2label[i]: round(p, 4) for i, p in enumerate(probs)},
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AfriAware hate speech model")
    parser.add_argument("--train",   action="store_true", help="Fine-tune the model")
    parser.add_argument("--predict", type=str,            help="Run inference on a text")
    parser.add_argument("--model",   type=str,            help="Path to saved model (for predict)")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        result = predict(args.predict, model_path=args.model)
        print(f"\nInput  : {args.predict}")
        print(f"Label  : {result['label']}  ({result['confidence']*100:.1f}% confidence)")
        print(f"Scores : {result['scores']}")
    else:
        parser.print_help()
