<div align="center">

# 🛡️ AfriAware

### Amharic Hate Speech Detection for Digital Wellbeing

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20Hub-yellow)](https://huggingface.co/YOUR_USERNAME/afriaware-amharic-hate-speech)
[![HF Space](https://img.shields.io/badge/🤗-Live%20Demo-orange)](https://huggingface.co/spaces/YOUR_USERNAME/afriaware-demo)
[![Dataset: AfriHate](https://img.shields.io/badge/dataset-AfriHate%20(NAACL%202025)-purple)](https://huggingface.co/datasets/afrihate/afrihate)

*Protecting African communities from online hate — one tweet at a time.*

[Live Demo](https://huggingface.co/spaces/YOUR_USERNAME/afriaware-demo) · [Model Weights](https://huggingface.co/YOUR_USERNAME/afriaware-amharic-hate-speech) · [Dataset](https://huggingface.co/datasets/afrihate/afrihate) · [Deployment Guide](DEPLOY.md)

</div>

---

## 🌍 Why This Matters

Hate speech detection fails African communities because existing tools:

- ❌ Were built on English/European data
- ❌ Ignore sociocultural context in African languages
- ❌ Rely on keyword spotting without understanding

**AfriAware** fixes this by fine-tuning a multilingual BERT model on [AfriHate](https://huggingface.co/datasets/afrihate/afrihate) — a dataset built *by* African researchers, annotated *by* native Amharic speakers, with full cultural context.

---

## 🎯 What It Does

```
Input: Amharic tweet or social media post
         │
         ▼
  ┌─────────────────────────────────┐
  │   Text Normalization            │  Remove URLs, mentions, emoji
  │   (Ethiopic-aware)              │  Unicode NFC normalisation
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │   mBERT Encoder                 │  Davlan/bert-base-multilingual
  │   fine-tuned on Amharic         │  -cased-finetuned-amharic
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │   Classification Head           │  Linear → Softmax
  │   3-class output                │
  └──────────────┬──────────────────┘
                 │
         ┌───────┴────────┐
         ▼                ▼
  🔴 hate            ← targets groups with hatred
  🟠 abusive         ← offensive, not group-targeted  
  🟢 normal          ← non-harmful content
```

---

## 📦 Dataset

| Dataset | Size | Labels | Source |
|---------|------|--------|--------|
| [AfriHate (amh)](https://huggingface.co/datasets/afrihate/afrihate) | ~3k Amharic tweets | hate / abusive / normal | NAACL 2025 |

AfriHate was built by African researchers and annotated by **native Amharic speakers** with deep sociocultural understanding — addressing the key failure mode of global moderation tools in African contexts.

> *"Hate speech detection often fails in Global South contexts due to absence of moderation, reliance on keyword spotting, and overlooking targeted campaigns against minority groups."*
> — Muhammad et al., AfriHate 2025

---

## 🏗️ Project Structure

```
afriaware/
│
├── 📁 app/
│   └── app.py                  ← Local Gradio demo (run this)
│
├── 📁 hf_space/
│   ├── app.py                  ← HuggingFace Space entry point
│   └── README.md               ← Space metadata (SDK, title, tags)
│
├── 📁 models/
│   └── final/                  ← Trained model weights live here
│       ├── model.safetensors   ← Main weights (you have this locally)
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── afriaware_config.json
│       └── training_args.bin
│
├── 📁 src/
│   ├── data/
│   │   └── preprocess.py       ← Amharic text normalization
│   ├── evaluation/
│   │   ├── metrics.py          ← F1, confusion matrix
│   │   └── explainability.py   ← SHAP token attribution
│   └── utils/
│       └── helpers.py          ← Shared config & utilities
│
├── 📁 docs/
│   └── model_card.md           ← HuggingFace model card
│
├── 📄 DEPLOY.md                ← Step-by-step HuggingFace deployment
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🚀 Quick Start

### 1 · Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/afriaware.git
cd afriaware
pip install -r requirements.txt
```

### 2 · Run the demo locally

> Make sure `models/final/model.safetensors` is present.

```bash
python app/app.py
# → http://localhost:7860
```

### 3 · Deploy to HuggingFace (see [DEPLOY.md](DEPLOY.md))

```bash
# Upload model weights
python upload_to_hub.py   # (script in DEPLOY.md)

# Push Space
cd hf_space && git push
# → https://huggingface.co/spaces/YOUR_USERNAME/afriaware-demo
```

---

## 🧠 Model Details

| Setting | Value |
|---------|-------|
| Base model | `Davlan/bert-base-multilingual-cased-finetuned-amharic` |
| Task | 3-class text classification |
| Max sequence length | 128 tokens |
| Training epochs | 5 (early stopping, patience=2) |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Optimizer | AdamW, weight decay=0.01, warmup ratio=0.1 |
| Seed | 42 |

### Why this base model?

`Davlan/bert-base-multilingual-cased-finetuned-amharic` was chosen because it:
- Is already pre-trained on Amharic text (Ethiopian news corpus)
- Handles Ethiopic Unicode script natively
- Achieves ~91.6% accuracy on Amharic NER tasks out of the box
- Fits in free Google Colab T4 GPU memory

---

## 📊 Results

| Metric | Score |
|--------|-------|
| F1 (macro) | *Run `notebooks/02_hate_speech_training.ipynb` to populate* |
| Accuracy | — |
| Baseline (AfriHate paper, mBERT) | ~0.74 F1 macro |

---

## 🌍 Roadmap

- [x] Amharic hate speech detection (AfriHate)
- [x] Gradio demo with confidence scores
- [x] HuggingFace Space deployment
- [ ] Extend to **Oromo** (`orm`) and **Tigrinya** (`tir`) via AfriHate
- [ ] ONNX export for low-bandwidth deployment in Ethiopia
- [ ] REST API (FastAPI) for NGO integration
- [ ] Cross-lingual zero-shot transfer to all 15 AfriHate languages

---

## 🤝 Intended Users

This tool is designed for:

- 📰 **Journalists & fact-checkers** monitoring Ethiopian social media
- 🏢 **NGOs** studying online hate patterns in the Horn of Africa
- 🎓 **Researchers** in African NLP and computational social science
- 🏛️ **Policymakers** needing data on digital harm trends

> ⚠️ **Ethical note:** This model should support — not replace — human moderation. Always apply cultural judgment and human review to flagged content.

---

## 📄 Citation

If you use this work, please cite the underlying dataset:

```bibtex
@inproceedings{muhammad-etal-2025-afrihate,
    title     = "{A}fri{H}ate: A Multilingual Collection of Hate Speech and
                  Abusive Language Datasets for {A}frican Languages",
    author    = {Muhammad, Shamsuddeen Hassan and others},
    booktitle = "Proceedings of NAACL 2025",
    year      = "2025",
    url       = "https://aclanthology.org/2025.naacl-long.92/"
}
```

---

## 📃 License

MIT License — open for research and NGO use.

---

<div align="center">

Built with ❤️ in **Addis Ababa** — open-source African NLP for digital safety.

*Fighting hate and disinformation in Amharic, one tweet at a time.*

</div>
