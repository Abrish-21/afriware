# AfriAware 🛡️
### Digital Wellbeing Intelligence for African Languages

AfriAware is an open-source NLP pipeline that detects **hate speech** and **disinformation** in Amharic — with a clear path to scale across 15 African languages via the AfriHate dataset.

This project demonstrates how culturally-grounded AI can protect communities from online harm.


## 🎯 What It Does

| Task | Input | Output |
|------|-------|--------|
| Hate speech detection | Amharic tweet | `hate` / `abusive` / `normal` + confidence |
| Fake news detection | Amharic news article | `fake` / `real` + confidence |
| Explainability | Either input | Top tokens driving the prediction (SHAP) |


## 📦 Datasets Used

| Dataset | Source | Size | Labels |
|---------|--------|------|--------|
| [AfriHate (amh)](https://huggingface.co/datasets/afrihate/afrihate) | HuggingFace | ~3k Amharic tweets | hate / abusive / normal |
| [RANLP Amharic](https://huggingface.co/datasets/uhhlt/amharichatespeechranlp) | HuggingFace | 15,100 tweets | hate / offensive / normal |


## 🏗️ Project Structure

```
afriaware/
├── data/
│   ├── raw/               # Downloaded datasets (gitignored)
│   └── processed/         # Cleaned, split datasets
├── src/
│   ├── data/
│   │   ├── load_datasets.py     # HuggingFace + local data loaders
│   │   └── preprocess.py        # Amharic text normalization
│   ├── models/
│   │   ├── hate_speech_model.py # Fine-tuning wrapper
│   │   └── fake_news_model.py   # Fine-tuning wrapper
│   ├── evaluation/
│   │   ├── metrics.py           # F1, precision, recall, confusion matrix
│   │   └── explainability.py    # SHAP token attribution
│   └── utils/
│       └── helpers.py           # Shared utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_hate_speech_training.ipynb
│   └── 03_fake_news_training.ipynb
├── app/
│   └── demo.py                  # Gradio web demo
├── tests/
│   └── test_pipeline.py
├── docs/
│   └── model_card.md            # HuggingFace model card
├── requirements.txt
├── setup.py
└── README.md
```


## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/abrish-21/afriaware.git
cd afriaware
pip install -r requirements.txt
```

### 2. Download datasets
```bash
python src/data/load_datasets.py
```

### 3. Train models
```bash
# Hate speech classifier
python src/models/hate_speech_model.py --train

# Fake news classifier
python src/models/fake_news_model.py --train
```

### 4. Run the demo
```bash
python app/demo.py
# Opens at http://localhost:7860
```


## 🧠 Model Architecture

Both classifiers share the same base:

```
Input (Amharic text)
    ↓
Tokenizer (bert-base-multilingual-cased)
    ↓
Encoder (mBERT or AfroXLMR fine-tuned on Amharic)
    ↓
Classification head (Linear → Softmax)
    ↓
Label + Confidence score
```

**Base model options** (configurable in `src/models/`):


## 📊 Baseline Results

| Model | Dataset | F1 (macro) |
|-------|---------|-----------|
| mBERT fine-tuned | AfriHate (amh) | ~0.74 (AfriHate paper) |
| mBERT fine-tuned | RANLP (amh) | ~0.917 (prior work) |


## 🌍 Roadmap



## 🤝 Citation

If you use this work, please also cite the underlying datasets:

```bibtex
@inproceedings{muhammad-etal-2025-afrihate,
    title = "{A}fri{H}ate: A Multilingual Collection of Hate Speech and Abusive Language Datasets for {A}frican Languages",
    author = {Muhammad, Shamsuddeen Hassan and others},
    booktitle = "Proceedings of NAACL 2025",
    year = "2025"
}
```


## 📄 License

MIT License — open for research and NGO use.


