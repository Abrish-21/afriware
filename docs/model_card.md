---
language:
  - am
tags:
  - hate-speech
  - text-classification
  - amharic
  - african-languages
  - digital-wellbeing
  - disinformation
license: mit
datasets:
  - afrihate/afrihate
  - uhhlt/amharichatespeechranlp
base_model: Davlan/bert-base-multilingual-cased-finetuned-amharic
---

# AfriAware — Amharic Hate Speech Classifier

Part of the **AfriAware** project: Digital Wellbeing Intelligence for African Languages.

## Model description

A fine-tuned version of `Davlan/bert-base-multilingual-cased-finetuned-amharic`
for 3-class Amharic hate speech detection.

**Labels:**
- `hate` — content targeting groups with hatred
- `abusive` — offensive but not directly targeting a group
- `normal` — non-harmful content

## Training data

- [AfriHate (amh)](https://huggingface.co/datasets/afrihate/afrihate) — tweets
  annotated by native Amharic speakers with sociocultural context
- [RANLP 2023 Amharic dataset](https://huggingface.co/datasets/uhhlt/amharichatespeechranlp)
  — 15,100 tweets, 3-class

## Intended uses and limitations

**Intended for:**
- Journalists and fact-checkers monitoring social media in Ethiopia
- NGOs studying hate speech patterns
- Research on low-resource African language NLP

**Not intended for:**
- Automated content removal without human review
- Any use without awareness of cultural context and model limitations
- Surveillance of individuals

## How to use

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="YOUR_USERNAME/afriaware-hate-speech-amharic"
)

result = classifier("ኢትዮጵያ ሀገራችን ናት")
# [{'label': 'normal', 'score': 0.94}]
```

## Training procedure

- Base model: `Davlan/bert-base-multilingual-cased-finetuned-amharic`
- Epochs: 5 (with early stopping, patience=2)
- Learning rate: 2e-5
- Batch size: 16
- Max sequence length: 128 tokens
- Optimizer: AdamW with weight decay 0.01

## Evaluation

| Metric | Score |
|--------|-------|
| F1 (macro) | TBD after training |
| Accuracy | TBD after training |

*Update this table after running `src/models/hate_speech_model.py --train`*

## Ethical considerations

Hate speech detection in low-resource languages carries significant risk
of false positives that could harm users from already marginalised communities.
This model should always be used with:

1. Human review of flagged content
2. Awareness of regional and dialectal variation in Amharic
3. Understanding that the training data reflects a specific time period
   (2020–2022) and socio-political context

## Citation

```bibtex
@inproceedings{muhammad-etal-2025-afrihate,
    title = "{A}fri{H}ate: A Multilingual Collection of Hate Speech and Abusive
             Language Datasets for {A}frican Languages",
    author = {Muhammad, Shamsuddeen Hassan and others},
    booktitle = "Proceedings of NAACL 2025",
    year = "2025"
}
```
