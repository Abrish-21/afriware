---
language:
  - am
tags:
  - hate-speech
  - text-classification
  - amharic
  - african-languages
  - bert
  - ai-for-good
license: mit
datasets:
  - afrihate/afrihate
base_model: Davlan/bert-base-multilingual-cased-finetuned-amharic
pipeline_tag: text-classification
---

# AfriAware — Amharic Hate Speech Classifier 🛡️

A fine-tuned Amharic hate speech classifier for digital safety research in Ethiopia and across Africa.

## Usage

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="YOUR_USERNAME/afriaware-amharic-hate-speech"
)

result = classifier("ኢትዮጵያ ሀገራችን ናት")
# [{'label': 'normal', 'score': 0.94}]
```

## Labels

| Label | ID | Meaning |
|-------|----|---------|
| `hate` | 0 | Hatred directed at a person or group |
| `abusive` | 1 | Offensive/aggressive language |
| `normal` | 2 | Non-harmful content |

## Training Details

| Setting | Value |
|---------|-------|
| Base model | `Davlan/bert-base-multilingual-cased-finetuned-amharic` |
| Dataset | [AfriHate (amh)](https://huggingface.co/datasets/afrihate/afrihate) |
| Epochs | 5 (early stopping, patience=2) |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max length | 128 tokens |
| Optimizer | AdamW, weight decay 0.01 |

## Dataset

Trained on the Amharic subset of [AfriHate](https://huggingface.co/datasets/afrihate/afrihate)
(Muhammad et al., NAACL 2025) — tweets annotated by native Amharic speakers with
full sociocultural context.

## Ethical Considerations

- Intended for **research and journalist support tools**, not automated removal
- Annotations reflect Ethiopian social media from 2020–2022
- Model should always be combined with human review
- Do not use for surveillance of individuals

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
