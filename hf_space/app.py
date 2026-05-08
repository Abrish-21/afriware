"""
app.py  –  HuggingFace Space entry point for AfriAware.

On HuggingFace Spaces, this file must be at the ROOT of the repo.
The model weights are loaded from the Hub (see README for upload steps).
"""

import json
import os
from pathlib import Path

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── Config ───────────────────────────────────────────────────────────────────

# On HF Spaces: set HF_MODEL_ID in Space secrets or hardcode after uploading weights
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "abrhamLearns/afriaware-amharic-hate-speech")

# Labels
ID2LABEL = {0: "hate", 1: "abusive", 2: "normal"}
LABEL2ID = {"hate": 0, "abusive": 1, "normal": 2}
MAX_LENGTH = 128

COLORS = {"hate": "#E8593C", "abusive": "#EF9F27", "normal": "#1D9E75"}
EMOJIS = {"hate": "🔴", "abusive": "🟠", "normal": "🟢"}
DESCRIPTIONS = {
    "hate":     "Content that expresses hatred toward a person or group.",
    "abusive":  "Offensive or aggressive language, not necessarily group-targeted.",
    "normal":   "Regular, non-harmful content.",
}

# ─── Load model ───────────────────────────────────────────────────────────────

print(f"Loading model from: {HF_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_ID,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)
model.eval()
print("✓ Model loaded")


# ─── Preprocessing ────────────────────────────────────────────────────────────

import re, unicodedata

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", " ", text
    )
    return re.sub(r"\s+", " ", text).strip()


# ─── Inference ────────────────────────────────────────────────────────────────

def analyse(text: str):
    if not text.strip():
        return (
            "<div style='color:#888;font-size:15px'>Enter some Amharic text above.</div>",
            None, "",
        )

    clean  = normalize(text)
    inputs = tokenizer(clean, return_tensors="pt", truncation=True,
                       padding=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs   = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id = int(torch.argmax(logits, dim=-1))
    label   = ID2LABEL[pred_id]
    conf    = probs[pred_id]
    scores  = {ID2LABEL[i]: probs[i] for i in range(3)}
    color   = COLORS[label]

    verdict_html = f"""
<div style="border-left:5px solid {color};background:{color}18;
            padding:16px 20px;border-radius:8px;margin-bottom:8px">
  <div style="font-size:28px;font-weight:700;color:{color}">
    {EMOJIS[label]} {label.upper()}
  </div>
  <div style="font-size:15px;color:#555;margin-top:4px">{DESCRIPTIONS[label]}</div>
  <div style="font-size:13px;color:#888;margin-top:8px">
    Confidence: <strong style="color:{color}">{conf*100:.1f}%</strong>
  </div>
</div>"""

    bars = ""
    for lbl, score in sorted(scores.items(), key=lambda x: -x[1]):
        c = COLORS[lbl]
        bars += f"""
<div style="margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
    <span>{EMOJIS[lbl]} <strong>{lbl}</strong></span>
    <span style="color:#666">{score*100:.1f}%</span>
  </div>
  <div style="background:#eee;border-radius:6px;height:10px;overflow:hidden">
    <div style="width:{score*100}%;background:{c};height:100%;border-radius:6px"></div>
  </div>
</div>"""

    scores_html = f"""<div style="padding:12px 0">
  <div style="font-size:12px;color:#888;margin-bottom:12px;text-transform:uppercase;
              letter-spacing:0.5px">All scores</div>{bars}</div>"""

    return verdict_html, scores_html, ""


# ─── UI ──────────────────────────────────────────────────────────────────────

CSS = ".gradio-container{max-width:820px!important;margin:auto} footer{display:none!important}"

HEADER = """
<div style="text-align:center;padding:24px 0 8px">
  <div style="font-size:42px">🛡️</div>
  <h1 style="font-size:28px;font-weight:800;margin:8px 0 4px;color:#111">AfriAware</h1>
  <p style="font-size:15px;color:#666;margin:0">
    Amharic Hate Speech Detection &nbsp;·&nbsp; Open-source African NLP
  </p>
  <div style="margin-top:10px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap">
    <span style="background:#f3f4f6;border-radius:20px;padding:3px 12px;font-size:12px;color:#444">🌍 AfriHate (NAACL 2025)</span>
    <span style="background:#f3f4f6;border-radius:20px;padding:3px 12px;font-size:12px;color:#444">🤗 mBERT fine-tuned</span>
    <span style="background:#f3f4f6;border-radius:20px;padding:3px 12px;font-size:12px;color:#444">🇪🇹 Amharic (አማርኛ)</span>
    <span style="background:#f3f4f6;border-radius:20px;padding:3px 12px;font-size:12px;color:#444">✅ 3-class classifier</span>
  </div>
</div>"""

EXAMPLES = [
    ["ኢትዮጵያ ሀገራችን ናት፣ ሁሉም ዜጋ እኩል ነው።"],
    ["ሁሉም ሰው ወንድም ናቸው፣ ተፋቀሩ።"],
    ["@user ይህ ጽሑፍ ጥሩ አይደለም!!!"],
    ["ሰላምና አንድነት ለሀገራችን አስፈላጊ ነው።"],
]

with gr.Blocks(css=CSS, title="AfriAware — Amharic Hate Speech") as demo:
    gr.HTML(HEADER)
    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Amharic text  (አማርኛ ጽሑፍ)",
                placeholder="ጽሑፉን እዚህ ያስገቡ…",
                lines=4,
            )
            with gr.Row():
                gr.ClearButton(text_input, value="🗑 Clear")
                submit_btn = gr.Button("🔍 Analyse", variant="primary", scale=2)
            gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")
        with gr.Column(scale=2):
            verdict_out = gr.HTML()
            scores_out  = gr.HTML()
            debug_out   = gr.HTML()

    submit_btn.click(fn=analyse, inputs=text_input,
                     outputs=[verdict_out, scores_out, debug_out])
    text_input.submit(fn=analyse, inputs=text_input,
                      outputs=[verdict_out, scores_out, debug_out])

    with gr.Accordion("ℹ️ About this model", open=False):
        gr.Markdown("""
**AfriAware** detects hate speech in Amharic social media posts.

| | |
|---|---|
| **Base model** | `Davlan/bert-base-multilingual-cased-finetuned-amharic` |
| **Training data** | [AfriHate](https://huggingface.co/datasets/afrihate/afrihate) — Amharic tweets |
| **Labels** | hate / abusive / normal |
| **Epochs** | 5 with early stopping |

> ⚠️ Always combine model outputs with human judgment.
        """)

demo.launch()
