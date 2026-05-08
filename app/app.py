"""
app.py  –  AfriAware Gradio Demo
Run locally:  python app/app.py
HuggingFace:  this file is also the HF Space entry point (app.py at root)
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data.preprocess import normalize_amharic
from src.utils.helpers import TASK_CONFIG, get_device

# ─── Model loading ────────────────────────────────────────────────────────────

MODEL_PATH = ROOT / "models" / "final"
_tokenizer = None
_model     = None


def load_model():
    global _tokenizer, _model
    if _model is not None:
        return True
    try:
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        _model     = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH),
            num_labels=TASK_CONFIG["num_labels"],
            id2label={str(k): v for k, v in TASK_CONFIG["id2label"].items()},
            label2id=TASK_CONFIG["label2id"],
            ignore_mismatched_sizes=True,
        )
        _model.eval()
        _model.to(get_device())
        return True
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        print("  Make sure models/final/model.safetensors exists.")
        return False


# ─── Inference ────────────────────────────────────────────────────────────────

def predict(text: str):
    """Returns (label, confidence, all_scores_dict) for one Amharic text."""
    if not text or not text.strip():
        return None

    clean  = normalize_amharic(text)
    inputs = _tokenizer(
        clean,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=TASK_CONFIG["max_length"],
    )
    inputs = {k: v.to(get_device()) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs   = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id = int(torch.argmax(logits, dim=-1))

    id2label = TASK_CONFIG["id2label"]
    return {
        "label":      id2label[pred_id],
        "confidence": probs[pred_id],
        "scores":     {id2label[i]: probs[i] for i in range(len(id2label))},
    }


# ─── Gradio handler ───────────────────────────────────────────────────────────

def analyse(text: str):
    if not text.strip():
        return (
            "<div style='color:#888;font-size:15px'>Enter some Amharic text above.</div>",
            None,
            "",
        )

    if not load_model():
        return (
            "<div style='color:#E8593C'>⚠️ Model weights not found.<br>"
            "Place <code>model.safetensors</code> in <code>models/final/</code></div>",
            None,
            "",
        )

    result = predict(text)
    label  = result["label"]
    conf   = result["confidence"]
    scores = result["scores"]

    cfg    = TASK_CONFIG
    emoji  = cfg["emojis"][label]
    color  = cfg["colors"][label]
    desc   = cfg["descriptions"][label]

    # ── Verdict HTML ──────────────────────────────────────────────────────────
    verdict_html = f"""
<div style="
  border-left: 5px solid {color};
  background: {color}18;
  padding: 16px 20px;
  border-radius: 8px;
  margin-bottom: 8px;
">
  <div style="font-size:28px;font-weight:700;color:{color}">
    {emoji} {label.upper()}
  </div>
  <div style="font-size:15px;color:#555;margin-top:4px">{desc}</div>
  <div style="font-size:13px;color:#888;margin-top:8px">
    Confidence: <strong style="color:{color}">{conf*100:.1f}%</strong>
  </div>
</div>
"""

    # ── Score bars ────────────────────────────────────────────────────────────
    bars = ""
    for lbl, score in sorted(scores.items(), key=lambda x: -x[1]):
        c   = cfg["colors"][lbl]
        pct = score * 100
        em  = cfg["emojis"][lbl]
        bars += f"""
<div style="margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
    <span>{em} <strong>{lbl}</strong></span>
    <span style="color:#666">{pct:.1f}%</span>
  </div>
  <div style="background:#eee;border-radius:6px;height:10px;overflow:hidden">
    <div style="width:{pct}%;background:{c};height:100%;border-radius:6px;
                transition:width 0.5s ease"></div>
  </div>
</div>"""

    scores_html = f"""
<div style="padding:12px 0">
  <div style="font-size:13px;color:#888;margin-bottom:12px;text-transform:uppercase;
              letter-spacing:0.5px">All scores</div>
  {bars}
</div>"""

    clean_note = f"<div style='font-size:12px;color:#aaa;margin-top:6px'>Normalised: {normalize_amharic(text)[:80]}</div>"

    return verdict_html, scores_html, clean_note


# ─── Example inputs ──────────────────────────────────────────────────────────

EXAMPLES = [
    ["ኢትዮጵያ ሀገራችን ናት፣ ሁሉም ዜጋ እኩል ነው።"],
    ["ሁሉም ሰው ወንድም ናቸው፣ ተፋቀሩ።"],
    ["@user ይህ ጽሑፍ ጥሩ አይደለም!!!"],
    ["ሰላምና አንድነት ለሀገራችን አስፈላጊ ነው።"],
]

# ─── UI ──────────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 820px !important; margin: auto }
.main-title { text-align: center; padding: 8px 0 4px }
footer { display: none !important }
"""

HEADER = """
<div style="text-align:center;padding:24px 0 8px">
  <div style="font-size:42px">🛡️</div>
  <h1 style="font-size:28px;font-weight:800;margin:8px 0 4px;color:#111">AfriAware</h1>
  <p style="font-size:15px;color:#666;margin:0">
    Amharic Hate Speech Detection &nbsp;·&nbsp; Open-source African NLP
  </p>
  <div style="margin-top:10px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap">
    <span style="background:#f3f4f6;border-radius:20px;padding:3px 12px;font-size:12px;color:#444">
      🌍 AfriHate Dataset
    </span>
    <span style="background:#f3f4f6;border-radius:20px;padding:3px 12px;font-size:12px;color:#444">
      🤗 mBERT fine-tuned
    </span>
    <span style="background:#f3f4f6;border-radius:20px;padding:3px 12px;font-size:12px;color:#444">
      🇪🇹 Amharic (አማርኛ)
    </span>
    <span style="background:#f3f4f6;border-radius:20px;padding:3px 12px;font-size:12px;color:#444">
      ✅ 3-class classifier
    </span>
  </div>
</div>
"""

FOOTER = """
<div style="text-align:center;margin-top:24px;padding:16px;
            border-top:1px solid #eee;font-size:12px;color:#aaa">
  Fine-tuned on <a href="https://huggingface.co/datasets/afrihate/afrihate" target="_blank"
  style="color:#3B8BD4">AfriHate</a> (NAACL 2025) &nbsp;·&nbsp;
  Base: <a href="https://huggingface.co/Davlan/bert-base-multilingual-cased-finetuned-amharic"
  target="_blank" style="color:#3B8BD4">Davlan/bert-base-multilingual-cased-finetuned-amharic</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/YOUR_USERNAME/afriaware" target="_blank"
  style="color:#3B8BD4">GitHub</a>
</div>
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="AfriAware — Amharic Hate Speech Detection") as demo:

        gr.HTML(HEADER)

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Amharic text  (አማርኛ ጽሑፍ)",
                    placeholder="ጽሑፉን እዚህ ያስገቡ…",
                    lines=4,
                    max_lines=8,
                )
                with gr.Row():
                    clear_btn  = gr.ClearButton(text_input, value="🗑 Clear")
                    submit_btn = gr.Button("🔍 Analyse", variant="primary", scale=2)

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=text_input,
                    label="Try an example",
                )

            with gr.Column(scale=2):
                verdict_out = gr.HTML(label="Result")
                scores_out  = gr.HTML(label="Scores")
                debug_out   = gr.HTML()

        submit_btn.click(
            fn=analyse,
            inputs=text_input,
            outputs=[verdict_out, scores_out, debug_out],
        )
        text_input.submit(
            fn=analyse,
            inputs=text_input,
            outputs=[verdict_out, scores_out, debug_out],
        )

        with gr.Accordion("ℹ️ About this model", open=False):
            gr.Markdown("""
**AfriAware** is an open-source Amharic hate speech classifier built to support digital safety in Ethiopia and across Africa.

| | |
|---|---|
| **Base model** | `Davlan/bert-base-multilingual-cased-finetuned-amharic` |
| **Training data** | AfriHate (amh) — tweets annotated by native Amharic speakers |
| **Labels** | `hate` / `abusive` / `normal` |
| **Max tokens** | 128 |
| **Training epochs** | 5 (early stopping, patience=2) |

**Ethical note:** This model is intended to support — not replace — human moderation.
Always combine model outputs with human judgment and cultural context.
            """)

        gr.HTML(FOOTER)

    return demo


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )
