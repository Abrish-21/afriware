"""
demo.py
AfriAware — Gradio web demo for hate speech and fake news detection.

Run: python app/demo.py
Opens at: http://localhost:7860
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.preprocess import normalize_amharic
from src.evaluation.explainability import gradient_attribution

# ─── Model loading ───────────────────────────────────────────────────────────

HATE_MODEL_PATH = ROOT / "models" / "hate_speech" / "final"
FAKE_MODEL_PATH = ROOT / "models" / "fake_news"  / "final"

HATE_ID2LABEL = {0: "hate", 1: "abusive", 2: "normal"}
FAKE_ID2LABEL = {0: "fake", 1: "real"}

HATE_EMOJI = {"hate": "🔴", "abusive": "🟠", "normal": "🟢"}
FAKE_EMOJI = {"fake": "⛔", "real":    "✅"}


def load_model(path: Path, id2label: dict):
    """Loads tokenizer + model from a saved directory."""
    if not path.exists():
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model     = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.eval()
    return tokenizer, model


def predict_label(text: str, tokenizer, model, id2label: dict, max_length: int = 256) -> dict:
    if tokenizer is None or model is None:
        return None
    text   = normalize_amharic(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=max_length)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id = int(torch.argmax(logits, dim=-1))
    return {
        "label":      id2label[pred_id],
        "confidence": probs[pred_id],
        "scores":     {id2label[i]: probs[i] for i in range(len(id2label))},
    }


# ─── Gradio inference functions ──────────────────────────────────────────────

_hate_tok, _hate_model = None, None
_fake_tok, _fake_model = None, None


def init_models():
    global _hate_tok, _hate_model, _fake_tok, _fake_model
    _hate_tok, _hate_model = load_model(HATE_MODEL_PATH, HATE_ID2LABEL)
    _fake_tok, _fake_model = load_model(FAKE_MODEL_PATH, FAKE_ID2LABEL)


def run_hate_detection(text: str) -> tuple:
    if not text.strip():
        return "Please enter some text.", "", ""

    result = predict_label(text, _hate_tok, _hate_model, HATE_ID2LABEL, max_length=128)
    if result is None:
        return (
            "⚠️ Model not loaded. Run training first:\n"
            "  python src/models/hate_speech_model.py --train",
            "", ""
        )

    label   = result["label"]
    emoji   = HATE_EMOJI.get(label, "")
    conf    = result["confidence"] * 100
    scores  = result["scores"]

    verdict = f"{emoji} **{label.upper()}** ({conf:.1f}% confidence)"
    score_txt = "\n".join(
        f"  {HATE_EMOJI.get(k, '')} {k:<10}: {v*100:.1f}%"
        for k, v in sorted(scores.items(), key=lambda x: -x[1])
    )

    return verdict, score_txt, f"Normalised input: {normalize_amharic(text)}"


def run_fake_news_detection(text: str) -> tuple:
    if not text.strip():
        return "Please enter some text.", "", ""

    result = predict_label(text, _fake_tok, _fake_model, FAKE_ID2LABEL, max_length=256)
    if result is None:
        return (
            "⚠️ Model not loaded. Run training first:\n"
            "  python src/models/fake_news_model.py --train",
            "", ""
        )

    label  = result["label"]
    emoji  = FAKE_EMOJI.get(label, "")
    conf   = result["confidence"] * 100
    scores = result["scores"]

    verdict = f"{emoji} **{label.upper()}** ({conf:.1f}% confidence)"
    score_txt = "\n".join(
        f"  {FAKE_EMOJI.get(k, '')} {k:<6}: {v*100:.1f}%"
        for k, v in sorted(scores.items(), key=lambda x: -x[1])
    )

    return verdict, score_txt, ""


def run_combined(text: str) -> tuple:
    """Runs both models and returns combined output."""
    hate_verdict, hate_scores, _ = run_hate_detection(text)
    fake_verdict, fake_scores, _ = run_fake_news_detection(text)
    return hate_verdict, hate_scores, fake_verdict, fake_scores


# ─── Sample inputs ───────────────────────────────────────────────────────────

HATE_EXAMPLES = [
    ["ይህ ሰው ጥሩ ነው።"],                         # "This person is good."
    ["ኢትዮጵያ ሀገራችን ናት።"],                      # "Ethiopia is our country."
]

FAKE_EXAMPLES = [
    ["ዜና፡ መንግስቱ አዲስ ፖሊሲ አወጀ።"],             # neutral news
    ["ብቸኛ ህክምና ተገኘ!!! ሁሉም ሰው ይሞክር!!!"],     # sensationalist
]


# ─── UI ──────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="AfriAware — Amharic Digital Wellbeing",
        theme=gr.themes.Soft(primary_hue="teal"),
    ) as demo:

        gr.Markdown("""
    # 🛡️ AfriAware
    ### Digital Wellbeing Intelligence for Amharic

    Detects **hate speech** and **disinformation** in Amharic text.
    Built on the [AfriHate](https://huggingface.co/datasets/afrihate/afrihate) dataset.
        """)

        with gr.Tabs():

            # ── Tab 1: Hate speech ──────────────────────────────────────────
            with gr.Tab("🔍 Hate speech detection"):
                gr.Markdown("Enter an Amharic tweet or social media post.")
                with gr.Row():
                    hs_input = gr.Textbox(
                        label="Amharic text",
                        placeholder="ጽሑፉን እዚህ ያስገቡ...",
                        lines=4,
                    )
                hs_btn = gr.Button("Analyse", variant="primary")
                with gr.Row():
                    hs_verdict = gr.Markdown(label="Verdict")
                with gr.Row():
                    hs_scores = gr.Textbox(label="Score breakdown", lines=4)
                    hs_debug  = gr.Textbox(label="Debug", lines=2)

                hs_btn.click(run_hate_detection, inputs=hs_input,
                             outputs=[hs_verdict, hs_scores, hs_debug])
                gr.Examples(HATE_EXAMPLES, inputs=hs_input)

            # ── Tab 2: Fake news ────────────────────────────────────────────
            with gr.Tab("📰 Fake news detection"):
                gr.Markdown("Enter an Amharic news article or headline.")
                with gr.Row():
                    fn_input = gr.Textbox(
                        label="Amharic news text",
                        placeholder="ዜናውን እዚህ ያስገቡ...",
                        lines=6,
                    )
                fn_btn = gr.Button("Analyse", variant="primary")
                with gr.Row():
                    fn_verdict = gr.Markdown(label="Verdict")
                with gr.Row():
                    fn_scores = gr.Textbox(label="Score breakdown", lines=3)
                    fn_debug  = gr.Textbox(label="Debug", lines=2)

                fn_btn.click(run_fake_news_detection, inputs=fn_input,
                             outputs=[fn_verdict, fn_scores, fn_debug])
                gr.Examples(FAKE_EXAMPLES, inputs=fn_input)

            # ── Tab 3: Combined ─────────────────────────────────────────────
            with gr.Tab("⚡ Combined analysis"):
                gr.Markdown(
                    "Run both classifiers simultaneously. "
                    "Useful for news content that may also contain hate speech."
                )
                cb_input = gr.Textbox(
                    label="Amharic text",
                    placeholder="ጽሑፉን እዚህ ያስገቡ...",
                    lines=5,
                )
                cb_btn = gr.Button("Analyse both", variant="primary")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Hate speech**")
                        cb_hate_verdict = gr.Markdown()
                        cb_hate_scores  = gr.Textbox(lines=4, label="Scores")
                    with gr.Column():
                        gr.Markdown("**Fake news**")
                        cb_fake_verdict = gr.Markdown()
                        cb_fake_scores  = gr.Textbox(lines=3, label="Scores")

                cb_btn.click(
                    run_combined, inputs=cb_input,
                    outputs=[cb_hate_verdict, cb_hate_scores,
                             cb_fake_verdict, cb_fake_scores],
                )

        gr.Markdown("""
---
**About AfriAware** · Built for the AI for Good Fellowship · 
[GitHub](https://github.com/YOUR_USERNAME/afriaware) · 
[Dataset: AfriHate](https://huggingface.co/datasets/afrihate/afrihate)
        """)

    return demo


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_models()
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # set True to get a public link via gradio.live
        show_error=True,
    )
