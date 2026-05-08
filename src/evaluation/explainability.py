"""
explainability.py
SHAP-based token attribution for AfriAware classifiers.

Shows which Amharic tokens most influenced a prediction — essential
for journalist-facing tools and NGO trust.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── SHAP explainer ──────────────────────────────────────────────────────────

def get_shap_explainer(model_path: str):
    """
    Builds a SHAP TextClassificationPipeline explainer.

    Requires: pip install shap
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Run: pip install shap")

    pipe = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1,
    )
    explainer = shap.Explainer(pipe)
    return explainer


def explain_prediction(
    text: str,
    model_path: str,
    task_name: str = "classifier",
    top_k: int = 10,
    save_plot: bool = True,
) -> dict:
    """
    Returns the top-k most influential tokens for a prediction.

    Output:
    {
        "prediction": "hate",
        "confidence": 0.92,
        "top_tokens": [("ሞት", 0.43), ("ጥፋ", 0.38), ...],
        "bottom_tokens": [("ሰላም", -0.21), ...]
    }
    """
    import shap

    explainer  = get_shap_explainer(model_path)
    shap_vals  = explainer([text])

    # shap_vals.values shape: (n_samples, n_tokens, n_classes)
    values = shap_vals.values[0]          # (n_tokens, n_classes)
    tokens = shap_vals.data[0]            # list of token strings

    # Find predicted class
    pred_class_idx = int(np.argmax(values.sum(axis=0)))

    token_scores = list(zip(tokens, values[:, pred_class_idx].tolist()))
    token_scores_sorted = sorted(token_scores, key=lambda x: abs(x[1]), reverse=True)

    top_positive = [(t, s) for t, s in token_scores_sorted if s > 0][:top_k]
    top_negative = [(t, s) for t, s in token_scores_sorted if s < 0][:top_k]

    result = {
        "prediction":     pred_class_idx,
        "top_tokens":     top_positive,
        "bottom_tokens":  top_negative,
    }

    if save_plot:
        _plot_token_importance(tokens, values[:, pred_class_idx], text, task_name)

    return result


# ─── Simple gradient-based fallback ──────────────────────────────────────────

def gradient_attribution(
    text: str,
    model_path: str,
    id2label: dict,
) -> dict:
    """
    Lightweight integrated-gradients-style attribution that works
    without the shap library. Useful for demos.

    Returns token importances for the predicted class.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs     = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids  = inputs["input_ids"]
    tokens     = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Embed + enable gradients
    embedding_layer = model.base_model.embeddings.word_embeddings
    embeddings      = embedding_layer(input_ids)
    embeddings.retain_grad()

    outputs = model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"])
    logits  = outputs.logits
    pred_id = int(torch.argmax(logits, dim=-1))

    # Backprop through predicted class
    logits[0, pred_id].backward()

    grads       = embeddings.grad[0]                       # (seq_len, hidden)
    importance  = grads.norm(dim=-1).detach().numpy()      # (seq_len,)

    # Normalise to [0, 1]
    if importance.max() > 0:
        importance = importance / importance.max()

    token_scores = list(zip(tokens, importance.tolist()))
    # Remove special tokens
    token_scores = [(t, s) for t, s in token_scores if t not in ("[CLS]", "[SEP]", "[PAD]")]

    return {
        "prediction": id2label[pred_id],
        "token_scores": sorted(token_scores, key=lambda x: x[1], reverse=True),
    }


# ─── Plotting ────────────────────────────────────────────────────────────────

def _plot_token_importance(
    tokens: list,
    scores: np.ndarray,
    text: str,
    task_name: str,
):
    """Saves a horizontal bar chart of token importances."""
    # Keep only top 15 by abs value
    pairs = sorted(zip(tokens, scores), key=lambda x: abs(x[1]), reverse=True)[:15]
    toks, vals = zip(*pairs)

    colors = ["#E8593C" if v > 0 else "#3B8BD4" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = range(len(toks))
    ax.barh(list(y_pos), vals, color=colors, height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(toks, fontsize=11)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("SHAP value (contribution to predicted class)")
    ax.set_title(f"Token attribution — {task_name}")

    pos_patch = mpatches.Patch(color="#E8593C", label="Increases prediction")
    neg_patch = mpatches.Patch(color="#3B8BD4", label="Decreases prediction")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9)

    plt.tight_layout()
    path = OUTPUT_DIR / f"{task_name}_token_attribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Token attribution plot saved → {path}")


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Swap in your trained model path
    MODEL_PATH = str(
        Path(__file__).resolve().parents[2] / "models" / "hate_speech" / "final"
    )
    ID2LABEL = {0: "hate", 1: "abusive", 2: "normal"}

    sample = "ይህ ሰው ጥሩ ነው"   # "This person is good" — should score as normal
    result = gradient_attribution(sample, MODEL_PATH, ID2LABEL)
    print(f"\nText       : {sample}")
    print(f"Prediction : {result['prediction']}")
    print("Top tokens :")
    for tok, score in result["token_scores"][:5]:
        print(f"  {tok:<20} {score:.3f}")
