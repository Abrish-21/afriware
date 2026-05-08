"""helpers.py – shared utilities."""
import torch

TASK_CONFIG = {
    "num_labels": 3,
    "id2label":   {0: "hate", 1: "abusive", 2: "normal"},
    "label2id":   {"hate": 0, "abusive": 1, "normal": 2},
    "max_length": 128,
    "colors":     {"hate": "#E8593C", "abusive": "#EF9F27", "normal": "#1D9E75"},
    "emojis":     {"hate": "🔴", "abusive": "🟠", "normal": "🟢"},
    "descriptions": {
        "hate":     "Content that expresses hatred toward a person or group.",
        "abusive":  "Offensive or aggressive language, not necessarily group-targeted.",
        "normal":   "Regular, non-harmful content.",
    },
}

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
