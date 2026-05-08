"""
upload_to_hub.py
One-click script to upload AfriAware model and Space to HuggingFace Hub.

Usage:
    1. Edit USERNAME below
    2. Run: python upload_to_hub.py
"""

from huggingface_hub import HfApi, login
from pathlib import Path

# ── EDIT THIS ─────────────────────────────────────────────────────────────────
USERNAME = "abrhamLearns"   # ← your HuggingFace username
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REPO = f"{USERNAME}/afriaware-amharic-hate-speech"
SPACE_REPO = f"{USERNAME}/afriaware-demo"
ROOT       = Path(__file__).resolve().parent

print("AfriAware — HuggingFace Upload")
print("=" * 45)

# Login (will prompt for token if not cached)
login()

api = HfApi()

# ── 1. Upload model weights ────────────────────────────────────────────────────
print(f"\n[1/2] Uploading model to {MODEL_REPO}...")
api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True, private=False)

# Upload model weights
api.upload_folder(
    folder_path=str(ROOT / "models" / "final"),
    repo_id=MODEL_REPO,
    repo_type="model",
    commit_message="Add AfriAware trained weights (mBERT fine-tuned on AfriHate amh)",
)

# Upload model card
api.upload_file(
    path_or_fileobj=str(ROOT / "docs" / "model_card.md"),
    path_in_repo="README.md",
    repo_id=MODEL_REPO,
    repo_type="model",
    commit_message="Add model card",
)

print(f"  ✓ Model live → https://huggingface.co/{MODEL_REPO}")

# ── 2. Upload HuggingFace Space ────────────────────────────────────────────────
print(f"\n[2/2] Creating Space at {SPACE_REPO}...")

# Patch app.py with the correct model ID before uploading
hf_app_path = ROOT / "hf_space" / "app.py"
content     = hf_app_path.read_text()
content     = content.replace(
    "YOUR_HF_USERNAME/afriaware-amharic-hate-speech",
    MODEL_REPO,
)
hf_app_path.write_text(content)

api.create_repo(
    repo_id=SPACE_REPO,
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
    private=False,
)
api.upload_folder(
    folder_path=str(ROOT / "hf_space"),
    repo_id=SPACE_REPO,
    repo_type="space",
    commit_message="Deploy AfriAware Gradio demo",
)

print(f"  ✓ Space live  → https://huggingface.co/spaces/{SPACE_REPO}")

print("\n" + "=" * 45)
print("✅ Deployment complete!")
print(f"\n  Model : https://huggingface.co/{MODEL_REPO}")
print(f"  Demo  : https://huggingface.co/spaces/{SPACE_REPO}")
print("\nShare these links in your project documentation!")
