# Deployment Guide — AfriAware

This guide covers three deployment paths:
1. **Local** — run on your machine
2. **HuggingFace Model Hub** — upload trained weights
3. **HuggingFace Space** — live public demo

---

## Step 1 — Run the demo locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo (make sure models/final/model.safetensors exists)
python app/app.py
# → Opens at http://localhost:7860
```

---

## Step 2 — Upload model weights to HuggingFace Hub

Your trained model lives in `models/final/`. Upload it as a public model repo.

### 2a. Install HuggingFace CLI

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

### 2b. Create the repo and upload

```python
from huggingface_hub import HfApi

api    = HfApi()
repo   = "YOUR_USERNAME/afriaware-amharic-hate-speech"

# Create repo (run once)
api.create_repo(repo_id=repo, repo_type="model", private=False)

# Upload all files from models/final/
api.upload_folder(
    folder_path="models/final",
    repo_id=repo,
    repo_type="model",
)

print(f"✓ Model live at: https://huggingface.co/{repo}")
```

Or use the CLI:

```bash
cd models/final
git init
git remote add origin https://huggingface.co/YOUR_USERNAME/afriaware-amharic-hate-speech
git add .
git commit -m "Add AfriAware trained weights"
git push
```

### 2c. Add the model card

Copy `docs/model_card.md` → rename to `README.md` and push it to the model repo.

---

## Step 3 — Deploy the HuggingFace Space (live demo)

### 3a. Create a new Space on HuggingFace

Go to https://huggingface.co/new-space and set:
- **Owner:** your username
- **Space name:** `afriaware-demo`
- **SDK:** Gradio
- **Visibility:** Public

### 3b. Push the Space files

```bash
cd hf_space/

# Edit app.py line 13: set your model ID
# HF_MODEL_ID = "YOUR_USERNAME/afriaware-amharic-hate-speech"

git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/afriaware-demo
git add .
git commit -m "AfriAware Gradio demo"
git push
```

The Space will auto-build. Live URL:
`https://huggingface.co/spaces/YOUR_USERNAME/afriaware-demo`

### 3c. Set the model ID as a Space secret (optional)

In your Space settings → **Secrets** → add:
```
HF_MODEL_ID = YOUR_USERNAME/afriaware-amharic-hate-speech
```

---

## Checklist

- [ ] `models/final/model.safetensors` exists locally
- [ ] `huggingface-cli login` done
- [ ] Model weights uploaded to Hub
- [ ] `hf_space/app.py` — updated `HF_MODEL_ID`
- [ ] Space created and code pushed
- [ ] Demo live and tested

---

## Quick upload script

Save as `upload_to_hub.py` and run once:

```python
from huggingface_hub import HfApi

USERNAME   = "YOUR_USERNAME"                        # ← change this
MODEL_REPO = f"{USERNAME}/afriaware-amharic-hate-speech"
SPACE_REPO = f"{USERNAME}/afriaware-demo"

api = HfApi()

# 1. Upload model
api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path="models/final", repo_id=MODEL_REPO, repo_type="model")
print(f"Model → https://huggingface.co/{MODEL_REPO}")

# 2. Upload Space
api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(folder_path="hf_space", repo_id=SPACE_REPO, repo_type="space")
print(f"Space → https://huggingface.co/spaces/{SPACE_REPO}")
```
