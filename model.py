import os
from unsloth import FastLanguageModel
from config import MODEL_PATH, MAX_SEQ_LENGTH, LOAD_IN_4BIT, BASE_MODEL, BASE_MODEL_REPO, ADAPTER_REPO

def _ensure_model_available():
    """
    Ensure base model and adapter are downloaded locally.
    Uses huggingface_hub snapshot_download to fetch if missing.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("huggingface_hub not available; skipping auto-download.")
        return

    # Base model
    if not os.path.exists(BASE_MODEL) or not os.listdir(BASE_MODEL):
        print(f"⬇️ Downloading base model {BASE_MODEL_REPO} to {BASE_MODEL} ...")
        os.makedirs(BASE_MODEL, exist_ok=True)
        snapshot_download(repo_id=BASE_MODEL_REPO, local_dir=BASE_MODEL, local_dir_use_symlinks=False)
        print("✅ Base model downloaded.")

    # Adapter (MODEL_PATH points to adapter dir)
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"⬇️ Downloading adapter {ADAPTER_REPO} to {MODEL_PATH} ...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        snapshot_download(repo_id=ADAPTER_REPO, local_dir=MODEL_PATH, local_dir_use_symlinks=False)
        print("✅ Adapter downloaded.")

def load_model():
    _ensure_model_available()
    print("🚀 Loading model, please wait...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    return model, tokenizer