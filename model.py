# model.py
import os
from unsloth import FastLanguageModel
from config import (
    MODEL_PATH, MAX_SEQ_LENGTH, LOAD_IN_4BIT,
    BASE_MODEL, BASE_MODEL_REPO, ADAPTER_REPO
)

def _ensure_model_available():
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return

    # Base model (có config.json)
    if not (os.path.isdir(BASE_MODEL) and os.listdir(BASE_MODEL)):
        os.makedirs(BASE_MODEL, exist_ok=True)
        snapshot_download(repo_id=BASE_MODEL_REPO, local_dir=BASE_MODEL, local_dir_use_symlinks=False)

    # Adapter (chỉ có adapter_config + weights)
    if not (os.path.isdir(MODEL_PATH) and os.listdir(MODEL_PATH)):
        os.makedirs(MODEL_PATH, exist_ok=True)
        snapshot_download(repo_id=ADAPTER_REPO, local_dir=MODEL_PATH, local_dir_use_symlinks=False)

def load_model():
    _ensure_model_available()
    print("🚀 Loading model, please wait...")

    # 1) Load BASE
    base_source = BASE_MODEL if (os.path.isdir(BASE_MODEL) and os.listdir(BASE_MODEL)) else BASE_MODEL_REPO
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_source,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # 2) Gắn PEFT LoRA adapter
    from peft import PeftModel
    adapter_source = MODEL_PATH if (os.path.isdir(MODEL_PATH) and os.listdir(MODEL_PATH)) else ADAPTER_REPO
    model = PeftModel.from_pretrained(model, adapter_source)  # is_trainable=False mặc định cho inference

    # 3) Bật chế độ suy luận nhanh của Unsloth (tùy chọn)
    FastLanguageModel.for_inference(model)

    print(f"✅ Base: {base_source} + Adapter: {adapter_source}")
    return model, tokenizer
