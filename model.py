from unsloth import FastLanguageModel
from config import MODEL_PATH, MAX_SEQ_LENGTH, LOAD_IN_4BIT

def load_model():
    print("🚀 Loading model, please wait...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    return model, tokenizer
    