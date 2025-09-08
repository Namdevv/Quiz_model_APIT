import re
import string
from typing import Tuple, Optional, Dict
import torch
import gc

def clear_cuda_cache():
    """
    Giải phóng bộ nhớ GPU trước khi load/chạy model.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("🧹 Cleared CUDA cache.")

def grade_mcq(gold, pred):
    return pred == gold

def _normalize_text(text: str) -> str:
    table = str.maketrans('', '', string.punctuation)
    return ' '.join(text.translate(table).lower().split())

def normalize_mcq_answer(model_answer: str, options: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """
    Chuẩn hoá câu trả lời MCQ thành ký tự A/B/C/D mà không gọi lại model.
    Trả về (raw_answer, normalized_choice_or_error).
    """
    if model_answer is None:
        return "", "model not anwser"

    raw = str(model_answer).strip()
    if raw == "":
        return raw, "model not anwser"

    upper_raw = raw.upper()

    if len(upper_raw) == 1 and upper_raw in {"A", "B", "C", "D"}:
        return raw, upper_raw

    keyword_pattern = r"(?:ANSWER|ĐÁP\s*ÁN|OPTION|CHỌN|LỰA\s*CHỌN|ANS|RES(?:ULT)?|FINAL)\s*[:\-=>]*\s*([A-D])\b"
    m = re.search(keyword_pattern, upper_raw)
    if m:
        return raw, m.group(1)

    for p in [r"\(([A-D])\)", r"\b([A-D])\.", r"\b([A-D])\)"]:
        m = re.search(p, upper_raw)
        if m:
            return raw, m.group(1)

    m = re.search(r"\b([A-D])\b", upper_raw)
    if m:
        return raw, m.group(1)

    # If regex patterns failed, try to map by matching option text
    if options:
        # Build normalized text map for options
        norm_to_choice = {}
        for choice, text in options.items():
            norm_txt = _normalize_text(str(text))
            if norm_txt:
                norm_to_choice[norm_txt] = choice.strip().upper()

        # Normalize model free-form answer and try contains/exact match
        norm_ans = _normalize_text(raw)
        # Exact normalized match
        if norm_ans in norm_to_choice:
            return raw, norm_to_choice[norm_ans]

        # Substring containment: if the option text appears inside the answer
        for norm_txt, choice in norm_to_choice.items():
            if norm_txt and norm_txt in norm_ans:
                return raw, choice

    return raw, "model not anwser"