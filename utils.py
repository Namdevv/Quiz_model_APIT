import re
from typing import Tuple, Optional
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

def normalize_mcq_answer(model_answer: str) -> Tuple[str, str]:
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

    return raw, "model not anwser"