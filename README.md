APIT Quiz Grader – LLM-based Evaluation & Analytics
==================================================

A lightweight system to grade quiz datasets with a local LLM (Llama 3B + LoRA adapter via Unsloth) and produce comprehensive analytics (charts, JSON reports) for MCQ and Writing tasks. Supports both Gradio UI and a headless CLI for batch processing. Automatically downloads models on first run.

Features
--------
- LLM inference with Unsloth (4-bit) for efficiency
- Auto-download base model and LoRA adapter if missing
- Gradio dashboard (upload JSON, run analysis, see charts, download reports)
- Headless CLI for large/batch runs without UI
- Robust MCQ answer normalization (maps free-form answers to A/B/C/D)
- Detailed analytics and visualizations saved under timestamped folders

Repository Layout
-----------------
- `interface.py`: Gradio UI and the main `run_quiz_ui` analysis pipeline
- `batch_run.py`: Headless CLI runner; reuses `run_quiz_ui`
- `model.py`: Model loader (with auto-download if not present)
- `config.py`: Configuration (model paths, generation params)
- `charts.py`: Creates the comprehensive analysis dashboard image
- `save_utils.py`: Writes enhanced MCQ/Writing JSON reports
- `utils.py`: Utilities (CUDA cache clearing, MCQ normalization, grading)
- `models/`: Local cache for base model and LoRA adapter
- `outputs/<YYYYMMDD_HHMMSS>/`: Timestamped run outputs

Requirements
------------
- Python 3.10+
- Recommended GPU with CUDA for fast inference (CPU works but slower)
- Dependencies (install into your virtual environment):

```
pip install unsloth transformers accelerate huggingface_hub gradio matplotlib seaborn scikit-learn numpy
# Optional if available on your platform:
pip install bitsandbytes
```

On Windows, if `bitsandbytes` is not available, Unsloth can still run but 4-bit quantization may require specific wheels. Refer to Unsloth/BnB docs if you encounter issues.

Configuration (`config.py`)
----------------------------
- `BASE_MODEL`: Local dir for the base model (download target)
- `MODEL_PATH`: Local dir for the LoRA adapter (download target)
- `BASE_MODEL_REPO`: HF repo id of the base model (e.g., `unsloth/Llama-3.2-3B-Instruct`)
- `ADAPTER_REPO`: HF repo id of the LoRA adapter
- `MAX_SEQ_LENGTH`: Input sequence length for the model (reduce to save memory)
- `LOAD_IN_4BIT`: Whether to load model in 4-bit (saves VRAM/RAM)
- `TEMPERATURE`, `TOP_P`, `TOP_K`, `MAX_NEW_TOKENS`: Generation parameters

Model Auto-Download
-------------------
On first run, `model.py` checks `BASE_MODEL` and `MODEL_PATH`. If missing/empty, it uses Hugging Face Hub to download:
- Base: `BASE_MODEL_REPO` → `BASE_MODEL`
- Adapter: `ADAPTER_REPO` → `MODEL_PATH`

If your repositories require authentication, log in first:

```
python -m huggingface_hub login
```

Input Format (Quiz JSON)
------------------------
Each item should include at least:
- `question_id`: unique id
- `question_type`: `MCQ` or `Writing`
- `question`: the prompt text
- `answer`: for MCQ, the correct choice label `A|B|C|D`; for Writing, reference text
- For MCQ: `options`: object mapping choices to text, e.g. `{ "A": "GET", "B": "POST", ... }`
- Optional: `category`, `difficulty`, and `explanation`

Example MCQ item:

```
{
  "question_id": "Q1",
  "question_type": "MCQ",
  "question": "Which HTTP method is idempotent?",
  "options": {"A": "GET", "B": "POST", "C": "PATCH", "D": "CONNECT"},
  "answer": "A",
  "category": "HTTP",
  "difficulty": "Easy"
}
```

How It Works
------------
1. `run_quiz_ui` loads the quiz file and iterates questions
2. For each question it builds a prompt and calls `ask_model` to get the model’s response
3. For MCQ:
   - The raw model output is normalized via `normalize_mcq_answer` to a choice `A|B|C|D`
   - If the model replies free-form (e.g., "GET is correct answer"), the system matches this text against option contents to infer the correct choice
   - The result is graded against the gold answer
4. For Writing:
   - The raw answer is recorded for review (no automatic grading here)
5. Aggregates stats by `category` and `difficulty`, builds visuals and JSON reports
6. Saves everything under `outputs/<timestamp>/`

MCQ Normalization Details (`utils.py`)
--------------------------------------
The function `normalize_mcq_answer(model_answer, options)` tries in order:
- Direct single-letter match (A/B/C/D)
- Regex patterns for phrases like "Answer: C", "Đáp án: B"
- General single-letter capture if unambiguous
- Free-form mapping: normalize punctuation/case and check if option text is contained in the model answer (or exact normalized match)
- Falls back to `"model not anwser"` if no match

This heuristic covers answers like:
- "C"
- "Answer: B"
- "GET is correct answer" → matches to option whose text is "GET"

You can strengthen it further (fuzzy match, token overlaps) if your data demands.

Running
-------

1) Gradio UI

```
# Windows PowerShell, in repo root
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt  # or install packages listed above
python interface.py
```

Then open the local URL printed by Gradio, upload your quiz JSON, and click "Run Comprehensive Analysis".

2) Headless CLI (no UI)

```
# Windows PowerShell, in repo root
venv\Scripts\python.exe batch_run.py path\to\quiz.json
```

This writes all artifacts into a timestamped folder and prints their paths.

Outputs
-------
All artifacts go under `outputs/<YYYYMMDD_HHMMSS>/`:
- `complete_quiz_report_<ts>.json`: main report (summary + all results)
- `mcq_detailed_analysis_<ts>.json`: MCQ-focused analytics
- `writing_detailed_analysis_<ts>.json`: Writing-focused analytics
- `comprehensive_analysis_<ts>.png`: dashboard image with 12 panels

The main report’s summary includes:
- total questions, counts per type
- MCQ accuracy
- breakdowns by `category` and `difficulty`

Memory & Performance Tips
-------------------------
- If you see the process get "Killed" or OOM errors:
  - Reduce `MAX_SEQ_LENGTH` (e.g., 1024 → 512)
  - Reduce `MAX_NEW_TOKENS` (e.g., 512 → 128/256)
  - Ensure `LOAD_IN_4BIT = True`
  - Close other GPU-heavy apps; prefer the headless CLI for large runs
- `interface.py` now passes an `attention_mask` to stabilize generation

Troubleshooting
---------------
- "huggingface_hub not available": install it with `pip install huggingface_hub`
- Private models: `python -m huggingface_hub login` before first run
- Slow generation on CPU: consider using a CUDA-capable GPU and recent PyTorch
- Free-form answers not mapped: inspect `utils.normalize_mcq_answer` and tune heuristics

Extending
---------
- Add new metrics or dimensions in `interface.py`’s summary build section
- Add more plots in `charts.py`
- Implement automatic grading for Writing if you have a rubric/scorer

License
-------
This project uses the respective licenses of the base model and adapter from Hugging Face. Review and comply with their terms for any redistribution or commercial use.


