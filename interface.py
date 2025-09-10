import json
import os
from datetime import datetime
from typing import Tuple, Optional
import gradio as gr
import torch

from model import load_model
from config import MODEL_PATH
from charts import create_comprehensive_charts  
from save_utils import save_enhanced_files      
from utils import normalize_mcq_answer, grade_mcq, clear_cuda_cache

clear_cuda_cache()
model, tokenizer = load_model()

from transformers import TextIteratorStreamer

def ask_model(
    prompt: str,
    *,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_new_tokens: int = 512,
    stream: bool = False,
    timeout_s: Optional[float] = None,
) -> str:
    """
    Gọi model chat và trả về *chính xác* phần sinh mới (không lẫn prompt).
    Nếu stream=True: vừa in ra màn hình vừa thu lại text, cuối cùng return full string.
    """

    # 1) Chuẩn bị chat input
    messages = [{"role": "user", "content": prompt}]
    # Trả về tensor luôn để không phải tokenize lần 2
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Độ dài token của prompt (để cắt phần sinh mới)
    prompt_len = inputs.shape[-1]

    gen_kwargs = dict(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=(temperature > 0.0),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
    )

    # Use built-in generation timeout to prevent hangs
    if timeout_s is not None:
        gen_kwargs["max_time"] = timeout_s

    if stream or timeout_s is not None:
        # 2A) Stream có thu hồi
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        import threading
        t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        t.start()

        chunks = []
        for piece in streamer:
            print(piece, end="", flush=True)  # hiển thị realtime
            chunks.append(piece)
        t.join()
        return "".join(chunks).strip()

    # 2B) Không stream: generate bình thường và cắt theo số token
    with torch.inference_mode():
        outputs = model.generate(**gen_kwargs)

    # Lấy phần token sinh mới sau prompt
    new_tokens = outputs[:, prompt_len:]
    answer = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return answer.strip()

# prompt ="""
# Question: Which HTTP status code means Not Found?\nA. 200\nB. 301\nC. 404\nD. 500 , Anwser only A or B or C or D
# """
# anwser = ask_model(prompt)
# print(normalize_mcq_answer(anwser))


def run_quiz_ui(quiz_file, debug: bool = False, timeout_s: Optional[float] = None, skip_charts: bool = False, progress=gr.Progress()):
    if quiz_file is None:
        return None, None, None, None, None, "Please upload a quiz file first!", ""
    
    with open(quiz_file.name, "r", encoding="utf-8") as f:
        quiz_data = json.load(f)
    
    results = []
    total = len(quiz_data)
    correct_mcq, total_mcq = 0, 0
    writing_review = []
    debug_lines = []

    def dbg(message: str):
        if debug:
            ts = datetime.now().strftime('%H:%M:%S')
            line = f"[{ts}] {message}"
            print(line, flush=True)
            debug_lines.append(line)
    
    # Model tag for filenames/metadata
    model_tag = os.path.basename(str(MODEL_PATH).rstrip(os.sep)).replace(" ", "_")

    # Create timestamp and output directory early to save logs incrementally
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"debug_{model_tag}_{timestamp}.log")
    dbg(f"Loaded quiz with {total} questions from {quiz_file.name}")

    # Process each question
    for idx, q in enumerate(quiz_data, start=1):
        try:
            qid = q["question_id"]
            qtype = q["question_type"]
            gold = q["answer"]
            dbg(f"Q{idx}/{total} [{qtype}] id={qid}")

            if qtype == "MCQ":
                options_text = "\n".join([f"{k}. {v}" for k,v in q["options"].items()])
                prompt = f"Question: {q['question']}\nOptions:\n{options_text}\nAnswer with only A or B or C or D."
            else:
                prompt = f"Question: {q['question']}\nAnswer in 1-3 sentences."

            dbg("Calling model.generate()...")
            model_answer = ask_model(prompt)

            # Process MCQ
            if qtype == "MCQ":
                total_mcq += 1
                raw_ans, norm_ans = normalize_mcq_answer(model_answer, q.get("options", {}))
                is_correct = grade_mcq(gold, norm_ans)
                if is_correct:
                    correct_mcq += 1

                results.append({
                    "question_id": qid,
                    "question": q["question"],
                    "type": "MCQ",
                    "category": q.get("category", "Unknown"),
                    "difficulty": q.get("difficulty", "Unknown"),
                    "options": q.get("options", {}),
                    "gold_answer": gold,
                    "model_raw_answer": raw_ans,
                    "model_normalized_answer": norm_ans,
                    "correct": is_correct,
                    "explanation": q.get("explanation", "")
                })
            else:
                results.append({
                    "question_id": qid,
                    "type": "Writing",
                    "gold": gold,
                    "model_answer": model_answer
                })

                writing_review.append({
                    "question_id": qid,
                    "question": q["question"],
                    "category": q.get("category", "Unknown"),
                    "difficulty": q.get("difficulty", "Unknown"),
                    "gold_answer": gold,
                    "model_answer": model_answer
                })
        except Exception as e:
            dbg(f"Error at Q{idx}: {e}")
        finally:
            # persist logs incrementally
            if debug:
                try:
                    with open(log_path, "w", encoding="utf-8") as lf:
                        lf.write("\n".join(debug_lines))
                except Exception:
                    pass
            progress(idx / total, f"Processing {idx}/{total} questions...")
    
    # Calculate metrics
    total_writing = len(writing_review)
    accuracy = (correct_mcq / total_mcq) if total_mcq else None
    
    # Create comprehensive analysis chart
    chart_file = None
    if not skip_charts:
        dbg("Generating charts...")
        try:
            chart_file = create_comprehensive_charts(results, writing_review, timestamp, output_dir, model_tag=model_tag)
            dbg(f"Charts saved to {chart_file}")
        except Exception as e:
            dbg(f"Chart generation failed: {e}")
            chart_file = None
    
    # Save enhanced files
    mcq_file, writing_file = save_enhanced_files(results, writing_review, quiz_data, timestamp, output_dir, model_tag=model_tag)
    
    # Create main report
    report = {
        "timestamp": timestamp,
        "model": model_tag,
        "summary": {
            "total_questions": total,
            "mcq_questions": total_mcq,
            "writing_questions": total_writing,
            "mcq_accuracy": accuracy,
            "mcq_correct": correct_mcq,
            "mcq_incorrect": total_mcq - correct_mcq
        },
        "results": results,
        "writing_for_review": writing_review
    }
    
    # Save main report
    main_report_filename = os.path.join(output_dir, f"complete_quiz_report_{model_tag}_{timestamp}.json")
    with open(main_report_filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Finalize debug logs
    if debug:
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(debug_lines))
        except Exception:
            pass
    
    # Create enhanced summary
    category_stats = {}
    difficulty_stats = {}
    mcq_results = [r for r in results if r["type"] == "MCQ"]
    
    for r in mcq_results:
        # category stats
        category = r.get('category', 'Unknown')
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0}
        category_stats[category]['total'] += 1
        if r['correct']:
            category_stats[category]['correct'] += 1
            
        # Difficulty stats
        diff = r.get('difficulty', 'Unknown')
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {'correct': 0, 'total': 0}
        difficulty_stats[diff]['total'] += 1
        if r['correct']:
            difficulty_stats[diff]['correct'] += 1
    
    # Create detailed summary
    summary_text = f"""
# 📊 Comprehensive Quiz Analysis Report

## Overall Performance
- **Total Questions:** {total}
- **MCQ Questions:** {total_mcq}
- **Writing Questions:** {total_writing}
- **MCQ Accuracy:** {accuracy*100:.1f}% ({correct_mcq}/{total_mcq})

## category Performance
"""
    
    for category, stats in category_stats.items():
        category_accuracy = stats['correct'] / stats['total'] * 100
        summary_text += f"- **{category}:** {category_accuracy:.1f}% ({stats['correct']}/{stats['total']})\n"
    
    summary_text += "\n## Difficulty Performance\n"
    for diff, stats in difficulty_stats.items():
        diff_accuracy = stats['correct'] / stats['total'] * 100
        summary_text += f"- **{diff}:** {diff_accuracy:.1f}% ({stats['correct']}/{stats['total']})\n"
    
    summary_text += f"""
## Files Generated
- **Comprehensive Analysis Chart:** {chart_file}
- **Complete Report:** {main_report_filename}
- **MCQ Detailed Analysis:** {mcq_file}
- **Writing Detailed Analysis:** {writing_file}
    """
    
    return (report, main_report_filename, mcq_file, writing_file, 
            chart_file, summary_text, "\n".join(debug_lines))

# ---- Enhanced UI ----
with gr.Blocks(theme=gr.themes.Soft(), title="APIT Quiz Grader - Advanced Analytics") as demo:
    gr.Markdown("# 🚀 APIT Quiz Grader - Advanced Analytics Dashboard")
    gr.Markdown("""
    Upload your quiz JSON file and get comprehensive analysis including:
    - **category Performance Analysis** - Performance breakdown by subject areas
    - **Difficulty Analysis** - How well the model handles different difficulty levels  
    - **Answer Pattern Analysis** - Confusion matrices and answer distribution
    - **Comprehensive Visualizations** - 12+ charts and graphs for deep insights
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            quiz_file = gr.File(
                label="📁 Upload Quiz JSON File", 
                file_types=[".json"],
                height=120
            )
            debug_toggle = gr.Checkbox(label="Enable debug logs", value=False)
            timeout_seconds = gr.Number(label="Generation timeout (seconds, optional)", value=None)
            skip_charts_toggle = gr.Checkbox(label="Skip chart generation (safe mode)", value=False)
            run_btn = gr.Button(
                "🔍 Run Comprehensive Analysis", 
                variant="primary",
                size="lg"
            )
    
    with gr.Row():
        with gr.Column():
            summary_output = gr.Markdown(label="📈 Comprehensive Analysis Summary")
    
    with gr.Row():
        with gr.Column():
            chart_output = gr.Image(
                label="📊 Comprehensive Analysis Dashboard",
                height=600
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📥 Download Detailed Reports")
            with gr.Row():
                complete_report = gr.File(label="📋 Complete Report JSON")
                mcq_results = gr.File(label="📈 MCQ Detailed Analysis")
                writing_results = gr.File(label="✍️ Writing Detailed Analysis")

    with gr.Row():
        with gr.Column():
            logs_output = gr.Textbox(label="🛠 Debug Logs", lines=12)
    
    with gr.Row():
        with gr.Column():
            detailed_output = gr.JSON(
                label="🔍 Raw Results (Preview)",
                height=500
            )
    
    # Event handler
    run_btn.click(
        fn=run_quiz_ui,
        inputs=[quiz_file, debug_toggle, timeout_seconds, skip_charts_toggle],
        outputs=[
            detailed_output,
            complete_report,
            mcq_results,
            writing_results,
            chart_output,
            summary_output,
            logs_output,
        ]
    )

if __name__ == "__main__":
    demo.launch(share=False)