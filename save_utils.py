import json
import os

def save_enhanced_files(results, writing_review, quiz_data, timestamp, output_dir=None, model_tag: str | None = None):
    """
    Lưu files với thông tin chi tiết hơn
    """
    mcq_results = [r for r in results if r["type"] == "MCQ"]
    
    # Enhanced MCQ Analysis
    mcq_analysis = {
        "metadata": {
            "timestamp": timestamp,
            "model": model_tag,
            "total_mcq_questions": len(mcq_results),
            "overall_accuracy": sum(1 for r in mcq_results if r["correct"]) / len(mcq_results) if mcq_results else 0
        },
        "topic_analysis": {},
        "difficulty_analysis": {},
        "detailed_results": mcq_results
    }
    
    # Analyze by topic
    topic_stats = {}
    for r in mcq_results:
        topic = r.get('topic', 'Unknown')
        if topic not in topic_stats:
            topic_stats[topic] = {'correct': 0, 'total': 0, 'questions': []}
        topic_stats[topic]['total'] += 1
        topic_stats[topic]['questions'].append(r['question_id'])
        if r['correct']:
            topic_stats[topic]['correct'] += 1
    
    for topic, stats in topic_stats.items():
        mcq_analysis['topic_analysis'][topic] = {
            'accuracy': stats['correct'] / stats['total'],
            'correct_count': stats['correct'],
            'total_count': stats['total'],
            'question_ids': stats['questions']
        }
    
    # Analyze by difficulty
    difficulty_stats = {}
    for r in mcq_results:
        diff = r.get('difficulty', 'Unknown')
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {'correct': 0, 'total': 0, 'questions': []}
        difficulty_stats[diff]['total'] += 1
        difficulty_stats[diff]['questions'].append(r['question_id'])
        if r['correct']:
            difficulty_stats[diff]['correct'] += 1
    
    for diff, stats in difficulty_stats.items():
        mcq_analysis['difficulty_analysis'][diff] = {
            'accuracy': stats['correct'] / stats['total'],
            'correct_count': stats['correct'],
            'total_count': stats['total'],
            'question_ids': stats['questions']
        }
    
    # Save enhanced MCQ analysis
    mcq_filename = f"mcq_detailed_analysis_{timestamp}.json"
    if model_tag:
        mcq_filename = f"mcq_detailed_analysis_{model_tag}_{timestamp}.json"
    if output_dir:
        mcq_filename = os.path.join(output_dir, mcq_filename)
    with open(mcq_filename, "w", encoding="utf-8") as f:
        json.dump(mcq_analysis, f, indent=2, ensure_ascii=False)
    
    # Enhanced Writing Analysis
    writing_analysis = {
        "metadata": {
            "timestamp": timestamp,
            "model": model_tag,
            "total_writing_questions": len(writing_review),
        },
        "topic_distribution": {},
        "difficulty_distribution": {},
        "detailed_results": writing_review
    }
    
    # Analyze writing questions by topic and difficulty if available
    writing_questions = [q for q in quiz_data if q.get('question_type') == 'Writing']
    for q in writing_questions:
        topic = q.get('topic', 'Unknown')
        diff = q.get('difficulty', 'Unknown')
        
        if topic not in writing_analysis['topic_distribution']:
            writing_analysis['topic_distribution'][topic] = 0
        writing_analysis['topic_distribution'][topic] += 1
        
        if diff not in writing_analysis['difficulty_distribution']:
            writing_analysis['difficulty_distribution'][diff] = 0
        writing_analysis['difficulty_distribution'][diff] += 1
    
    # Save enhanced writing analysis
    writing_filename = f"writing_detailed_analysis_{timestamp}.json"
    if model_tag:
        writing_filename = f"writing_detailed_analysis_{model_tag}_{timestamp}.json"
    if output_dir:
        writing_filename = os.path.join(output_dir, writing_filename)
    with open(writing_filename, "w", encoding="utf-8") as f:
        json.dump(writing_analysis, f, indent=2, ensure_ascii=False)
    
    return mcq_filename, writing_filename