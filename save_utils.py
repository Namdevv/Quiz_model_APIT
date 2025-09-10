import json
import os

def save_enhanced_files(results, writing_review, quiz_data, timestamp, output_dir=None):
    """
    Lưu files với thông tin chi tiết hơn
    """
    mcq_results = [r for r in results if r["type"] == "MCQ"]
    
    # Enhanced MCQ Analysis
    mcq_analysis = {
        "metadata": {
            "timestamp": timestamp,
            "total_mcq_questions": len(mcq_results),
            "overall_accuracy": sum(1 for r in mcq_results if r["correct"]) / len(mcq_results) if mcq_results else 0
        },
        "category_analysis": {},
        "difficulty_analysis": {},
        "detailed_results": mcq_results
    }
    
    # Analyze by category
    category_stats = {}
    for r in mcq_results:
        category = r.get('category', 'Unknown')
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0, 'questions': []}
        category_stats[category]['total'] += 1
        category_stats[category]['questions'].append(r['question_id'])
        if r['correct']:
            category_stats[category]['correct'] += 1
    
    for category, stats in category_stats.items():
        mcq_analysis['category_analysis'][category] = {
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
    if output_dir:
        mcq_filename = os.path.join(output_dir, mcq_filename)
    with open(mcq_filename, "w", encoding="utf-8") as f:
        json.dump(mcq_analysis, f, indent=2, ensure_ascii=False)
    
    # Enhanced Writing Analysis
    writing_analysis = {
        "metadata": {
            "timestamp": timestamp,
            "total_writing_questions": len(writing_review),
        },
        "category_distribution": {},
        "difficulty_distribution": {},
        "detailed_results": writing_review
    }
    
    # Analyze writing questions by category and difficulty if available
    writing_questions = [q for q in quiz_data if q.get('question_type') == 'Writing']
    for q in writing_questions:
        category = q.get('category', 'Unknown')
        diff = q.get('difficulty', 'Unknown')
        
        if category not in writing_analysis['category_distribution']:
            writing_analysis['category_distribution'][category] = 0
        writing_analysis['category_distribution'][category] += 1
        
        if diff not in writing_analysis['difficulty_distribution']:
            writing_analysis['difficulty_distribution'][diff] = 0
        writing_analysis['difficulty_distribution'][diff] += 1
    
    # Save enhanced writing analysis
    writing_filename = f"writing_detailed_analysis_{timestamp}.json"
    if output_dir:
        writing_filename = os.path.join(output_dir, writing_filename)
    with open(writing_filename, "w", encoding="utf-8") as f:
        json.dump(writing_analysis, f, indent=2, ensure_ascii=False)
    
    return mcq_filename, writing_filename