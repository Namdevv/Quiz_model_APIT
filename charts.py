import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix

import os

def create_comprehensive_charts(results, writing_review, timestamp, output_dir=None):
    """
    Tạo biểu đồ phân tích toàn diện với nhiều khía cạnh
    """
    # Filter MCQ results for analysis
    mcq_results = [r for r in results if r["type"] == "MCQ"]
    
    if not mcq_results:
        return None
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Overall Accuracy Pie Chart
    ax1 = plt.subplot(3, 4, 1)
    correct_count = sum(1 for r in mcq_results if r["correct"])
    incorrect_count = len(mcq_results) - correct_count
    
    if len(mcq_results) > 0:
        ax1.pie([correct_count, incorrect_count], 
               labels=[f'Correct ({correct_count})', f'Incorrect ({incorrect_count})'],
               colors=['#2E8B57', '#DC143C'],
               autopct='%1.1f%%',
               startangle=90)
        accuracy = correct_count / len(mcq_results) * 100
        ax1.set_title(f'Overall MCQ Accuracy\n{accuracy:.1f}%', fontweight='bold')
    
    # 2. Question Type Distribution
    ax2 = plt.subplot(3, 4, 2)
    question_types = ['MCQ', 'Writing']
    question_counts = [len(mcq_results), len(writing_review)]
    colors = ['#4169E1', '#FF6347']
    
    bars = ax2.bar(question_types, question_counts, color=colors, alpha=0.8)
    ax2.set_title('Question Type Distribution', fontweight='bold')
    ax2.set_ylabel('Number of Questions')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Accuracy by Topic
    ax3 = plt.subplot(3, 4, 3)
    topic_stats = {}
    for r in mcq_results:
        topic = r.get('topic', 'Unknown')
        if topic not in topic_stats:
            topic_stats[topic] = {'correct': 0, 'total': 0}
        topic_stats[topic]['total'] += 1
        if r['correct']:
            topic_stats[topic]['correct'] += 1
    
    if topic_stats:
        topics = list(topic_stats.keys())
        accuracies = [topic_stats[topic]['correct'] / topic_stats[topic]['total'] * 100 
                     for topic in topics]
        
        bars = ax3.barh(topics, accuracies, color='skyblue', alpha=0.8)
        ax3.set_xlabel('Accuracy (%)')
        ax3.set_title('Accuracy by Topic', fontweight='bold')
        ax3.set_xlim(0, 100)
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 1, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center', fontweight='bold')
    
    # 4. Accuracy by Difficulty
    ax4 = plt.subplot(3, 4, 4)
    difficulty_stats = {}
    for r in mcq_results:
        diff = r.get('difficulty', 'Unknown')
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {'correct': 0, 'total': 0}
        difficulty_stats[diff]['total'] += 1
        if r['correct']:
            difficulty_stats[diff]['correct'] += 1
    
    if difficulty_stats:
        difficulties = list(difficulty_stats.keys())
        diff_accuracies = [difficulty_stats[diff]['correct'] / difficulty_stats[diff]['total'] * 100 
                          for diff in difficulties]
        
        bars = ax4.bar(difficulties, diff_accuracies, color=['#90EE90', '#FFD700', '#FF6B6B'], alpha=0.8)
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Accuracy by Difficulty', fontweight='bold')
        ax4.set_ylim(0, 100)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Topic Distribution (Donut Chart)
    ax5 = plt.subplot(3, 4, 5)
    topic_counts = Counter(r.get('topic', 'Unknown') for r in mcq_results)
    if topic_counts:
        wedges, texts, autotexts = ax5.pie(topic_counts.values(), 
                                          labels=topic_counts.keys(),
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          wedgeprops=dict(width=0.5))
        ax5.set_title('Questions by Topic\n(Distribution)', fontweight='bold')
    
    # 6. Difficulty Distribution
    ax6 = plt.subplot(3, 4, 6)
    diff_counts = Counter(r.get('difficulty', 'Unknown') for r in mcq_results)
    if diff_counts:
        colors_diff = {'Easy': '#90EE90', 'Medium': '#FFD700', 'Hard': '#FF6B6B'}
        colors_list = [colors_diff.get(d, '#808080') for d in diff_counts.keys()]
        
        bars = ax6.bar(diff_counts.keys(), diff_counts.values(), color=colors_list, alpha=0.8)
        ax6.set_ylabel('Number of Questions')
        ax6.set_title('Questions by Difficulty', fontweight='bold')
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Answer Distribution (A, B, C, D)
    ax7 = plt.subplot(3, 4, 7)
    correct_answers = Counter(r.get('gold_answer', 'Unknown') for r in mcq_results)
    model_answers = Counter(r.get('model_normalized_answer', 'Unknown') for r in mcq_results)
    
    if correct_answers:
        options = ['A', 'B', 'C', 'D']
        correct_counts = [correct_answers.get(opt, 0) for opt in options]
        model_counts = [model_answers.get(opt, 0) for opt in options]
        
        x = np.arange(len(options))
        width = 0.35
        
        ax7.bar(x - width/2, correct_counts, width, label='Correct Answer', alpha=0.8, color='lightblue')
        ax7.bar(x + width/2, model_counts, width, label='Model Answer', alpha=0.8, color='lightcoral')
        
        ax7.set_xlabel('Answer Options')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Answer Distribution Comparison', fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(options)
        ax7.legend()
    
    # 8. Performance Heatmap (Topic vs Difficulty)
    ax8 = plt.subplot(3, 4, 8)
    if topic_stats and difficulty_stats:
        # Create matrix for heatmap
        topics_list = list(topic_stats.keys())
        difficulties_list = list(difficulty_stats.keys())
        
        # Create performance matrix
        performance_matrix = []
        for topic in topics_list:
            topic_results = [r for r in mcq_results if r.get('topic') == topic]
            row = []
            for diff in difficulties_list:
                topic_diff_results = [r for r in topic_results if r.get('difficulty') == diff]
                if topic_diff_results:
                    accuracy = sum(1 for r in topic_diff_results if r['correct']) / len(topic_diff_results) * 100
                else:
                    accuracy = 0
                row.append(accuracy)
            performance_matrix.append(row)
        
        if performance_matrix:
            sns.heatmap(performance_matrix, 
                       xticklabels=difficulties_list,
                       yticklabels=topics_list,
                       annot=True, 
                       fmt='.1f',
                       cmap='RdYlGn',
                       ax=ax8,
                       cbar_kws={'label': 'Accuracy (%)'})
            ax8.set_title('Performance Heatmap\n(Topic vs Difficulty)', fontweight='bold')
    
    # 9. Confidence Analysis (Based on raw answer length - proxy for confidence)
    ax9 = plt.subplot(3, 4, 9)
    answer_lengths = [len(r.get('model_raw_answer', '')) for r in mcq_results]
    correct_status = [r['correct'] for r in mcq_results]
    
    if answer_lengths:
        correct_lengths = [length for length, correct in zip(answer_lengths, correct_status) if correct]
        incorrect_lengths = [length for length, correct in zip(answer_lengths, correct_status) if not correct]
        
        ax9.hist([correct_lengths, incorrect_lengths], 
                bins=10, 
                label=['Correct', 'Incorrect'], 
                alpha=0.7, 
                color=['green', 'red'])
        ax9.set_xlabel('Response Length (characters)')
        ax9.set_ylabel('Frequency')
        ax9.set_title('Response Length Distribution', fontweight='bold')
        ax9.legend()
    
    # 10. Summary Statistics
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')
    
    # Calculate key metrics
    total_questions = len(mcq_results) + len(writing_review)
    mcq_accuracy = correct_count / len(mcq_results) * 100 if mcq_results else 0
    
    most_difficult_topic = min(topic_stats.items(), 
                              key=lambda x: x[1]['correct']/x[1]['total'])[0] if topic_stats else "N/A"
    easiest_topic = max(topic_stats.items(), 
                       key=lambda x: x[1]['correct']/x[1]['total'])[0] if topic_stats else "N/A"
    
    summary_text = f"""
    QUIZ PERFORMANCE SUMMARY
    ========================
    
    Total Questions: {total_questions}
    MCQ Questions: {len(mcq_results)}
    Writing Questions: {len(writing_review)}
    
    MCQ Accuracy: {mcq_accuracy:.1f}%
    Correct Answers: {correct_count}/{len(mcq_results)}
    
    Most Challenging Topic: {most_difficult_topic}
    Easiest Topic: {easiest_topic}
    
    Topics Covered: {len(topic_stats)}
    Difficulty Levels: {len(difficulty_stats)}
    """
    
    ax10.text(0.1, 0.9, summary_text, transform=ax10.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # 11. Confusion Matrix for Answer Patterns
    ax11 = plt.subplot(3, 4, 11)
    from sklearn.metrics import confusion_matrix
    
    if mcq_results:
        true_answers = [r.get('gold_answer', 'A') for r in mcq_results]
        pred_answers = [r.get('model_normalized_answer', 'A') for r in mcq_results]
        
        labels = ['A', 'B', 'C', 'D']
        cm = confusion_matrix(true_answers, pred_answers, labels=labels)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax11)
        ax11.set_xlabel('Model Predicted')
        ax11.set_ylabel('Correct Answer')
        ax11.set_title('Answer Confusion Matrix', fontweight='bold')
    
    # 12. Performance Trend (by question order - if available)
    ax12 = plt.subplot(3, 4, 12)
    if mcq_results:
        # Sort by question_id to show performance trend
        sorted_results = sorted(mcq_results, key=lambda x: x.get('question_id', ''))
        performance_trend = [1 if r['correct'] else 0 for r in sorted_results]
        
        # Create moving average for trend
        window_size = max(3, len(performance_trend) // 10)
        if len(performance_trend) >= window_size:
            moving_avg = np.convolve(performance_trend, np.ones(window_size)/window_size, mode='valid')
            ax12.plot(range(len(moving_avg)), moving_avg, 'b-', linewidth=2, alpha=0.8)
            ax12.fill_between(range(len(moving_avg)), moving_avg, alpha=0.3)
        
        ax12.set_xlabel('Question Sequence')
        ax12.set_ylabel('Success Rate')
        ax12.set_title('Performance Trend', fontweight='bold')
        ax12.set_ylim(0, 1)
        ax12.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    
    # Save chart
    chart_filename = f"comprehensive_analysis_{timestamp}.png"
    if output_dir:
        chart_filename = os.path.join(output_dir, chart_filename)
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_filename