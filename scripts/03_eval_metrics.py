"""
Comprehensive evaluation with multiple NLP metrics and thorough analysis.
Includes BLEU, ROUGE-L, F1-score, perplexity, semantic similarity, and qualitative testing.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def generate_response(model, tokenizer, user_query: str, max_length=128) -> str:
    """Generate response for a user query."""
    input_text = f"finance: {user_query}"
    input_ids = tokenizer.encode(input_text, return_tensors='tf', max_length=max_length, truncation=True)
    
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score."""
    bleu = BLEU()
    references_formatted = [[ref] for ref in references]
    score = bleu.corpus_score(predictions, references_formatted)
    return score.score

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        rouge1_scores.append(score['rouge1'].fmeasure)
        rouge2_scores.append(score['rouge2'].fmeasure)
        rougeL_scores.append(score['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores)
    }

def compute_perplexity(model, tokenizer, data: List[Dict], batch_size=16, max_length=128) -> float:
    """Compute perplexity on dataset."""
    total_loss = 0
    total_samples = 0
    
    for i in tqdm(range(0, len(data), batch_size), desc="Computing perplexity"):
        batch = data[i:i+batch_size]
        
        inputs = [f"finance: {item['user']}" for item in batch]
        targets = [item['assistant'] for item in batch]
        
        input_encodings = tokenizer(
            inputs,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        target_encodings = tokenizer(
            targets,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        labels = target_encodings['input_ids'].numpy().copy()
        labels[labels == tokenizer.pad_token_id] = -100
        
        outputs = model(
            input_ids=input_encodings['input_ids'],
            attention_mask=input_encodings['attention_mask'],
            labels=tf.constant(labels),
            training=False
        )
        
        total_loss += outputs.loss.numpy()
        total_samples += 1
    
    avg_loss = total_loss / total_samples
    perplexity = np.exp(avg_loss)
    
    return perplexity

def predict_intent(response: str, intent_keywords: Dict[str, List[str]]) -> str:
    """
    Predict intent from generated response using keyword matching.
    This is a simple heuristic for intent classification.
    """
    response_lower = response.lower()
    
    # Count keyword matches for each intent
    intent_scores = {}
    for intent, keywords in intent_keywords.items():
        score = sum(1 for kw in keywords if kw.lower() in response_lower)
        intent_scores[intent] = score
    
    # Return intent with highest score, or 'unknown' if no matches
    if max(intent_scores.values()) > 0:
        return max(intent_scores, key=intent_scores.get)
    return 'unknown'

def compute_intent_f1(predictions: List[str], references: List[str], 
                      true_intents: List[str]) -> Dict[str, float]:
    """
    Compute F1-score for intent classification.
    Uses keyword-based intent prediction from generated responses.
    """
    # Define intent keywords for classification
    intent_keywords = {
        'account_balance': ['balance', 'account', 'funds', 'available'],
        'transaction_history': ['transaction', 'history', 'statement', 'activity'],
        'transfer_funds': ['transfer', 'send', 'move', 'payment'],
        'loan_inquiry': ['loan', 'borrow', 'credit', 'interest rate'],
        'investment_advice': ['invest', 'portfolio', 'stocks', 'bonds', 'diversify'],
        'credit_score': ['credit score', 'fico', 'credit report'],
        'fraud_alert': ['fraud', 'suspicious', 'unauthorized', 'security'],
        'bill_payment': ['bill', 'pay', 'utility', 'invoice'],
        'savings_account': ['savings', 'save', 'interest', 'deposit'],
        'mortgage_info': ['mortgage', 'home loan', 'property'],
        'tax_info': ['tax', 'irs', 'deduction', '1099'],
        'retirement_planning': ['retirement', '401k', 'ira', 'pension'],
        'currency_exchange': ['currency', 'exchange', 'forex', 'conversion'],
        'insurance_query': ['insurance', 'coverage', 'premium', 'policy'],
    }
    
    # Predict intents from generated responses
    predicted_intents = [predict_intent(pred, intent_keywords) for pred in predictions]
    
    # Filter out OOD samples for intent classification
    filtered_predicted = []
    filtered_true = []
    for pred_intent, true_intent in zip(predicted_intents, true_intents):
        if true_intent != 'out_of_domain':
            filtered_predicted.append(pred_intent)
            filtered_true.append(true_intent)
    
    # Compute metrics
    if len(filtered_true) > 0:
        precision, recall, f1, support = precision_recall_fscore_support(
            filtered_true, filtered_predicted, average='weighted', zero_division=0
        )
        
        # Also compute per-intent F1
        per_intent_f1 = precision_recall_fscore_support(
            filtered_true, filtered_predicted, average=None, zero_division=0,
            labels=list(set(filtered_true))
        )
        
        return {
            'f1_weighted': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'per_intent_f1': {intent: float(score) for intent, score in 
                             zip(list(set(filtered_true)), per_intent_f1[2])},
            'confusion_matrix': confusion_matrix(filtered_true, filtered_predicted).tolist(),
            'intent_labels': sorted(list(set(filtered_true)))
        }
    
    return {'f1_weighted': 0.0, 'precision': 0.0, 'recall': 0.0}

def compute_semantic_similarity(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute semantic similarity using sentence embeddings.
    Measures how semantically similar predictions are to references.
    """
    print("  Loading sentence transformer for semantic similarity...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode sentences
    pred_embeddings = model.encode(predictions, show_progress_bar=True)
    ref_embeddings = model.encode(references, show_progress_bar=False)
    
    # Compute cosine similarities
    similarities = []
    for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
        sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
        similarities.append(sim)
    
    return {
        'semantic_similarity_mean': float(np.mean(similarities)),
        'semantic_similarity_std': float(np.std(similarities)),
        'semantic_similarity_min': float(np.min(similarities)),
        'semantic_similarity_max': float(np.max(similarities))
    }

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute exact match accuracy."""
    matches = sum(1 for pred, ref in zip(predictions, references) 
                  if pred.strip().lower() == ref.strip().lower())
    return matches / len(predictions) if predictions else 0.0

def compute_response_quality_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute response quality metrics:
    - Average response length
    - Length ratio (pred/ref)
    - Vocabulary diversity (unique tokens / total tokens)
    """
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = [len(ref.split()) for ref in references]
    
    # Vocabulary diversity
    all_tokens = ' '.join(predictions).split()
    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    
    return {
        'avg_pred_length': float(np.mean(pred_lengths)),
        'avg_ref_length': float(np.mean(ref_lengths)),
        'length_ratio': float(np.mean(pred_lengths) / np.mean(ref_lengths)) if np.mean(ref_lengths) > 0 else 0.0,
        'vocabulary_diversity': float(unique_tokens / total_tokens) if total_tokens > 0 else 0.0,
        'pred_length_std': float(np.std(pred_lengths))
    }

def analyze_per_intent_performance(predictions: List[str], references: List[str], 
                                   intents: List[str]) -> Dict[str, Dict]:
    """Analyze performance metrics per intent."""
    intent_groups = {}
    
    # Group by intent
    for pred, ref, intent in zip(predictions, references, intents):
        if intent not in intent_groups:
            intent_groups[intent] = {'predictions': [], 'references': []}
        intent_groups[intent]['predictions'].append(pred)
        intent_groups[intent]['references'].append(ref)
    
    # Compute metrics per intent
    per_intent_metrics = {}
    for intent, data in intent_groups.items():
        preds = data['predictions']
        refs = data['references']
        
        bleu = compute_bleu(preds, refs)
        rouge = compute_rouge(preds, refs)
        
        per_intent_metrics[intent] = {
            'count': len(preds),
            'bleu': float(bleu),
            'rougeL': float(rouge['rougeL']),
        }
    
    return per_intent_metrics

def identify_failure_cases(predictions: List[str], references: List[str], 
                          test_data: List[Dict], n_worst=10) -> List[Dict]:
    """Identify worst performing examples for error analysis."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Compute per-example scores
    examples_with_scores = []
    for pred, ref, data in zip(predictions, references, test_data):
        score = scorer.score(ref, pred)['rougeL'].fmeasure
        examples_with_scores.append({
            'user': data['user'],
            'reference': ref,
            'prediction': pred,
            'intent': data['intent'],
            'is_ood': data.get('is_ood', False),
            'score': score
        })
    
    # Sort by score and get worst cases
    examples_with_scores.sort(key=lambda x: x['score'])
    
    return examples_with_scores[:n_worst]

def generate_qualitative_examples(model, tokenizer, test_data: List[Dict], n_examples=15) -> List[Dict]:
    """Generate diverse qualitative conversation examples."""
    examples = []
    
    # Get diverse examples: in-domain and OOD
    in_domain = [d for d in test_data if not d.get('is_ood', False)]
    ood = [d for d in test_data if d.get('is_ood', False)]
    
    # Sample examples from different intents
    intent_samples = {}
    for item in in_domain:
        intent = item['intent']
        if intent not in intent_samples:
            intent_samples[intent] = []
        intent_samples[intent].append(item)
    
    # Get 1-2 examples per intent
    selected = []
    for intent, items in intent_samples.items():
        selected.extend(items[:2])
    
    # Add OOD examples
    selected.extend(ood[:3])
    
    for item in tqdm(selected[:n_examples], desc="Generating examples"):
        prediction = generate_response(model, tokenizer, item['user'])
        
        examples.append({
            'user': item['user'],
            'reference': item['assistant'],
            'prediction': prediction,
            'intent': item['intent'],
            'is_ood': item.get('is_ood', False),
        })
    
    return examples

def plot_confusion_matrix(cm: List[List[int]], labels: List[str], output_path: Path):
    """Plot confusion matrix for intent classification."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Intent Classification Confusion Matrix')
    plt.ylabel('True Intent')
    plt.xlabel('Predicted Intent')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {output_path}")

def plot_per_intent_performance(per_intent_metrics: Dict, output_path: Path):
    """Plot performance metrics per intent."""
    intents = list(per_intent_metrics.keys())
    bleu_scores = [per_intent_metrics[i]['bleu'] for i in intents]
    rougeL_scores = [per_intent_metrics[i]['rougeL'] for i in intents]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # BLEU scores
    ax1.barh(intents, bleu_scores, color='skyblue')
    ax1.set_xlabel('BLEU Score')
    ax1.set_title('BLEU Score by Intent')
    ax1.grid(axis='x', alpha=0.3)
    
    # ROUGE-L scores
    ax2.barh(intents, rougeL_scores, color='lightcoral')
    ax2.set_xlabel('ROUGE-L Score')
    ax2.set_title('ROUGE-L Score by Intent')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved per-intent performance to {output_path}")

def evaluate_model(model_dir: Path, test_data_path: Path, output_dir: Path) -> Dict:
    """Evaluate model with comprehensive metrics and thorough analysis."""
    print(f"Loading model from {model_dir}...")
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    print(f"Loading test data from {test_data_path}...")
    test_data = load_jsonl(test_data_path)
    print(f"  Test samples: {len(test_data)}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    references = []
    intents = []
    
    for item in tqdm(test_data[:500], desc="Generating"):  # Limit for speed
        pred = generate_response(model, tokenizer, item['user'])
        predictions.append(pred)
        references.append(item['assistant'])
        intents.append(item['intent'])
    
    # Compute comprehensive metrics
    print("\nComputing comprehensive metrics...")
    
    print("  - BLEU...")
    bleu_score = compute_bleu(predictions, references)
    
    print("  - ROUGE (1, 2, L)...")
    rouge_scores = compute_rouge(predictions, references)
    
    print("  - Perplexity...")
    perplexity = compute_perplexity(model, tokenizer, test_data[:500])
    
    print("  - Intent F1-score...")
    intent_metrics = compute_intent_f1(predictions, references, intents)
    
    print("  - Semantic similarity...")
    semantic_metrics = compute_semantic_similarity(predictions, references)
    
    print("  - Exact match...")
    exact_match = compute_exact_match(predictions, references)
    
    print("  - Response quality metrics...")
    quality_metrics = compute_response_quality_metrics(predictions, references)
    
    print("  - Per-intent performance...")
    per_intent_metrics = analyze_per_intent_performance(predictions, references, intents)
    
    print("  - Failure case analysis...")
    failure_cases = identify_failure_cases(predictions, references, test_data[:500], n_worst=10)
    
    # Generate qualitative examples
    print("\nGenerating qualitative examples...")
    examples = generate_qualitative_examples(model, tokenizer, test_data, n_examples=15)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'confusion_matrix' in intent_metrics and intent_metrics['confusion_matrix']:
        plot_confusion_matrix(
            intent_metrics['confusion_matrix'],
            intent_metrics['intent_labels'],
            output_dir / 'confusion_matrix.png'
        )
    
    plot_per_intent_performance(per_intent_metrics, output_dir / 'per_intent_performance.png')
    
    # Compile all metrics
    metrics = {
        # Core NLP metrics
        'bleu': float(bleu_score),
        'rouge1': float(rouge_scores['rouge1']),
        'rouge2': float(rouge_scores['rouge2']),
        'rougeL': float(rouge_scores['rougeL']),
        'perplexity': float(perplexity),
        
        # Intent classification metrics
        'intent_f1': float(intent_metrics.get('f1_weighted', 0.0)),
        'intent_precision': float(intent_metrics.get('precision', 0.0)),
        'intent_recall': float(intent_metrics.get('recall', 0.0)),
        'per_intent_f1': intent_metrics.get('per_intent_f1', {}),
        
        # Semantic metrics
        'semantic_similarity': float(semantic_metrics['semantic_similarity_mean']),
        'semantic_similarity_std': float(semantic_metrics['semantic_similarity_std']),
        
        # Response quality
        'exact_match': float(exact_match),
        'avg_response_length': float(quality_metrics['avg_pred_length']),
        'length_ratio': float(quality_metrics['length_ratio']),
        'vocabulary_diversity': float(quality_metrics['vocabulary_diversity']),
        
        # Analysis
        'per_intent_performance': per_intent_metrics,
        'failure_cases': failure_cases,
        'test_samples': len(test_data),
        'examples': examples,
    }
    
    return metrics

def print_comprehensive_report(metrics: Dict):
    """Print comprehensive evaluation report."""
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*70)
    
    print("\n📊 CORE NLP METRICS:")
    print("-" * 70)
    print(f"  BLEU Score:           {metrics['bleu']:.2f}")
    print(f"  ROUGE-1:              {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2:              {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L:              {metrics['rougeL']:.4f}")
    print(f"  Perplexity:           {metrics['perplexity']:.2f}")
    
    print("\n🎯 INTENT CLASSIFICATION METRICS:")
    print("-" * 70)
    print(f"  F1-Score (Weighted):  {metrics['intent_f1']:.4f}")
    print(f"  Precision:            {metrics['intent_precision']:.4f}")
    print(f"  Recall:               {metrics['intent_recall']:.4f}")
    
    print("\n🔍 SEMANTIC SIMILARITY:")
    print("-" * 70)
    print(f"  Mean Similarity:      {metrics['semantic_similarity']:.4f}")
    print(f"  Std Deviation:        {metrics['semantic_similarity_std']:.4f}")
    
    print("\n📝 RESPONSE QUALITY:")
    print("-" * 70)
    print(f"  Exact Match:          {metrics['exact_match']:.4f}")
    print(f"  Avg Response Length:  {metrics['avg_response_length']:.1f} tokens")
    print(f"  Length Ratio:         {metrics['length_ratio']:.2f}")
    print(f"  Vocab Diversity:      {metrics['vocabulary_diversity']:.4f}")
    
    print("\n📈 PER-INTENT PERFORMANCE (Top 5):")
    print("-" * 70)
    sorted_intents = sorted(
        metrics['per_intent_performance'].items(),
        key=lambda x: x[1]['bleu'],
        reverse=True
    )[:5]
    for intent, perf in sorted_intents:
        print(f"  {intent:25s} | BLEU: {perf['bleu']:5.2f} | ROUGE-L: {perf['rougeL']:.4f} | Count: {perf['count']}")
    
    print("\n❌ FAILURE CASES (Top 3):")
    print("-" * 70)
    for i, case in enumerate(metrics['failure_cases'][:3], 1):
        print(f"\n  Case {i} ({case['intent']}, Score: {case['score']:.3f}):")
        print(f"    User:       {case['user'][:80]}...")
        print(f"    Reference:  {case['reference'][:80]}...")
        print(f"    Prediction: {case['prediction'][:80]}...")
    
    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--model_dir', type=str, default='models/t5-small-finance', help='Model directory')
    parser.add_argument('--test', type=str, default='data/processed/test.jsonl', help='Test data path')
    parser.add_argument('--out', type=str, default='experiments/runs.csv', help='Output CSV path')
    parser.add_argument('--output_dir', type=str, default='data/reports', help='Output directory for reports')
    parser.add_argument('--append', action='store_true', help='Append to existing CSV')
    
    args = parser.parse_args()
    
    # Evaluate
    metrics = evaluate_model(
        Path(args.model_dir), 
        Path(args.test),
        Path(args.output_dir)
    )
    
    # Print comprehensive report
    print_comprehensive_report(metrics)
    
    # Print qualitative examples
    print("\n💬 QUALITATIVE EXAMPLES:")
    print("="*70)
    for i, ex in enumerate(metrics['examples'][:5], 1):
        print(f"\nExample {i} ({'OOD' if ex['is_ood'] else ex['intent']}):")
        print(f"  User:       {ex['user']}")
        print(f"  Reference:  {ex['reference']}")
        print(f"  Prediction: {ex['prediction']}")
        print("-" * 70)
    
    # Save detailed results
    results_path = Path("experiments/evaluation_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Detailed results saved to {results_path}")
    
    # Update runs CSV if requested
    if args.append and Path(args.out).exists():
        df = pd.read_csv(args.out)
        if len(df) > 0:
            # Update last row with test metrics
            df.loc[df.index[-1], 'test_bleu'] = metrics['bleu']
            df.loc[df.index[-1], 'test_rougel'] = metrics['rougeL']
            df.loc[df.index[-1], 'test_ppl'] = metrics['perplexity']
            df.loc[df.index[-1], 'test_f1'] = metrics['intent_f1']
            df.loc[df.index[-1], 'test_semantic_sim'] = metrics['semantic_similarity']
            df.to_csv(args.out, index=False)
            print(f"✓ Updated {args.out} with test metrics")

if __name__ == "__main__":
    main()
