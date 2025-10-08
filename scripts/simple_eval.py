"""
Simple evaluation script that computes key metrics without Unicode issues.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from tqdm import tqdm
import pandas as pd

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
    """Compute ROUGE scores."""
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
    
    return float(perplexity)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='models/t5-small-finance')
    parser.add_argument('--test', default='data/processed/test.jsonl')
    parser.add_argument('--out', default='experiments/evaluation_results.json')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_dir}...")
    model = TFAutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    print(f"Loading test data from {args.test}...")
    test_data = load_jsonl(Path(args.test))
    print(f"  Test samples: {len(test_data)}")
    
    print("\\nGenerating predictions...")
    predictions = []
    references = []
    
    for item in tqdm(test_data, desc="Generating"):
        pred = generate_response(model, tokenizer, item['user'])
        predictions.append(pred)
        references.append(item['assistant'])
    
    print("\\nComputing metrics...")
    
    # Core metrics
    bleu_score = compute_bleu(predictions, references)
    rouge_scores = compute_rouge(predictions, references)
    perplexity = compute_perplexity(model, tokenizer, test_data)
    
    # Compile results
    results = {
        'bleu': float(bleu_score),
        'rouge1': float(rouge_scores['rouge1']),
        'rouge2': float(rouge_scores['rouge2']),
        'rougeL': float(rouge_scores['rougeL']),
        'perplexity': float(perplexity),
        'num_samples': len(test_data)
    }
    
    # Save results
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"BLEU Score:     {results['bleu']:.4f}")
    print(f"ROUGE-1:        {results['rouge1']:.4f}")
    print(f"ROUGE-2:        {results['rouge2']:.4f}")
    print(f"ROUGE-L:        {results['rougeL']:.4f}")
    print(f"Perplexity:     {results['perplexity']:.4f}")
    print(f"Test Samples:   {results['num_samples']}")
    print("="*60)
    print(f"Results saved to: {args.out}")

if __name__ == "__main__":
    main()