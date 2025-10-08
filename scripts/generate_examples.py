"""
Generate qualitative examples for notebook
"""

import json
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def generate_response(model, tokenizer, user_query, max_length=128):
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

# Load model and tokenizer
print("Loading model...")
model = TFAutoModelForSeq2SeqLM.from_pretrained('models/t5-small-finance')
tokenizer = AutoTokenizer.from_pretrained('models/t5-small-finance')

# Load test data
test_data = load_jsonl('data/processed/test.jsonl')

# Generate examples
examples = []
for i, item in enumerate(test_data[:10]):
    prediction = generate_response(model, tokenizer, item['user'])
    examples.append({
        'user': item['user'],
        'reference': item['assistant'],
        'prediction': prediction,
        'intent': item['intent'],
        'is_ood': item['is_ood']
    })

# Save examples
with open('experiments/qualitative_examples.json', 'w') as f:
    json.dump(examples, f, indent=2)

print("Examples generated and saved to experiments/qualitative_examples.json")