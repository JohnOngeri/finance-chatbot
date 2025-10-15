import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

print("Loading model...")
model = TFAutoModelForSeq2SeqLM.from_pretrained('models/t5-small-finance-chatgpt')
tokenizer = AutoTokenizer.from_pretrained('models/t5-small-finance-chatgpt')
print("Model loaded!")

def test_query(query):
    input_text = f'finance: {query}'
    input_ids = tokenizer.encode(input_text, return_tensors='tf', max_length=128, truncation=True)
    outputs = model.generate(
        input_ids, 
        max_length=128, 
        num_beams=3, 
        early_stopping=True, 
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test finance questions
print("\n" + "="*60)
print("TESTING FINANCE QUESTIONS")
print("="*60)

questions = [
    "What is budgeting?",
    "How can I improve my credit score?", 
    "What are investment strategies?",
    "How do I save money?",
    "What is compound interest?"
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {test_query(q)}")

# Test weather/OOD questions  
print("\n" + "="*60)
print("TESTING WEATHER/OOD QUESTIONS")
print("="*60)

ood_questions = [
    "What's the weather today?",
    "Will it rain tomorrow?",
    "What movies are playing?",
    "How do I cook pasta?",
    "Who won the game?"
]

for q in ood_questions:
    print(f"\nQ: {q}")
    print(f"A: {test_query(q)}")