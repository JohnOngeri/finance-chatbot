"""
Simple CLI demo for the finance chatbot.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from pathlib import Path
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

def generate_response(model, tokenizer, user_query: str, max_length=128, explain=False) -> str:
    """Generate response for a user query."""
    input_text = f"finance: {user_query}"
    
    if explain:
        print(f"\n[Explanation]")
        print(f"  Input prompt: '{input_text}'")
        print(f"  Max length: {max_length}")
        print(f"  Decoding: Beam search (num_beams=4)")
    
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

def main():
    parser = argparse.ArgumentParser(description='Finance Chatbot CLI Demo')
    parser.add_argument('--model_dir', type=str, default='models/t5-small-finance', help='Model directory')
    parser.add_argument('--explain', action='store_true', help='Show explanation for responses')
    
    args = parser.parse_args()
    
    print("Loading model...")
    model = TFAutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    print("✓ Model loaded!\n")
    
    print("="*60)
    print("Finance Chatbot CLI Demo")
    print("="*60)
    print("Ask me anything about finance! (Type 'quit' to exit)")
    print("Toggle explanation with 'explain on/off'")
    print("="*60 + "\n")
    
    explain = args.explain
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'explain on':
            explain = True
            print("✓ Explanation mode enabled\n")
            continue
        
        if user_input.lower() == 'explain off':
            explain = False
            print("✓ Explanation mode disabled\n")
            continue
        
        response = generate_response(model, tokenizer, user_input, explain=explain)
        print(f"\nBot: {response}\n")

if __name__ == "__main__":
    main()
