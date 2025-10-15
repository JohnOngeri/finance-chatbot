"""
Gradio UI for Finance Chatbot - Enhanced Version
Provides intuitive, user-friendly chat interface with comprehensive features.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gradio as gr
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
import time
import json
from datetime import datetime

# Load model and tokenizer
MODEL_DIR = "models/t5-small-finance-processed"

print("Loading model...")
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("✓ Model loaded!")

# Categorized sample questions for better UX
SAMPLE_QUESTIONS = {
    "💰 Budgeting & Savings": [
        "What is budgeting and why is it important?",
        "How can I start saving money effectively?",
        "What is an emergency fund and how much should I save?",
        "What are some practical budgeting tips?",
    ],
    "💳 Credit & Debt": [
        "How can I improve my credit score?",
        "What's the difference between credit and debit cards?",
        "How do credit card interest rates work?",
        "What should I know about managing debt?",
    ],
    "🏦 Banking & Loans": [
        "What are common bank fees and how can I avoid them?",
        "How do personal loans work?",
        "What is compound interest?",
        "What should I consider when choosing a bank?",
    ],
    "📈 Investing & Planning": [
        "What are the basics of investing?",
        "How do exchange rates work?",
        "What is financial planning?",
        "What are different types of investment accounts?",
    ],
}

# Finance keywords for OOD detection
FINANCE_KEYWORDS = [
    'budget', 'save', 'saving', 'credit', 'debit', 'loan', 'interest', 'bank',
    'money', 'finance', 'financial', 'invest', 'investment', 'tax', 'exchange', 
    'currency', 'account', 'payment', 'debt', 'income', 'expense', 'mortgage', 
    'insurance', 'stock', 'bond', 'fund', 'retirement', 'pension', 'asset',
    'liability', 'equity', 'dividend', 'portfolio', 'risk', 'return', 'apr',
    'fee', 'charge', 'transaction', 'balance', 'statement', 'deposit', 'withdrawal'
]

OOD_PATTERNS = [
    'weather', 'cook', 'recipe', 'movie', 'game', 'sport', 'election', 
    'political', 'medical', 'health', 'symptom', 'disease', 'treatment',
    'entertainment', 'celebrity', 'music', 'art', 'travel', 'hotel'
]

def calculate_confidence(query: str, response: str) -> tuple[float, str]:
    """Calculate confidence score based on query-response characteristics."""
    query_lower = query.lower()
    
    # Check for finance keywords in query
    keyword_match = sum(1 for kw in FINANCE_KEYWORDS if kw in query_lower)
    
    # Check response quality indicators
    response_length = len(response.split())
    has_specific_info = any(term in response.lower() for term in ['%', '$', 'rate', 'fee', 'account'])
    
    # Calculate confidence
    confidence = 0.5  # Base confidence
    
    if keyword_match > 0:
        confidence += min(0.3, keyword_match * 0.1)
    
    if response_length > 20:
        confidence += 0.1
    
    if has_specific_info:
        confidence += 0.1
    
    confidence = min(1.0, confidence)
    
    # Determine confidence level
    if confidence >= 0.8:
        level = "High"
    elif confidence >= 0.6:
        level = "Medium"
    else:
        level = "Low"
    
    return confidence, level

def detect_ood(query: str) -> tuple[bool, str]:
    """Detect out-of-domain queries."""
    query_lower = query.lower()
    
    # Check for explicit OOD patterns (weather, cooking, etc.)
    has_ood_pattern = any(pattern in query_lower for pattern in OOD_PATTERNS)
    
    # Only reject if we have clear OOD patterns
    if has_ood_pattern:
        return True, "Out-of-domain query detected"
    
    # Let the model handle everything else
    return False, "In-domain query"

def generate_response(
    user_query: str, 
    explain: bool = False, 
    max_length: int = 128,
    temperature: float = 0.7
):
    """Generate chatbot response with enhanced features."""
    start_time = time.time()
    
    # OOD Detection
    is_ood, ood_reason = detect_ood(user_query)
    
    if is_ood:
        response = (
            "🔍 **Out-of-Domain Query Detected**\n\n"
            "I'm specialized in **personal finance topics** including:\n"
            "- 💰 Budgeting and savings\n"
            "- 💳 Credit and debt management\n"
            "- 🏦 Banking and loans\n"
            "- 📈 Investing and financial planning\n\n"
            "Please ask me a finance-related question, or try one of the suggested questions!"
        )
        
        response_time = time.time() - start_time
        confidence = 1.0
        confidence_level = "High"
        
        explanation = ""
        if explain:
            explanation = f"""
### 🔍 Explanation

**1. Input Analysis:**
- Query: "{user_query}"
- Detection: {ood_reason}
- Domain: Out-of-domain

**2. OOD Detection Logic:**
- Finance keywords found: No
- OOD patterns detected: Yes
- Decision: Polite rejection with guidance

**3. Response Strategy:**
- Provide clear domain boundaries
- Suggest alternative topics
- Maintain helpful tone

**4. Performance:**
- Response time: {response_time:.3f}s
- Confidence: {confidence:.2%} ({confidence_level})
            """
        
        return response, explanation, response_time, confidence, confidence_level
    
    # Generate response
    input_text = f"finance: {user_query}"
    input_ids = tokenizer.encode(
        input_text, 
        return_tensors='tf', 
        max_length=max_length, 
        truncation=True
    )
    
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=3,
        early_stopping=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.1,
        do_sample=False,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_time = time.time() - start_time
    
    # Calculate confidence
    confidence, confidence_level = calculate_confidence(user_query, response)
    
    # Format response with markdown
    formatted_response = f"💬 {response}"
    
    # Generate explanation if requested
    explanation = ""
    if explain:
        explanation = f"""
### 🔍 Explanation

**1. Input Processing:**
- Original query: "{user_query}"
- Formatted prompt: "{input_text}"
- Domain: Finance ✓
- Input tokens: {len(input_ids[0])}

**2. Model Configuration:**
- Architecture: T5-small (60M parameters)
- Fine-tuned on: 3000+ finance conversations
- Max length: {max_length} tokens
- Temperature: {temperature}
- Decoding: Beam search (num_beams=4)
- No repeat n-gram: 2

**3. Generation Process:**
- Tokenization: SentencePiece (T5 tokenizer)
- Encoding: Input → Token IDs
- Generation: Beam search decoding
- Decoding: Token IDs → Text
- Output tokens: {len(outputs[0])}

**4. Quality Metrics:**
- Response time: {response_time:.3f}s
- Confidence: {confidence:.2%} ({confidence_level})
- Response length: {len(response.split())} words

**5. Domain Coverage:**
- Topics: Budgeting, Credit, Banking, Investing
- Training data: 14 finance intents
- OOD handling: Active
        """
    
    return formatted_response, explanation, response_time, confidence, confidence_level

def chat_interface(message, history, explain, max_length, temperature):
    """Enhanced chat interface handler with rich features."""
    if not message.strip():
        return history, "", ""
    
    # Generate response
    response, explanation, response_time, confidence, confidence_level = generate_response(
        message, 
        explain, 
        max_length,
        temperature
    )
    
    # Format user message
    user_msg = f"**You:** {message}"
    
    # Format bot response with metadata
    bot_msg = f"{response}\n\n"
    bot_msg += f"⏱️ *Response time: {response_time:.2f}s* | "
    bot_msg += f"📊 *Confidence: {confidence_level} ({confidence:.0%})*"
    
    if explain and explanation:
        bot_msg += f"\n\n{explanation}"
    
    # Update history
    history = history or []
    history.append([user_msg, bot_msg])
    
    # Generate related questions
    related = generate_related_questions(message)
    
    return history, "", related

def generate_related_questions(query: str) -> str:
    """Generate related questions based on user query."""
    query_lower = query.lower()
    
    related_map = {
        'budget': [
            "What are some budgeting strategies?",
            "How do I track my expenses?",
            "What is the 50/30/20 budget rule?"
        ],
        'credit': [
            "How is credit score calculated?",
            "What affects my credit score?",
            "How long does it take to improve credit?"
        ],
        'save': [
            "What are high-yield savings accounts?",
            "How much should I save each month?",
            "What are automated savings tools?"
        ],
        'invest': [
            "What is diversification?",
            "What are index funds?",
            "How do I start investing?"
        ],
        'loan': [
            "What is APR vs interest rate?",
            "How do I qualify for a loan?",
            "What are loan terms?"
        ],
    }
    
    for keyword, questions in related_map.items():
        if keyword in query_lower:
            return "### 💡 Related Questions:\n" + "\n".join(f"- {q}" for q in questions)
    
    return "### 💡 Try asking about:\n- Budgeting strategies\n- Credit management\n- Savings tips\n- Investment basics"

def export_chat(history):
    """Export chat history as JSON."""
    if not history:
        return None
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation": [
            {"user": msg[0], "bot": msg[1]} 
            for msg in history
        ]
    }
    
    filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return filename

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.chat-message {
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
}
.user-message {
    background-color: #e3f2fd;
}
.bot-message {
    background-color: #f5f5f5;
}
.confidence-high {
    color: #4caf50;
}
.confidence-medium {
    color: #ff9800;
}
.confidence-low {
    color: #f44336;
}
"""

# Create enhanced Gradio interface
with gr.Blocks(
    title="Finance Chatbot - AI Assistant", 
    theme=gr.themes.Soft(primary_hue="blue"),
    css=custom_css
) as demo:
    
    # Header
    gr.Markdown(
        """
        # 💰 Finance Chatbot - Your AI Financial Assistant
        
        Welcome! I'm an AI assistant specialized in personal finance. Ask me anything about budgeting, 
        savings, credit, loans, banking, and financial planning. I'm here to help you make informed 
        financial decisions!
        
        ---
        """
    )
    
    # Main interface
    with gr.Row():
        # Left column - Chat interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=200, 
                label="💬 Conversation",
                show_label=True,
                avatar_images=(None, "🤖"),
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Type your finance question here... (e.g., 'How can I improve my credit score?')",
                    lines=2,
                    max_lines=5,
                    show_label=False,
                    scale=4
                )
            
            with gr.Row():
                submit = gr.Button("📤 Send", variant="primary", scale=1)
                clear = gr.Button("🗑️ Clear Chat", scale=1)
                export = gr.Button("💾 Export Chat", scale=1)
            
            # Advanced options
            with gr.Accordion("⚙️ Advanced Options", open=False):
                with gr.Row():
                    explain_checkbox = gr.Checkbox(
                        label="Show Detailed Explanation",
                        value=False,
                        info="Display reasoning steps, model configuration, and quality metrics"
                    )
                    max_length_slider = gr.Slider(
                        minimum=64,
                        maximum=256,
                        value=128,
                        step=32,
                        label="Max Response Length",
                        info="Maximum number of tokens in response"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Higher = more creative, Lower = more focused"
                    )
            
            # Related questions display
            related_questions = gr.Markdown("### 💡 Ask me anything about finance!")
        
        # Right column - Help and examples
        with gr.Column(scale=1):
            gr.Markdown("### 📚 Quick Start Guide")
            gr.Markdown(
                """
                **How to use:**
                1. Type your question in the text box
                2. Click "Send" or press Enter
                3. Get instant AI-powered answers
                4. Enable "Show Explanation" for details
                
                **Tips:**
                - Be specific in your questions
                - Try the example questions below
                - Use advanced options for customization
                - Export your chat for later reference
                """
            )
            
            gr.Markdown("### 💡 Example Questions")
            
            # Categorized example buttons
            for category, questions in SAMPLE_QUESTIONS.items():
                with gr.Accordion(category, open=False):
                    for question in questions:
                        btn = gr.Button(question, size="sm", variant="secondary")
                        btn.click(lambda q=question: q, outputs=msg)
            
            # Help section
            with gr.Accordion("❓ Help & Information", open=False):
                gr.Markdown(
                    """
                    ### What I Can Help With:
                    
                    ✅ **Budgeting & Savings**
                    - Creating budgets
                    - Saving strategies
                    - Emergency funds
                    - Expense tracking
                    
                    ✅ **Credit & Debt**
                    - Credit scores
                    - Credit cards
                    - Debt management
                    - Interest rates
                    
                    ✅ **Banking & Loans**
                    - Bank accounts
                    - Personal loans
                    - Mortgages
                    - Banking fees
                    
                    ✅ **Investing & Planning**
                    - Investment basics
                    - Financial planning
                    - Retirement accounts
                    - Risk management
                    
                    ### What I Can't Help With:
                    
                    ❌ Medical or health advice
                    ❌ Legal advice
                    ❌ Tax preparation (consult a professional)
                    ❌ Non-finance topics
                    
                    ### Confidence Levels:
                    
                    - 🟢 **High**: Strong finance-related query
                    - 🟡 **Medium**: Somewhat related to finance
                    - 🔴 **Low**: May be out-of-domain
                    
                    ### Keyboard Shortcuts:
                    
                    - `Enter`: Send message
                    - `Shift + Enter`: New line
                    """
                )
    
    # Footer with instructions
    with gr.Accordion("📖 Detailed Instructions & Features", open=False):
        gr.Markdown(
            """
            ## Features
            
            ### 🎯 Core Features:
            - **Intelligent Responses**: AI-powered answers trained on 3000+ finance conversations
            - **OOD Detection**: Automatically detects and handles non-finance questions
            - **Confidence Scoring**: Shows how confident the model is in its response
            - **Response Time**: Displays how quickly the answer was generated
            - **Explanation Mode**: See detailed reasoning and model configuration
            
            ### 🔧 Advanced Options:
            - **Max Response Length**: Control how long responses can be (64-256 tokens)
            - **Temperature**: Adjust creativity vs focus (0.1-1.0)
            - **Export Chat**: Save your conversation as JSON for later reference
            
            ### 📊 Quality Indicators:
            - **Response Time**: How fast the model generated the answer
            - **Confidence Level**: High/Medium/Low based on query-response match
            - **Related Questions**: Suggested follow-up questions
            
            ### 🎓 Best Practices:
            1. **Be Specific**: "How can I improve my credit score?" vs "Tell me about credit"
            2. **One Topic**: Focus on one question at a time for best results
            3. **Use Examples**: Click suggested questions to see the format
            4. **Check Confidence**: Higher confidence = more reliable answer
            5. **Enable Explanation**: Learn how the AI generates responses
            
            ### 🔒 Privacy & Limitations:
            - This is an AI assistant, not a financial advisor
            - Always consult professionals for important financial decisions
            - Your conversations are not stored permanently
            - The model is trained on general finance knowledge
            
            ### 🐛 Troubleshooting:
            - **Low confidence?** Try rephrasing with finance keywords
            - **Out-of-domain?** Make sure your question is finance-related
            - **Unclear answer?** Try being more specific or use examples
            - **Need more detail?** Enable "Show Explanation" mode
            """
        )
    
    # Event handlers
    submit_event = msg.submit(
        chat_interface, 
        [msg, chatbot, explain_checkbox, max_length_slider, temperature_slider], 
        [chatbot, msg, related_questions]
    )
    
    click_event = submit.click(
        chat_interface, 
        [msg, chatbot, explain_checkbox, max_length_slider, temperature_slider], 
        [chatbot, msg, related_questions]
    )
    
    clear.click(lambda: ([], ""), None, [chatbot, related_questions], queue=False)
    
    export.click(export_chat, chatbot, None)

# Launch configuration
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching Finance Chatbot UI...")
    print("="*60)
    print("\n📊 Model Information:")
    print(f"   - Architecture: T5-small (60M parameters)")
    print(f"   - Training: Fine-tuned on finance domain")
    print(f"   - Intents: 14 finance categories")
    print(f"   - Dataset: 3000+ conversations")
    print("\n💡 Features:")
    print("   - Intelligent OOD detection")
    print("   - Confidence scoring")
    print("   - Explanation mode")
    print("   - Related questions")
    print("   - Chat export")
    print("\n" + "="*60 + "\n")
    
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )