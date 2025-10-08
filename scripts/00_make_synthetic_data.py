"""
Generate synthetic finance domain conversational data.
Creates ≥2,500 conversation pairs across ≥12 intents with OOD examples.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import argparse

# Set seed for reproducibility
random.seed(42)

# Intent taxonomy for Finance domain
INTENTS = [
    "budgeting_definition",
    "budgeting_tips",
    "savings_advice",
    "interest_compound",
    "credit_score",
    "credit_vs_debit",
    "loan_info",
    "bank_fees",
    "fraud_awareness",
    "mobile_money",
    "exchange_rates",
    "tax_basics",
    "investment_basics",
    "financial_planning"
]

# Templates for each intent
TEMPLATES = {
    "budgeting_definition": [
        ("What is budgeting?", "Budgeting is the process of creating a plan to spend your money. It helps you determine in advance whether you will have enough money to do the things you need or would like to do. A budget helps you track income and expenses."),
        ("Can you explain budgeting?", "Budgeting means making a financial plan for your income and expenses. It's about deciding how to allocate your money across different categories like housing, food, savings, and entertainment."),
        ("Define budget", "A budget is a financial plan that outlines expected income and expenses over a specific period. It helps you control spending and work toward financial goals."),
        ("What does it mean to budget?", "To budget means to create a spending plan based on your income and expenses. It involves tracking where your money goes and making intentional decisions about spending and saving."),
    ],
    "budgeting_tips": [
        ("How can I budget better?", "Start with the 50/30/20 rule: allocate 50% of income to needs, 30% to wants, and 20% to savings and debt repayment. Track all expenses for a month to understand your spending patterns, then adjust accordingly."),
        ("Give me budgeting advice", "Create a realistic budget by listing all income sources and expenses. Use budgeting apps to track spending automatically. Review your budget monthly and adjust as needed. Always pay yourself first by setting aside savings before spending."),
        ("What's a good budgeting method?", "The 50/30/20 rule is popular and effective: 50% for necessities like rent and groceries, 30% for discretionary spending, and 20% for savings and debt. Another method is zero-based budgeting where every dollar has a purpose."),
        ("Tips for managing my budget?", "Track every expense, even small ones. Use the envelope method for cash spending. Automate savings transfers. Cut unnecessary subscriptions. Review and adjust your budget monthly based on actual spending."),
    ],
    "savings_advice": [
        ("How much should I save?", "Financial experts recommend saving at least 20% of your income. Start with an emergency fund covering 3-6 months of living expenses, then focus on retirement and other goals."),
        ("What's an emergency fund?", "An emergency fund is money set aside specifically for unexpected expenses like medical bills, car repairs, or job loss. It should cover 3-6 months of essential living expenses and be kept in an easily accessible savings account."),
        ("Where should I keep my savings?", "Keep emergency funds in a high-yield savings account for easy access and some interest. For long-term savings, consider certificates of deposit (CDs) or investment accounts depending on your timeline and risk tolerance."),
        ("How do I start saving money?", "Start small with automatic transfers from checking to savings each payday. Even $25-50 per paycheck adds up. Cut one unnecessary expense and redirect that money to savings. Build the habit first, then increase amounts."),
    ],
    "interest_compound": [
        ("What is compound interest?", "Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods. Essentially, it's earning interest on your interest, which accelerates wealth growth over time."),
        ("Explain compound interest", "Compound interest means your money grows exponentially because you earn returns on both your original investment and previous earnings. For example, $1,000 at 5% annual interest becomes $1,050 after year one, then $1,102.50 after year two."),
        ("How does compound interest work?", "Compound interest works by adding earned interest back to the principal, so future interest calculations are based on a larger amount. The more frequently interest compounds (daily, monthly, annually), the faster your money grows."),
        ("Why is compound interest important?", "Compound interest is powerful for long-term wealth building. Starting early gives your money more time to compound. Even small amounts can grow significantly over decades due to the exponential nature of compounding."),
    ],
    "credit_score": [
        ("What is a credit score?", "A credit score is a three-digit number (typically 300-850) that represents your creditworthiness. It's calculated based on payment history, credit utilization, length of credit history, types of credit, and recent inquiries."),
        ("How can I improve my credit score?", "Pay all bills on time, keep credit card balances below 30% of limits, don't close old credit accounts, limit new credit applications, and regularly check your credit report for errors. Payment history is the most important factor."),
        ("What affects credit score?", "Five main factors affect your credit score: payment history (35%), amounts owed (30%), length of credit history (15%), new credit (10%), and credit mix (10%). Late payments and high balances hurt your score most."),
        ("What's a good credit score?", "Credit scores range from 300-850. Generally, 670-739 is good, 740-799 is very good, and 800+ is excellent. Scores above 670 typically qualify for favorable loan terms and interest rates."),
    ],
    "credit_vs_debit": [
        ("What's the difference between credit and debit cards?", "Credit cards let you borrow money up to a limit and pay it back later, potentially with interest. Debit cards withdraw money directly from your bank account. Credit cards can help build credit history, while debit cards help avoid debt."),
        ("Should I use credit or debit?", "Use credit cards for building credit, earning rewards, and purchase protection, but only if you pay the full balance monthly. Use debit cards if you struggle with overspending or want to avoid debt entirely."),
        ("Credit card vs debit card?", "Credit cards offer rewards, fraud protection, and credit building but can lead to debt if misused. Debit cards provide spending control and no interest charges but offer less fraud protection and don't build credit."),
        ("When to use credit card?", "Use credit cards for large purchases with buyer protection, online shopping for security, earning rewards on regular expenses, and building credit history. Always pay the full balance to avoid interest charges."),
    ],
    "loan_info": [
        ("What is a personal loan?", "A personal loan is an unsecured loan from a bank or lender that can be used for various purposes like debt consolidation, home improvements, or major purchases. Interest rates depend on your credit score and income."),
        ("How do mortgages work?", "A mortgage is a loan specifically for purchasing real estate where the property serves as collateral. You make monthly payments over 15-30 years covering principal and interest. If you default, the lender can foreclose on the property."),
        ("What is APR?", "APR (Annual Percentage Rate) is the yearly cost of borrowing money, expressed as a percentage. It includes the interest rate plus fees and other charges, giving you the true cost of a loan for comparison shopping."),
        ("Should I get a loan?", "Consider a loan only if you can afford the monthly payments, the interest rate is reasonable, and the purpose is worthwhile (education, home, debt consolidation). Avoid loans for depreciating assets or discretionary spending."),
    ],
    "bank_fees": [
        ("What are bank fees?", "Bank fees are charges for services like monthly account maintenance, overdrafts, ATM usage outside your network, wire transfers, paper statements, and minimum balance violations. Fees vary by bank and account type."),
        ("How can I avoid bank fees?", "Maintain minimum balance requirements, use in-network ATMs, set up overdraft protection, opt for electronic statements, and choose accounts with no monthly fees. Many online banks offer fee-free checking and savings accounts."),
        ("What is an overdraft fee?", "An overdraft fee is charged when you spend more money than you have in your account and the bank covers the transaction. Fees typically range from $25-35 per transaction. You can opt out of overdraft coverage to avoid these fees."),
        ("Why do banks charge fees?", "Banks charge fees to cover operational costs and generate revenue. However, many fees are avoidable by meeting account requirements, using services properly, or choosing banks with fewer fees like online banks or credit unions."),
    ],
    "fraud_awareness": [
        ("What is phishing?", "Phishing is a fraudulent attempt to obtain sensitive information like passwords, credit card numbers, or social security numbers by disguising as a trustworthy entity via email, text, or phone. Never click suspicious links or share personal info."),
        ("How do I protect against fraud?", "Use strong, unique passwords for each account. Enable two-factor authentication. Monitor bank and credit card statements regularly. Don't share personal information via email or phone unless you initiated contact. Shred financial documents before disposal."),
        ("What is identity theft?", "Identity theft occurs when someone uses your personal information (name, SSN, credit card numbers) without permission to commit fraud like opening accounts, making purchases, or filing false tax returns. It can severely damage your credit and finances."),
        ("Signs of credit card fraud?", "Watch for unauthorized charges, unfamiliar accounts on your credit report, missing bills or statements, denied credit applications, or calls about purchases you didn't make. Report suspicious activity to your bank immediately."),
    ],
    "mobile_money": [
        ("What is mobile banking?", "Mobile banking allows you to access banking services through your smartphone or tablet. You can check balances, transfer money, pay bills, deposit checks, and manage accounts anytime, anywhere with an internet connection."),
        ("Is mobile banking safe?", "Mobile banking is generally safe when you use official bank apps, strong passwords, biometric authentication, and secure networks. Avoid public Wi-Fi for banking. Enable notifications for all transactions to catch unauthorized activity quickly."),
        ("What is a digital wallet?", "A digital wallet is an electronic service that stores payment information on your device for contactless payments. Examples include Apple Pay, Google Pay, and Samsung Pay. They use encryption and tokenization for security."),
        ("How does mobile payment work?", "Mobile payments use NFC (near-field communication) technology to transmit encrypted payment information from your phone to a payment terminal. Your actual card number isn't shared, making it more secure than physical cards."),
    ],
    "exchange_rates": [
        ("What are exchange rates?", "Exchange rates represent the value of one currency in terms of another. For example, if USD/EUR is 0.85, one US dollar equals 0.85 euros. Rates fluctuate based on economic factors, interest rates, and market demand."),
        ("Why do exchange rates change?", "Exchange rates change due to supply and demand, interest rate differences, economic performance, political stability, inflation rates, and market speculation. Central bank policies and international trade also influence currency values."),
        ("How to get best exchange rate?", "Compare rates from multiple sources. Use credit cards with no foreign transaction fees for purchases abroad. Avoid airport currency exchanges which have poor rates. Consider online currency exchange services or withdraw local currency from ATMs."),
        ("What affects currency value?", "Currency value is affected by interest rates, inflation, economic growth, political stability, trade balances, government debt, and market sentiment. Strong economies with stable governments typically have stronger currencies."),
    ],
    "tax_basics": [
        ("What is income tax?", "Income tax is a tax levied by governments on income earned by individuals and businesses. The amount depends on your income level and tax bracket. Taxes fund public services like infrastructure, education, and healthcare."),
        ("What are tax deductions?", "Tax deductions are expenses that can be subtracted from your gross income to reduce taxable income. Common deductions include mortgage interest, charitable donations, student loan interest, and certain business expenses."),
        ("When are taxes due?", "In the US, federal income tax returns are typically due on April 15th for the previous calendar year. You can file for an extension until October 15th, but any taxes owed must still be paid by April 15th to avoid penalties."),
        ("What is a tax bracket?", "A tax bracket is a range of income taxed at a specific rate in a progressive tax system. Higher income portions are taxed at higher rates. For example, the first $10,000 might be taxed at 10%, while income over $50,000 is taxed at 22%."),
    ],
    "investment_basics": [
        ("What is diversification?", "Diversification means spreading investments across various assets (stocks, bonds, real estate) to reduce risk. The principle is 'don't put all eggs in one basket.' If one investment performs poorly, others may offset the loss."),
        ("What is a stock?", "A stock represents partial ownership in a company. When you buy stock, you become a shareholder and may receive dividends and benefit from price appreciation. Stock prices fluctuate based on company performance and market conditions."),
        ("What is a bond?", "A bond is a fixed-income investment where you loan money to a government or corporation for a set period at a fixed interest rate. Bonds are generally less risky than stocks but offer lower potential returns."),
        ("Should I invest?", "Investing is important for long-term wealth building and retirement. Start after establishing an emergency fund and paying off high-interest debt. Begin with low-cost index funds for diversification. The earlier you start, the more time for compound growth."),
    ],
    "financial_planning": [
        ("What is net worth?", "Net worth is your total assets (cash, investments, property) minus total liabilities (debts, loans). It's a snapshot of your overall financial health. Calculate it annually to track progress toward financial goals."),
        ("How do I set financial goals?", "Set SMART goals: Specific, Measurable, Achievable, Relevant, and Time-bound. Categorize as short-term (< 1 year), medium-term (1-5 years), or long-term (5+ years). Examples: save $5,000 emergency fund in 12 months, or save $50,000 for house down payment in 5 years."),
        ("What is financial planning?", "Financial planning is the process of setting goals, assessing your current financial situation, and creating strategies to achieve those goals. It includes budgeting, saving, investing, insurance, tax planning, and retirement planning."),
        ("When should I start financial planning?", "Start financial planning as soon as you have income. The earlier you begin, the more time for compound growth and achieving goals. Even small steps like budgeting and building an emergency fund make a significant difference over time."),
    ],
}

# Out-of-domain queries with appropriate rejections
OOD_EXAMPLES = [
    ("What's the weather today?", "I'm specialized in finance topics like budgeting, savings, credit, and loans. I can't help with weather information."),
    ("How do I bake a cake?", "I'm specialized in finance topics. I can't provide cooking advice, but I'd be happy to help with budgeting for groceries or meal planning costs."),
    ("Who won the election?", "I'm specialized in finance and avoid political topics. I can help with financial planning, budgeting, or investment questions instead."),
    ("What are symptoms of flu?", "I'm specialized in finance and can't provide medical advice. For health concerns, please consult a healthcare professional. I can help with health insurance or medical expense budgeting questions."),
    ("Recommend a good movie", "I'm specialized in finance topics. I can't recommend entertainment, but I can help you budget for entertainment expenses or subscription services."),
    ("How to fix my car?", "I'm specialized in finance and can't provide automotive repair advice. However, I can help you budget for car maintenance or understand auto loan options."),
    ("What's the capital of France?", "I'm specialized in finance topics. I can't help with geography, but I'd be happy to discuss currency exchange rates or international banking."),
    ("Tell me a joke", "I'm specialized in finance topics like budgeting, savings, and investing. I focus on providing helpful financial information rather than entertainment."),
    ("What's the meaning of life?", "I'm specialized in finance and focus on practical financial advice. I can help with financial planning, budgeting, or achieving your financial goals."),
    ("How do I learn Python?", "I'm specialized in finance topics. For programming questions, I'd recommend coding resources. I can help with budgeting for online courses or education expenses though."),
    ("What's the best restaurant nearby?", "I'm specialized in finance and can't provide restaurant recommendations. However, I can help you budget for dining out or track food expenses."),
    ("How to train my dog?", "I'm specialized in finance topics. I can't provide pet training advice, but I can help you budget for pet expenses or understand pet insurance options."),
    ("What's the score of the game?", "I'm specialized in finance and don't track sports scores. I can help with budgeting, savings, credit, loans, and other financial topics."),
    ("Translate this to Spanish", "I'm specialized in finance and don't provide translation services. I can help with financial topics like budgeting, savings, or understanding exchange rates."),
    ("What's the latest celebrity gossip?", "I'm specialized in finance topics. I focus on providing helpful financial advice rather than entertainment news."),
]

# Paraphrase variations
PARAPHRASES = {
    "What is": ["What's", "Can you explain", "Define", "Tell me about", "Help me understand"],
    "How can I": ["How do I", "What's the best way to", "Tips for", "Help me", "Guide me to"],
    "Should I": ["Is it good to", "Is it wise to", "Would you recommend", "Do you suggest"],
}

def apply_paraphrase(text: str) -> str:
    """Apply random paraphrasing to text."""
    for original, variations in PARAPHRASES.items():
        if text.startswith(original):
            return text.replace(original, random.choice(variations), 1)
    return text

def generate_conversations(n_samples: int = 3000) -> List[Dict]:
    """Generate synthetic conversation pairs."""
    conversations = []
    conv_id = 0
    
    # Calculate samples per intent
    n_intents = len(INTENTS)
    n_ood = int(n_samples * 0.12)  # 12% OOD
    n_in_domain = n_samples - n_ood
    samples_per_intent = n_in_domain // n_intents
    
    # Generate in-domain conversations
    for intent in INTENTS:
        templates = TEMPLATES[intent]
        subintents = [f"{intent}_{i}" for i in range(len(templates))]
        
        for _ in range(samples_per_intent):
            # Select random template
            template_idx = random.randint(0, len(templates) - 1)
            user_text, assistant_text = templates[template_idx]
            
            # Apply paraphrasing 40% of the time
            if random.random() < 0.4:
                user_text = apply_paraphrase(user_text)
            
            conversations.append({
                "id": f"conv_{conv_id:05d}",
                "user": user_text,
                "assistant": assistant_text,
                "intent": intent,
                "subintent": subintents[template_idx],
                "source": "synthetic",
                "is_ood": False,
            })
            conv_id += 1
    
    # Generate OOD conversations
    for _ in range(n_ood):
        user_text, assistant_text = random.choice(OOD_EXAMPLES)
        conversations.append({
            "id": f"conv_{conv_id:05d}",
            "user": user_text,
            "assistant": assistant_text,
            "intent": "ood",
            "subintent": "ood",
            "source": "synthetic",
            "is_ood": True,
        })
        conv_id += 1
    
    # Add multi-turn conversations (10% of total)
    n_multiturn = int(n_samples * 0.10)
    for _ in range(n_multiturn):
        # Create a 2-3 turn conversation
        n_turns = random.randint(2, 3)
        intent = random.choice(INTENTS)
        templates = TEMPLATES[intent]
        
        for turn in range(n_turns):
            template_idx = random.randint(0, len(templates) - 1)
            user_text, assistant_text = templates[template_idx]
            
            conversations.append({
                "id": f"conv_{conv_id:05d}",
                "user": user_text,
                "assistant": assistant_text,
                "intent": intent,
                "subintent": f"{intent}_{template_idx}",
                "source": "synthetic_multiturn",
                "is_ood": False,
            })
            conv_id += 1
    
    # Shuffle conversations
    random.shuffle(conversations)
    
    return conversations

def stratified_split(conversations: List[Dict], train_ratio=0.8, val_ratio=0.1):
    """Split data with stratification by intent."""
    from collections import defaultdict
    
    # Group by intent
    intent_groups = defaultdict(list)
    for conv in conversations:
        intent_groups[conv["intent"]].append(conv)
    
    train_data, val_data, test_data = [], [], []
    
    for intent, items in intent_groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])
    
    # Shuffle each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def save_jsonl(data: List[Dict], filepath: Path):
    """Save data to JSONL format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic finance conversation data')
    parser.add_argument('--domain', type=str, default='finance', help='Domain name')
    parser.add_argument('--n', type=int, default=3000, help='Number of samples to generate')
    parser.add_argument('--out', type=str, default='data/raw/seed_facts.md', help='Output path')
    args = parser.parse_args()
    
    print(f"Generating {args.n} synthetic conversations for {args.domain} domain...")
    
    # Generate conversations
    conversations = generate_conversations(args.n)
    
    print(f"Generated {len(conversations)} total conversations")
    print(f"  - In-domain: {sum(1 for c in conversations if not c['is_ood'])}")
    print(f"  - OOD: {sum(1 for c in conversations if c['is_ood'])}")
    print(f"  - Unique intents: {len(set(c['intent'] for c in conversations))}")
    
    # Split data
    train_data, val_data, test_data = stratified_split(conversations)
    
    print(f"\nSplit sizes:")
    print(f"  - Train: {len(train_data)}")
    print(f"  - Val: {len(val_data)}")
    print(f"  - Test: {len(test_data)}")
    
    # Save splits
    output_dir = Path("data/processed")
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "val.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")
    
    print(f"\n✓ Data saved to {output_dir}/")
    print(f"✓ Seed facts already exist at data/raw/seed_facts.md")

if __name__ == "__main__":
    main()
