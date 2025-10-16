# Finance Chatbot Dataset Report

## Overview

This dataset contains synthetic conversational data for a finance domain chatbot, 
preprocessed using a comprehensive pipeline to ensure high quality and domain specificity.

## Preprocessing Pipeline

### 1. Text Normalization

- **Unicode Normalization:** Applied NFC (Canonical Decomposition followed by Canonical Composition)
- **Whitespace Handling:** Collapsed multiple spaces, tabs, and newlines into single spaces
- **Control Character Removal:** Removed non-printable control characters
- **Rationale:** Ensures consistent text representation and removes hidden characters that could affect model training

### 2. Noise Removal

- **HTML/XML Tags:** Stripped all markup tags using regex patterns
- **URLs:** Removed web addresses that don't contribute to financial understanding
- **Email Addresses:** Removed to protect privacy and reduce noise
- **Punctuation Normalization:** Standardized spacing around punctuation marks
- **Rationale:** Removes irrelevant information and standardizes text format

### 3. Financial Term Preservation

- **Protected Terms:** 29 financial acronyms and terms (APR, IRA, 401k, etc.)
- **Case Preservation:** Maintains uppercase for financial acronyms
- **Rationale:** Preserves domain-specific terminology critical for financial understanding

### 4. Quality Filtering

- **User Query Length:** 5-500 characters
- **Assistant Response Length:** 20-1000 characters
- **Content Validation:** Ensures meaningful alphanumeric content
- **Structure Validation:** Verifies proper sentence structure
- **Rationale:** Filters out malformed, too brief, or excessively long samples

### 5. Deduplication

- **Method:** Exact matching on (user, assistant) pairs
- **Rationale:** Prevents model from memorizing repeated patterns


## Tokenization Analysis

### Tokenizer: T5 SentencePiece

**Why T5 SentencePiece?**
- **Subword Tokenization:** Handles rare financial terms by breaking them into meaningful subwords
- **Language-Agnostic:** Works well with mixed-case text and special characters common in finance
- **Vocabulary Efficiency:** 32,000 token vocabulary provides good coverage without excessive size
- **Sequence-to-Sequence Optimized:** Designed for text generation tasks like conversational AI
- **Robust to OOV:** Rare terms are decomposed rather than mapped to unknown tokens

### Tokenization Statistics (Train Set)

- **Vocabulary Size:** 32,100 tokens
- **Unique Tokens Used:** 961 (2.99% coverage)
- **Financial Terms Found:** 7 unique terms
- **Avg Tokens per Financial Term:** 1.43

**User Query Tokens:** Mean=7.7, Median=7, Range=[4-13]
**Assistant Response Tokens:** Mean=45.8, Median=46, Range=[22-80]

### Tokenization Examples

\`\`\`
Example 1: 'What is APR?'
Tokens: ['▁What', '▁is', '▁AP', 'R', '?']

Example 2: 'How does compound interest work?'
Tokens: ['▁How', '▁does', '▁compound', '▁interest', '▁work', '?']

Example 3: 'Explain 401k retirement accounts'
Tokens: ['▁Explain', '▁', '4', '0', '1', 'k', '▁retirement', '▁accounts']
\`\`\`

*Note: ▁ represents space character in SentencePiece*

## Intent Distribution (Train Set)

| Intent | Count | Percentage |
|--------|-------|------------|
| investment_basics | 23 | 13.0% |
| loan_info | 18 | 10.2% |
| ood | 15 | 8.5% |
| credit_score | 14 | 7.9% |
| mobile_money | 14 | 7.9% |
| bank_fees | 14 | 7.9% |
| tax_basics | 13 | 7.3% |
| fraud_awareness | 12 | 6.8% |
| financial_planning | 12 | 6.8% |
| budgeting_definition | 9 | 5.1% |
| interest_compound | 9 | 5.1% |
| credit_vs_debit | 8 | 4.5% |
| budgeting_tips | 8 | 4.5% |
| exchange_rates | 4 | 2.3% |
| savings_advice | 4 | 2.3% |

## Sample Conversations

### In-Domain Examples (After Preprocessing)


**Example 1** (Intent: `tax_basics`)
- **User:** When are taxes due?
- **Assistant:** In the US, federal income tax returns are typically due on April 15th for the previous calendar year. You can file for an extension until October 15th, but any taxes owed must still be paid by April 15th to avoid penalties.


**Example 2** (Intent: `fraud_awareness`)
- **User:** How do I protect against fraud?
- **Assistant:** Use strong, unique passwords for each account. Enable two-factor authentication. Monitor bank and credit card statements regularly. Don't share personal information via email or phone unless you initiated contact. Shred financial documents before disposal.


**Example 3** (Intent: `tax_basics`)
- **User:** What is income tax?
- **Assistant:** Income tax is a tax levied by governments on income earned by individuals and businesses. The amount depends on your income level and tax bracket. Taxes fund public services like infrastructure, education, and healthcare.


**Example 4** (Intent: `financial_planning`)
- **User:** How do I set financial goals?
- **Assistant:** Set SMART goals: Specific, Measurable, Achievable, Relevant, and Time-bound. Categorize as short-term (< 1 year), medium-term (1-5 years), or long-term (5+ years). Examples: save $5, 000 emergency fund in 12 months, or save $50, 000 for house down payment in 5 years.


**Example 5** (Intent: `credit_score`)
- **User:** What's a good credit score?
- **Assistant:** Credit scores range from 300-850. Generally, 670-739 is good, 740-799 is very good, and 800+ is excellent. Scores above 670 typically qualify for favorable loan terms and interest rates.

### Out-of-Domain Examples


**Example 1**
- **User:** Recommend a good movie
- **Assistant:** I'm specialized in finance topics. I can't recommend entertainment, but I can help you budget for entertainment expenses or subscription services.


**Example 2**
- **User:** How do I learn Python?
- **Assistant:** I'm specialized in finance topics. For programming questions, I'd recommend coding resources. I can help with budgeting for online courses or education expenses though.


**Example 3**
- **User:** Tell me a joke
- **Assistant:** I'm specialized in finance topics like budgeting, savings, and investing. I focus on providing helpful financial information rather than entertainment.


## Visualizations

See the following generated plots in `data/reports/`:
- `token_length_distributions.png` - Token length histograms for all splits
- `intent_distribution.png` - Intent distribution bar chart
- `split_sizes.png` - Comparison of split sizes

## Data Quality Summary

✓ **High-Quality Dataset:** Comprehensive preprocessing ensures clean, consistent data
✓ **Domain-Specific:** Financial terminology preserved and properly handled
✓ **Well-Tokenized:** T5 SentencePiece provides excellent coverage and subword handling
✓ **Balanced Distribution:** Stratified splits maintain intent distribution across train/val/test
✓ **Quality Filtered:** Malformed and low-quality samples removed
✓ **Deduplicated:** No repeated conversation pairs

