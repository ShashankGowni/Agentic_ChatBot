"""Budget analyzer node - DYNAMIC with LLM"""

from src.lang_graph_chatbot.state.state import State
from langchain_core.messages import AIMessage
import re

class BudgetAnalyzerNode:
    def __init__(self, model):
        self.llm = model
    
    def extract_financial_data(self, text: str) -> dict:
        """Extract income and expenses with 'k' suffix support"""
        text_lower = text.lower()
        numbers = []
        
        # Find numbers with optional 'k' suffix
        pattern = r'(\d+(?:,\d+)*(?:\.\d+)?)\s*k?\b'
        matches = re.findall(pattern, text_lower)
        
        for match in matches:
            num = float(match.replace(',', ''))
            if text_lower.find(match + 'k') != -1 or text_lower.find(match + ' k') != -1:
                num *= 1000
            numbers.append(num)
        
        income = None
        expenses = None
        
        if 'income' in text_lower or 'salary' in text_lower or 'earn' in text_lower:
            if len(numbers) >= 1:
                income = numbers[0]
        
        if 'expense' in text_lower or 'spend' in text_lower:
            if len(numbers) >= 2:
                expenses = numbers[1]
            elif len(numbers) == 1 and income is None:
                expenses = numbers[0]
        
        if len(numbers) >= 2:
            if income is None:
                income = numbers[0]
            if expenses is None:
                expenses = numbers[1]
        
        return {"income": income, "expenses": expenses}
    
    def process(self, state: State) -> dict:
        """Process budget analysis - DYNAMIC with STRUCTURED output"""
        messages = state.get("messages", [])
        user_message = messages[-1].content if messages else ""
        
        print(f"[Budget Analyzer] Processing: {user_message}")
        
        # Extract data
        data = self.extract_financial_data(user_message)
        
        if data["income"] and data["expenses"]:
            income = data["income"]
            expenses = data["expenses"]
            savings = income - expenses
            savings_rate = (savings / income * 100) if income > 0 else 0
            
            # Calculate 50-30-20 rule
            suggested_needs = income * 0.50
            suggested_wants = income * 0.30
            suggested_savings = income * 0.20
            emergency_fund = expenses * 6
            
            # DYNAMIC PROMPT with STRICT FORMATTING
            prompt = f"""You are a Budget Analysis Expert. Provide STRUCTURED, POINT-WISE analysis.

**Customer Data:**
- Monthly Income: ₹{income:,.0f}
- Monthly Expenses: ₹{expenses:,.0f}
- Monthly Savings: ₹{savings:,.0f}
- Savings Rate: {savings_rate:.1f}%

**Benchmarks:**
- Recommended Needs (50%): ₹{suggested_needs:,.0f}
- Recommended Wants (30%): ₹{suggested_wants:,.0f}
- Recommended Savings (20%): ₹{suggested_savings:,.0f}
- Emergency Fund Target: ₹{emergency_fund:,.0f}

**Use this EXACT structure with bullet points:**

📊 **Budget Analysis Report**

💵 **Current Financial Status:**
• Monthly Income: ₹{income:,.0f}
• Monthly Expenses: ₹{expenses:,.0f}
• Monthly Savings: ₹{savings:,.0f}
• Savings Rate: {savings_rate:.1f}%

📈 **Financial Health:**
• Rating: [Use 🟢 Excellent (30%+), 🟡 Good (20-30%), 🟠 Fair (10-20%), 🔴 Poor (<10%)]
• Status: [One sentence assessment]

💡 **50-30-20 Rule Analysis:**

**Recommended:**
• Needs (50%): ₹{suggested_needs:,.0f}
• Wants (30%): ₹{suggested_wants:,.0f}
• Savings (20%): ₹{suggested_savings:,.0f}

**Your Current Status:**
• [Compare and advise]

🎯 **Emergency Fund Planning:**
• Target amount: ₹{emergency_fund:,.0f} (6 months expenses)
• Current savings: ₹{savings:,.0f}/month
• Time to build: [calculate] months
• Importance: [one line]

📋 **Improvement Actions:**

**Immediate Steps (This Month):**
• [Specific action with numbers]
• [Specific action with numbers]
• [Specific action with numbers]

**Medium Term (3-6 Months):**
• [Specific goal]
• [Specific goal]

💰 **What You Can Do With ₹{savings:,.0f}/Month:**
• Emergency Fund: Build in [X] months
• Gold Investment: Buy ~[X] grams/month
• Mutual Fund SIP: Invest ₹[X]/month
• Fixed Deposit: Save ₹[X]/month

Use ONLY bullet points (•), NOT paragraphs. Be specific."""

        else:
            # No data - ask for it
            prompt = f"""You are a Budget Analyzer. User asked: "{user_message}"

They need to provide budget details.

**Respond in this structure:**

📊 **Budget Analysis Tool**

To analyze your budget, I need:

**Required Information:**
• Monthly income/salary (example: 60k or 60000)
• Monthly expenses (example: 40k or 40000)

**Example Queries:**
Analyze my budget: income 80k, expenses 60k

**What I'll Analyze:**
• Savings rate calculation
• 50-30-20 rule comparison
• Emergency fund planning
• Optimization suggestions
• Investment opportunities

**I Can Help With:**
• Identifying overspending
• Creating savings plan
• Building emergency fund
• Investment allocation

Please provide your income and expenses. Use bullet points."""

        # Let LLM generate response
        print(f"[Budget Analyzer] Sending to LLM...")
        response_text = self.llm.invoke(prompt)
        print(f"[Budget Analyzer] Response received")
        
        return {
            "messages": [AIMessage(content=response_text)]
        }