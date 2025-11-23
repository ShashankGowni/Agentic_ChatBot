"""Gold investment advisor node - DYNAMIC with LLM"""

from src.lang_graph_chatbot.state.state import State
from langchain_core.messages import AIMessage
import re

class GoldAdvisorNode:
    def __init__(self, model):
        self.llm = model
    
    def extract_numbers(self, text: str) -> dict:
        """Extract financial numbers with k/K support"""
        text_lower = text.lower()
        numbers = []
        
        # Find numbers with optional 'k' suffix (60k = 60000)
        pattern = r'(\d+(?:,\d+)*(?:\.\d+)?)\s*k?\b'
        matches = re.findall(pattern, text_lower)
        
        for match in matches:
            num = float(match.replace(',', ''))
            # Check if 'k' follows this number
            if text_lower.find(match + 'k') != -1 or text_lower.find(match + ' k') != -1:
                num *= 1000
            numbers.append(num)
        
        salary = None
        expenses = None
        target_grams = None
        months = 4  # default
        
        # Identify salary
        if 'salary' in text_lower or 'income' in text_lower or 'earn' in text_lower:
            if len(numbers) >= 1:
                salary = numbers[0]
        
        # Identify expenses
        if 'expense' in text_lower or 'spend' in text_lower:
            if len(numbers) >= 2:
                expenses = numbers[1]
            elif len(numbers) == 1 and salary is None:
                expenses = numbers[0]
        
        # If 2 numbers but not assigned, assume first is salary
        if len(numbers) >= 2 and salary is None:
            salary = numbers[0]
            expenses = numbers[1]
        
        # Look for grams
        if 'gram' in text_lower:
            gram_matches = re.findall(r'(\d+)\s*gram', text_lower)
            if gram_matches:
                target_grams = float(gram_matches[0])
        
        # Look for months
        month_match = re.search(r'in (\d+) month', text_lower)
        if month_match:
            months = int(month_match.group(1))
        
        result = {
            "salary": salary,
            "expenses": expenses,
            "target_grams": target_grams,
            "months": months
        }
        
        print(f"[GoldAdvisor] Extracted: {result}")
        return result
    
    def process(self, state: State) -> dict:
        """Process gold investment queries - DYNAMIC with STRUCTURED output"""
        messages = state.get("messages", [])
        user_message = messages[-1].content if messages else ""
        
        print(f"[Gold Advisor] Processing: {user_message}")
        
        # Extract financial data
        extracted = self.extract_numbers(user_message)
        
        # Build dynamic prompt for LLM
        if extracted["salary"] and extracted["expenses"]:
            salary = extracted["salary"]
            expenses = extracted["expenses"]
            savings = salary - expenses
            months = extracted["months"]
            gold_price = 6500  # Current approximate price
            total_savings = savings * months
            affordable_grams = total_savings / gold_price
            
            # DYNAMIC PROMPT with STRICT FORMATTING
            prompt = f"""You are a Gold Investment Advisor. Provide STRUCTURED, POINT-WISE advice.

**Customer Data:**
- Monthly Salary: ₹{salary:,.0f}
- Monthly Expenses: ₹{expenses:,.0f}
- Monthly Savings: ₹{savings:,.0f}
- Timeline: {months} months
- Total Savings: ₹{total_savings:,.0f}
- Current Gold Price: ₹{gold_price}/gram
- Affordable: {affordable_grams:.1f} grams

**IMPORTANT: Use this EXACT structure with bullet points:**

🏆 **Gold Investment Analysis**

💰 **Your Financial Summary:**
• Monthly Salary: ₹{salary:,.0f}
• Monthly Expenses: ₹{expenses:,.0f}
• Monthly Savings: ₹{savings:,.0f}
• Savings Rate: [calculate]%

📅 **{months}-Month Purchase Plan:**
• Total you'll save: ₹{total_savings:,.0f}
• Gold you can afford: {affordable_grams:.1f} grams
• Approximate value: ₹[calculate]

🏆 **Monthly Breakdown:**
• Savings per month: ₹{savings:,.0f}
• Gold per month: [calculate] grams
• Current gold price: ₹{gold_price}/gram

💡 **Investment Options:**

**1. Physical Gold (Jewelry/Coins)**
• ✅ Tangible asset
• ✅ Good for occasions
• ⚠️ Making charges: 10-25%
• ⚠️ Storage needed

**2. Digital Gold**
• ✅ No making charges
• ✅ Buy from ₹1
• ✅ Easy to sell
• ⚠️ Delivery fee if converted

**3. Sovereign Gold Bonds (SGB)**
• ✅ 2.5% interest yearly
• ✅ Government backed
• ✅ Tax benefits
• ⚠️ 8-year lock-in

📋 **My Recommendation:**
• [Specific advice for their ₹{savings:,.0f}/month savings]
• [Whether they should buy physical/digital/SGB]
• [Mention SIP approach - buying monthly]
• [Timeline advice]

🎯 **Action Steps:**
1. [Specific action]
2. [Specific action]
3. [Specific action]

Use ONLY bullet points (•), NOT paragraphs. Be specific with numbers."""

        else:
            # No data - ask for it
            prompt = f"""You are a Gold Investment Advisor. User asked: "{user_message}"

They need to provide financial details.

**Respond in this structure:**

🏆 **Gold Investment Planning**

To help you, I need:

**Required Information:**
• Your monthly salary/income (example: 60k or 60000)
• Your monthly expenses (example: 35k or 35000)
• Timeline (optional: "in 4 months")

**Example Query:**
I want to buy gold. Salary 60k, expenses 35k, in 4 months

**What I'll Help With:**
• Calculate how much gold you can afford
• Compare investment options
• Create purchase timeline
• Suggest best approach (Physical/Digital/SGB)

**Quick Gold Facts:**
• Current price: ~₹6,500/gram
• SGB gives 2.5% interest
• Digital gold has no making charges
• SIP approach reduces risk

Ask them friendly to provide the information. Use bullet points."""

        # Let LLM generate response
        print(f"[Gold Advisor] Sending to LLM...")
        response_text = self.llm.invoke(prompt)
        print(f"[Gold Advisor] Response received")
        
        return {
            "messages": [AIMessage(content=response_text)]
        }