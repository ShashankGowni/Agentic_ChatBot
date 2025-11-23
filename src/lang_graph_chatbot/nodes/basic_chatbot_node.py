"""Basic chatbot node - DYNAMIC (Paragraphs OK for general chat)"""

from src.lang_graph_chatbot.state.state import State
from langchain_core.messages import AIMessage

class BasicChatBot:
    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """Process general queries - Paragraphs OK"""
        messages = state.get("messages", [])
        
        if messages:
            if hasattr(messages[-1], "content"):
                user_message = messages[-1].content
            else:
                user_message = str(messages[-1])
        else:
            user_message = ""

        print(f"[BasicChatBot] Processing: {user_message}")

        # DYNAMIC PROMPT - Paragraphs OK for chat
        prompt = f"""You are an AI Financial Advisor Assistant.

**Your Expertise:**
• Personal finance and budgeting
• Gold and precious metal investments
• Investment planning (mutual funds, stocks, FDs)
• Savings strategies
• Income and expense management

**User Query:** "{user_message}"

**Response Guidelines:**

**IF this is a greeting or general question:**
- Respond warmly in paragraph format (natural conversation)
- Introduce yourself briefly
- Mention you can help with:
  • Gold purchase planning
  • Budget analysis
  • Investment advice
  • Financial planning
- Ask what they need help with

**IF this is an investment question:**
Use this structure:

📊 **Investment Analysis**

**Your Query:** [Summarize their question]

**Options to Consider:**

**1. [Investment Type]**
• ✅ [Benefit]
• ✅ [Benefit]
• ⚠️ [Risk/Consideration]

**2. [Investment Type]**
• ✅ [Benefit]
• ⚠️ [Risk]

**My Recommendation:**
• [Specific advice based on context]
• [Ask for details if needed: age, risk tolerance, amount]

**IF they ask about specific topics but without numbers:**
- Provide educational response in paragraph format
- Use bullet points for lists
- Be conversational
- Ask clarifying questions

**Format:**
- Greetings/General: Paragraphs are fine
- Financial Advice: Use bullet points and structure
- Use emojis moderately (💰 📊 🏆 📈 💡)
- Keep it professional but friendly
- Use Indian Rupee (₹) when discussing money

Generate helpful response now:"""

        # LLM generates response
        response_text = self.llm.invoke(prompt)
        print(f"[BasicChatBot] Response generated")

        return {"messages": [AIMessage(content=response_text)]}