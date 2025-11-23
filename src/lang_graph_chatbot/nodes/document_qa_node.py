"""Document Q&A node"""

from src.lang_graph_chatbot.state.state import State
from langchain_core.messages import AIMessage
from src.lang_graph_chatbot.tools import document_parser

class DocumentQANode:
    def __init__(self, model):
        self.llm = model
    
    def process(self, state: State) -> dict:
        """Process document-related queries"""
        messages = state.get("messages", [])
        user_message = messages[-1].content if messages else ""
        
        print(f"[Document Q&A] Processing: {user_message}")
        
        # Parse text for financial data
        parsed_data = document_parser.parse_text_input(user_message)
        
        if parsed_data["found"]:
            response = f"""📄 **Document Analysis**

Extracted Information:
"""
            if parsed_data["income"]:
                response += f"- Income: ₹{parsed_data['income']:,.0f}\n"
            if parsed_data["expenses"]:
                response += f"- Expenses: ₹{parsed_data['expenses']:,.0f}\n"
            
            response += "\n💡 I can help you analyze this further. Would you like budget advice or investment suggestions?"
        
        else:
            response = """📄 **Document Upload Feature**

Currently, I can analyze financial information from your text.

Please share:
- Salary slips (copy relevant numbers)
- Expense details
- Bank statements (amounts)

Example: "My salary is 70000, rent is 20000, food is 10000, other expenses 15000"

*Advanced PDF upload feature coming soon!*"""

        print(f"[Document Q&A] Response generated")
        
        return {
            "messages": [AIMessage(content=response)]
        }