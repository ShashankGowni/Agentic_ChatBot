"""Query classifier node - determines user intent"""

from src.lang_graph_chatbot.state.state import State
from langchain_core.messages import AIMessage

class ClassifierNode:
    def __init__(self, model):
        self.llm = model
    
    def process(self, state: State) -> dict:
        """Classify user query and route to appropriate node"""
        messages = state.get("messages", [])
        
        if not messages:
            return {"next_node": "basic_chat"}
        
        # Get last user message
        last_message = messages[-1]
        user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        print(f"[Classifier] Analyzing query: {user_input}")
        
        # Create classification prompt
        classification_prompt = f"""Classify this user query into ONE category:

User Query: "{user_input}"

Categories:
1. GOLD - Questions about buying gold, gold investment, gold prices
2. BUDGET - Questions about budget, savings, expenses, income analysis
3. INVESTMENT - Questions about where to invest, investment advice
4. DOCUMENT - User wants to analyze a document or uploaded file
5. GENERAL - General financial questions or greetings

Reply with ONLY ONE WORD: GOLD, BUDGET, INVESTMENT, DOCUMENT, or GENERAL"""

        try:
            response = self.llm.invoke(classification_prompt)
            intent = response.strip().upper()
            
            print(f"[Classifier] Detected intent: {intent}")
            
            # Map intent to node
            intent_mapping = {
                "GOLD": "gold_advisor",
                "BUDGET": "budget_analyzer",
                "INVESTMENT": "financial_advisor",
                "DOCUMENT": "document_qa",
                "GENERAL": "basic_chat"
            }
            
            next_node = intent_mapping.get(intent, "basic_chat")
            
            return {
                "messages": state["messages"],
                "next_node": next_node,
                "user_intent": intent
            }
            
        except Exception as e:
            print(f"[Classifier] Error: {e}")
            return {
                "messages": state["messages"],
                "next_node": "basic_chat",
                "user_intent": "GENERAL"
            }