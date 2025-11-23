### Start of ./src/lang_graph_chatbot\main.py ###
import streamlit as st
from src.lang_graph_chatbot.ui.streamlitui.loadui import LoadStreamlitUi
from src.lang_graph_chatbot.LLMS.huggingface import HuggingFace
from src.lang_graph_chatbot.LLMS.gemini_llm import GeminiLLM
from src.lang_graph_chatbot.graph.graph_builder import GraphBuilder
from src.lang_graph_chatbot.ui.streamlitui.display_result import DisplayResult
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def langchain_chatbot():
    ui = LoadStreamlitUi()
    user_inputs = ui.load_Streamlit_Ui()

    if not user_inputs:
        st.error("Error: Failed to load user input from the UI.")
        return

    user_message = st.chat_input("💬 Ask me about finance, gold, budgeting, investments...")
    
    if user_message:
        print(f"[Main] User Input: {user_message}")

        try:
            # Get selected LLM provider
            selected_provider = user_inputs.get("Selected_llm", "gemini")
            print(f"[Main] Selected Provider: {selected_provider}")
            
            # Initialize model based on provider
            if selected_provider == "gemini":
                # Get Gemini API key
                api_token = user_inputs.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
                
                if not api_token:
                    st.warning(" Please enter your Gemini API key in the sidebar.")
                    st.info("👉 Get a FREE key at https://aistudio.google.com/app/apikey")
                    return
                
                print("[Main] Using Gemini")
                obj_llm_config = GeminiLLM(user_input=user_inputs, api_token=api_token)
                model = obj_llm_config.get_llm_model()
            
            else:  # huggingface
                # Get HuggingFace API token
                api_token = user_inputs.get("huggingface_api_token") or os.getenv("HUGGINGFACE_API_TOKEN")
                
                if not api_token:
                    st.warning(" Please enter your HuggingFace API token in the sidebar.")
                    st.info("👉 Get a FREE token at https://huggingface.co/settings/tokens")
                    return
                
                print("[Main] Using HuggingFace")
                obj_llm_config = HuggingFace(user_input=user_inputs, api_token=api_token)
                model = obj_llm_config.get_llm_model()

            if not model:
                st.error(" Error: LLM model could not be configured.")
                return

            # Get selected use case
            usecase = user_inputs.get("selected_user_option")
            if not usecase:
                st.error(" Error: No use case is selected.")
                return

            print(f"[Main] Use case: {usecase}")

            # Build graph based on use case
            graph_builder = GraphBuilder(model)
            compiled_graph = graph_builder.setup_graph(usecase)

            # Process message through graph
            display = DisplayResult(usecase, user_message, compiled_graph)
            display.display_result_ui()

        except Exception as e:
            st.error(f" Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            st.error("Please check your API key and try again.")
            return
### End of ./src/lang_graph_chatbot\main.py ###

### Start of ./src/lang_graph_chatbot\__init__.py ###

### End of ./src/lang_graph_chatbot\__init__.py ###

### Start of ./src/lang_graph_chatbot\graph\graph_builder.py ###
from langgraph.graph import StateGraph
from lang_graph_chatbot.state.state import State
from langgraph.graph import START, END
from lang_graph_chatbot.nodes.basic_chatbot_node import BasicChatBot
from lang_graph_chatbot.nodes.classifier_node import ClassifierNode
from lang_graph_chatbot.nodes.gold_advisor_node import GoldAdvisorNode
from lang_graph_chatbot.nodes.budget_analyzer_node import BudgetAnalyzerNode
from lang_graph_chatbot.nodes.document_qa_node import DocumentQANode

class GraphBuilder:
    def __init__(self, model):
        self.llm = model
        self.graph = StateGraph(State)
        self.basic_chat_bot_node = None

    def basic_chat_bot(self):
        """Simple chatbot without routing"""
        self.basic_chat_bot_node = BasicChatBot(self.llm)
        self.graph.add_node("Chatbot", self.basic_chat_bot_node.process)
        self.graph.add_edge(START, "Chatbot")
        self.graph.add_edge("Chatbot", END)
        compiled_graph = self.graph.compile()
        return compiled_graph

    def financial_advisor_bot(self):
        """Financial advisor with multiple specialized nodes"""
        
        # Initialize all nodes
        classifier = ClassifierNode(self.llm)
        gold_advisor = GoldAdvisorNode(self.llm)
        budget_analyzer = BudgetAnalyzerNode(self.llm)
        document_qa = DocumentQANode(self.llm)
        basic_chat = BasicChatBot(self.llm)
        
        # Add nodes to graph
        self.graph.add_node("classifier", classifier.process)
        self.graph.add_node("gold_advisor", gold_advisor.process)
        self.graph.add_node("budget_analyzer", budget_analyzer.process)
        self.graph.add_node("document_qa", document_qa.process)
        self.graph.add_node("basic_chat", basic_chat.process)
        
        # Define routing function
        def route_query(state: State) -> str:
            """Route to appropriate node based on classification"""
            next_node = state.get("next_node", "basic_chat")
            print(f"[Router] Routing to: {next_node}")
            return next_node
        
        # Add edges
        self.graph.add_edge(START, "classifier")
        
        # Conditional routing from classifier
        self.graph.add_conditional_edges(
            "classifier",
            route_query,
            {
                "gold_advisor": "gold_advisor",
                "budget_analyzer": "budget_analyzer",
                "document_qa": "document_qa",
                "basic_chat": "basic_chat",
                "financial_advisor": "basic_chat"  # fallback
            }
        )
        
        # All specialist nodes go to END
        self.graph.add_edge("gold_advisor", END)
        self.graph.add_edge("budget_analyzer", END)
        self.graph.add_edge("document_qa", END)
        self.graph.add_edge("basic_chat", END)
        
        compiled_graph = self.graph.compile()
        return compiled_graph

    def setup_graph(self, usecase: str):
        """Setup graph based on selected use case"""
        print(f"[GraphBuilder] Setting up graph for: {usecase}")
        
        if usecase == "Basic Chatbot":
            return self.basic_chat_bot()
        elif usecase == "Financial Advisor":
            return self.financial_advisor_bot()
        elif usecase == "Gold Investment Advisor":
            return self.financial_advisor_bot()
        elif usecase == "Budget Analyzer":
            return self.financial_advisor_bot()
        else:
            # Default to basic chatbot
            return self.basic_chat_bot()
### End of ./src/lang_graph_chatbot\graph\graph_builder.py ###

### Start of ./src/lang_graph_chatbot\graph\__init__.py ###

### End of ./src/lang_graph_chatbot\graph\__init__.py ###

### Start of ./src/lang_graph_chatbot\LLMS\gemini_llm.py ###
import streamlit as st
import google.generativeai as genai
import traceback

class GeminiModelWrapper:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        
        try:
            genai.configure(api_key=api_key)
            # Use the actual model name from selection
            self.model = genai.GenerativeModel(model_name)
            print(f"✅ Gemini Model initialized: {model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            raise

    def invoke(self, input_text: str) -> str:
        print(f"📝 Gemini Input: {input_text[:100]}...")
        
        try:
            response = self.model.generate_content(
                input_text,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.7,
                )
            )
            
            if response and response.text:
                print(f"✅ Gemini Success: {len(response.text)} characters")
                return response.text
            elif response and response.prompt_feedback:
                print(f"⚠️ Content blocked: {response.prompt_feedback}")
                return "⚠️ Content was blocked by safety filters. Please rephrase your query."
            else:
                return "No response generated. Please try again."
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Gemini Error: {error_msg}")
            print(traceback.format_exc())
            
            if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
                st.error("🔑 Invalid API Key. Get a new key at: https://aistudio.google.com/app/apikey")
                return "❌ Invalid API key"
            
            elif "403" in error_msg:
                st.error("⚠️ API not enabled. Enable at: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
                return "❌ API not enabled"
            
            elif "quota" in error_msg.lower() or "429" in error_msg:
                st.warning("⚠️ Rate limit reached. Wait 60 seconds.")
                return "⏳ Rate limit reached. Please wait."
            
            elif "404" in error_msg or "not found" in error_msg.lower():
                st.error(f"❌ Model not found. Available models listed at startup.")
                return f"❌ Model not available: {self.model_name}"
            
            else:
                st.error(f"❌ Error: {error_msg}")
                return f"Error: {error_msg}"


class GeminiLLM:
    def __init__(self, user_input, api_token):
        self.user_input = user_input
        self.api_token = api_token

    def get_llm_model(self):
        if not self.api_token:
            st.error(" Please enter your Gemini API key")
            return None
        
        # Get selected model from UI
        selected_model = self.user_input.get("Selected_model", "gemini-2.5-flash")
        st.write(f"✅ Using Gemini Model: **{selected_model}**")
        
        try:
            return GeminiModelWrapper(selected_model, self.api_token)
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {e}")
            return None
### End of ./src/lang_graph_chatbot\LLMS\gemini_llm.py ###

### Start of ./src/lang_graph_chatbot\LLMS\groq_llm.py ###
# import streamlit as st
# from groq import Groq
# import traceback

# class GroqModelWrapper:
#     def __init__(self, model_name, api_token):
#         self.model_name = model_name
#         self.api_token = api_token
#         self.client = Groq(api_key=api_token)
#         print(f"[Groq] Initialized model: {self.model_name}")

#     def invoke(self, input_text: str) -> str:
#         """Call Groq API with user input"""
#         print(f"[Groq] Input: {input_text[:100]}...")
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {"role": "user", "content": input_text}
#                 ],
#                 max_tokens=1024,
#                 temperature=0.7
#             )
            
#             content = response.choices[0].message.content
#             print(f"[Groq] Response: {content[:100]}...")
#             return content

#         except Exception as e:
#             error_msg = f"Groq API Error: {str(e)}"
#             print(error_msg)
#             print(traceback.format_exc())
#             st.error(error_msg)
#             return f"Error: {str(e)}"


# class GroqLLM:
#     def __init__(self, user_input, api_token):
#         self.user_input = user_input
#         self.api_token = api_token

#     def get_llm_model(self):
#         """Get the selected Groq model"""
#         selected_model = self.user_input.get("Selected_model", None)
#         if selected_model:
#             st.write(f"✅ Using Groq Model: **{selected_model}**")
#             return GroqModelWrapper(selected_model, self.api_token)
#         else:
#             st.error(" No model selected.")
#             return None
### End of ./src/lang_graph_chatbot\LLMS\groq_llm.py ###

### Start of ./src/lang_graph_chatbot\LLMS\huggingface.py ###
import streamlit as st
from huggingface_hub import InferenceClient
import traceback

class HuggingFaceModelWrapper:
    def __init__(self, model_name, api_token):
        self.model_name = model_name
        self.api_token = api_token
        self.client = InferenceClient(token=api_token)
        print(f"HuggingFace Model Name: {self.model_name}")

    def invoke(self, input_text: str) -> str:
        print(f"User Input to HuggingFace: {input_text}")
        
        # Try chat_completion first
        try:
            print(f"Attempting chat_completion with: {self.model_name}")
            
            messages = [{"role": "user", "content": input_text}]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=512,
                temperature=0.7
            )
            
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                print(f"✅ Success: {content}")
                return content
            else:
                return str(response)

        except (StopIteration, ValueError) as e:
            print(f" chat_completion not available for this model, trying text_generation...")
            
        except Exception as e:
            print(f" chat_completion error: {e}")
            if "stopiteration" in str(type(e)).lower() or "provider" in str(e).lower():
                print(f"Trying text_generation as fallback...")
            else:
                error_msg = f"Unexpected error: {str(e)}"
                print(error_msg)
                return f"Error: {str(e)}"
        
        # Fallback to text_generation
        try:
            print(f"Attempting text_generation with: {self.model_name}")
            response = self.client.text_generation(
                input_text,
                model=self.model_name,
                max_new_tokens=512,
                temperature=0.7,
                return_full_text=False
            )
            print(f"✅ text_generation success: {response}")
            return response if response else "No content returned."
            
        except Exception as e2:
            error_msg = f"Model {self.model_name} is not available"
            print(f" Both methods failed: {str(e2)}")
            print(traceback.format_exc())
            
            st.warning(f" {self.model_name} is currently unavailable. Using default model.")
            
            # Final fallback: use Mistral-7B which we know works
            if self.model_name != "mistralai/Mistral-7B-Instruct-v0.2":
                print(f"Falling back to mistralai/Mistral-7B-Instruct-v0.2")
                try:
                    fallback_messages = [{"role": "user", "content": input_text}]
                    fallback_response = self.client.chat_completion(
                        messages=fallback_messages,
                        model="mistralai/Mistral-7B-Instruct-v0.2",
                        max_tokens=512,
                        temperature=0.7
                    )
                    content = fallback_response.choices[0].message.content
                    print(f"✅ Fallback successful: {content}")
                    st.info(" Using Mistral-7B-Instruct-v0.2 (your selected model was unavailable)")
                    return content
                except Exception as e3:
                    print(f" Even fallback failed: {e3}")
                    return "Sorry, I'm having trouble connecting to the models right now. Please try again."
            
            return f" Unable to get response. Please try selecting mistralai/Mistral-7B-Instruct-v0.2"


class HuggingFace:
    def __init__(self, user_input, api_token):
        self.user_input = user_input
        self.api_token = api_token

    def get_llm_model(self):
        selected_hf_model = self.user_input.get("Selected_model", None)
        if selected_hf_model:
            st.write(f"✅ Selected HuggingFace Model: {selected_hf_model}")
            return HuggingFaceModelWrapper(selected_hf_model, self.api_token)
        else:
            # Default to Mistral if no model selected
            st.info(" No model selected, using Mistral-7B-Instruct-v0.2 (default)")
            return HuggingFaceModelWrapper("mistralai/Mistral-7B-Instruct-v0.2", self.api_token)
### End of ./src/lang_graph_chatbot\LLMS\huggingface.py ###

### Start of ./src/lang_graph_chatbot\LLMS\ollama.py ###
import ollama
import streamlit as st

class OllamaModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Model Name: {self.model_name}")

    def invoke(self, input_text: str) -> str:
        print(f"User Input to Ollama: {input_text}")
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": input_text}]
            )
            print(f"Full raw response from Ollama: {response}")

            # Extract content from response object attributes
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                content = response.message.content
                print(f"Extracted assistant response content: {content}")
                return content or "No content returned."
            elif isinstance(response, dict):
                return response.get("message", "No message key in response")
            elif isinstance(response, str):
                return response
            else:
                return str(response)

        except Exception as e:
            print(f"Error during Ollama invoke: {e}")
            st.error(f"Error: {e}")
            return "Sorry, I failed to respond."

class Ollama:
    def __init__(self, user_input):
        self.user_input = user_input

    def get_llm_model(self):
        selected_ollama_model = self.user_input.get("Selected_model", None)
        if selected_ollama_model:
            st.write(f"✅ Selected Ollama Model: {selected_ollama_model}")
            return OllamaModelWrapper(selected_ollama_model)
        else:
            st.error("Error: No model selected.")
            return None

### End of ./src/lang_graph_chatbot\LLMS\ollama.py ###

### Start of ./src/lang_graph_chatbot\LLMS\__init__.py ###

### End of ./src/lang_graph_chatbot\LLMS\__init__.py ###

### Start of ./src/lang_graph_chatbot\nodes\basic_chatbot_node.py ###
from src.lang_graph_chatbot.state.state import State
from langchain_core.messages import AIMessage

class BasicChatBot:
    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """Process general financial queries and greetings"""
        messages = state.get("messages", [])
        print(f"[BasicChatBot] State messages: {messages}")
        
        if messages:
            if hasattr(messages[-1], "content"):
                user_message = messages[-1].content
            else:
                user_message = str(messages[-1])
        else:
            user_message = ""

        print(f"[BasicChatBot] Processing user message: {user_message}")

        # Enhanced prompt for financial context
        financial_context_prompt = f"""You are a friendly and professional Financial Advisor AI assistant.

Your expertise includes:
- Personal finance and budgeting
- Gold and precious metal investments  
- Savings and investment planning
- Income and expense management

User Query: {user_message}

Provide helpful, accurate, and friendly financial advice. 

If the user is greeting you or asking general questions, respond warmly and let them know you can help with:
- Gold purchase planning
- Budget analysis
- Investment advice
- Financial planning

If they ask specific financial questions, provide clear guidance and ask for details if needed (like income, expenses, etc.).

Response:"""

        response_text = self.llm.invoke(financial_context_prompt)
        print(f"[BasicChatBot] LLM response: {response_text}")

        return {"messages": [AIMessage(content=response_text)]}
### End of ./src/lang_graph_chatbot\nodes\basic_chatbot_node.py ###

### Start of ./src/lang_graph_chatbot\nodes\budget_analyzer_node.py ###
"""Budget analyzer node"""

from src.lang_graph_chatbot.state.state import State
from langchain_core.messages import AIMessage
from src.lang_graph_chatbot.tools import budget_calculator
import re

class BudgetAnalyzerNode:
    def __init__(self, model):
        self.llm = model
    
    def extract_financial_data(self, text: str) -> dict:
        """Extract income and expenses from text"""
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', text)
        numbers = [float(n.replace(',', '')) for n in numbers]
        
        # Try to identify income vs expenses from context
        text_lower = text.lower()
        
        income = None
        expenses = None
        
        if len(numbers) >= 2:
            # Assume first number is income, second is expenses
            income = numbers[0]
            expenses = numbers[1]
        elif len(numbers) == 1:
            if 'income' in text_lower or 'salary' in text_lower:
                income = numbers[0]
            elif 'expense' in text_lower or 'spend' in text_lower:
                expenses = numbers[0]
        
        return {"income": income, "expenses": expenses}
    
    def process(self, state: State) -> dict:
        """Process budget analysis queries"""
        messages = state.get("messages", [])
        user_message = messages[-1].content if messages else ""
        
        print(f"[Budget Analyzer] Processing: {user_message}")
        
        # Extract financial data
        data = self.extract_financial_data(user_message)
        
        if data["income"] and data["expenses"]:
            # Perform budget analysis
            analysis = budget_calculator.calculate_savings(data["income"], data["expenses"])
            
            # Get budget suggestion using 50-30-20 rule
            suggestion = budget_calculator.suggest_budget_plan(data["income"], analysis["savings"])
            
            # Generate response
            response = f"""📊 **Budget Analysis Report**

💵 **Current Financial Status:**
- Monthly Income: ₹{analysis['income']:,.0f}
- Monthly Expenses: ₹{analysis['expenses']:,.0f}
- Monthly Savings: ₹{analysis['savings']:,.0f}
- Savings Rate: {analysis['savings_percentage']:.1f}%

📈 **Financial Health:**
{"🟢 Excellent! You're saving well." if analysis['savings_percentage'] >= 30 else
 "🟡 Good, but can improve." if analysis['savings_percentage'] >= 20 else
 "🟠 Fair, try to save more." if analysis['savings_percentage'] >= 10 else
 "🔴 Low savings. Need to reduce expenses."}

💡 **Recommended Budget (50-30-20 Rule):**
- Needs (50%): ₹{suggestion['suggested_needs']:,.0f}
- Wants (30%): ₹{suggestion['suggested_wants']:,.0f}
- Savings (20%): ₹{suggestion['suggested_savings']:,.0f}

🎯 **Emergency Fund Target:**
₹{analysis['expenses'] * 6:,.0f} (6 months of expenses)

📋 **Recommendations:**
1. Build an emergency fund
2. Reduce unnecessary expenses
3. Invest your savings wisely
4. Track expenses monthly

Would you like investment suggestions for your savings?"""

        else:
            # Ask for information
            response = """To analyze your budget, please provide:

1. Your monthly income/salary
2. Your monthly expenses

Example: "My income is 60000 and expenses are 40000"

Or you can ask general budget questions!"""

        print(f"[Budget Analyzer] Response generated")
        
        return {
            "messages": [AIMessage(content=response)]
        }
### End of ./src/lang_graph_chatbot\nodes\budget_analyzer_node.py ###

### Start of ./src/lang_graph_chatbot\nodes\classifier_node.py ###
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
### End of ./src/lang_graph_chatbot\nodes\classifier_node.py ###

### Start of ./src/lang_graph_chatbot\nodes\document_qa_node.py ###
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
### End of ./src/lang_graph_chatbot\nodes\document_qa_node.py ###

### Start of ./src/lang_graph_chatbot\nodes\gold_advisor_node.py ###
"""Gold investment advisor node"""

from src.lang_graph_chatbot.state.state import State
from langchain_core.messages import AIMessage
from src.lang_graph_chatbot.tools import gold_advisor
import re

class GoldAdvisorNode:
    def __init__(self, model):
        self.llm = model
    
    def extract_numbers(self, text: str) -> dict:
        """Extract financial numbers from text"""
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', text)
        numbers = [float(n.replace(',', '')) for n in numbers]
        
        result = {
            "salary": numbers[0] if len(numbers) > 0 else None,
            "expenses": numbers[1] if len(numbers) > 1 else None,
            "target_grams": numbers[2] if len(numbers) > 2 else None
        }
        return result
    
    def process(self, state: State) -> dict:
        """Process gold investment queries"""
        messages = state.get("messages", [])
        user_message = messages[-1].content if messages else ""
        
        print(f"[Gold Advisor] Processing: {user_message}")
        
        # Extract financial data from user message
        extracted = self.extract_numbers(user_message)
        
        # If we have salary and expenses, provide gold advice
        if extracted["salary"] and extracted["expenses"]:
            salary = extracted["salary"]
            expenses = extracted["expenses"]
            
            # Calculate affordable gold
            affordable = gold_advisor.calculate_affordable_gold(salary, expenses)
            
            # Create purchase plan if target mentioned
            target_grams = extracted["target_grams"] or affordable["affordable_grams_per_month"] * 6
            plan = gold_advisor.create_gold_purchase_plan(
                affordable["monthly_savings"],
                target_grams
            )
            
            # Generate response
            response = f"""📊 **Gold Investment Analysis**

💰 **Your Financial Summary:**
- Monthly Salary: ₹{salary:,.0f}
- Monthly Expenses: ₹{expenses:,.0f}
- Monthly Savings: ₹{affordable['monthly_savings']:,.0f}

🏆 **Gold Purchase Capacity:**
- Current Gold Price: ₹{affordable['gold_price_per_gram']:,.0f}/gram
- You can afford: {affordable['affordable_grams_per_month']:.2f} grams/month

📅 **Recommended Purchase Plan:**
- Target: {plan['target_grams']:.0f} grams
- Total Cost: ₹{plan['total_cost']:,.0f}
- Duration: {plan['months_needed']:.1f} months
- Monthly Investment: ₹{plan['suggested_monthly_investment']:,.0f}

💡 **Advice:**
Gold is a stable long-term investment. Consider buying in small quantities monthly (SIP in gold) to average out price fluctuations.

Would you like to know more about gold investment strategies?"""

        else:
            # Ask for more information
            response = """To help you with gold investment planning, I need some information:

Please provide:
1. Your monthly salary/income
2. Your monthly expenses

Example: "My salary is 50000 and expenses are 30000"

Or ask me general questions about gold investment!"""

        print(f"[Gold Advisor] Response generated")
        
        return {
            "messages": [AIMessage(content=response)]
        }
### End of ./src/lang_graph_chatbot\nodes\gold_advisor_node.py ###

### Start of ./src/lang_graph_chatbot\nodes\__init__.py ###

### End of ./src/lang_graph_chatbot\nodes\__init__.py ###

### Start of ./src/lang_graph_chatbot\state\state.py ###
from typing_extensions import TypedDict, List
from langgraph.graph.message import add_messages
from typing import Annotated, Optional

class State(TypedDict):
    """A state of the graph, which is a dictionary of nodes and their corresponding values."""
    messages: Annotated[List, add_messages]
    next_node: Optional[str]  
    user_intent: Optional[str]  
### End of ./src/lang_graph_chatbot\state\state.py ###

### Start of ./src/lang_graph_chatbot\state\__init__.py ###

### End of ./src/lang_graph_chatbot\state\__init__.py ###

### Start of ./src/lang_graph_chatbot\tools\budget_calculator.py ###
"""Budget calculation and analysis tools"""

def calculate_savings(income: float, expenses: float) -> dict:
    """Calculate monthly savings"""
    savings = income - expenses
    savings_percentage = (savings / income * 100) if income > 0 else 0
    
    return {
        "income": income,
        "expenses": expenses,
        "savings": savings,
        "savings_percentage": round(savings_percentage, 2)
    }

def analyze_budget(income: float, expenses_dict: dict) -> dict:
    """Analyze detailed budget breakdown"""
    total_expenses = sum(expenses_dict.values())
    savings = income - total_expenses
    savings_percentage = (savings / income * 100) if income > 0 else 0
    
    # Calculate emergency fund (3-6 months of expenses)
    emergency_fund = total_expenses * 3
    
    # Recommendations
    if savings_percentage >= 30:
        recommendation = "Excellent! You're saving well."
    elif savings_percentage >= 20:
        recommendation = "Good savings rate. Consider investing more."
    elif savings_percentage >= 10:
        recommendation = "Fair. Try to increase savings by reducing expenses."
    else:
        recommendation = "Low savings. Review and cut unnecessary expenses."
    
    return {
        "income": income,
        "total_expenses": total_expenses,
        "expense_breakdown": expenses_dict,
        "savings": savings,
        "savings_percentage": round(savings_percentage, 2),
        "emergency_fund_target": emergency_fund,
        "recommendation": recommendation
    }

def suggest_budget_plan(income: float, savings_goal: float) -> dict:
    """Suggest budget allocation using 50-30-20 rule"""
    needs = income * 0.50  # 50% for needs
    wants = income * 0.30  # 30% for wants
    savings = income * 0.20  # 20% for savings
    
    return {
        "income": income,
        "suggested_needs": needs,
        "suggested_wants": wants,
        "suggested_savings": savings,
        "custom_savings_goal": savings_goal,
        "rule": "50-30-20 Rule (Needs-Wants-Savings)"
    }
### End of ./src/lang_graph_chatbot\tools\budget_calculator.py ###

### Start of ./src/lang_graph_chatbot\tools\document_parser.py ###
"""Document parsing tools (basic version)"""

def parse_text_input(text: str) -> dict:
    """Parse financial data from text input"""
    # Simple parser - can be enhanced with NLP
    data = {
        "income": None,
        "expenses": None,
        "found": False
    }
    
    # Simple keyword extraction
    text_lower = text.lower()
    
    # Try to find income
    if "salary" in text_lower or "income" in text_lower:
        # Extract numbers (simple approach)
        import re
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', text)
        if numbers:
            data["income"] = float(numbers[0].replace(',', ''))
            data["found"] = True
    
    # Try to find expenses
    if "expense" in text_lower or "spend" in text_lower:
        import re
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', text)
        if len(numbers) > 1:
            data["expenses"] = float(numbers[1].replace(',', ''))
        elif len(numbers) == 1 and data["income"] is None:
            data["expenses"] = float(numbers[0].replace(',', ''))
        data["found"] = True
    
    return data

# For PDF parsing, you'd need: pip install PyPDF2
# def extract_from_pdf(pdf_file):
#     # Implementation for PDF extraction
#     pass
### End of ./src/lang_graph_chatbot\tools\document_parser.py ###

### Start of ./src/lang_graph_chatbot\tools\financial_analyzer.py ###
"""General financial analysis tools"""

def analyze_income_expenditure(income: float, expenses: float) -> dict:
    """Comprehensive income vs expenditure analysis"""
    savings = income - expenses
    savings_rate = (savings / income * 100) if income > 0 else 0
    
    # Financial health score (0-100)
    if savings_rate >= 30:
        health_score = 90
        health_status = "Excellent"
    elif savings_rate >= 20:
        health_score = 75
        health_status = "Good"
    elif savings_rate >= 10:
        health_score = 50
        health_status = "Average"
    else:
        health_score = 25
        health_status = "Poor"
    
    return {
        "income": income,
        "expenses": expenses,
        "savings": savings,
        "savings_rate": round(savings_rate, 2),
        "financial_health_score": health_score,
        "financial_health_status": health_status,
        "recommendations": [
            "Build emergency fund (3-6 months expenses)",
            "Invest in diversified portfolio",
            "Consider insurance coverage"
        ]
    }

def investment_recommendation(age: int, risk_tolerance: str, amount: float) -> dict:
    """Recommend investment strategy based on profile"""
    risk_tolerance = risk_tolerance.lower()
    
    if risk_tolerance == "high" and age < 35:
        allocation = {
            "equity": 70,
            "debt": 20,
            "gold": 10
        }
        expected_return = "12-15%"
    elif risk_tolerance == "medium" or (age >= 35 and age < 50):
        allocation = {
            "equity": 50,
            "debt": 35,
            "gold": 15
        }
        expected_return = "10-12%"
    else:  # low risk or age >= 50
        allocation = {
            "equity": 30,
            "debt": 50,
            "gold": 20
        }
        expected_return = "8-10%"
    
    return {
        "age": age,
        "risk_tolerance": risk_tolerance,
        "investment_amount": amount,
        "recommended_allocation": allocation,
        "expected_annual_return": expected_return,
        "advice": f"Diversified portfolio suitable for {risk_tolerance} risk profile"
    }

def emergency_fund_calculator(monthly_expenses: float, months: int = 6) -> dict:
    """Calculate emergency fund requirement"""
    emergency_fund = monthly_expenses * months
    
    return {
        "monthly_expenses": monthly_expenses,
        "recommended_months": months,
        "emergency_fund_target": round(emergency_fund, 2),
        "advice": f"Keep ₹{emergency_fund:,.0f} in liquid savings for emergencies"
    }
### End of ./src/lang_graph_chatbot\tools\financial_analyzer.py ###

### Start of ./src/lang_graph_chatbot\tools\gold_advisor.py ###
"""Gold investment advisory tools"""

def get_current_gold_price() -> float:
    """Get current gold price (simulated - you can integrate real API)"""
    # In production, use real API like: https://www.goldapi.io/
    return 6500.0  # ₹ per gram (example price)

def calculate_affordable_gold(salary: float, expenses: float, gold_price_per_gram: float = None) -> dict:
    """Calculate how much gold user can afford to buy"""
    if gold_price_per_gram is None:
        gold_price_per_gram = get_current_gold_price()
    
    monthly_savings = salary - expenses
    affordable_grams = monthly_savings / gold_price_per_gram if gold_price_per_gram > 0 else 0
    
    return {
        "monthly_salary": salary,
        "monthly_expenses": expenses,
        "monthly_savings": monthly_savings,
        "gold_price_per_gram": gold_price_per_gram,
        "affordable_grams_per_month": round(affordable_grams, 2),
        "affordable_amount": round(monthly_savings, 2)
    }

def create_gold_purchase_plan(monthly_savings: float, target_grams: float, gold_price_per_gram: float = None) -> dict:
    """Create a purchase plan for gold"""
    if gold_price_per_gram is None:
        gold_price_per_gram = get_current_gold_price()
    
    total_cost = target_grams * gold_price_per_gram
    months_needed = (total_cost / monthly_savings) if monthly_savings > 0 else 0
    monthly_investment = total_cost / months_needed if months_needed > 0 else 0
    
    return {
        "target_grams": target_grams,
        "gold_price_per_gram": gold_price_per_gram,
        "total_cost": round(total_cost, 2),
        "monthly_savings_available": monthly_savings,
        "months_needed": round(months_needed, 1),
        "suggested_monthly_investment": round(monthly_investment, 2)
    }

def gold_investment_advice(amount: float, duration_months: int) -> dict:
    """Provide gold investment advice"""
    gold_price = get_current_gold_price()
    total_grams = amount / gold_price
    
    # Simulated returns (gold typically 8-10% annually)
    annual_return_rate = 0.09  # 9%
    future_value = amount * (1 + (annual_return_rate * duration_months / 12))
    
    return {
        "investment_amount": amount,
        "duration_months": duration_months,
        "current_gold_price": gold_price,
        "grams_purchased": round(total_grams, 2),
        "expected_return_rate": f"{annual_return_rate * 100}%",
        "estimated_future_value": round(future_value, 2),
        "advice": "Gold is a stable investment for long-term wealth preservation."
    }
### End of ./src/lang_graph_chatbot\tools\gold_advisor.py ###

### Start of ./src/lang_graph_chatbot\tools\__init__.py ###

### End of ./src/lang_graph_chatbot\tools\__init__.py ###

### Start of ./src/lang_graph_chatbot\ui\uiconfigfile.py ###
from configparser import ConfigParser

class Config:
    def __init__(self, config_file="./src/lang_graph_chatbot/ui/uiconfig.ini"):
        self.config = ConfigParser()
        self.config.read(config_file)

    def get_llm_models(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(", ")

    def get_Usecase_Options(self):
        return self.config["DEFAULT"].get("USECASE_OPTIONS").split(", ")

    def get_Groq_MODEL_OPTIONS(self):
        """Kept for backward compatibility, but not used"""
        return self.config["DEFAULT"].get("Groq_MODEL_OPTIONS", "").split(", ")
    
    def get_HuggingFace_MODEL_OPTIONS(self):
        """Get HuggingFace model options"""
        return self.config["DEFAULT"].get("HuggingFace_MODEL_OPTIONS").split(", ")
    
    def get_Gemini_MODEL_OPTIONS(self):
        """Get Gemini model options"""
        return self.config["DEFAULT"].get("Gemini_MODEL_OPTIONS").split(", ")

    def get_PAGE_TITLE(self):
        return self.config["DEFAULT"].get("PAGE_TITLE")
### End of ./src/lang_graph_chatbot\ui\uiconfigfile.py ###

### Start of ./src/lang_graph_chatbot\ui\__init__.py ###

### End of ./src/lang_graph_chatbot\ui\__init__.py ###

### Start of ./src/lang_graph_chatbot\ui\streamlitui\display_result.py ###
import streamlit as st
from langchain_core.messages import HumanMessage

class DisplayResult:
    def __init__(self, usecase, user_message, compiled_graph):
        self.usecase = usecase
        self.user_message = user_message
        self.compiled_graph = compiled_graph

    def display_result_ui(self):
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Append user message to session
        st.session_state["messages"].append({"role": "user", "content": self.user_message})

        # Show loading spinner while processing
        with st.spinner("🤔 Analyzing your query..."):
            try:
                # Prepare state for graph
                state = {"messages": [HumanMessage(content=self.user_message)]}
                
                # Invoke the compiled graph
                result = self.compiled_graph.invoke(state)
                
                print(f"[DisplayResult] Graph result: {result}")

                # Extract response from result
                if "messages" in result and len(result["messages"]) > 0:
                    last_message = result["messages"][-1]
                    response = last_message.content if hasattr(last_message, "content") else str(last_message)
                else:
                    response = "I apologize, but I couldn't generate a response. Please try again."

            except Exception as e:
                error_msg = str(e)
                print(f"[DisplayResult] Error: {error_msg}")
                import traceback
                print(traceback.format_exc())
                
                # User-friendly error message
                if "rate limit" in error_msg.lower():
                    response = "⚠️ Rate limit reached. Please wait a moment and try again."
                elif "api key" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                    response = "🔑 API Key Error. Please check your Groq API key in the sidebar."
                elif "connection" in error_msg.lower():
                    response = "🌐 Connection error. Please check your internet connection."
                else:
                    response = f"❌ Error: {error_msg}\n\nPlease try again or rephrase your question."
                
                # Show detailed error in expander for debugging
                with st.expander("🔍 Technical Details"):
                    st.error(traceback.format_exc())

        # Append assistant response to session
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # Display all messages in chat format
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])  # Use markdown for better formatting
### End of ./src/lang_graph_chatbot\ui\streamlitui\display_result.py ###

### Start of ./src/lang_graph_chatbot\ui\streamlitui\loadui.py ###
import streamlit as st
from src.lang_graph_chatbot.ui.uiconfigfile import Config

class LoadStreamlitUi:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_Streamlit_Ui(self):
        page_title = self.config.get_PAGE_TITLE()
        if not page_title:
            page_title = "Financial Advisor ChatBot"
        
        st.set_page_config(page_title="💰 " + page_title, layout='wide')
        
        # Custom header
        st.markdown("""
            <h1 style='text-align: center; color: #1f77b4;'>
                💰 AI Financial Advisor
            </h1>
            <p style='text-align: center; color: #666;'>
                Powered by LangGraph • Gemini 🔷 • HuggingFace 🤗
            </p>
        """, unsafe_allow_html=True)

        with st.sidebar:
            st.markdown("## ⚙️ Settings")
            
            # LLM Provider Selection (NEW!)
            st.markdown("### 🤖 AI Provider")
            provider_choice = st.radio(
                "Choose your AI provider:",
                ["🔷 Google Gemini (Fast)", "🤗 HuggingFace (Unlimited)"],
                index=0,
                help="Gemini: Faster responses, 1500/day limit\nHuggingFace: Unlimited, slower"
            )
            
            # Set provider based on choice
            if "Gemini" in provider_choice:
                self.user_controls["Selected_llm"] = "gemini"
                
                # Gemini model selection
                gemini_models = self.config.get_Gemini_MODEL_OPTIONS()
                self.user_controls["Selected_model"] = st.selectbox(
                    "📦 Select Model",
                    gemini_models,
                    index=0,
                    help="gemini-1.5-flash: Fastest, FREE\ngemini-1.5-pro: Best quality, FREE"
                )
                
                # Gemini API Key input
                self.user_controls["gemini_api_key"] = st.text_input(
                    "🔑 Gemini API Key",
                    type="password",
                    help="Get FREE key at https://aistudio.google.com/app/apikey"
                )
                
                # Token validation
                if self.user_controls["gemini_api_key"]:
                    if len(self.user_controls["gemini_api_key"]) > 30:
                        st.success("✅ API key format looks good!")
                    else:
                        st.warning("⚠️ API key seems too short")
                
                # Provider info
                st.info("🔷 **Gemini Free Tier**\n\n• 60 requests/min\n• 1500 requests/day\n• Fast responses ⚡")
                
            else:  # HuggingFace
                self.user_controls["Selected_llm"] = "huggingface"
                
                # HuggingFace model selection (only working models)
                hf_models = self.config.get_HuggingFace_MODEL_OPTIONS()
                self.user_controls["Selected_model"] = st.selectbox(
                    "📦 Select Model",
                    hf_models,
                    index=0,
                    help="Mistral & Zephyr are most reliable"
                )
                
                # HuggingFace API Token input
                self.user_controls["huggingface_api_token"] = st.text_input(
                    "🔑 HuggingFace API Token",
                    type="password",
                    help="Get FREE token at https://huggingface.co/settings/tokens"
                )
                
                # Token validation
                if self.user_controls["huggingface_api_token"]:
                    if self.user_controls["huggingface_api_token"].startswith("hf_"):
                        st.success("✅ Token format looks good!")
                    else:
                        st.warning("⚠️ Token should start with 'hf_'")
                
                # Provider info
                st.success("🤗 **HuggingFace Free Tier**\n\n• Unlimited requests\n• No credit card\n• 100% FREE forever")
            
            # Quick setup guides
            with st.expander("🚀 Quick Setup Guide"):
                if self.user_controls["Selected_llm"] == "gemini":
                    st.markdown("""
                    **Get FREE Gemini API Key (30 sec):**
                    
                    1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
                    2. Click **"Get API Key"**
                    3. Click **"Create API key"**
                    4. Copy and paste above ☝️
                    
                    **No credit card required!** ✅
                    """)
                else:
                    st.markdown("""
                    **Get FREE HuggingFace Token (30 sec):**
                    
                    1. Go to [HuggingFace](https://huggingface.co/join)
                    2. Sign up (FREE)
                    3. Go to [Settings → Tokens](https://huggingface.co/settings/tokens)
                    4. Click **"New token"**
                    5. Select **"Read"** access
                    6. Copy and paste above ☝️
                    """)
            
            st.markdown("---")
            
            # Use case selection
            user_option = self.config.get_Usecase_Options()
            self.user_controls["selected_user_option"] = st.selectbox(
                "📋 Select Use Case",
                user_option,
                help="Choose the type of financial assistance you need"
            )
            
            st.markdown("---")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                if "messages" in st.session_state:
                    st.session_state["messages"] = []
                st.rerun()
            
            st.markdown("---")
            
            # Features info
            with st.expander("ℹ️ What Can I Do?"):
                st.markdown("""
                **Financial Services:**
                - 🏆 Gold purchase planning
                - 💰 Budget analysis & optimization
                - 📊 Investment recommendations
                - 💵 Savings strategies
                - 📈 Financial goal planning
                
                **Example Questions:**
                ```
                "I want to buy gold, salary 50k, expenses 30k"
                
                "Analyze my budget: income 80k, expenses 60k"
                
                "Should I invest in gold or mutual funds?"
                
                "Help me save 20k per month"
                ```
                """)
            
            # Provider comparison
            with st.expander("⚖️ Provider Comparison"):
                st.markdown("""
                | Feature | Gemini | HuggingFace |
                |---------|--------|-------------|
                | Speed | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ |
                | Limit | 1500/day | Unlimited |
                | Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
                | Setup | 30 seconds | 30 seconds |
                | Cost | FREE | FREE |
                
                **Recommendation:** Use Gemini for best experience!
                """)
            
            st.markdown("---")
            
            # Branding
            st.markdown("""
                <div style='text-align: center; color: #666; font-size: 12px;'>
                    <p>Built with ❤️ using</p>
                    <p>LangGraph • Gemini 🔷 • HuggingFace 🤗</p>
                    <p style='margin-top: 10px; font-size: 10px;'>
                        100% Free & Open Source
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        return self.user_controls
### End of ./src/lang_graph_chatbot\ui\streamlitui\loadui.py ###

