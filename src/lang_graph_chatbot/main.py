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