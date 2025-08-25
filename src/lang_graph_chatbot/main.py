import streamlit as st
from src.lang_graph_chatbot.ui.streamlitui.loadui import LoadStreamlitUi
from src.lang_graph_chatbot.LLMS.ollama import Ollama
from src.lang_graph_chatbot.graph.graph_builder import GraphBuilder
from src.lang_graph_chatbot.ui.streamlitui.display_result import DisplayResult

def langchain_chatbot():
    ui = LoadStreamlitUi()
    user_inputs = ui.load_Streamlit_Ui()

    if not user_inputs:
        st.error("Error: Failed to load user input from the UI.")
        return

    user_message = st.chat_input("Enter message : ")
    if user_message:
        print(f"User Input captured in main: {user_message}")

        try:
            # Create OllamaModelWrapper instance
            obj_llm_config = Ollama(user_input=user_inputs)
            model = obj_llm_config.get_llm_model()
            if not model:
                st.error("Error: LLM model could not be configured.")
                return

            # Create the DisplayResult instance and pass the Ollama model wrapper
            usecase = user_inputs.get("selected_user_option")
            if not usecase:
                st.error("Error: No use case is selected.")
                return

            # Create DisplayResult instance and pass the model
            display = DisplayResult(usecase, user_message, model)
            display.display_result_ui()

        except Exception as e:
            st.error(f"Error: Failed to setup LLM or Graph - {str(e)}")
            return
