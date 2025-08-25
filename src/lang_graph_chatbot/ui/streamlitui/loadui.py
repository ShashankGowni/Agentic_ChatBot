import streamlit as st
from src.lang_graph_chatbot.ui.uiconfigfile import Config

class LoadStreamlitUi:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_Streamlit_Ui(self):
        page_title = self.config.get_PAGE_TITLE()
        if not page_title:
            page_title = "Default Page Title"
        
        st.set_page_config(page_title=" 🧑‍🚀 " + page_title, layout='wide')
        st.header(" 🧑‍🚀 " + page_title)

        with st.sidebar:
            llm_options = self.config.get_llm_models()
            user_option = self.config.get_Usecase_Options()
            
            # Select LLM and model
            self.user_controls["Selected_llm"] = st.selectbox("Select LLM", llm_options)
            if self.user_controls["Selected_llm"] == "ollama":
                model_options = self.config.get_Ollama_MODEL_OPTIONS()
                self.user_controls["Selected_model"] = st.selectbox("Select Model", model_options)
                st.info("Ollama runs locally. Ensure Ollama is running and the selected model is downloaded.")
            
            # Select Use Case
            self.user_controls["selected_user_option"] = st.selectbox("Select Usecases", user_option)
        
        return self.user_controls
