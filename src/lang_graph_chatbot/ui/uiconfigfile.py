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