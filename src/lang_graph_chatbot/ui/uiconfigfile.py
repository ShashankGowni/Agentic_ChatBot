from configparser import ConfigParser

class Config:
    def __init__(self, config_file="./src/lang_graph_chatbot/ui/uiconfig.ini"):
        self.config = ConfigParser()
        self.config.read(config_file)

    def get_llm_models(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(", ")

    def get_Usecase_Options(self):
        return self.config["DEFAULT"].get("USECASE_OPTIONS").split(", ")

    def get_Ollama_MODEL_OPTIONS(self):
        return self.config["DEFAULT"].get("Ollama_MODEL_OPTIONS").split(", ")

    def get_PAGE_TITLE(self):
        return self.config["DEFAULT"].get("PAGE_TITLE")
