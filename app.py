import sys
import os

# Add 'src' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Now import langchain_chatbot after updating the sys.path
from src.lang_graph_chatbot.main import langchain_chatbot

if __name__ == "__main__":
    # Create a chatbot instance
    langchain_chatbot()
