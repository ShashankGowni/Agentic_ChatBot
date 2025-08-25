from langgraph.graph import StateGraph
from lang_graph_chatbot.state.state import State
from langgraph.graph import START, END
from lang_graph_chatbot.nodes.basic_chatbot_node import BasicChatBot

class GraphBuilder:
    def __init__(self, model):
        self.llm = model
        self.graph = StateGraph(State)
        self.basic_chat_bot_node = None

    def basic_chat_bot(self):
        self.basic_chat_bot_node = BasicChatBot(self.llm)
        self.graph.add_node("Chatbot", self.basic_chat_bot_node.process)
        self.graph.add_edge(START, "Chatbot")
        self.graph.add_edge("Chatbot", END)
        compiled_graph = self.graph.compile()
        return compiled_graph

    def setup_graph(self, usecase: str):
        if usecase == "Basic Chatbot":
            return self.basic_chat_bot()
        else:
            raise ValueError(f"Unknown use case: {usecase}")
