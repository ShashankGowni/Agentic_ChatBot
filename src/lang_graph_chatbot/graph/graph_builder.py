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