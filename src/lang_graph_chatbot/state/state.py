from typing_extensions import TypedDict, List
from langgraph.graph.message import add_messages
from typing import Annotated, Optional

class State(TypedDict):
    """A state of the graph, which is a dictionary of nodes and their corresponding values."""
    messages: Annotated[List, add_messages]
    next_node: Optional[str]  
    user_intent: Optional[str]  