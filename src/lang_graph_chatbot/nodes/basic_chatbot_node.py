from src.lang_graph_chatbot.state.state import State

class BasicChatBot:
    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        # Extract messages from state
        messages = state.get("messages", [])
        print(f"[BasicChatBot] State messages: {messages}")

        # Get the last user message (if exists)
        if messages:
            # If it's a HumanMessage object, get its content attribute
            if hasattr(messages[-1], "content"):
                user_message = messages[-1].content
            else:
                user_message = str(messages[-1])
        else:
            user_message = ""

        print(f"[BasicChatBot] Processing user message: {user_message}")

        # Get the response from the language model
        response = self.llm.invoke(user_message)
        print(f"[BasicChatBot] Raw response from LLM: {response}")  # Debugging the model's full raw response

        # Check if the response contains assistant's message
        if isinstance(response, dict) and "messages" in response:
            assistant_message = next((msg['content'] for msg in response['messages'] if msg['role'] == 'assistant'), None)
            if assistant_message:
                response_text = assistant_message
            else:
                response_text = "Sorry, no assistant response found."
        else:
            response_text = "Error: Response format is unexpected."

        # If no response is generated, return a fallback message
        if not response_text:
            response_text = "Sorry, I couldn't generate a response."

        return {"message": response_text}
