import streamlit as st

class DisplayResult:
    def __init__(self, usecase, user_message, ollama_model_wrapper):
        self.usecase = usecase
        self.user_message = user_message
        self.ollama_model_wrapper = ollama_model_wrapper  

    def display_result_ui(self):
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Append user message to session state
        st.session_state["messages"].append({"role": "user", "content": self.user_message})

        # Prepare state dict for model invoke
        state = {"messages": st.session_state["messages"]}

        try:
            # Get the model's response by invoking the model
            response = self.ollama_model_wrapper.invoke(self.user_message)

            # Debugging: Check raw result
            # st.write(f"Raw Model Response: {response}")  # Display raw response in UI

            # Ensure we are getting the assistant's response
            # st.write(f"Assistant Response: {response}")

        except Exception as e:
            response = f"Error: {e}"

        # Append assistant response to session state
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # Render all messages in Streamlit chat
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"], unsafe_allow_html=True)
