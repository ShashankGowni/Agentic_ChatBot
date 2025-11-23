import streamlit as st
from langchain_core.messages import HumanMessage

class DisplayResult:
    def __init__(self, usecase, user_message, compiled_graph):
        self.usecase = usecase
        self.user_message = user_message
        self.compiled_graph = compiled_graph

    def display_result_ui(self):
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Append user message to session
        st.session_state["messages"].append({"role": "user", "content": self.user_message})

        # Show loading spinner while processing
        with st.spinner("🤔 Analyzing your query..."):
            try:
                # Prepare state for graph
                state = {"messages": [HumanMessage(content=self.user_message)]}
                
                # Invoke the compiled graph
                result = self.compiled_graph.invoke(state)
                
                print(f"[DisplayResult] Graph result: {result}")

                # Extract response from result
                if "messages" in result and len(result["messages"]) > 0:
                    last_message = result["messages"][-1]
                    response = last_message.content if hasattr(last_message, "content") else str(last_message)
                else:
                    response = "I apologize, but I couldn't generate a response. Please try again."

            except Exception as e:
                error_msg = str(e)
                print(f"[DisplayResult] Error: {error_msg}")
                import traceback
                print(traceback.format_exc())
                
                # User-friendly error message
                if "rate limit" in error_msg.lower():
                    response = "⚠️ Rate limit reached. Please wait a moment and try again."
                elif "api key" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                    response = "🔑 API Key Error. Please check your Groq API key in the sidebar."
                elif "connection" in error_msg.lower():
                    response = "🌐 Connection error. Please check your internet connection."
                else:
                    response = f"❌ Error: {error_msg}\n\nPlease try again or rephrase your question."
                
                # Show detailed error in expander for debugging
                with st.expander("🔍 Technical Details"):
                    st.error(traceback.format_exc())

        # Append assistant response to session
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # Display all messages in chat format
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])  # Use markdown for better formatting