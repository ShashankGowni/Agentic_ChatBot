import ollama
import streamlit as st

class OllamaModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Model Name: {self.model_name}")

    def invoke(self, input_text: str) -> str:
        print(f"User Input to Ollama: {input_text}")
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": input_text}]
            )
            print(f"Full raw response from Ollama: {response}")

            # Extract content from response object attributes
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                content = response.message.content
                print(f"Extracted assistant response content: {content}")
                return content or "No content returned."
            elif isinstance(response, dict):
                return response.get("message", "No message key in response")
            elif isinstance(response, str):
                return response
            else:
                return str(response)

        except Exception as e:
            print(f"Error during Ollama invoke: {e}")
            st.error(f"Error: {e}")
            return "Sorry, I failed to respond."

class Ollama:
    def __init__(self, user_input):
        self.user_input = user_input

    def get_llm_model(self):
        selected_ollama_model = self.user_input.get("Selected_model", None)
        if selected_ollama_model:
            st.write(f"✅ Selected Ollama Model: {selected_ollama_model}")
            return OllamaModelWrapper(selected_ollama_model)
        else:
            st.error("Error: No model selected.")
            return None
