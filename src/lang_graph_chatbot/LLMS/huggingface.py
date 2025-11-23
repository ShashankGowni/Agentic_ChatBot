import streamlit as st
from huggingface_hub import InferenceClient
import traceback

class HuggingFaceModelWrapper:
    def __init__(self, model_name, api_token):
        self.model_name = model_name
        self.api_token = api_token
        self.client = InferenceClient(token=api_token)
        print(f"HuggingFace Model Name: {self.model_name}")

    def invoke(self, input_text: str) -> str:
        print(f"User Input to HuggingFace: {input_text}")
        
        # Try chat_completion first
        try:
            print(f"Attempting chat_completion with: {self.model_name}")
            
            messages = [{"role": "user", "content": input_text}]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=512,
                temperature=0.7
            )
            
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                print(f"✅ Success: {content}")
                return content
            else:
                return str(response)

        except (StopIteration, ValueError) as e:
            print(f" chat_completion not available for this model, trying text_generation...")
            
        except Exception as e:
            print(f" chat_completion error: {e}")
            if "stopiteration" in str(type(e)).lower() or "provider" in str(e).lower():
                print(f"Trying text_generation as fallback...")
            else:
                error_msg = f"Unexpected error: {str(e)}"
                print(error_msg)
                return f"Error: {str(e)}"
        
        # Fallback to text_generation
        try:
            print(f"Attempting text_generation with: {self.model_name}")
            response = self.client.text_generation(
                input_text,
                model=self.model_name,
                max_new_tokens=512,
                temperature=0.7,
                return_full_text=False
            )
            print(f"✅ text_generation success: {response}")
            return response if response else "No content returned."
            
        except Exception as e2:
            error_msg = f"Model {self.model_name} is not available"
            print(f" Both methods failed: {str(e2)}")
            print(traceback.format_exc())
            
            st.warning(f" {self.model_name} is currently unavailable. Using default model.")
            
            # Final fallback: use Mistral-7B which we know works
            if self.model_name != "mistralai/Mistral-7B-Instruct-v0.2":
                print(f"Falling back to mistralai/Mistral-7B-Instruct-v0.2")
                try:
                    fallback_messages = [{"role": "user", "content": input_text}]
                    fallback_response = self.client.chat_completion(
                        messages=fallback_messages,
                        model="mistralai/Mistral-7B-Instruct-v0.2",
                        max_tokens=512,
                        temperature=0.7
                    )
                    content = fallback_response.choices[0].message.content
                    print(f"✅ Fallback successful: {content}")
                    st.info(" Using Mistral-7B-Instruct-v0.2 (your selected model was unavailable)")
                    return content
                except Exception as e3:
                    print(f" Even fallback failed: {e3}")
                    return "Sorry, I'm having trouble connecting to the models right now. Please try again."
            
            return f" Unable to get response. Please try selecting mistralai/Mistral-7B-Instruct-v0.2"


class HuggingFace:
    def __init__(self, user_input, api_token):
        self.user_input = user_input
        self.api_token = api_token

    def get_llm_model(self):
        selected_hf_model = self.user_input.get("Selected_model", None)
        if selected_hf_model:
            st.write(f"✅ Selected HuggingFace Model: {selected_hf_model}")
            return HuggingFaceModelWrapper(selected_hf_model, self.api_token)
        else:
            # Default to Mistral if no model selected
            st.info(" No model selected, using Mistral-7B-Instruct-v0.2 (default)")
            return HuggingFaceModelWrapper("mistralai/Mistral-7B-Instruct-v0.2", self.api_token)