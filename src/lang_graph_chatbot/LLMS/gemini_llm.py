import streamlit as st
import google.generativeai as genai
import traceback

class GeminiModelWrapper:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        
        try:
            genai.configure(api_key=api_key)
            # Use the actual model name from selection
            self.model = genai.GenerativeModel(model_name)
            print(f"✅ Gemini Model initialized: {model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            raise

    def invoke(self, input_text: str) -> str:
        print(f"📝 Gemini Input: {input_text[:100]}...")
        
        try:
            response = self.model.generate_content(
                input_text,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.7,
                )
            )
            
            if response and response.text:
                print(f"✅ Gemini Success: {len(response.text)} characters")
                return response.text
            elif response and response.prompt_feedback:
                print(f"⚠️ Content blocked: {response.prompt_feedback}")
                return "⚠️ Content was blocked by safety filters. Please rephrase your query."
            else:
                return "No response generated. Please try again."
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Gemini Error: {error_msg}")
            print(traceback.format_exc())
            
            if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
                st.error("🔑 Invalid API Key. Get a new key at: https://aistudio.google.com/app/apikey")
                return "❌ Invalid API key"
            
            elif "403" in error_msg:
                st.error("⚠️ API not enabled. Enable at: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
                return "❌ API not enabled"
            
            elif "quota" in error_msg.lower() or "429" in error_msg:
                st.warning("⚠️ Rate limit reached. Wait 60 seconds.")
                return "⏳ Rate limit reached. Please wait."
            
            elif "404" in error_msg or "not found" in error_msg.lower():
                st.error(f"❌ Model not found. Available models listed at startup.")
                return f"❌ Model not available: {self.model_name}"
            
            else:
                st.error(f"❌ Error: {error_msg}")
                return f"Error: {error_msg}"


class GeminiLLM:
    def __init__(self, user_input, api_token):
        self.user_input = user_input
        self.api_token = api_token

    def get_llm_model(self):
        if not self.api_token:
            st.error(" Please enter your Gemini API key")
            return None
        
        # Get selected model from UI
        selected_model = self.user_input.get("Selected_model", "gemini-2.5-flash")
        st.write(f"✅ Using Gemini Model: **{selected_model}**")
        
        try:
            return GeminiModelWrapper(selected_model, self.api_token)
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {e}")
            return None