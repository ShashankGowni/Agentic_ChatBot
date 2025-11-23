# import streamlit as st
# from groq import Groq
# import traceback

# class GroqModelWrapper:
#     def __init__(self, model_name, api_token):
#         self.model_name = model_name
#         self.api_token = api_token
#         self.client = Groq(api_key=api_token)
#         print(f"[Groq] Initialized model: {self.model_name}")

#     def invoke(self, input_text: str) -> str:
#         """Call Groq API with user input"""
#         print(f"[Groq] Input: {input_text[:100]}...")
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {"role": "user", "content": input_text}
#                 ],
#                 max_tokens=1024,
#                 temperature=0.7
#             )
            
#             content = response.choices[0].message.content
#             print(f"[Groq] Response: {content[:100]}...")
#             return content

#         except Exception as e:
#             error_msg = f"Groq API Error: {str(e)}"
#             print(error_msg)
#             print(traceback.format_exc())
#             st.error(error_msg)
#             return f"Error: {str(e)}"


# class GroqLLM:
#     def __init__(self, user_input, api_token):
#         self.user_input = user_input
#         self.api_token = api_token

#     def get_llm_model(self):
#         """Get the selected Groq model"""
#         selected_model = self.user_input.get("Selected_model", None)
#         if selected_model:
#             st.write(f"✅ Using Groq Model: **{selected_model}**")
#             return GroqModelWrapper(selected_model, self.api_token)
#         else:
#             st.error(" No model selected.")
#             return None