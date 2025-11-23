from huggingface_hub import InferenceClient

api_token = "your_api_token"
model = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"Testing new token...")
print(f"Token: {api_token[:10]}...")
print(f"Model: {model}")

try:
    client = InferenceClient(token=api_token)
    
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    
    response = client.chat_completion(
        messages=messages,
        model=model,
        max_tokens=100
    )
    
    reply = response.choices[0].message.content
    print(f"✅ SUCCESS! Response: {reply}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()