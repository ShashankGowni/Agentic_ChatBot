import google.generativeai as genai

# Replace with your ACTUAL API key
API_KEY = "Your_Api_Key"  

print("🔍 Testing Gemini API...")
print(f"Key starts with: {API_KEY[:10]}...")

try:
    # Configure
    genai.configure(api_key=API_KEY)
    print("✅ API key configured")
    
    # List available models
    print("\n📋 Available models:")
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  ✅ {model.name}")
    
    # Try to use gemini-pro
    print("\n🧪 Testing gemini-pro...")
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Say hello in one sentence")
    
    print("✅ SUCCESS!")
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    print("\nFull error:")
    print(traceback.format_exc())