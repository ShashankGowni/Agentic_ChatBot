import ollama

try:
    response = ollama.chat(
        model="gemma:2b",
        messages=[{"role": "user", "content": "what is the capital city of india"}]
    )
    print(response)
except Exception as e:
    print("Error:", e)
