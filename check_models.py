import os
import google.generativeai as genai

# Ensure your key is set in the environment or paste it here for testing
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

if not GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("🔍 Scanning available models for your API key...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ Available: {m.name}")
    except Exception as e:
        print(f"❌ Error: {e}")
