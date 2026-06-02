import sys
import os

try:
    from google import genai
    print("[OK] google-genai SDK is installed.")
except ImportError:
    print("[ERROR] google-genai is NOT installed.")
    sys.exit(1)

if len(sys.argv) < 2:
    print("\nUsage: python test_generation.py <YOUR_GEMINI_API_KEY>")
    sys.exit(1)

api_key = sys.argv[1].strip()
client = genai.Client(api_key=api_key)

models_to_test = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash"
]

print("Testing generation capabilities for available models...")
for model_name in models_to_test:
    print(f"\nTesting model: {model_name}...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'OK' in one word."
        )
        print(f"  [OK] Success! Response: {response.text.strip()}")
    except Exception as e:
        print(f"  [ERROR] Failed: {str(e)[:200]}...")
