import sys
import os

try:
    from google import genai
    print("[OK] google-genai SDK is installed.")
except ImportError:
    print("[ERROR] google-genai is NOT installed.")
    sys.exit(1)

if len(sys.argv) < 2:
    print("\nUsage: python diagnose_gemini.py <YOUR_GEMINI_API_KEY>")
    sys.exit(1)

api_key = sys.argv[1].strip()

print("\nChecking API connection and listing available models using google-genai SDK...")
try:
    client = genai.Client(api_key=api_key)
    models = list(client.models.list())
    
    print("\n[OK] Connection Successful! Models available to this API key:")
    found_embedding = False
    for m in models:
        actions = getattr(m, 'supported_actions', [])
        action_names = []
        for a in actions:
            if isinstance(a, str):
                action_names.append(a)
            else:
                action_names.append(str(a))
                
        print(f" - {m.name} (Actions: {', '.join(action_names)})")
        if "embedContent" in action_names or "embedText" in action_names or "embedding" in m.name.lower():
            found_embedding = True
            
    if not found_embedding:
        print("\n[WARN] Warning: No embedding models were returned for this API key. Make sure the Generative Language API is enabled.")
except Exception as e:
    print(f"\n[ERROR] Connection failed.")
    print(f"Details: {str(e)}")
    print("\nPossible Causes:")
    print("1. The API key is invalid or has a typo.")
    print("2. The key belongs to a GCP project where the 'Generative Language API' has not been enabled in the API library.")
    print("3. There is a network or firewall block preventing connection to Google's API servers.")
