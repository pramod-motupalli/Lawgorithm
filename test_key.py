import google.generativeai as genai

# --- PASTE YOUR KEY HERE ---
API_KEY = "AIzaSyDfCfU0OIXg5V-cHxKvv4zF4Wxy5AnYJLE"
# ---------------------------

if API_KEY == "PASTE_YOUR_KEY_HERE":
    print("Please replace 'PASTE_YOUR_KEY_HERE' with your actual Google API Key in the script.")
    exit(1)

print(f"Testing Key: {API_KEY[:5]}...{API_KEY[-5:]}")

try:
    genai.configure(api_key=API_KEY)
    print("Successfully configured. Listing models...")
    
    models = list(genai.list_models())
    print(f"Success! Found {len(models)} models.")
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f" - {m.name}")
            
except Exception as e:
    print("\n[ERROR] The API Key is invalid or there is a connection issue.")
    print(f"Details: {e}")
