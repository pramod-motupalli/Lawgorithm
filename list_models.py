import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv("lawgorithm/.env")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Try to read from config.py if not in env
    try:
        from lawgorithm import config
        api_key = config.GOOGLE_API_KEY
    except ImportError:
        pass

if not api_key:
    print("Error: GOOGLE_API_KEY not found.")
    exit(1)

print(f"API Key found: {api_key[:5]}...{api_key[-5:]}")

genai.configure(api_key=api_key)

print("Listing available models...")
try:
    models = list(genai.list_models())
    print(f"Found {len(models)} models.")
    for m in models:
        print(f"Model: {m.name}")
        print(f"Methods: {m.supported_generation_methods}")
except Exception as e:
    print(f"Error listing models: {e}")
    import traceback
    traceback.print_exc()
