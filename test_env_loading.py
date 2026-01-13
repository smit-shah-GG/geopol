import os
from dotenv import load_dotenv

print("5. Testing .env loading:")
print(f"   Before load_dotenv: GEMINI_API_KEY = {os.getenv('GEMINI_API_KEY')}")

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
print(f"   After load_dotenv: GEMINI_API_KEY = {'[SET]' if api_key else 'None'}")
if api_key:
    print(f"   API key length: {len(api_key)} chars")
    print(f"   First 10 chars: {api_key[:10]}...")
