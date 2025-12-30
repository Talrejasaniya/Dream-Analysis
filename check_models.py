import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("--- Available Models List ---")
try:
    # Hum seedha list print karenge bina kisi condition ke
    for model in client.models.list():
        # Sirf model ka naam print karenge
        print(model.name)
except Exception as e:
    print(f"Error fetching models: {e}")