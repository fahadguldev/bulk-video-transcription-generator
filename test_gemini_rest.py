import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY_1")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY_1 not found in .env")

url = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/gemini-embedding-2:embedContent"
)

payload = {
    "model": "models/gemini-embedding-2",
    "content": {
        "parts": [
            {
                "text": "This is a Second Brain embedding test."
            }
        ]
    }
}

response = requests.post(
    url,
    headers={
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    },
    json=payload,
)

print("Status:", response.status_code)
print(response.text)
