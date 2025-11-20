import requests
import json

API_KEY = "MtdNuztiHz74Qc34xc0VhxAVgRxz0F7t8cPQybeU"

# ✅ Use v1 (not v1beta)
URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={API_KEY}"

data = {
    "contents": [
        {"parts": [{"text": "Hello Gemini Flash! Please confirm you're working."}]}
    ]
}

response = requests.post(URL, json=data)

print("Status:", response.status_code)
print(json.dumps(response.json(), indent=2))
