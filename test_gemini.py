import google.generativeai as genai
import json

# ----------------------------------------
# Gemini API Setup
# ----------------------------------------
genai.configure(api_key="AIzaSyBfvJVOK9idpF-c0q1TS1-jlUn4uyyhR8w")

# Use the latest valid model name
MODEL_NAME = "gemini-1.5-pro"

try:
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"✅ Gemini model '{MODEL_NAME}' initialized successfully!")

    # Simple test prompt
    prompt = """
    You are a classification AI.
    Classify the following image or text type of issue:
    Example: "There is a traffic accident on the road"

    Return output as valid JSON only:
    {
      "label": "Public Safety",
      "reason": "Traffic accident is a safety concern",
      "Priority": "2"
    }
    """

    # Test model text generation
    response = model.generate_content(prompt)
    print("\n🔹 Gemini Response:\n", response.text)

    # Try to parse the output
    try:
        json_data = json.loads(response.text)
        print("\n✅ JSON parsed successfully:\n", json.dumps(json_data, indent=2))
    except json.JSONDecodeError:
        print("\n⚠️ Response is not valid JSON but model is working fine.")

except Exception as e:
    print(f"❌ Gemini API test failed: {e}")
