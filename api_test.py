import json
from dotenv import load_dotenv
from openrouter_client import chat_completion, get_openrouter_client

# --- CONFIGURATION ---
# Load environment variables from your .env file
load_dotenv()

get_openrouter_client()
MODEL = "llama3.1-8b-instruct"

def test_lambda_api():
    """
    Sends a simple test request to verify OpenRouter connectivity.
    """
    print("▶️  Sending test request to OpenRouter chat completions…")

    try:
        response = chat_completion(
            model=MODEL,
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            max_tokens=50,
        )
        message = response.choices[0].message.content
        print("\n🎉 SUCCESS! The API endpoint and key are working correctly.")
        print("\n--- Model Response ---")
        print(message.strip())
    except Exception as e:
        print("\n❌ FAILED. The OpenRouter call raised an error.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    test_lambda_api()
