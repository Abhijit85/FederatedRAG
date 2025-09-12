import os
import requests
import json
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Load environment variables from your .env file
load_dotenv()

# Get your API key from the environment variables
API_KEY = os.environ.get("LAMDA_API_KEY")

# --- IMPORTANT: This is the corrected API endpoint URL ---
API_URL ="https://api.lambda.ai/v1/chat/completions"

def test_lambda_api():
    """
    Sends a simple test request to the Lambda Labs API to verify the endpoint and key.
    """
    if not API_KEY:
        print("❌ Error: LAMDA_API_KEY not found in your .env file.")
        print("Please make sure your .env file is in the same directory and contains your key.")
        return

    # 1. Set up the authorization headers
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 2. Create a simple payload for the request
    # We use a small, fast model for a quick response.
    payload = {
        "model": "llama3.1-8b-instruct",
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        "max_tokens": 50
    }

    print(f"▶️  Sending test request to: {API_URL}")

    try:
        # 3. Make the POST request
        response = requests.post(API_URL, headers=headers, json=payload)

        # 4. Check the response from the server
        print(f"✅ Request sent. Server responded with status code: {response.status_code}")

        if response.status_code == 200:
            print("\n🎉 SUCCESS! The API endpoint and key are working correctly.")
            print("\n--- Full Response ---")
            # Use json.dumps for pretty printing the JSON response
            print(json.dumps(response.json(), indent=2))
        else:
            print("\n❌ FAILED. The server returned an error.")
            print("This could be due to an incorrect API key, an invalid model name, or a server-side issue.")
            print("\n--- Error Details ---")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"\n❌ FAILED. A network error occurred while trying to connect to the API.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    test_lambda_api()
