# check_logs.py
import os
import pprint
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING

def view_recent_logs(num_logs: int = 5):
    """
    Connects to MongoDB and prints the most recent log entries.
    """
    # 1. Load environment variables from your .env file
    load_dotenv()
    mongo_uri = os.environ.get("MONGO_URI")
    db_name = "FredRag"
    collection_name = "logs"

    if not mongo_uri:
        print("❌ MONGO_URI not found in .env file. Please set it.")
        return

    try:
        # 2. Connect to the MongoDB client
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        print(f"✅ Successfully connected to '{db_name}.{collection_name}'")

        # 3. Find the most recent logs
        # We sort by the 'timestamp_entry' field in descending order to get the latest ones first.
        recent_logs = collection.find({}).sort("timestamp_entry", DESCENDING).limit(num_logs)
        
        # 4. Print the logs in a readable format
        print(f"--- Displaying the last {num_logs} log entries ---")
        log_count = 0
        for log in recent_logs:
            log_count += 1
            pprint.pprint(log)
            print("-" * 20)

        if log_count == 0:
            print("No logs found in the collection.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    view_recent_logs()