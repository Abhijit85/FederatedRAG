import unittest
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from build_compendium import main as run_database_setup
from vector_search import VectorSearchFilter
from math_qa import MongoRAGManager as MathRAGManager
from science_qa import MongoRAGManager as ScienceRAGManager
from populate_vector_store import JinaAIClient
from jina_key_manager import get_available_jina_api_keys, get_named_jina_api_keys

# --- TEST CONFIGURATION ---
load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"
JINA_API_KEYS = get_available_jina_api_keys(allow_empty=True)

class TestMongoDatabaseSetup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        This method runs once before all tests.
        It runs the main database setup process to populate the collections.
        """
        print("--- (SETUP) Running the full database setup process... ---")
        if not all([MONGO_URI, DB_NAME]) or not JINA_API_KEYS:
            raise unittest.SkipTest("MONGO_URI, DB_NAME, or a JINA_API_KEY not set. Skipping tests.")
        
        run_database_setup()
        
        cls.client = MongoClient(MONGO_URI)
        cls.db = cls.client[DB_NAME]
        print("--- (SETUP) Setup complete. ---")

    @classmethod
    def tearDownClass(cls):
        """
        This method runs once after all tests.
        It closes the database connection.
        """
        if hasattr(cls, 'client'):
            cls.client.close()
            print("\n--- (TEARDOWN) MongoDB connection closed. ---")

    def test_01_collections_created(self):
        """
        Test 1: Verify that all expected collections have been created.
        """
        print("\n--- TEST 1: Checking for collection creation ---")
        # CORRECTED: Changed 'science_images' to 'science_problems'
        expected_collections = ["vectors", "math_problems", "science_problems", "logs"]
        existing_collections = self.db.list_collection_names()
        
        for collection in expected_collections:
            with self.subTest(collection=collection):
                self.assertIn(collection, existing_collections, f"Collection '{collection}' was NOT created.")
                print(f"✅ Collection '{collection}' exists.")

    def test_02_collections_populated(self):
        """
        Test 2: Verify that the main collections are not empty.
        """
        print("\n--- TEST 2: Checking if collections are populated ---")
        # CORRECTED: Changed 'science_images' to 'science_problems'
        collections_to_check = ["vectors", "math_problems", "science_problems"]
        
        for collection_name in collections_to_check:
            with self.subTest(collection=collection_name):
                collection = self.db[collection_name]
                doc_count = collection.count_documents({})
                self.assertGreater(doc_count, 0, f"Collection '{collection_name}' is empty.")
                print(f"✅ Collection '{collection_name}' contains {doc_count} documents.")

    def test_03_tool_vector_search(self):
        """
        Test 3: Perform a live vector search on the 'vectors' collection for tool selection.
        """
        print("\n--- TEST 3: Testing Tool Selection Vector Search ---")
        # UPDATED: Now uses the new MongoDB-based search filter
        search_filter = VectorSearchFilter()
        test_query = "how to calculate the area of a circle"
        
        print(f"Performing search with query: '{test_query}'")
        search_results = search_filter.search(query=test_query, top_k=1)
        
        self.assertIsNotNone(search_results, "Search returned None.")
        self.assertIsInstance(search_results, list, "Search should return a list.")
        # CORRECTED: Now we expect a list of strings, not metadata objects
        self.assertEqual(len(search_results), 1, "Search should return 1 result for top_k=1.")
        
        first_result = search_results[0]
        self.assertIsInstance(first_result, str)
        self.assertIn("Tool Scenario:", first_result, "Search result text is not in the expected format.")
        
        print(f"✅ Vector search successful. Top result: '{first_result[:80]}...'")

    def test_04_math_rag_search(self):
        """
        Test 4: Perform a live RAG search on the 'math_problems' collection.
        """
        print("\n--- TEST 4: Testing Math RAG Vector Search ---")
        jina_client = JinaAIClient(get_named_jina_api_keys())
        math_rag = MathRAGManager(jina_client)
        test_query = "A train travels at 60 mph for 3 hours. How far did it go?"
        
        print(f"Performing RAG search with query: '{test_query}'")
        # Use the main search method which relies on the Atlas Vector Search index
        rag_results = math_rag.query(user_query=test_query, n_results=1)

        self.assertIsNotNone(rag_results, "RAG search returned None.")
        self.assertEqual(len(rag_results), 1, "RAG search should return 1 result.")
        
        first_result = rag_results[0]
        self.assertIn("text", first_result)
        self.assertIn("embedding", first_result)
        self.assertIn("Problem:", first_result['text'])
        
        print("✅ Math RAG search successful.")

if __name__ == '__main__':
    unittest.main()
