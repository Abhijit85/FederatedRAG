import os
import json
import re
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from agenttools import BaseTool, ToolUsageExample
from mongo_utils import MongoVectorStore
from typing import Dict, Optional, List
from datasets import load_dataset
from tqdm import tqdm

# --- 1. CONFIGURATION ---
load_dotenv()
API_KEY = os.environ.get("JINA_API_KEY")
if not API_KEY:
    raise ValueError("JINA_API_KEY environment variable not set.")

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"
GSM8K_COLLECTION = os.environ.get("GSM8K_COLLECTION", "gsm8k_problems")

JINA_EMBED_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_API_URL = "https://api.jina.ai/v1/rerank"
JINA_CHAT_API_URL = "https://api.jina.ai/v1/chat/completions"

# Jina AI limits
MAX_BATCH_SIZE = 50  # Conservative batch size for safety
MAX_TEXT_LENGTH = 6000  # Conservative character limit per text

# --- 2. RAG SYSTEM COMPONENTS ---

class JinaAIClient:
    """A client to interact with Jina AI APIs for embeddings and reranking."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # Test the API connection on initialization
        self._test_connection()

    def _test_connection(self):
        """Test Jina API with a simple embedding request."""
        print("🔍 Testing Jina API connection...")
        try:
            response = requests.post(
                JINA_EMBED_API_URL,
                headers=self.headers,
                json={"model": "jina-embeddings-v2-base-en", "input": ["test"]},
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ Jina API connection successful!")
                return True
            else:
                print(f"⚠️  Jina API test returned status {response.status_code}")
                print(f"   Response: {response.text[:300]}")
                return False
                
        except Exception as e:
            print(f"❌ Jina API test failed: {e}")
            raise ValueError(f"Cannot connect to Jina API. Please check your JINA_API_KEY. Error: {e}")

    def get_embeddings(self, texts, batch_size=MAX_BATCH_SIZE):
        """Get embeddings with batching and text length validation."""
        all_embeddings = []
        
        # Validate inputs
        if not texts:
            return []
        
        # Truncate texts if needed and clean
        processed_texts = []
        for text in texts:
            if not text or not isinstance(text, str):
                text = " "  # Replace empty/invalid with space
            # Strip and truncate
            text = text.strip()[:MAX_TEXT_LENGTH]
            if not text:
                text = " "  # Ensure non-empty after strip
            processed_texts.append(text)
        
        print(f"Embedding {len(processed_texts)} texts in batches of {batch_size}...")
        
        # Process in batches
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(processed_texts) + batch_size - 1) // batch_size
            
            try:
                print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...", end=" ")
                response = requests.post(
                    JINA_EMBED_API_URL, 
                    headers=self.headers,
                    json={"model": "jina-embeddings-v2-base-en", "input": batch},
                    timeout=30
                )
                
                # Check for specific error messages
                if response.status_code != 200:
                    error_detail = response.text
                    print(f"\n❌ Batch {batch_num} failed with status {response.status_code}")
                    print(f"   Error details: {error_detail[:200]}")
                    all_embeddings.extend([None] * len(batch))
                    continue
                
                response.raise_for_status()
                batch_embeddings = [item['embedding'] for item in response.json()['data']]
                all_embeddings.extend(batch_embeddings)
                print("✓")
                
                # Rate limiting: small delay between batches
                if i + batch_size < len(processed_texts):
                    time.sleep(0.5)
                    
            except requests.exceptions.RequestException as e:
                print(f"\n❌ Error embedding batch {batch_num}: {e}")
                # Return None embeddings for failed batch
                all_embeddings.extend([None] * len(batch))
        
        successful = sum(1 for emb in all_embeddings if emb is not None)
        print(f"Embedding complete: {successful}/{len(all_embeddings)} successful")
        return all_embeddings

    def rerank_documents(self, query, documents):
        """Rerank documents with error handling."""
        try:
            # Truncate query and documents if needed
            query = query[:MAX_TEXT_LENGTH]
            documents = [doc[:MAX_TEXT_LENGTH] for doc in documents]
            
            response = requests.post(
                JINA_RERANK_API_URL, 
                headers=self.headers,
                json={
                    "model": "jina-reranker-v2-base-multilingual", 
                    "query": query, 
                    "documents": documents, 
                    "top_n": len(documents)
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['results']
        except requests.exceptions.RequestException as e:
            print(f"[!] Reranking failed: {e}. Using original order.")
            # Return documents in original order with dummy scores
            return [{"document": {"text": doc}, "relevance_score": 1.0 - (i * 0.1)} 
                    for i, doc in enumerate(documents)]

    def generate_chat_response(self, prompt, *, max_retries: int = 3, base_delay: float = 2.0):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    JINA_CHAT_API_URL,
                    headers=self.headers,
                    json={
                        "model": "jina-deepsearch-v1",
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                    timeout=60,
                )
                if response.status_code == 524:
                    raise requests.exceptions.RequestException(
                        "Gateway timeout (524) from Jina chat endpoint."
                    )
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt == max_retries:
                    break
                delay = base_delay * (2 ** (attempt - 1))
                print(f"[!] Jina chat call failed (attempt {attempt}/{max_retries}): {exc}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
        raise last_error


class MongoRAGManager:
    """A manager for the GSM8K RAG system using MongoDB."""
    def __init__(self, jina_client, collection_name=GSM8K_COLLECTION):
        self.vector_store = MongoVectorStore(MONGO_URI, DB_NAME, collection_name)
        self.jina_client = jina_client
        print(f"✅ MongoDB RAG collection '{collection_name}' is ready.")

    def count(self):
        return self.vector_store.collection.count_documents({})

    def add_documents(self, documents_df):
        """Add documents with progress tracking and error handling."""
        print(f"Embedding {len(documents_df)} documents for RAG with Jina AI...")
        
        # Get embeddings with batching
        embeddings = self.jina_client.get_embeddings(
            documents_df["text_for_embedding"].tolist()
        )
        
        # Filter out failed embeddings
        documents_to_insert = []
        failed_count = 0
        
        for i, row in documents_df.iterrows():
            if embeddings[i] is None:
                failed_count += 1
                continue
                
            doc = {
                "_id": row["id"],
                "text": row["text_for_embedding"],
                "embedding": embeddings[i],
                "metadata": {
                    "question": row["question"],
                    "solution_steps": row["solution_steps"],
                    "final_answer": row["final_answer"],
                    "reasoning_category": row["reasoning_category"]
                }
            }
            documents_to_insert.append(doc)
        
        if failed_count > 0:
            print(f"⚠️  {failed_count} documents failed to embed and were skipped.")
        
        if documents_to_insert:
            # Insert in batches to avoid overwhelming MongoDB
            batch_size = 500
            for i in range(0, len(documents_to_insert), batch_size):
                batch = documents_to_insert[i:i + batch_size]
                try:
                    self.vector_store.collection.insert_many(batch, ordered=False)
                except Exception as e:
                    print(f"[!] Error inserting batch {i//batch_size + 1}: {e}")
            
            print(f"✅ Added {len(documents_to_insert)} documents to MongoDB RAG collection.")
        else:
            print("❌ No documents were successfully embedded.")

    def query(self, user_query, n_results=5):
        query_embedding = self.jina_client.get_embeddings([user_query])[0]
        
        if query_embedding is None:
            print("[!] Failed to embed query. Cannot perform search.")
            return []
        
        try:
            results = self.vector_store.search(query_embedding, num_results=n_results)
            if results:
                return results
        except Exception as exc:
            print(f"[!] Vector search failed ({exc}). Falling back to manual search.")

        print("[!] No vector-search results found; using manual cosine similarity.")
        return self.vector_store.search_manual(query_embedding, num_results=n_results)


class GSM8KRAGSystem:
    """Orchestrates the RAG process using MongoDB for GSM8K dataset."""
    def __init__(self, api_key, use_local_file: bool = False, local_file_path: str = None):
        self.jina_client = JinaAIClient(api_key)
        self.db_manager = MongoRAGManager(self.jina_client)
        
        if self.db_manager.count() == 0:
            print("GSM8K RAG collection is empty. Populating with new data...")
            if use_local_file and local_file_path:
                training_data = self._load_from_local_file(local_file_path)
            else:
                training_data = self._load_from_huggingface()
            
            processed_docs_df = self._load_and_preprocess_data(training_data)
            self.db_manager.add_documents(processed_docs_df)
        else:
            print(f"GSM8K RAG system already has {self.db_manager.count()} problems.")

    def _load_from_huggingface(self):
        """Load GSM8K dataset from HuggingFace."""
        print("📥 Loading GSM8K dataset from HuggingFace...")
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        return [{"question": item["question"], "answer": item["answer"]} for item in dataset]

    def _load_from_local_file(self, file_path: str):
        """Load GSM8K dataset from local JSON file."""
        print(f"📥 Loading GSM8K dataset from local file: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def _categorize_problem(self, question: str) -> str:
        """Categorize GSM8K problems based on reasoning type."""
        question_lower = question.lower()
        
        # Multi-step arithmetic patterns
        if any(word in question_lower for word in ['total', 'altogether', 'sum', 'combined']):
            return "multi_step_arithmetic"
        
        # Ratio/grouping patterns
        if any(word in question_lower for word in ['each', 'per', 'every', 'split', 'divide', 'share']):
            return "ratio_grouping"
        
        # Temporal reasoning
        if any(word in question_lower for word in ['age', 'years', 'days', 'hours', 'time', 'after', 'before']):
            return "temporal_reasoning"
        
        # Money/cost
        if any(word in question_lower for word in ['$', 'dollar', 'cost', 'price', 'pay', 'spent', 'earn']):
            return "money_cost_reasoning"
        
        # Counting/comparison
        if any(word in question_lower for word in ['more than', 'less than', 'fewer', 'how many']):
            return "count_compare"
        
        return "general_reasoning"

    def _extract_solution_components(self, answer_text: str) -> tuple:
        """
        Extract solution steps and final numerical answer from GSM8K answer format.
        GSM8K format: "Step 1\nStep 2\n#### 42"
        """
        # Split by #### to separate steps from final answer
        parts = answer_text.split('####')
        
        if len(parts) == 2:
            solution_steps = parts[0].strip()
            final_answer = parts[1].strip()
        else:
            # Fallback if format is unexpected
            solution_steps = answer_text.strip()
            # Try to extract last number as answer
            numbers = re.findall(r'-?\d+\.?\d*', answer_text)
            final_answer = numbers[-1] if numbers else "unknown"
        
        return solution_steps, final_answer

    def _load_and_preprocess_data(self, training_data: List[Dict]) -> pd.DataFrame:
        """
        Preprocess GSM8K data for embedding and storage.
        Creates CONCISE embedding text to avoid length limits.
        """
        processed_docs = []
        
        print("Processing GSM8K problems...")
        for i, item in tqdm(enumerate(training_data), total=len(training_data)):
            question = item['question']
            answer_text = item['answer']
            
            # Extract solution and final answer
            solution_steps, final_answer = self._extract_solution_components(answer_text)
            
            # Categorize the problem
            category = self._categorize_problem(question)
            
            # Create CONCISE embedding text (to stay under limits)
            # Truncate solution steps if too long
            max_solution_length = 300  # Very conservative
            if len(solution_steps) > max_solution_length:
                solution_steps_short = solution_steps[:max_solution_length] + "..."
            else:
                solution_steps_short = solution_steps
            
            # Create compact embedding format
            text_for_embedding = (
                f"Q: {question[:400]}\n"  # Limit question length too
                f"Sol: {solution_steps_short}\n"
                f"Ans: {final_answer}"
            )
            
            # Final safety check on length
            if len(text_for_embedding) > MAX_TEXT_LENGTH:
                text_for_embedding = text_for_embedding[:MAX_TEXT_LENGTH]
            
            processed_docs.append({
                "id": f"gsm8k_{i}",
                "text_for_embedding": text_for_embedding,
                "question": question,
                "solution_steps": solution_steps,  # Store full solution
                "final_answer": final_answer,
                "reasoning_category": category
            })
        
        print(f"✅ Preprocessed {len(processed_docs)} GSM8K problems.")
        return pd.DataFrame(processed_docs)

    def answer_question(self, user_query: str, scenario: str = None) -> str:
        """
        Answers a user's math question using the full RAG pipeline.
        Returns the complete LLM response with step-by-step reasoning.
        """
        print(f"\n🔎 Querying GSM8K RAG system for: '{user_query}'")
        
        # Retrieve similar problems (increased to 5 for better context)
        retrieved_docs = self.db_manager.query(user_query, n_results=5)

        if retrieved_docs:
            docs_to_rerank = []
            for doc in retrieved_docs:
                # Build complete example text from metadata
                metadata = doc.get('metadata', {})
                example_text = (
                    f"Question: {metadata.get('question', '')}\n"
                    f"Solution:\n{metadata.get('solution_steps', '')}\n"
                    f"Final Answer: {metadata.get('final_answer', '')}"
                )
                docs_to_rerank.append(example_text)
            
            docs_to_rerank = [txt for txt in docs_to_rerank if txt]
            print(f"\n🔄 Reranking {len(docs_to_rerank)} documents for relevance...")
            reranked_results = self.jina_client.rerank_documents(user_query, docs_to_rerank)
            print("✅ Reranking complete.")

            # Build context with top examples
            context_chunks = []
            for i, doc in enumerate(reranked_results[:3]):  # Use top 3 after reranking
                relevance = doc.get('relevance_score', 0.0)
                document_payload = doc.get('document')
                
                if isinstance(document_payload, dict):
                    text = document_payload.get('text', '')
                else:
                    text = document_payload or doc.get('text', '') or ''
                
                context_chunks.append(
                    f"Example {i+1} (Relevance: {relevance:.2f}):\n{text}"
                )
            
            context_str = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant examples found."
        else:
            print("[!] No relevant documents retrieved; proceeding with direct reasoning.")
            context_str = "No relevant examples were retrieved from the knowledge base."
        
        # Create scenario guidance if provided
        scenario_guidance = ""
        if scenario:
            scenario_guidance = f"This problem involves {scenario}. Pay special attention to this aspect in your reasoning."

        # Construct prompt for GSM8K (free-form numerical answer)
        prompt = f"""You are an expert math tutor specializing in grade-school mathematics. Your task is to solve the given math word problem step-by-step, showing clear reasoning at each stage.

**Context: Similar Solved Problems**
Below are examples of similar problems and their solutions from the knowledge base:

{context_str}

**Student's Question:**
{user_query}

{scenario_guidance}

**Instructions:**
1. Read the problem carefully and identify what is being asked
2. Break down the problem into logical steps
3. Show your calculations clearly at each step
4. Verify your arithmetic is correct
5. State your final numerical answer clearly

**Your Response:**
Provide a complete step-by-step solution. End your response with the final answer in this exact format:
#### [numerical answer]

For example: #### 42 or #### 3.5
"""

        print("\n🤖 Generating solution with Jina DeepSearch...")
        return self.jina_client.generate_chat_response(prompt)


# --- 3. THE GSM8K TOOL FOR THE AGENT ---

class GSM8KTool(BaseTool):
    """
    A tool for solving grade-school math word problems using a RAG system.
    Designed for GSM8K dataset with free-form numerical answers.
    """
    def __init__(self, use_local_file: bool = False, local_file_path: str = None):
        super().__init__("gsm8k")
        self.description = (
            "A tool for solving grade-school math word problems (grades 3-6) "
            "using multi-step reasoning and chain-of-thought. Returns numerical answers."
        )
        
        try:
            self.rag_system = GSM8KRAGSystem(
                API_KEY, 
                use_local_file=use_local_file, 
                local_file_path=local_file_path
            )
            print("✅ GSM8K RAG system initialized successfully.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not initialize GSM8K RAG system: {e}")
            import traceback
            traceback.print_exc()
            self.rag_system = None

    def run(self, user_query: str, data_item: Optional[Dict] = None, 
            recommended_scenario: str = None) -> ToolUsageExample:
        """
        Executes the math problem-solving logic using the GSM8K RAG system.
        
        Args:
            user_query: The math word problem to solve
            data_item: Optional metadata about the problem
            recommended_scenario: Optional hint about problem type (e.g., "ratio reasoning")
        
        Returns:
            ToolUsageExample with complete solution and extracted numerical answer
        """
        if not self.rag_system:
            return self._create_error_response(user_query, "RAG system not initialized.")

        try:
            full_response_text = self.rag_system.answer_question(user_query, recommended_scenario)
            parsed_output = self._parse_llm_response(full_response_text)

            return ToolUsageExample(
                tool_name=self.name,
                user_query=user_query,
                raw_prompt="[Prompt managed by internal GSM8K RAG system]",
                llm_response=full_response_text,
                parsed_output=parsed_output
            )
        except Exception as e:
            return self._create_error_response(user_query, f"An error occurred in the RAG system: {e}")

    def _parse_llm_response(self, response_text: str) -> dict:
        """
        Extract the final numerical answer from the LLM's response.
        Looks for the GSM8K format: #### [number]
        """
        # Primary pattern: #### [number]
        match = re.search(r'####\s*(-?\d+\.?\d*)', response_text)
        if match:
            answer = match.group(1)
            return {
                "final_answer": answer,
                "answer_type": "numerical",
                "full_solution": response_text
            }
        
        # Fallback: Look for explicit answer statements
        fallback_patterns = [
            r'final answer is\s*(-?\d+\.?\d*)',
            r'answer:\s*(-?\d+\.?\d*)',
            r'therefore,?\s*(-?\d+\.?\d*)',
            r'the answer is\s*(-?\d+\.?\d*)'
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                answer = match.group(1)
                return {
                    "final_answer": answer,
                    "answer_type": "numerical",
                    "full_solution": response_text,
                    "extraction_method": "fallback"
                }
        
        # Last resort: extract the last number in the response
        numbers = re.findall(r'-?\d+\.?\d*', response_text)
        if numbers:
            return {
                "final_answer": numbers[-1],
                "answer_type": "numerical",
                "full_solution": response_text,
                "extraction_method": "last_number",
                "warning": "Answer extracted as last number - may be inaccurate"
            }
        
        return {
            "final_answer": None,
            "answer_type": "numerical",
            "full_solution": response_text,
            "error": "Could not extract numerical answer"
        }

    def _create_error_response(self, user_query: str, error_message: str) -> ToolUsageExample:
        """Helper to create a consistent error object."""
        print(f"❌ {error_message}")
        return ToolUsageExample(
            tool_name=self.name,
            user_query=user_query,
            raw_prompt="[Error occurred]",
            llm_response=error_message,
            parsed_output={
                "error": error_message, 
                "final_answer": None,
                "answer_type": "numerical"
            }
        )


# --- 4. USAGE EXAMPLE ---
if __name__ == "__main__":
    # Initialize the tool
    # Option 1: Load from HuggingFace (default)
    tool = GSM8KTool()
    
    # Option 2: Load from local JSON file
    # tool = GSM8KTool(use_local_file=True, local_file_path="gsm8k_train.json")
    
    # Test with a sample problem
    test_problem = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning 
    and bakes muffins for her friends every day with four. She sells the remainder at the farmers' 
    market daily for $2 per fresh duck egg. How much in dollars does she make every day at the 
    farmers' market?"""
    
    result = tool.run(test_problem)
    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)
    print(f"Final Answer: {result.parsed_output.get('final_answer')}")
    print(f"\nFull Solution:\n{result.llm_response}")