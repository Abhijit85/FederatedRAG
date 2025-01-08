from openai import OpenAI
import numpy as np
import os
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience

# Set your OpenAI API key here
os.environ['OPENAI_API_KEY'] = "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"
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

"""
Generates a response from the OpenAI GPT-3.5-turbo model based on the given prompt.
Args:
    prompt (str): The input prompt to generate a response for.
Returns:
    str: The generated response from the model.
"""
# Function implementation here
def get_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
 
    return response.choices[0].message.content

# Set your Neo4j credentials here
neo4j_uri = "bolt://localhost:7689"
neo4j_user = "neo4j"
neo4j_password = "ilovemovies"

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))


# Recommendation function to search the knowledge graph for a movie title containing the given prompt
def search_knowledge_graph(prompt):
    with driver.session() as session:
        result = session.run(
            "MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre {name: ($prompt)}) "
            "RETURN m.title as name ORDER BY m.rating ASC LIMIT 5",
            prompt=prompt
        )
        titles = []
        for record in result:
            titles.append(record["name"])
        return titles
    
from jinja2 import Template

class PromptTemplate:
    ''' A class to represent a template for generating prompts with variables
    Attributes:
        template (str): The template string with variables
        input_variables (list): A list of the variable names in the template
    '''
    def __init__(self, template, input_variables):
        self.template = Template(template)
        self.input_variables = input_variables
    
    def format(self, **kwargs):
        return self.template.render(**kwargs)        


# Define a template for a complex prompt, this templates works for both item recommendation and graph completion.   AZ          
complex_template = PromptTemplate(
        template="""You are a movie recommender.
        You will be given two movie titles: {{movie1}}, {{movie2}} and {{movie3}}.
        You will find the top most common factor between the two movies.
        Do not provide additional information.
        Follow the below steps to find the common factor:
        Step 1: List out the main elements or themes each movies.
        Step 2: Identify the top most common factor between the above two movies.
        Step 3: Classify the genre of the movies based on the common factor.
        Step 4: If more than one genre is identified, choose one genre that is most relevant to the common factor.
        Step 5: Provide the genre as the final answer.
        Do not broadcast your steps. Provide the final answer and cypher query only.
        """,
                                                                                                                      
        input_variables=["movie1", "movie2", "movie3"]
    )

# Function to generate embeddings
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Function to generate embeddings for movie titles
def convert_to_embedding(*args):
    if len(args) < 2 or len(args) > 3:
        raise ValueError("Function requires either two or three arguments")
    
    prompt = "Convert the following to an embedding:\n"
    if len(args) == 2:
        movie_field1, genre = args
        prompt += f"Movie: {movie_field1}\nGenre: {genre}"
    else:
        movie_field1, movie_field2, genre = args
        prompt += f"Movie 1: {movie_field1}\nMovie 2: {movie_field2}\nGenre: {genre}"
    
    response = get_embedding(prompt)
    # embedding = response.strip()
    # embedding = [elem.strip() for elem in response]
    # return embedding
    return response

# Function to recommend the movies based on the genre
from neo4j_graphrag.indexes import create_vector_index
create_vector_index(driver, name="text_embeddings", label="Chunk",
                   embedding_property="embedding", dimensions=1536, similarity_fn="cosine")

# GraphRAG Vector Cypher Retriever
# from neo4j_graphrag.retrievers import VectorCypherRetriever
# from openai import LLM
# from neo4j_graphrag.templates import RagTemplate
# import numpy as np

# def GraphRAG(query, top_k=5):
#     retriever = VectorCypherRetriever(
#         driver=driver,
#         index_name="text_embeddings",
#         retrieval_query="""
#     //1) Go out 2-3 hops in the entity graph and get relationships
#     WITH node AS chunk
#     MATCH (chunk)<-[:FROM_CHUNK]-(entity)-[relList:!FROM_CHUNK]-{1,2}(nb)
#     UNWIND relList AS rel

#     //2) collect relationships and text chunks
#     WITH collect(DISTINCT chunk) AS chunks, collect(DISTINCT rel) AS rels

#     //3) format and return context
#     RETURN apoc.text.join([c in chunks | c.text], '\n') +
#     apoc.text.join([r in rels |
#     startNode(r).name+' - '+type(r)+' '+r.details+' -> '+endNode(r).name],
#     '\n') AS info
#     """
#         query=query,
#         top_k=top_k
#     )
#     llm = LLM(model_name="gpt-3.5-turbo",  model_params={"temperature": 0.0})
#     rag_template = RagTemplate(template='''Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned.

#     # Question:
#     {query_text}

#     # Context:
#     {context}

#     # Answer:
#     ''', expected_inputs=['query_text', 'context'])
#     response = retriever.retrieve(query)
#     context = response['info']
#     answer = llm.generate(rag_template.format(query_text=query, context=context))
#     return answer

# def get_common_movies(movie_list):
#     if len(movie_list) < 3:
#         raise ValueError("At least two movies are required")
    
#     movie1, movie2,movie3 = movie_list[:3]
#     prompt = complex_template.format(movie1=movie1, movie2=movie2,movie3=movie3)
#     response = get_response(prompt)
#     genre = response.strip()
#     movies = search_knowledge_graph(genre)
    
#     return genre, movies


def find_nearest_neighbors(embedding, top_k=5):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (m:Movie)
            WITH m, gds.alpha.similarity.euclideanDistance(m.embedding, $embedding) AS distance
            RETURN m.title AS title, distance
            ORDER BY distance ASC
            LIMIT $top_k
            """,
            embedding=embedding,
            top_k=top_k
        )
        neighbors = []
        for record in result:
            neighbors.append((record["title"], record["distance"]))
        return neighbors




# Example usage
if __name__ == "__main__":

#     movie_list = []
#     for i in range(3):
#         movie = input(f"Enter the title of movie {i+1}: ")
#         movie_list.append(movie)

#     common_movies = get_common_movies(movie_list)
#     genre, movies = common_movies
#     print(common_movies)
#     print(f"Genre: {genre}")
#     print("Movies:")
#     for movie in movies:
#         print(movie)
 movie_triples = []
for i in range(1):
    movie_name = input(f"Enter the name of movie {i+1}: ")
    movie_id = input(f"Enter the ID of movie {i+1}: ")
    genre = input(f"Enter the genre of movie {i+1}: ")
    movie_triples.append((movie_name, movie_id, genre))

embeddings = []
for triple in movie_triples:
    embedding = convert_to_embedding(*triple)
    embeddings.append(embedding)
    # Combine embeddings into a single embedding

    combined_embedding = np.mean(embeddings, axis=0).tolist()

# query = "Generate recommendations based on the following embeddings."
# recommendations = GraphRAG(query, top_k=5)

neighbors = find_nearest_neighbors(combined_embedding, top_k=5)

print("Nearest Neighbors:")
for neighbor in neighbors:
    print(f"Title: {neighbor[0]}, Distance: {neighbor[1]}")

# print("Recommendations:")
# print(recommendations)
  


