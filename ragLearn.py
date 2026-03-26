import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

!pip install chromadb sentence-transformers
!pip install open-clip-torch

class MyColabEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()

client = chromadb.Client()
custom_ef = MyColabEmbeddingFunction()

collection = client.get_or_create_collection(
    name="my_first_rag_collection",
    embedding_function=custom_ef
)

print("Adding documents and metadata to the vector database...")
collection.upsert(
    documents=[
        "Astronauts on the Apollo missions ate freeze-dried ice cream.",
        "A myocardial infarction, or heart attack, requires immediate medical attention."
    ],
    ids=["doc_space", "doc_medical"],

    metadatas=[
        {"category": "space", "author": "NASA"},
        {"category": "health", "author": "WebMD"}
    ]
)

print("\n--- 1. THE 'GET' API (Exact Match, No AI) ---")
get_results = collection.get(
    ids=["doc_space"],
    include=["documents", "metadatas"]
)
print("Get Results:", get_results['documents'])


print("\n--- 2. VECTOR SEARCH WITH FILTERING (AI + SQL) ---")
query_results = collection.query(
    query_texts=["What kind of vehicle they travel in?"],
    n_results=1,
    where={"category": "space"},
    include=["documents", "distances", "metadatas"]
)

print("Best Match:", query_results['documents'][0][0])
print("Distance Score:", query_results['distances'][0][0])
print("Metadata:", query_results['metadatas'][0][0])


print("Adding more documents to test text searching...")
collection.upsert(
    documents=[
        "Please contact support@nasa.gov for more information about the Apollo missions.",
        "The space shuttle is a reusable low Earth orbital spacecraft.",
        "Medical emergencies should be reported to 911 immediately."
    ],
    ids=["doc_support", "doc_shuttle", "doc_911"],
    metadatas=[
        {"category": "space", "type": "contact"},
        {"category": "space", "type": "vehicle"},
        {"category": "health", "type": "emergency"}
    ]
)

print("\n--- 1. FULL TEXT SEARCH ($contains) ---")
contains_results = collection.get(
    where_document={"$contains": "spacecraft"}
)
print("Documents containing 'spacecraft':", contains_results['documents'])


print("\n--- 2. REGEX SEARCH ($regex) ---")
regex_results = collection.get(
    where_document={"$regex": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}
)
print("Documents containing an Email:", regex_results['documents'])


print("\n--- 3. THE ULTIMATE COMBO (Vector + Metadata + Full Text) ---")
combo_results = collection.query(
    query_texts=["Who do I contact?"],
    n_results=1,
    where={"category": "space"},
    where_document={"$contains": "support"}
)
print("Ultimate Combo Match:", combo_results['documents'][0][0])


print("Adding advanced metadata documents (Movies)...")
collection.upsert(
    documents=[
        "The Martian is a movie about an astronaut surviving alone on Mars.",
        "Interstellar explores black holes, time dilation, and relativity.",
        "Apollo 13 is a historical drama about a real lunar mission crisis.",
        "Inception is a mind-bending dream heist thriller."
    ],
    ids=["movie_martian", "movie_interstellar", "movie_apollo13", "movie_inception"],
    metadatas=[
        {"rating": 8.0, "genres": ["sci-fi", "survival"], "year": 2015},
        {"rating": 8.6, "genres": ["sci-fi", "drama"], "year": 2014},
        {"rating": 7.6, "genres": ["history", "drama", "space"], "year": 1995},
        {"rating": 8.8, "genres": ["sci-fi", "thriller"], "year": 2010}
    ]
)

print("\n--- 1. NUMERIC FILTERING ($gte) ---")
highly_rated = collection.get(
    where={"rating": {"$gte": 8.5}}
)
print("Highly Rated Movies:", highly_rated['documents'])


print("\n--- 2. ARRAY FILTERING ($contains) ---")
drama_movies = collection.get(
    where={"genres": {"$contains": "drama"}}
)
print("Drama Movies:", drama_movies['documents'])


print("\n--- 3. LOGICAL OPERATORS ($and) ---")
classic_scifi = collection.get(
    where={
        "$and": [
            {"genres": {"$contains": "sci-fi"}},
            {"year": {"$lt": 2015}}
        ]
    }
)
print("Sci-Fi before 2015:", classic_scifi['documents'])


image_path = "/content/sample_data/dog.webp"

print("\nLoading OpenCLIP AI Model (this takes a few seconds)...")
client = chromadb.Client()
clip_ef = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

multimodal_db = client.get_or_create_collection(
    name="my_multimodal_collection",
    embedding_function=clip_ef,
    data_loader=image_loader
)

print("Adding text documents to the database...")
multimodal_db.upsert(
    ids=["txt_dog", "txt_space", "txt_food"],
    documents=[
        "A description of a furry canine animal, like a dog or a puppy.",
        "People used to travel with this to the entire world",
        "A delicious recipe for baking a chocolate cake."
    ]
)
print("\n--- IMAGE-TO-TEXT SEARCH ---")
print(f"Reading pixels from {image_path}...")
print("Asking the database: 'Which text document best matches this picture?'")

results = multimodal_db.query(
    query_uris=[image_path],
    n_results=1,
    include=["documents", "distances"]
)

print("\n--- THE RESULT ---\nThe AI looked at the image and chose this text:")
print(f"🏆 '{results['documents'][0][0]}'")
