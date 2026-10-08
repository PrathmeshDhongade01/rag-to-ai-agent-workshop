import chromadb
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

# -----------------------------
# GEMINI CLIENT
# -----------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)


# -----------------------------
# CHROMADB
# -----------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="kbtcoe"
)


# -----------------------------
# CREATE QUERY EMBEDDING
# -----------------------------

def create_embedding(text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


# -----------------------------
# RETRIEVE RELEVANT CHUNKS
# -----------------------------

def search_rag(query, top_k=3):

    query_embedding = create_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieved_chunks = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        retrieved_chunks.append({
            "text": document,
            "page": metadata["page"]
        })

    return retrieved_chunks


# -----------------------------
# GENERATE ANSWER
# -----------------------------

def generate_answer(query, retrieved_chunks):

    context = "\n\n".join(
        [
            f"Page {chunk['page']}:\n{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are a helpful college information assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:

"I couldn't find that information in the provided document."

Keep the answer clear and concise.

Context:
{context}

User Question:
{query}

Answer:
"""

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    return interaction.output_text


# -----------------------------
# TEST COMPLETE RAG
# -----------------------------

if __name__ == "__main__":

    query = input("\nAsk a question about KBTCOE: ")

    # Retrieve
    results = search_rag(query)

    # Generate
    answer = generate_answer(
        query,
        results
    )

    print("\n========== AI ANSWER ==========\n")
    print(answer)

    print("\n========== SOURCES ==========\n")

    for result in results:

        print(
            f"Page {result['page']}"
        )