import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(
    api_key=api_key
)

text = "KBTCOE was established in 1999."

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
)

embedding = response.embeddings[0].values

print("\n========== EMBEDDING TEST ==========\n")
print("Text:", text)
print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])