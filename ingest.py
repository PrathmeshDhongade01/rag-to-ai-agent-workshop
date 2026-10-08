import os
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

# Gemini client
client = genai.Client(api_key=api_key)

# -----------------------------
# 1. READ PDF
# -----------------------------

pdf_path = "data/KBTCOE_Collegedemodata.pdf"

reader = PdfReader(pdf_path)

print(f"\nPDF pages found: {len(reader.pages)}")

# -----------------------------
# 2. EXTRACT TEXT
# -----------------------------

pages = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if text:
        pages.append({
            "page": page_number,
            "text": text
        })

print(f"Pages with text: {len(pages)}")

# -----------------------------
# 3. CHUNK TEXT
# -----------------------------

chunks = []

chunk_size = 1000
overlap = 150

for page in pages:

    text = page["text"]

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

        start += chunk_size - overlap

print(f"Total chunks created: {len(chunks)}")

# -----------------------------
# 4. CREATE GEMINI EMBEDDINGS
# -----------------------------

print("\nCreating Gemini embeddings...")

embeddings = []

batch_size = 20

for i in range(0, len(chunks), batch_size):

    batch = chunks[i:i + batch_size]

    texts = [item["text"] for item in batch]

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )

    batch_embeddings = [
        embedding.values
        for embedding in response.embeddings
    ]

    embeddings.extend(batch_embeddings)

    print(
        f"Embedded {min(i + batch_size, len(chunks))}"
        f"/{len(chunks)} chunks"
    )

# -----------------------------
# 5. CREATE CHROMA DATABASE
# -----------------------------

print("\nCreating ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Remove old collection if it exists
try:
    chroma_client.delete_collection("kbtcoe")
except Exception:
    pass

collection = chroma_client.create_collection(
    name="kbtcoe"
)

# -----------------------------
# 6. STORE DATA
# -----------------------------

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

documents = [
    chunk["text"]
    for chunk in chunks
]

metadatas = [
    {
        "page": chunk["page"]
    }
    for chunk in chunks
]

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("\n================================")
print("RAG INGESTION COMPLETE")
print("================================")
print(f"Documents stored: {collection.count()}")
print("Database: ./chroma_db")