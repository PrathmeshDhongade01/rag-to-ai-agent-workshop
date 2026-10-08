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

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Explain RAG in two simple sentences."
)

print("\n========== GEMINI RESPONSE ==========\n")
print(interaction.output_text)