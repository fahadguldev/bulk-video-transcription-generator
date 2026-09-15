import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

INPUT_FILE = "knowledge_base/eval_retrieval.jsonl"
    
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f][:5]

for i, record in enumerate(records, 1):
    text = record["content"]["text"]

    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
    )

    embedding = result.embeddings[0].values

    print(f"Record {i}")
    print(f"ID: {record['id']}")
    print(f"Text length: {len(text)} characters")
    print(f"Embedding dimensions: {len(embedding)}")
    print()