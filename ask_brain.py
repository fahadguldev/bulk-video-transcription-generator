import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIG
# ============================================================

with open("system_prompt.md", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

MODEL = "gemini-3.1-flash-lite"

KNOWLEDGE_FILE = Path("knowledge_base/records.jsonl")
EMBEDDINGS_FILE = Path("knowledge_base/gemini_embeddings.npz")

TOP_K = 5


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


def load_api_keys():
    """Load all configured Gemini API keys."""

    keys = []

    for i in range(1, 6):
        key = os.getenv(f"GEMINI_API_KEY_{i}")

        if key:
            keys.append(
                {
                    "number": i,
                    "key": key,
                }
            )

    if not keys:
        raise RuntimeError(
            "No Gemini API keys found in .env"
        )

    return keys


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_knowledge_base():

    records = []

    with KNOWLEDGE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            records.append(record)

    return records


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    data = np.load(
        EMBEDDINGS_FILE,
        allow_pickle=False,
    )

    ids = data["ids"].tolist()

    embeddings = data["embeddings"].astype(
        np.float32
    )

    return ids, embeddings


# ============================================================
# GEMINI CLIENT
# ============================================================

def create_client(api_key):

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# EMBEDDING
# ============================================================

def embed_question(client, question):

    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=question,
    )

    return np.asarray(
        result.embeddings[0].values,
        dtype=np.float32,
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(query, embeddings):

    query_norm = np.linalg.norm(query)

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    similarities = (
        embeddings @ query
    ) / (
        embedding_norms * query_norm
    )

    return similarities


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question,
    client,
    records,
    embedding_ids,
    embeddings,
):

    query_embedding = embed_question(
        client,
        question,
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings,
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:TOP_K]

    results = []

    id_to_record = {
        record["id"]: record
        for record in records
    }

    for index in top_indices:

        record_id = embedding_ids[index]

        record = id_to_record.get(
            record_id
        )

        if record is None:
            continue

        results.append(
            {
                "id": record_id,
                "score": float(
                    similarities[index]
                ),
                "record": record,
            }
        )

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):

        record = result["record"]

        text = record.get(
            "content",
            {}
        ).get(
            "text",
            ""
        )

        classification = record.get(
            "classification",
            {}
        )

        context_parts.append(
            f"""
SOURCE {i}
ID: {result['id']}
SIMILARITY: {result['score']:.4f}
DOMAIN: {classification.get('domain', 'unknown')}
TOPICS: {classification.get('topics', [])}

CONTENT:
{text}
""".strip()
        )

    return "\n\n" + "\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context,
    client,
):

    prompt = f"""
USER QUESTION:
{question}

RELEVANT KNOWLEDGE:
{context}

Answer the user's question based on the retrieved knowledge baseon on these instructions:
1. Find the retrieved record(s) that directly answer the question.
2. Use their exact meaning and direction.
3. Do not generalize from a related topic.
4. Do not reverse, soften, or reinterpret the user's viewpoint.
5. If the retrieved knowledge does not contain enough evidence, do not invent an answer.
6. For unknown topics, use the user's deferral style.
7. Keep the final answer short and natural.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
        }
    )

    return response.text.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SECOND BRAIN")
    print("=" * 60)

    print("\nLoading knowledge base...")

    records = load_knowledge_base()

    print(
        f"Loaded {len(records)} knowledge records."
    )

    print("\nLoading embeddings...")

    embedding_ids, embeddings = load_embeddings()

    print(
        f"Loaded {len(embedding_ids)} embeddings."
    )

    if len(records) != len(embedding_ids):

        raise RuntimeError(
            "Knowledge records and embeddings "
            "have different counts."
        )

    # --------------------------------------------------------
    # API keys
    # --------------------------------------------------------

    api_keys = load_api_keys()

    # For now use the first working key.
    # If it fails because of quota, try another key.
    clients = []

    for item in api_keys:

        try:

            client = create_client(
                item["key"]
            )

            clients.append(
                {
                    "number": item["number"],
                    "client": client,
                }
            )

        except Exception:
            pass

    if not clients:

        raise RuntimeError(
            "Could not create a Gemini client."
        )

    print(
        f"Gemini clients available: "
        f"{len(clients)}"
    )

    # --------------------------------------------------------
    # Interactive loop
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("READY")
    print("=" * 60)

    print(
        "\nAsk your Second Brain anything."
    )

    print(
        "Type 'exit' to quit.\n"
    )

    while True:

        try:

            question = input(
                "You: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print("\n\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
        }:

            print("\nGoodbye.")
            break

        # ----------------------------------------------------
        # Try available Gemini clients
        # ----------------------------------------------------

        answer = None
        last_error = None

        for client_info in clients:

            client = client_info["client"]

            try:

                # Retrieve
                results = retrieve(
                    question,
                    client,
                    records,
                    embedding_ids,
                    embeddings,
                )

                # Build context
                context = build_context(
                    results
                )

                # Generate answer
                answer = generate_answer(
                    question,
                    context,
                    client,
                )

                break

            except Exception as error:

                last_error = error

                print(
                    f"\nAPI key "
                    f"{client_info['number']} "
                    f"failed. Trying another..."
                )

        if answer is None:

            print(
                "\nERROR:"
            )

            print(last_error)

            continue

        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        print(
            "\nSecond Brain:"
        )

        print(answer)

        print(
            "\n" + "-" * 60
        )

        print(
            "Ask another question "
            "(or type 'exit'):\n"
        )


if __name__ == "__main__":
    main()