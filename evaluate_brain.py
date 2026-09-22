import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIG
# ============================================================

MODEL = "gemini-3.1-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-2"

KNOWLEDGE_FILE = Path("knowledge_base/records.jsonl")
EMBEDDINGS_FILE = Path("knowledge_base/gemini_embeddings.npz")
EVAL_FILE = Path("knowledge_base/eval_retrieval.jsonl")
OUTPUT_FILE = Path("knowledge_base/eval_generation_results.jsonl")

TOP_K = 5


# ============================================================
# SYSTEM PROMPT
# ============================================================

with open("system_prompt.md", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


def load_api_keys():

    keys = []

    for i in range(1, 6):

        key = os.getenv(f"GEMINI_API_KEY_{i}")

        if key:

            keys.append({
                "number": i,
                "key": key,
            })

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

            records.append(
                json.loads(line)
            )

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
# LOAD EVALUATION QUESTIONS
# ============================================================

def load_eval_questions():

    questions = []

    with EVAL_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            questions.append(
                json.loads(line)
            )

    return questions


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

def embed_question(
    client,
    question,
):

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
    )

    return np.asarray(
        result.embeddings[0].values,
        dtype=np.float32,
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    query,
    embeddings,
):

    query_norm = np.linalg.norm(query)

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    return (
        embeddings @ query
    ) / (
        embedding_norms * query_norm
    )


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

    id_to_record = {
        record["id"]: record
        for record in records
    }

    results = []

    for index in top_indices:

        record_id = embedding_ids[index]

        record = id_to_record.get(
            record_id
        )

        if record is None:
            continue

        results.append({
            "id": record_id,
            "score": float(
                similarities[index]
            ),
            "record": record,
        })

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

Answer the user's question based on the retrieved knowledge base.

Instructions:
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
        },
    )

    return response.text.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SECOND BRAIN — GENERATION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading knowledge base...")

    records = load_knowledge_base()

    print(
        f"Loaded {len(records)} records."
    )

    print("\nLoading embeddings...")

    embedding_ids, embeddings = load_embeddings()

    print(
        f"Loaded {len(embedding_ids)} embeddings."
    )

    print("\nLoading evaluation questions...")

    eval_questions = load_eval_questions()

    print(
        f"Loaded {len(eval_questions)} evaluation questions."
    )

    if len(records) != len(embedding_ids):

        raise RuntimeError(
            "Knowledge records and embeddings "
            "have different counts."
        )

    # --------------------------------------------------------
    # Create clients
    # --------------------------------------------------------

    api_keys = load_api_keys()

    clients = []

    for item in api_keys:

        try:

            client = create_client(
                item["key"]
            )

            clients.append({
                "number": item["number"],
                "client": client,
            })

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
    # Remove previous results
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():

        OUTPUT_FILE.unlink()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)

    results_for_file = []

    for number, evaluation in enumerate(
        eval_questions,
        start=1,
    ):

        eval_id = evaluation["eval_id"]
        question = evaluation["question"]

        print(
            f"\n[{number}/{len(eval_questions)}] "
            f"{eval_id}"
        )

        print(
            f"Question: {question}"
        )

        answer = None
        retrieved = None
        last_error = None

        for client_info in clients:

            client = client_info["client"]

            try:

                retrieved = retrieve(
                    question,
                    client,
                    records,
                    embedding_ids,
                    embeddings,
                )

                context = build_context(
                    retrieved
                )

                answer = generate_answer(
                    question,
                    context,
                    client,
                )

                break

            except Exception as error:

                last_error = error

                print(
                    f"API key "
                    f"{client_info['number']} "
                    f"failed. Trying another..."
                )

        if answer is None:

            print(
                f"ERROR: {last_error}"
            )

            continue

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"Answer: {answer}"
        )

        print("Retrieved:")

        for rank, item in enumerate(
            retrieved,
            start=1,
        ):

            print(
                f"  {rank}. "
                f"{item['id']} "
                f"({item['score']:.4f})"
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_record = {
            "eval_id": eval_id,
            "question": question,
            "expected": {
                "ground_truth_id": evaluation.get(
                    "ground_truth_id"
                ),
                "expected_topic": evaluation.get(
                    "expected_topic"
                ),
                "domain": evaluation.get(
                    "domain"
                ),
                "difficulty": evaluation.get(
                    "difficulty"
                ),
            },
            "answer": answer,
            "retrieved": [
                {
                    "id": item["id"],
                    "score": item["score"],
                }
                for item in retrieved
            ],
        }

        results_for_file.append(
            output_record
        )

        with OUTPUT_FILE.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nCompleted: "
        f"{len(results_for_file)} / "
        f"{len(eval_questions)}"
    )

    print(
        f"Results saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()