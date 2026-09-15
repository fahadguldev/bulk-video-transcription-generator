import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError


# ============================================================
# CONFIG
# ============================================================

MODEL = "gemini-embedding-2"
BATCH_SIZE = 25

KNOWLEDGE_FILE = Path("knowledge_base/records.jsonl")
EVAL_FILE = Path("knowledge_base/eval_retrieval.jsonl")

OUTPUT_FILE = Path("knowledge_base/gemini_embeddings.npz")
CHECKPOINT_FILE = Path("knowledge_base/gemini_embeddings_checkpoint.npz")


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


def load_api_keys():
    """Load GEMINI_API_KEY_1 ... GEMINI_API_KEY_5."""
    keys = []

    for i in range(1, 6):
        key = os.getenv(f"GEMINI_API_KEY_{i}")

        if key:
            keys.append((i, key))

    return keys


# ============================================================
# GEMINI HELPERS
# ============================================================

def create_client(api_key):
    return genai.Client(api_key=api_key)


def is_quota_error(error):
    """Return True if the error is a Gemini quota/rate-limit error."""
    text = str(error).upper()

    return (
        "429" in text
        or "RESOURCE_EXHAUSTED" in text
        or "QUOTA_EXCEEDED" in text
    )


def is_auth_error(error):
    """Return True if the error looks like an authentication error."""
    text = str(error).upper()

    return (
        "401" in text
        or "UNAUTHENTICATED" in text
        or "INVALID_ARGUMENT" in text
    )


def embed_single(client, text):
    """Small test embedding used during key preflight."""
    result = client.models.embed_content(
        model=MODEL,
        contents=text,
    )

    if not result.embeddings:
        raise RuntimeError("Gemini returned no embedding.")

    return result.embeddings[0].values


def embed_batch(client, texts):
    """
    Embed multiple independent texts in a single API request.

    Each text is wrapped in its own Content object so Gemini
    returns one separate embedding per input.
    """

    contents = [
        types.Content(
            parts=[
                types.Part.from_text(text=text)
            ]
        )
        for text in texts
    ]

    result = client.models.embed_content(
        model=MODEL,
        contents=contents,
    )

    if not result.embeddings:
        raise RuntimeError("Gemini returned no embeddings.")

    if len(result.embeddings) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings, "
            f"but Gemini returned {len(result.embeddings)}."
        )

    return [
        embedding.values
        for embedding in result.embeddings
    ]


# ============================================================
# KEY PREFLIGHT
# ============================================================

def test_api_keys(api_keys):
    """
    Test every configured key before starting the expensive
    knowledge-base embedding process.
    """

    print("=" * 60)
    print("GEMINI API KEY PREFLIGHT")
    print("=" * 60)

    working_keys = []

    for key_number, api_key in api_keys:
        print(f"\nTesting API key {key_number}...")

        try:
            client = create_client(api_key)

            vector = embed_single(
                client,
                "Second Brain embedding preflight test."
            )

            print("✓ SUCCESS")
            print(f"  Dimensions: {len(vector)}")

            working_keys.append(
                {
                    "number": key_number,
                    "key": api_key,
                    "client": client,
                }
            )

        except Exception as error:
            print("✗ FAILED")

            if is_quota_error(error):
                print("  Reason: quota exhausted / rate limit")
            elif is_auth_error(error):
                print("  Reason: authentication / invalid key")
            else:
                print(f"  Reason: {error}")

    print("\n" + "-" * 60)
    print(f"Working keys: {len(working_keys)}")

    for item in working_keys:
        print(f"✓ API key {item['number']}")

    if not working_keys:
        raise RuntimeError(
            "No working Gemini API keys were found."
        )

    return working_keys


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_knowledge_base():
    print("\n" + "=" * 60)
    print("LOADING KNOWLEDGE BASE")
    print("=" * 60)

    records = []

    with KNOWLEDGE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(
                    f"WARNING: Could not parse line "
                    f"{line_number}: {error}"
                )
                continue

            record_id = record.get("id")

            content = record.get("content", {})
            text = content.get("text", "")

            if not record_id:
                print(
                    f"WARNING: Missing ID on line {line_number}"
                )
                continue

            if not text or not text.strip():
                print(
                    f"WARNING: Empty text for record {record_id}"
                )
                continue

            records.append(
                {
                    "id": record_id,
                    "text": text.strip(),
                }
            )

    print(f"Knowledge records: {len(records)}")

    return records


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

def load_eval_questions():
    print("\n" + "=" * 60)
    print("LOADING EVALUATION DATA")
    print("=" * 60)

    questions = []

    if not EVAL_FILE.exists():
        print("Evaluation file not found.")
        return questions

    with EVAL_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            questions.append(record)

    print(f"Evaluation questions: {len(questions)}")

    return questions


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(ids, embeddings):
    """
    Save current progress.

    IDs are stored as strings and embeddings as float32 to keep
    the checkpoint reasonably small.
    """

    np.savez_compressed(
        CHECKPOINT_FILE,
        ids=np.array(ids, dtype=str),
        embeddings=np.array(
            embeddings,
            dtype=np.float32,
        ),
    )


def load_checkpoint():
    """Load an existing checkpoint if available."""

    if not CHECKPOINT_FILE.exists():
        return [], []

    print("\nExisting checkpoint found.")
    print(f"Loading: {CHECKPOINT_FILE}")

    data = np.load(
        CHECKPOINT_FILE,
        allow_pickle=False,
    )

    ids = data["ids"].tolist()
    embeddings = data["embeddings"]

    embeddings = [
        vector
        for vector in embeddings
    ]

    print(f"Checkpoint records: {len(ids)}")

    return ids, embeddings


# ============================================================
# FINAL SAVE
# ============================================================

def save_final(ids, embeddings):
    np.savez_compressed(
        OUTPUT_FILE,
        ids=np.array(ids, dtype=str),
        embeddings=np.array(
            embeddings,
            dtype=np.float32,
        ),
    )

    print("\n" + "=" * 60)
    print("FINAL EMBEDDINGS SAVED")
    print("=" * 60)

    print(f"File: {OUTPUT_FILE}")
    print(f"Records: {len(ids)}")

    if embeddings:
        print(
            f"Dimensions: {len(embeddings[0])}"
        )


# ============================================================
# EMBED KNOWLEDGE BASE
# ============================================================

def embed_knowledge_base(records, working_keys):
    """
    Embed the entire knowledge base in batches.

    Key behavior:
    - 25 records per API request
    - checkpoint after every successful batch
    - automatically rotates keys on 429
    - removes failed keys from the active pool
    """

    checkpoint_ids, checkpoint_embeddings = load_checkpoint()

    processed_ids = set(checkpoint_ids)

    ids = checkpoint_ids
    embeddings = checkpoint_embeddings

    remaining_records = [
        record
        for record in records
        if record["id"] not in processed_ids
    ]

    print("\n" + "=" * 60)
    print("EMBEDDING KNOWLEDGE BASE")
    print("=" * 60)

    print(f"Total records: {len(records)}")
    print(f"Already embedded: {len(ids)}")
    print(f"Remaining: {len(remaining_records)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Active API keys: {len(working_keys)}")

    if not remaining_records:
        print("\nEverything is already embedded.")
        return ids, embeddings

    key_index = 0

    while remaining_records:

        if not working_keys:
            print("\nERROR: No working API keys remain.")
            print("Progress has been saved in the checkpoint.")
            raise RuntimeError(
                "All Gemini API keys are unavailable."
            )

        current_key = working_keys[key_index]

        batch = remaining_records[:BATCH_SIZE]

        batch_ids = [
            record["id"]
            for record in batch
        ]

        batch_texts = [
            record["text"]
            for record in batch
        ]

        current_position = len(ids) + 1
        total_records = len(records)

        print(
            f"\nEmbedding "
            f"{current_position}-{current_position + len(batch) - 1}"
            f"/{total_records} "
            f"using API key {current_key['number']}..."
        )

        try:
            batch_embeddings = embed_batch(
                current_key["client"],
                batch_texts,
            )

            ids.extend(batch_ids)
            embeddings.extend(batch_embeddings)

            # Remove successfully processed records.
            remaining_records = remaining_records[len(batch):]

            # Save checkpoint immediately.
            save_checkpoint(
                ids,
                embeddings,
            )

            print(
                f"✓ Batch complete "
                f"({len(ids)}/{total_records})"
            )

        except Exception as error:

            if is_quota_error(error):

                print(
                    f"⚠ API key {current_key['number']} "
                    f"hit quota."
                )

                # Remove exhausted key.
                working_keys.pop(key_index)

                if working_keys:
                    key_index %= len(working_keys)

                    print(
                        f"→ Switching to API key "
                        f"{working_keys[key_index]['number']}"
                    )

                # Retry the exact same batch.
                continue

            elif is_auth_error(error):

                print(
                    f"⚠ API key {current_key['number']} "
                    f"failed authentication."
                )

                working_keys.pop(key_index)

                if working_keys:
                    key_index %= len(working_keys)

                    print(
                        f"→ Switching to API key "
                        f"{working_keys[key_index]['number']}"
                    )

                continue

            else:

                print("\nERROR while embedding batch:")
                print(error)

                print(
                    "\nCheckpoint has been saved."
                )

                raise

        # Move to next key for normal rotation.
        if working_keys:
            key_index = (
                key_index + 1
            ) % len(working_keys)

    return ids, embeddings


# ============================================================
# EVALUATION
# ============================================================

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def evaluate_retrieval(
    records,
    ids,
    embeddings,
    eval_questions,
    working_keys,
):
    """
    Embed evaluation questions and calculate
    Top-1 / Top-3 / Top-5 retrieval accuracy.
    """

    if not eval_questions:
        print("\nNo evaluation questions found.")
        return

    print("\n" + "=" * 60)
    print("EVALUATING RETRIEVAL")
    print("=" * 60)

    # Map ID -> embedding index.
    id_to_index = {
        record_id: index
        for index, record_id in enumerate(ids)
    }

    knowledge_matrix = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    eval_texts = [
        item["question"]
        for item in eval_questions
    ]

    # Use one batch for all 30 evaluation questions.
    key_index = 0

    while True:

        if not working_keys:
            raise RuntimeError(
                "No Gemini API keys available for evaluation."
            )

        current_key = working_keys[key_index]

        print(
            f"Embedding {len(eval_texts)} evaluation questions "
            f"using API key {current_key['number']}..."
        )

        try:
            eval_embeddings = embed_batch(
                current_key["client"],
                eval_texts,
            )
            break

        except Exception as error:

            if is_quota_error(error) or is_auth_error(error):

                print(
                    f"⚠ API key {current_key['number']} "
                    f"unavailable."
                )

                working_keys.pop(key_index)

                if not working_keys:
                    raise RuntimeError(
                        "No API keys remain for evaluation."
                    )

                key_index %= len(working_keys)

            else:
                raise

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    results = []

    for eval_item, query_embedding in zip(
        eval_questions,
        eval_embeddings,
    ):

        similarities = np.array(
            [
                cosine_similarity(
                    query_embedding,
                    knowledge_vector,
                )
                for knowledge_vector in knowledge_matrix
            ],
            dtype=np.float32,
        )

        ranking = np.argsort(
            similarities
        )[::-1]

        top5_indices = ranking[:5]

        top5_ids = [
            ids[index]
            for index in top5_indices
        ]

        ground_truth = eval_item[
            "ground_truth_id"
        ]

        rank = None

        for position, record_id in enumerate(
            top5_ids,
            start=1,
        ):
            if record_id == ground_truth:
                rank = position
                break

        if rank == 1:
            top1_correct += 1

        if rank is not None and rank <= 3:
            top3_correct += 1

        if rank is not None and rank <= 5:
            top5_correct += 1

        results.append(
            {
                "eval_id": eval_item["eval_id"],
                "question": eval_item["question"],
                "ground_truth": ground_truth,
                "rank": rank,
                "top5": top5_ids,
                "difficulty": eval_item.get(
                    "difficulty",
                    "unknown",
                ),
                "language": eval_item.get(
                    "language",
                    "unknown",
                ),
            }
        )

    total = len(eval_questions)

    print("\n" + "-" * 60)
    print("RETRIEVAL RESULTS")
    print("-" * 60)

    print(
        f"Top-1 accuracy: "
        f"{top1_correct}/{total} "
        f"({top1_correct / total:.1%})"
    )

    print(
        f"Top-3 recall:   "
        f"{top3_correct}/{total} "
        f"({top3_correct / total:.1%})"
    )

    print(
        f"Top-5 recall:   "
        f"{top5_correct}/{total} "
        f"({top5_correct / total:.1%})"
    )

    # --------------------------------------------------------
    # Breakdown by difficulty
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("BREAKDOWN BY DIFFICULTY")
    print("-" * 60)

    difficulties = sorted(
        set(
            result["difficulty"]
            for result in results
        )
    )

    for difficulty in difficulties:

        subset = [
            result
            for result in results
            if result["difficulty"] == difficulty
        ]

        total_difficulty = len(subset)

        top1 = sum(
            result["rank"] == 1
            for result in subset
        )

        top3 = sum(
            result["rank"] is not None
            and result["rank"] <= 3
            for result in subset
        )

        top5 = sum(
            result["rank"] is not None
            and result["rank"] <= 5
            for result in subset
        )

        print(
            f"{difficulty}: "
            f"Top1 {top1}/{total_difficulty}, "
            f"Top3 {top3}/{total_difficulty}, "
            f"Top5 {top5}/{total_difficulty}"
        )

    # --------------------------------------------------------
    # Breakdown by language
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("BREAKDOWN BY LANGUAGE")
    print("-" * 60)

    languages = sorted(
        set(
            result["language"]
            for result in results
        )
    )

    for language in languages:

        subset = [
            result
            for result in results
            if result["language"] == language
        ]

        total_language = len(subset)

        top1 = sum(
            result["rank"] == 1
            for result in subset
        )

        top3 = sum(
            result["rank"] is not None
            and result["rank"] <= 3
            for result in subset
        )

        top5 = sum(
            result["rank"] is not None
            and result["rank"] <= 5
            for result in subset
        )

        print(
            f"{language}: "
            f"Top1 {top1}/{total_language}, "
            f"Top3 {top3}/{total_language}, "
            f"Top5 {top5}/{total_language}"
        )

    # --------------------------------------------------------
    # Show failures
    # --------------------------------------------------------

    failures = [
        result
        for result in results
        if result["rank"] is None
        or result["rank"] > 1
    ]

    print("\n" + "-" * 60)
    print(
        f"RETRIEVAL CASES NOT TOP-1: "
        f"{len(failures)}/{total}"
    )
    print("-" * 60)

    for result in failures:

        print(
            f"\n{result['eval_id']} "
            f"(difficulty={result['difficulty']}, "
            f"language={result['language']})"
        )

        print(
            f"Question: {result['question']}"
        )

        print(
            f"Ground truth: "
            f"{result['ground_truth']}"
        )

        print(
            f"Rank: {result['rank']}"
        )

        print(
            "Top 5:"
        )

        for position, record_id in enumerate(
            result["top5"],
            start=1,
        ):
            print(
                f"  {position}. {record_id}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("SECOND BRAIN — GEMINI EMBEDDING EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # API keys
    # --------------------------------------------------------

    api_keys = load_api_keys()

    if not api_keys:
        raise RuntimeError(
            "No GEMINI_API_KEY_1 ... GEMINI_API_KEY_5 "
            "found in .env"
        )

    print(
        f"\nConfigured API keys: {len(api_keys)}"
    )

    working_keys = test_api_keys(
        api_keys
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    records = load_knowledge_base()

    eval_questions = load_eval_questions()

    if not records:
        raise RuntimeError(
            "Knowledge base is empty."
        )

    # --------------------------------------------------------
    # Embed knowledge base
    # --------------------------------------------------------

    ids, embeddings = embed_knowledge_base(
        records,
        working_keys,
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if len(ids) != len(records):
        raise RuntimeError(
            f"Embedding count mismatch: "
            f"{len(ids)} embeddings for "
            f"{len(records)} records."
        )

    if len(embeddings) != len(records):
        raise RuntimeError(
            f"Vector count mismatch: "
            f"{len(embeddings)} embeddings for "
            f"{len(records)} records."
        )

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    save_final(
        ids,
        embeddings,
    )

    # --------------------------------------------------------
    # Remove checkpoint only after successful completion
    # --------------------------------------------------------

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

        print(
            f"\nCheckpoint removed: "
            f"{CHECKPOINT_FILE}"
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluate_retrieval(
        records,
        ids,
        embeddings,
        eval_questions,
        working_keys,
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Total time: "
        f"{elapsed / 60:.1f} minutes"
    )

    print(
        f"Embeddings: "
        f"{len(embeddings)}"
    )

    print(
        f"Dimensions: "
        f"{len(embeddings[0])}"
    )

    print(
        f"Output: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()