#!/usr/bin/env python3
"""Prepare cleaned JSONL Second Brain data for RAG.

Reads knowledge_base/records.jsonl (unchanged) and writes
knowledge_base/rag_records.jsonl with one line per record:

  {"id": ..., "text": <searchable representation>, "metadata": {<all other fields>}}

Searchable text per source type (verbatim, no rewriting/summarizing):
  - video_transcript      -> transcript text
  - tiktok_comment/tiktok_dm -> question + response (separate lines)
  - linkedin_post         -> post text

`metadata` preserves everything else from the original record (source,
classification, language, question) so it can be used for filtering and
retrieval display, separate from the embedded text.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "knowledge_base", "records.jsonl")
OUT = os.path.join(ROOT, "knowledge_base", "rag_records.jsonl")

PAIR_TYPES = {"tiktok_comment_reply", "tiktok_dm"}


def searchable_text(record):
    stype = record["source"]["type"]
    content = record["content"]
    if stype == "video_transcript":
        return content["text"]
    if stype in PAIR_TYPES:
        q = content.get("question")
        r = content["text"]
        return (q + "\n" + r) if q else r
    if stype == "linkedin_post":
        return content["text"]
    return content["text"]


def metadata_of(record):
    content = record["content"]
    meta = {
        "source": record["source"],
        "classification": record["classification"],
        "language": content.get("language"),
    }
    if content.get("question") is not None:
        meta["question"] = content["question"]           # original wording
        meta["question_language"] = content.get("question_language")
    return meta


def main():
    records = [json.loads(line) for line in open(SRC, encoding="utf-8") if line.strip()]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for rec in records:
            r = {"id": rec["id"],
                 "text": searchable_text(rec),
                 "metadata": metadata_of(rec)}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    empties = [r for r in records if not searchable_text(r).strip()]
    print(f"wrote {len(records)} records -> knowledge_base/rag_records.jsonl")
    print(f"empty-text records: {len(empties)}")


if __name__ == "__main__":
    main()