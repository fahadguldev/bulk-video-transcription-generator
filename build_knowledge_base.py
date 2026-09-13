#!/usr/bin/env python3
"""Build knowledge_base/records.jsonl from all Second Brain data sources.

Sources (read-only, originals untouched):
  - transcripts/hinglish/cleaned/**: 119 usable video transcripts (one record each)
  - cleaned/tiktok-comments/comments/qa_pairs.json + qa_pairs_style_only.json
  - cleaned/tiktok-comments/dms/dm_pairs.json + dm_pairs_style_only.json
  - cleaned/linkedin/linkedin_posts.json

Excluded:
  - *_filtered.json (25 qa + 7 dm) -> DM-deflection replies ("check dm"), no content
  - transcripts rejected in the cleaned/ pass (never reached this folder)

Rules honored:
  - no modifications to raw files; no translation; wording preserved verbatim
  - question -> response preserved as separate fields on the same record
  - duplicate transcripts (near-identical normalized text) and duplicate
    LinkedIn posts removed (first occurrence kept)
  - empty / whitespace-only records dropped
"""
import json
import os
import re
import hashlib
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "knowledge_base", "records.jsonl")

TRANSCRIPTS = os.path.join(ROOT, "transcripts", "hinglish", "cleaned")
CLEANED = os.path.join(ROOT, "cleaned")

SUB_FOLDERS = ["aws", "books", "game", "questions", "questions-2", "random"]

DEPTH_0 = re.compile("[" + "\u0900-\u097F" + "]")   # Devanagari
ARABIC = re.compile("[\u0600-\u06FF\u0750-\u077F]")
QUOTES = re.compile(r"\s+")
EMOJI_ONLY = re.compile(r"^[\W\d\s\u00a9-\u329f]*$")

HINGLISH_MARKERS = {
    "kya", "hai", "nahi", "nhi", "mein", "eain", "main", "bhai", "kar", "ka", "ki", "ke",
    "ho", "hain", "aap", "tum", "diya", "tha", "the", "hoga", "hoon", "ypua", "ma", "ko",
    "se", "ab", "to", "bhi", "krske", "chahiye", "chahye", "chahiay", "ja", "ga", "raha",
    "rahy", "rahay", "rahe", "ay", "haiya", "dono", "sath", "saath", "milke", "apna",
    "apni", "mere", "mera", "meri", "log", "batch", "krna", "karna", "karo", "krta",
    "karta", "krte", "sakta", "sakti", "sakte", "bna", "bana", "asha", "umeed", "zarur",
    "zyada", "zyaada", "kam", "aqalmand", "samajh", "samjh", "beter", "bahter", "sakhti",
    "shuru", "khud", "dia", "dijiye", "kijiye", "krlo", "denge", "milta", "milte",
}
HINGLISH_IGNORE = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "it", "for"}

TOPIC_LEXICON = {
    "python": ["python"], "java": ["java"], "c++": ["c\\+\\+"], "csharp": ["c#", "csharp", ".net"],
    "aws": ["aws", "amazon web"], "cloud": ["cloud", "serverless", "ec2", "s3", "lambda"],
    "cybersecurity": ["cyber", "security", "hacking", "ethical"], "ai": ["artificial intelligence", "\bai\b"],
    "ml": ["machine learning"], "datascience": ["data science", "datascience", "data sci"] ,
    "webdev": ["web dev", "webdev", "frontend", "backend", "front end", "back end", "html", "css", "react"],
    "gamedev": ["unity", "game dev", "gamedev", "game object", "unreal"],
    "networking": ["network", "networking"], "database": ["database", "sql", "mysql", "mongodb"],
    "dba": ["dbms", "database admin"], "csa": ["computer science"], "software-engineering": ["software engineering", "software dev", "software developer"],
    "career": ["career", "salary", "job", "jobs", "scope", "market"], "university": ["university", "nust",
        "fast", "lums", "luw", "numerical", "college"], "education": ["degree", "semester", "cgpa", "gpa",
        "fsc", "ics", "matric", "12th", "student", "admission", "fee"],
    "devops": ["devops", "docker", "kubernetes", "ci/cd", "container"], "certification": ["certification", "cert"],
    "interview": ["interview"], "gaming": ["gaming", "laptop", "pc build"], "linux": ["linux", "ubuntu"],
}

STYLE_KEYWORDS = ["happy", "great", "excited", "proud", "hamdulillah", "alhamdulillah", "inshallah", "insha'allah",
                  "congrats", "wow", "amazing", "awesome", "love", "best", "progress", "ready", "available"]
OPINION_PATTERNS = [
    r"\b(i|my|mere|mera)\b.*\b(think|believe|say|advice|suggest|recommend)\b",
    r"\bin my (opinion|view|experience)\b",
    r"\bi (would|will) (recommend|suggest|advise)\b",
    r"\brather\b", r"\bpersonally\b", r"\bmy advice is\b",
    r"\breally (should|must|need to)\b", r"\bbetter (to|if)\b",
]
EXPERIENCE_PATTERNS = [
    r"\bwhen i\b", r"\bi (started|did|worked|earned|studied|took|completed|made|learned|start)\b",
    r"\bmy (gpa|semester|degree|experience|first year|journey)\b", r"\bi am a\b", r"\bi am (doing|studying)\b",
    r"\bmain ne\b", r"\bmaine\b", r"\bma ne\b", r"\bi am in\b", r"\bim in\b", r"\bone of the first\b",
]


def slugify(name):
    return name.replace(".txt", "").replace(" ", "-").replace("/", "-")


def detect_language(text):
    if not text or not text.strip():
        return None
    if DEPTH_0.search(text):
        return "hindi"
    if ARABIC.search(text):
        return "urdu"
    if EMOJI_ONLY.match(text.strip()):
        return None
    tokens = [t.lower() for t in re.findall(r"[\w'\u2019]+", text)]
    if not tokens:
        return None
    sig = [t for t in tokens if t not in HINGLISH_IGNORE]
    hits = sum(1 for t in sig if t in HINGLISH_MARKERS)
    return "hinglish" if hits >= 2 else "english"


def make_topics(text, filename=""):
    topics = []
    low = text.lower()
    for topic, pats in TOPIC_LEXICON.items():
        if any(re.search(re.compile(p), low) for p in pats):
            topics.append(topic)
    if not topics and filename:
        name = slugify(filename)
        for topic in list(TOPIC_LEXICON.keys()):
            if topic in name:
                topics.append(topic)
    return topics


def infer_booleans(text):
    low = text.lower()
    opinion = bool(re.search(r"(?:%s)" % "|".join(OPINION_PATTERNS), low))
    experience = bool(re.search(r"(?:%s)" % "|".join(EXPERIENCE_PATTERNS), low))
    return experience, opinion


def norm_time(value):
    if not value:
        return None
    s = str(value).strip()
    if s.isdigit():
        try:
            return datetime.fromtimestamp(int(s), tz=timezone.utc).date().isoformat()
        except (ValueError, OSError):
            return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None


def norm_key(text):
    return re.sub(r"\s+", " ", text.lower().strip())


def emit(records, seen, rec_id, source_type, source_file, date, text, language,
         question=None, extra_source=None, ai_assisted=False,
         knowledge=True, style=True):
    if text is None or not text.strip():
        return
    key = norm_key(text)
    if question:
        key = norm_key(question) + " || " + key
    if not key or key in seen:
        return
    seen.add(key)
    qlang = detect_language(question) if question else None
    relang = language if language is not None else detect_language(text)
    exp, opi = infer_booleans(text)
    rec = {
        "id": rec_id,
        "source": {
            "type": source_type,
            "file": source_file,
        },
        "content": {
            "text": text,
            "language": relang,
        },
        "classification": {
            "domain": "professional",
            "topics": make_topics(text + " " + (question or "")),
            "knowledge": knowledge,
            "experience": exp,
            "opinion": opi,
            "style": style,
            "ai_assisted": ai_assisted,
        },
    }
    if date:
        rec["source"]["date"] = date
    if extra_source:
        rec["source"].update(extra_source)
    if question:
        rec["content"]["question"] = question
        rec["content"]["question_language"] = qlang
    records.append(rec)


def main():
    records = []
    seen = set()
    counter = [0]

    def next_id(prefix):
        counter[0] += 1
        return f"{prefix}.{counter[0]}"

    # ---- transcripts: one record per file ----
    for folder in SUB_FOLDERS:
        d = os.path.join(TRANSCRIPTS, folder)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".txt"):
                continue
            path = os.path.join(d, name)
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            emit(records, seen, next_id(f"video_transcript.{folder}"), "video_transcript",
                 os.path.join("transcripts", "hinglish", "cleaned", folder, name),
                 mtime_date(path), text,
                 language=detect_language(text),
                 style=True, ai_assisted=False,
                 extra_source={"folder": folder, "file_name": name})

    # ---- tiktok comment replies ----
    for fname, prefix in [("qa_pairs.json", "comment_reply"),
                          ("qa_pairs_style_only.json", "comment_reply_style")]:
        p = os.path.join(CLEANED, "tiktok-comments", "comments", fname)
        data = json.load(open(p, encoding="utf-8"))
        for x in data:
            emit(records, seen, next_id(prefix), "tiktok_comment_reply",
                 os.path.join("cleaned", "tiktok-comments", "comments", fname),
                 norm_time(x.get("date")), x["reply_text"],
                 language=detect_language(x["reply_text"]),
                 question=x.get("question_text"),
                 ai_assisted=bool(x.get("ai_assisted")),
                 extra_source={"url": x.get("video_url"), "video_id": x.get("video_id"),
                               "likes": x.get("likes"), "value_type": x.get("value_type")})

    # ---- tiktok dms ----
    for fname, prefix in [("dm_pairs.json", "tiktok_dm"),
                          ("dm_pairs_style_only.json", "tiktok_dm_style")]:
        p = os.path.join(CLEANED, "tiktok-comments", "dms", fname)
        data = json.load(open(p, encoding="utf-8"))
        for x in data:
            emit(records, seen, next_id(prefix), "tiktok_dm",
                 os.path.join("cleaned", "tiktok-comments", "dms", fname),
                 norm_time(x.get("date")), x["reply_text"],
                 language=detect_language(x["reply_text"]),
                 question=x.get("question_text"),
                 ai_assisted=bool(x.get("ai_assisted")),
                 extra_source={"value_type": x.get("value_type")})

    # ---- linkedin posts ----
    p = os.path.join(CLEANED, "linkedin", "linkedin_posts.json")
    data = json.load(open(p, encoding="utf-8"))
    for x in data:
        emit(records, seen, next_id("linkedin_post"), "linkedin_post",
             os.path.join("cleaned", "linkedin", "linkedin_posts.json"),
             norm_time(x.get("date")), x["text"],
             language=detect_language(x["text"]),
             ai_assisted=True, style=False,
             extra_source={"url": x.get("share_link")})

    os.makedirs(os.path.join(ROOT, "knowledge_base"), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} records -> {os.path.relpath(OUT, ROOT)}")
    from collections import Counter
    print(Counter(r["source"]["type"] for r in records))


def mtime_date(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).date().isoformat()
    except OSError:
        return None


if __name__ == "__main__":
    main()