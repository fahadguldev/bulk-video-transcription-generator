#!/usr/bin/env python3
"""Build knowledge_base/eval_retrieval.jsonl — real-query retrieval eval set.

Each item is a REAL question/comment/DM from the Second Brain data (verbatim,
never generated or paraphrased) paired with the records.jsonl id of the record
containing the creator's relevant previous response (ground truth).

Output schema (one item per line):
  eval_id, question, ground_truth_id, expected_topic, domain,
  language, difficulty, why_relevant
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "knowledge_base", "records.jsonl")
OUT = os.path.join(ROOT, "knowledge_base", "eval_retrieval.jsonl")

# (record_id, expected_topic, domain, language, difficulty, why_relevant)
CURATED = [
    # ---- easy: direct keyword-to-topic matches ----
    ("comment_reply.439", "networking career for CS students", "professional", "english", "easy",
     "Question explicitly names networking and matches the record where the creator encouraged a 6th-semester CS student to pursue it."),
    ("comment_reply.397", "game development career", "professional", "english", "easy",
     "Direct 'game developer' question; the ground-truth record is the creator's only substantive response on game dev as a career."),
    ("comment_reply.301", "programming language before university (python vs C++)", "professional", "hinglish", "easy",
     "Asks whether Python or C++ should be learned before university; the record contains the creator's direct answer on this."),
    ("comment_reply.524", "BS Computer Science vs BS Data Science", "professional", "hinglish", "easy",
     "The question is a CS-vs-Data-Science degree choice; ground truth is the record where the creator recommended CS as broader."),
    ("comment_reply.290", "AWS prerequisites", "professional", "english", "easy",
     "'Prerequisites to learn AWS' maps directly to the record where the creator answered that AWS has no real prerequisites."),
    # ---- medium: paraphrased / multi-topic, still keyword-linked ----
    ("comment_reply.355", "cloud computing for beginners and jobs", "professional", "english", "medium",
     "Cloud + beginner + job keywords appear and the record directly addresses whether a beginner can learn cloud computing and get a job."),
    ("comment_reply.1002", "cloud computing roadmap", "professional", "english", "medium",
     "'Proper road map for cloud computing' is a stated goal; the ground truth is the record advising Linux, networking, then AWS/Azure/GCP."),
    ("comment_reply.299", "cybersecurity vs data science future-proofing", "professional", "english", "medium",
     "Compares two named fields for future-proofing; the record gives the interest-based decision, matching the comparison."),
    ("comment_reply.364", "cybersecurity: university vs academy certification", "professional", "english", "medium",
     "Question is university-vs-academy path for cyber; record answers with university-first advice, same concept, slightly different words."),
    ("comment_reply.825", "self-learning cybersecurity at home (TryHackMe/HackTheBox)", "professional", "hinglish", "medium",
     "Asks if cyber security can be self-learned using platforms; record confirms with home-learning + certification guidance."),
    ("comment_reply.318", "UET vs PU for CS undergraduate", "professional", "english", "medium",
     "University comparison; record is the creator's UET-vs-PU assessment for CS, matching despite no repetition of the exact wording."),
    ("comment_reply.136", "career direction after high CGPA credentials", "professional", "english", "medium",
     "Multi-CGPA background question; the record is the creator's only direct 'what should I do now' career guidance response."),
    ("comment_reply.281", "future of software engineering jobs", "professional", "hinglish", "medium",
     "Hinglish phrasing about SE jobs ending; record is the creator's reassurance that SE won't disappear though jobs shrink."),
    ("comment_reply.346", "field choice after 2nd year while learning JavaScript", "professional", "english", "medium",
     "Long contextual question; ground truth is the record advising full-stack web dev + projects for someone learning JavaScript."),
    ("comment_reply.414", "software engineering at age 15 (future field)", "professional", "english", "medium",
     "Teenage career-choice question; record is the creator's response weighing AI impact on CS/SE roles for the same user."),
    ("tiktok_dm.1899", "CS student pivoting away from coding", "professional", "english", "medium",
     "DM asking for direction as a CS student who dislikes coding; record is the creator's reply offering a no-coding path."),
    ("tiktok_dm.2559", "MERN developer without software-house job", "professional", "english", "medium",
     "Experience-based interview/job question; ground truth is the creator's interview-prep answer for the same situation."),
    ("tiktok_dm.2214", "best Pakistani universities for BS admission", "professional", "urdu", "medium",
     "Urdu-script university-ranking question; the record lists the creator's private/govt university rankings."),
    ("tiktok_dm.2065", "BS CS admission feasibility from marks", "professional", "mixed", "medium",
     "Mixed Urdu-script + English marks question; ground truth record is the creator's admission-feasibility exchange."),
    # ---- hard: correct answer uses different wording / no keyword overlap ----
    ("comment_reply.288", "transition from CCNA/MCSE networking background to AWS", "professional", "hinglish", "hard",
     "No 'AWS learning path' keywords; correct context is the creator noting this exact networking-cert-to-AWS question."),
    ("comment_reply.291", "Linux first vs directly starting AWS", "professional", "hinglish", "hard",
     "Wording is a Linux-vs-AWS choice; correct record answers with Linux basics implicit in starting AWS — paraphrased response."),
    ("comment_reply.428", "why Docker containers are lightweight", "professional", "english", "hard",
     "Question only says 'lightweight' without naming Docker; ground truth is the Docker resource/architecture explanation."),
    ("comment_reply.543", "future-proof field that AI won't replace", "professional", "english", "hard",
     "'Won't be replaced by AI' ≠ a named topic; correct record recommends a specific degree in response, requiring semantic mapping."),
    ("comment_reply.370", "switching from frontend to cybersecurity due to AI", "professional", "hinglish", "hard",
     "The topic 'cybersecurity' is implied by switching away from frontend; ground truth confirms cyber less AI-impacted."),
    ("comment_reply.474", "non-coding skill to learn", "professional", "english", "hard",
     "Very short and concept-based (no coding keyword); correct record lists no-coding career domains like cloud/QA/business."),
    ("comment_reply.1001", "can a pre-medical student do AI", "professional", "hinglish", "hard",
     "Topic is eligibility of a pre-medical background for AI; correct record discusses AI saturation & feasibility for such students."),
    ("comment_reply.431", "AWS scope in Pakistan", "professional", "hinglish", "hard",
     "'Scope' requires mapping to job roles; correct record names cloud architect/devops as the AWS career answer."),
    ("tiktok_dm.2787", "vague future-securing direction", "professional", "english", "hard",
     "Open-ended 'secure my future' statement; correct record is the creator's multi-path guidance, no topic keywords."),
    ("tiktok_dm.1745", "learning Python via courses with zero CS background", "professional", "hinglish", "hard",
     "Asks about course route rather than naming outcomes; ground truth is the creator's whole-year self-learning advice."),
    ("tiktok_dm.2145", "creator's own study background and university", "professional", "hinglish", "hard",
     "Personal identity question about the creator; only the profile-description record is the correct retrieval target."),
]

TRUTH_TYPES = {"tiktok_comment_reply", "tiktok_dm", "video_transcript", "linkedin_post"}


def main():
    records = {r["id"]: r for r in (json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip())}
    items = []
    used = set()
    n = 0
    for rid, topic, domain, lang, diff, why in CURATED:
        rec = records.get(rid)
        if not rec:
            print("MISSING", rid)
            continue
        q = rec["content"].get("question")
        if not q:
            print("NO QUESTION", rid)
            continue
        if q in used:
            print("DUP QUESTION", rid)
            continue
        used.add(q)
        n += 1
        items.append({
            "eval_id": f"ev_{n:03d}",
            "question": q,
            "ground_truth_id": rid,
            "expected_topic": topic,
            "domain": domain,
            "language": lang,
            "difficulty": diff,
            "why_relevant": why,
        })
    with open(OUT, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(items)} eval items -> knowledge_base/eval_retrieval.jsonl")
    print("difficulty:", Counter(it["difficulty"] for it in items))
    print("language:", Counter(it["language"] for it in items))
    print("ground-truth source types:", Counter(records[it["ground_truth_id"]]["source"]["type"] for it in items))
    print("topics:", len({it["expected_topic"] for it in items}), "distinct topics")


if __name__ == "__main__":
    main()