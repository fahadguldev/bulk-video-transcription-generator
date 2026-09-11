import json
import glob
import os
import re

CREATOR_UNIQUE_ID = "fahadgul.dev"
INPUT_GLOB = "comments/*.json"  # folder where you drop all your exported JSON files
OUTPUT_FILE = "qa_pairs.json"
STYLE_ONLY_FILE = "qa_pairs_style_only.json"
LOW_VALUE_FILE = "qa_pairs_filtered.json"

MIN_REPLY_LENGTH = 15  # characters, only applies to potential "content" pairs
LOW_VALUE_PATTERNS = [
    r"^\s*(chk|check)\s*dm\s*$",
    r"^\s*dm\s*me\s*$",
    r"^\s*plz\s*dm\s*me\s*$",
]

EMOJI_ONLY_PATTERN = re.compile(
    r"^[\s\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF✨️️♀♂]+$"
)


def is_emoji_only(text):
    stripped = text.strip()
    return bool(stripped) and bool(EMOJI_ONLY_PATTERN.match(stripped))


def classify_reply(text):
    """
    Returns one of: 'content' (knowledge + style), 'style_only' (style signal,
    no real knowledge, e.g. emoji reactions), 'filtered' (not useful for either).
    """
    stripped = text.strip()

    if is_emoji_only(stripped):
        return "style_only"

    for pattern in LOW_VALUE_PATTERNS:
        if re.match(pattern, stripped, flags=re.IGNORECASE):
            return "filtered"

    if len(stripped) < MIN_REPLY_LENGTH:
        # short but not emoji and not a dm-deflection -> could still carry
        # style (e.g. "thanks dear!", "helooo") even if content value is low
        return "style_only"

    return "content"


def find_node_by_id(nodes, target_id):
    """Search a list of comment/reply nodes (and their nested replies) for a comment_id match."""
    for node in nodes:
        if node.get("comment_id") == target_id:
            return node
        found = find_node_by_id(node.get("replies", []), target_id)
        if found:
            return found
    return None


def walk_thread(node, parent_text, parent_author, all_top_level, rows, video_id, video_url):
    """
    Recursively walk a comment/reply tree.
    Whenever we find a node authored by the creator, pair it with the thing
    it was actually replying to (parent_text), then continue walking deeper.
    """
    author = node.get("author", {})
    is_creator = node.get("is_creator", False)
    text = node.get("text", "")

    # Determine what this node was actually replying to
    reply_to_id = node.get("reply_to_reply_id")
    actual_parent_text = parent_text
    actual_parent_author = parent_author

    if reply_to_id and reply_to_id != "0":
        # It's replying to a specific reply deeper in the thread, not the top-level parent
        target = find_node_by_id(all_top_level, reply_to_id)
        if target:
            actual_parent_text = target.get("text", parent_text)
            actual_parent_author = target.get("author", {}).get("unique_id", parent_author)

    if is_creator and actual_parent_text:
        rows.append({
            "video_id": video_id,
            "video_url": video_url,
            "question_text": actual_parent_text,
            "question_author": actual_parent_author,
            "reply_text": text,
            "reply_comment_id": node.get("comment_id", ""),
            "date": node.get("created_at", ""),
            "likes": node.get("likes", 0),
            "source_type": "tiktok_comment_reply",
            "ai_assisted": False,
        })

    # Recurse into this node's own replies, using this node as the new "parent" context
    for child in node.get("replies", []):
        walk_thread(child, text, author.get("unique_id", ""), all_top_level, rows, video_id, video_url)


def process_file(filepath, rows):
    with open(filepath, encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "comments" not in data:
        print(f"  Skipping (not a comment export): {filepath}")
        return

    video_id = data.get("video_id", "")
    video_url = data.get("video_url", "")
    top_level_comments = data.get("comments", [])

    for comment in top_level_comments:
        # Top-level comments have no reply_to_reply_id; their "parent" is the question itself
        walk_thread(
            comment,
            parent_text=None,
            parent_author=None,
            all_top_level=top_level_comments,
            rows=rows,
            video_id=video_id,
            video_url=video_url,
        )


def main():
    files = glob.glob(INPUT_GLOB)
    if not files:
        # fallback: also allow a single file passed via this same folder for testing
        files = glob.glob("*.json")

    print(f"Found {len(files)} JSON export files.")
    rows = []
    for filepath in files:
        print(f"Processing: {filepath}")
        process_file(filepath, rows)

    content_rows = []
    style_only_rows = []
    filtered_rows = []

    for r in rows:
        category = classify_reply(r["reply_text"])
        r["value_type"] = category
        r["domain"] = "professional"  # default; re-tag to 'personal' manually where relevant
        r["usage"] = "content_and_style" if category == "content" else (
            "style_only" if category == "style_only" else "excluded"
        )
        if category == "content":
            content_rows.append(r)
        elif category == "style_only":
            style_only_rows.append(r)
        else:
            filtered_rows.append(r)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(content_rows, f, ensure_ascii=False, indent=2)

    with open(STYLE_ONLY_FILE, "w", encoding="utf-8") as f:
        json.dump(style_only_rows, f, ensure_ascii=False, indent=2)

    with open(LOW_VALUE_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered_rows, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(rows)} total pairs.")
    print(f"  -> {len(content_rows)} content pairs (knowledge + style) saved to {OUTPUT_FILE}")
    print(f"  -> {len(style_only_rows)} style-only pairs (emoji/short reactions) saved to {STYLE_ONLY_FILE}")
    print(f"  -> {len(filtered_rows)} filtered pairs (dm-deflections, not useful) saved to {LOW_VALUE_FILE}")


if __name__ == "__main__":
    main()