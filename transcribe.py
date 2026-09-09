# from faster_whisper import WhisperModel

# VIDEO = "videos/test.mp4"

# model = WhisperModel(
#     "small",
#     device="cpu",
#     compute_type="int8"
# )

# segments, info = model.transcribe(
#     VIDEO,
#     language=None,
#     vad_filter=True
# )

# with open("transcripts/test.txt", "w", encoding="utf-8") as f:
#     for segment in segments:
#         f.write(segment.text.strip() + "\n")

# print(f"Detected language: {info.language}")
# print(f"Probability: {info.language_probability:.2f}")
# print("Transcription complete.")



import os
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")

os.makedirs("transcripts", exist_ok=True)

for filename in os.listdir("videos"):
    if not filename.lower().endswith((".mp4", ".mov")):
        continue

    video_path = os.path.join("videos", filename)
    out_path = os.path.join("transcripts", os.path.splitext(filename)[0] + ".txt")

    if os.path.exists(out_path):
        print(f"Skipping (already done): {filename}")
        continue

    print(f"Transcribing: {filename}")
    segments, info = model.transcribe(video_path, language=None, vad_filter=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment.text.strip() + "\n")

    print(f"  -> {info.language} ({info.language_probability:.2f})")

print("All done.")