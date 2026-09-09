from faster_whisper import WhisperModel

VIDEO = "videos/test.mp4"

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(
    VIDEO,
    language=None,
    vad_filter=True
)

with open("transcripts/test.txt", "w", encoding="utf-8") as f:
    for segment in segments:
        f.write(segment.text.strip() + "\n")

print(f"Detected language: {info.language}")
print(f"Probability: {info.language_probability:.2f}")
print("Transcription complete.")
