# code/model_comparison.py
import os, time, json
from faster_whisper import WhisperModel
from pathlib import Path
import yaml

# ---------------------- CONFIG ----------------------
# Base directory = project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load model config
config_path = BASE_DIR / "models" / "model_configs.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_cfg = config["models"]["faster-whisper"]
model = WhisperModel(model_cfg["name"], device=model_cfg["device"])

# Paths
AUDIO_DIR = BASE_DIR / "tests" / "test_audio_samples"
OUTPUT_DIR = BASE_DIR / "results" / "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []

# ---------------------- RUN TRANSCRIPTION ----------------------
for audio_file in AUDIO_DIR.glob("*"):
    print(f"🎧 Processing: {audio_file.name}")
    start = time.time()
    segments, info = model.transcribe(str(audio_file))
    duration = time.time() - start

    text = " ".join([seg.text for seg in segments]).strip()
    result = {
        "model": model_cfg["name"],
        "file": audio_file.name,
        "duration": info.duration,
        "processing_time": duration,
        "transcript": text
    }
    results.append(result)

    # Save human-readable transcript
    txt_path = OUTPUT_DIR / f"{audio_file.stem}_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"   📄 Saved transcript: {txt_path.name}")

# ---------------------- SAVE JSON FOR METRICS ----------------------
output_path = OUTPUT_DIR / "faster-whisper_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Transcription complete!")
print(f"📁 JSON report saved to: {output_path}")
print(f"📂 Individual transcripts saved in: {OUTPUT_DIR}")
