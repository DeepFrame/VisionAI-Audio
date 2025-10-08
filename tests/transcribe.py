import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

import argparse
from faster_whisper import WhisperModel

parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True)
parser.add_argument("--out", default="results/transcript.txt")
args = parser.parse_args()

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe(args.audio, beam_size=15)

with open(args.out, "w", encoding="utf-8") as f:
    for seg in segments:
        f.write(seg.text.strip() + " ")
print(f"✅ Transcription saved to {args.out}")
