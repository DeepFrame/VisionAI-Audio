import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")
import yaml
from pathlib import Path
from models.faster_whisper.faster_whisper_model import transcribe_audio as faster_whisper_run
from models.openai_whisper.openai_whisper_model import transcribe_audio as openai_whisper_run
from models.whisper_cpp.whisper_cpp_model import transcribe_audio as whisper_cpp_run   # 👈 add this

BASE_DIR = Path(__file__).resolve().parent.parent
config_path = BASE_DIR / "models" / "model_configs.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

AUDIO_DIR = BASE_DIR / "tests" / "test_audio_samples"
OUTPUT_DIR = BASE_DIR / "results" / "reports"

# Run each model
print("\n🚀 Running Faster-Whisper...")
faster_output = faster_whisper_run(config["models"]["faster-whisper"], AUDIO_DIR, OUTPUT_DIR)

print("\n🚀 Running OpenAI Whisper...")
openai_output = openai_whisper_run(config["models"]["openai-whisper"], AUDIO_DIR, OUTPUT_DIR)

# print("\n🚀 Running Whisper.cpp...")
# cpp_output = whisper_cpp_run(config["models"]["whisper-cpp"], AUDIO_DIR, OUTPUT_DIR)

print("\n✅ All models completed!")
print(f"📁 Reports saved to: {OUTPUT_DIR}")
