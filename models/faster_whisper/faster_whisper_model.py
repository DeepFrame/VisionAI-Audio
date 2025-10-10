import time, json, os
from pathlib import Path
from faster_whisper import WhisperModel
from pydub import AudioSegment


def transcribe_audio(model_cfg, audio_dir, output_dir):
    model_variant = "faster-whisper"
    model_name = model_cfg["name"]
    framework = model_cfg["framework"]
    device = model_cfg["device"]

    print(f"\n🚀 Running {model_variant} ({framework}, model={model_name}, device={device})")

    model = WhisperModel(model_name, device=device)
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for audio_file in Path(audio_dir).glob("*"):
        print(f"🎧 Processing: {audio_file.name}")
        audio = AudioSegment.from_file(audio_file)
        duration = len(audio) / 1000.0

        start = time.time()
        segments, info = model.transcribe(str(audio_file))
        proc_time = time.time() - start
        text = " ".join([seg.text for seg in segments])
        rtf = proc_time / duration

        results.append({
            "variant": model_variant,
            "framework": framework,
            "model": model_name,
            "device": device,
            "file": audio_file.name,
            "duration": round(duration, 4),
            "processing_time": round(proc_time, 6),
            "rtf": round(rtf, 4),
            "transcript": text
        })

    json_path = Path(output_dir) / f"{model_variant}_{model_name}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Transcription complete for {model_variant} ({model_name})")
    print(f"📁 Results saved to: {json_path}")
    return json_path
