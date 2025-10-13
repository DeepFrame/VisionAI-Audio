import time, json, os
from pathlib import Path
import torch
import whisperx
from pydub import AudioSegment


def transcribe_audio(model_cfg, audio_dir, output_dir):
    model_variant = "whisperx"
    model_name = model_cfg["name"]        # e.g. "large-v2"
    framework = model_cfg.get("framework", "whisperx")
    device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    compute_type = model_cfg.get("compute_type", "float16" if device == "cuda" else "int8")

    # ✅ Log configuration
    print("\n========== MODEL CONFIGURATION ==========")
    print(f"Model Variant : {model_variant}")
    print(f"Framework     : {framework}")
    print(f"Model Name    : {model_name}")
    print(f"Device        : {device}")
    print(f"Compute Type  : {compute_type}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"WhisperX Ver. : {whisperx.__version__ if hasattr(whisperx, '__version__') else 'N/A'}")
    print("=========================================\n")

    os.makedirs(output_dir, exist_ok=True)
    results = []

    # 🔹 Load WhisperX model
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    for audio_file in Path(audio_dir).glob("*"):
        print(f"🎧 Processing: {audio_file.name}")
        audio = AudioSegment.from_file(audio_file)
        duration = len(audio) / 1000.0
        audio_data = whisperx.load_audio(str(audio_file))

        start = time.time()
        result = model.transcribe(audio_data)
        proc_time = time.time() - start

        # 🔹 Optional alignment
        print("⏱️ Aligning timestamps...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        aligned = whisperx.align(result["segments"], model_a, metadata, audio_data, device)

        text = " ".join(seg["text"].strip() for seg in aligned["segments"])
        rtf = proc_time / duration

        results.append({
            "variant": model_variant,
            "framework": framework,
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "file": audio_file.name,
            "duration_sec": round(duration, 2),
            "processing_time_sec": round(proc_time, 4),
            "rtf": round(rtf, 4),
            "language": result.get("language", "unknown"),
            "transcript": text,
        })

    # 🔹 Save all results
    json_path = Path(output_dir) / f"{model_variant}_{model_name}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Transcription complete for {model_variant} ({model_name})")
    print(f"📁 Results saved to: {json_path}")
    return json_path
