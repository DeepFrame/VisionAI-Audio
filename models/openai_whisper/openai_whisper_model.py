import whisper, time, json, os
from pathlib import Path
from pydub import AudioSegment
import torch
from code.resource_monitor import ResourceMonitor

def transcribe_audio(model_cfg, audio_dir, output_dir):
    model_variant = "openai-whisper"
    model_name = model_cfg["name"]
    framework = model_cfg["framework"]
    device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"

    print("\n========== MODEL CONFIGURATION ==========")
    print(f"Model Variant : {model_variant}")
    print(f"Framework     : {framework}")
    print(f"Model Name    : {model_name}")
    print(f"Device        : {device}")
    print(f"Compute Type  : {compute_type}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("=========================================\n")

    os.makedirs(output_dir, exist_ok=True)
    model = whisper.load_model(model_name, device=device)
    results = []

    for audio_file in Path(audio_dir).glob("*"):
        print(f"🎧 Processing: {audio_file.name}")
        audio = AudioSegment.from_file(audio_file)
        duration = len(audio) / 1000.0

        monitor = ResourceMonitor(interval=1)
        monitor.start()

        start = time.time()
        result = model.transcribe(str(audio_file))
        proc_time = time.time() - start

        monitor.stop()
        resource_stats = monitor.get_summary()

        text = result["text"].strip()
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
            **resource_stats  # 👈 adds CPU, RAM, GPU stats
        })
        print(f"✅ Done: {audio_file.name} | Time: {proc_time:.2f}s | CPU: {resource_stats['avg_cpu']:.1f}% | GPU: {resource_stats.get('avg_gpu', 0):.1f}%")

    json_path = Path(output_dir) / f"{model_variant}_{model_name}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Transcription complete for {model_variant} ({model_name})")
    print(f"📁 Results saved to: {json_path}")
    return json_path
