import time
import json
import os
from pathlib import Path
from pywhispercpp.model import Model
from pydub import AudioSegment
import tempfile
from code.resource_monitor import ResourceMonitor

# --- 🔧 Helper: Ensure audio is 16-bit PCM, mono, 16kHz for Whisper.cpp ---
def prepare_audio(input_path, target_sr=16000):
    """Convert audio to 16-bit PCM mono WAV (required by Whisper.cpp)."""
    audio = AudioSegment.from_file(input_path)
    original_info = {
        "frame_rate": audio.frame_rate,
        "channels": audio.channels,
        "sample_width": audio.sample_width * 8  # in bits
    }

    print(f"🔍 Original audio info: {original_info}")

    # Ensure correct format
    audio = audio.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")

    print(f"✅ Converted to 16-bit PCM mono @ {target_sr}Hz -> {tmp.name}")
    return tmp.name, original_info


# --- 🚀 Main transcription function ---
def transcribe_audio(model_cfg, audio_dir, output_dir):
    model_variant = "whisper-cpp"
    model_name = model_cfg["name"]
    framework = model_cfg["framework"]
    device = model_cfg.get("device", "cpu")
    compute_type = "int8"

    print("\n========== MODEL CONFIGURATION ==========")
    print(f"Model Variant : {model_variant}")
    print(f"Framework     : {framework}")
    print(f"Model Name    : {model_name}")
    print(f"Device        : {device}")
    print(f"Compute Type  : {compute_type}")
    print("=========================================\n")

    os.makedirs(output_dir, exist_ok=True)
    model = Model(model_name)
    results = []

    for audio_file in Path(audio_dir).glob("*"):
        print(f"\n🎧 Processing: {audio_file.name}")

        # --- Prepare and log audio info ---
        input_path, original_info = prepare_audio(str(audio_file))

        # --- Duration calculation ---
        audio = AudioSegment.from_file(input_path)
        duration = len(audio) / 1000.0

        monitor = ResourceMonitor(interval=1)
        monitor.start()

        # --- Run Whisper.cpp transcription ---
        start = time.time()
        segments = model.transcribe(input_path)
        proc_time = time.time() - start

        monitor.stop()
        resource_stats = monitor.get_summary()

        # --- Merge all text segments ---
        text = " ".join([seg.text for seg in segments])
        rtf = proc_time / duration if duration > 0 else 0

        # --- Append results ---
        results.append({
            "variant": model_variant,
            "framework": framework,
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "file": audio_file.name,
            "original_audio_info": original_info,
            "duration_sec": round(duration, 2),
            "processing_time_sec": round(proc_time, 4),
            "rtf": round(rtf, 4),
            "language": "unknown",
            "transcript": text,
            **resource_stats
        })
        print(
            f"✅ Done: {audio_file.name} | Time: {proc_time:.2f}s | CPU: {resource_stats['avg_cpu']:.1f}% | GPU: {resource_stats.get('avg_gpu', 0):.1f}%")

        # --- Clean up ---
        if os.path.exists(input_path):
            os.remove(input_path)

    # --- Save results ---
    json_path = Path(output_dir) / f"{model_variant}_{model_name}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Transcription complete for {model_variant} ({model_name})")
    print(f"📁 Results saved to: {json_path}")
    return json_path
