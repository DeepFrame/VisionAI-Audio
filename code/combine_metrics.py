import os
import json
from glob import glob
from collections import defaultdict

def combine_all_reports():
    # ✅ Use correct path relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/reports"))
    output_path = os.path.join(base_dir, "combined_summary.json")

    combined = defaultdict(lambda: {"models": {}})

    print(f"🔍 Searching in: {base_dir}")

    # --- 1️⃣ Model result files
    model_files = glob(os.path.join(base_dir, "*_results.json"))
    print(f"📂 Found {len(model_files)} model result files")

    for mfile in model_files:
        try:
            with open(mfile, "r", encoding="utf-8") as f:
                model_data = json.load(f)
                if isinstance(model_data, dict):
                    model_data = [model_data]

                model_name = os.path.basename(mfile).split("_results.json")[0]
                for entry in model_data:
                    file_name = entry.get("file")
                    if not file_name:
                        continue

                    combined[file_name]["models"][model_name] = {
                        "framework": entry.get("framework"),
                        "device": entry.get("device"),
                        "compute_type": entry.get("compute_type"),
                        "duration_sec": entry.get("duration_sec"),
                        "processing_time_sec": entry.get("processing_time_sec"),
                        "rtf": entry.get("rtf"),
                        "avg_cpu": entry.get("avg_cpu"),
                        "max_cpu": entry.get("max_cpu"),
                        "avg_mem": entry.get("avg_mem"),
                        "max_mem": entry.get("max_mem"),
                        "avg_gpu": entry.get("avg_gpu"),
                        "max_gpu": entry.get("max_gpu"),
                        "avg_gpu_mem": entry.get("avg_gpu_mem"),
                        "max_gpu_mem": entry.get("max_gpu_mem")
                    }
        except Exception as e:
            print(f"⚠️ Error reading model file {mfile}: {e}")

    # --- 2️⃣ Metrics files
    metric_files = glob(os.path.join(base_dir, "metrics_*.json"))
    print(f"📊 Found {len(metric_files)} metric files")

    # Mapping short names (used in metrics) → full model names
    model_name_map = {
        "cpp": "whisper-cpp_large-v3",
        "faster": "faster-whisper_large-v3",
        "whisperx": "whisperx_large-v2"
    }

    for mfile in metric_files:
        try:
            with open(mfile, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                if isinstance(metrics_data, dict):
                    metrics_data = [metrics_data]

                base_name = os.path.basename(mfile)
                short_model_name = base_name.replace("metrics_", "").replace("_vs_baseline.json", "")
                full_model_name = model_name_map.get(short_model_name, short_model_name)

                for entry in metrics_data:
                    file_name = entry.get("file")
                    if not file_name:
                        continue

                    if full_model_name not in combined[file_name]["models"]:
                        combined[file_name]["models"][full_model_name] = {}

                    combined[file_name]["models"][full_model_name].update({
                        "WER": entry.get("WER"),
                        "CER": entry.get("CER"),
                        "reference_len_chars": entry.get("reference_len_chars"),
                        "prediction_len_chars": entry.get("prediction_len_chars")
                    })
        except Exception as e:
            print(f"⚠️ Error reading metrics file {mfile}: {e}")

    # --- 3️⃣ Save clean combined summary
    final_data = [{"file": fname, "models": models["models"]} for fname, models in combined.items()]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2)

    # --- 4️⃣ Print summary
    all_models = sorted({m for v in combined.values() for m in v["models"].keys()})
    print("\n✅ Combined summary saved to:", output_path)
    print(f"📊 Files combined: {len(final_data)}")
    print(f"🧩 Models detected: {all_models}")

if __name__ == "__main__":
    combine_all_reports()
