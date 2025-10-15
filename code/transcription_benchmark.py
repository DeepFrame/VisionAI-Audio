# # code/transcription_benchmark.py
# from pathlib import Path
# from code.metrics_calculator import evaluate_json
#
# BASE_DIR = Path(__file__).resolve().parent.parent
#
# # Example filenames (these may be relative and will be resolved by evaluate_json)
# reference = "results/reports/openai-whisper_large-v3_results.json"
# prediction = "results/reports/faster-whisper_large-v3_results.json"
# output = "results/reports/metrics_faster_vs_baseline.json"
#
# # You can pass repo_root to make resolution explicit
# evaluate_json(reference_json=reference, prediction_json=prediction, output_path=output)
from pathlib import Path
from code.metrics_calculator import evaluate_json

BASE_DIR = Path(__file__).resolve().parent.parent
reports = BASE_DIR / "results" / "reports"

pairs = [
    ("openai-whisper_large-v3_results.json", "faster-whisper_large-v3_results.json", "metrics_faster_vs_baseline.json"),
    ("openai-whisper_large-v3_results.json", "whisperx_large-v2_results.json", "metrics_whisperx_vs_baseline.json"),
    ("openai-whisper_large-v3_results.json", "whisper-cpp_large-v3_results.json", "metrics_cpp_vs_baseline.json"),
]

for ref, pred, out in pairs:
    print(f"\n🔍 Comparing {pred} vs {ref}")
    evaluate_json(
        reference_json=reports / ref,
        prediction_json=reports / pred,
        output_path=reports / out
    )

# from code.metrics_calculator import evaluate_json
#
# pairs = [
#     ("results/reports/openai_whisper_large-v3_results.json",
#      "results/reports/faster_whisper_large-v3_results.json",
#      "results/reports/metrics_faster_vs_baseline.json"),
#
#     ("results/reports/openai_whisper_large-v3_results.json",
#      "results/reports/whisperx_results.json",
#      "results/reports/metrics_whisperx_vs_baseline.json"),
#
#     ("results/reports/openai_whisper_large-v3_results.json",
#      "results/reports/whisper_cpp_results.json",
#      "results/reports/metrics_cpp_vs_baseline.json"),
# ]
#
# for ref, pred, out in pairs:
#     print(f"\n🔍 Comparing {pred} vs {ref}")
#     evaluate_json(reference_json=ref, prediction_json=pred, output_path=out)

