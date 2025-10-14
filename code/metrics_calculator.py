# code/metrics_calculator.py
"""
metrics_calculator.py
----------------------------------
Utility functions to compute transcription accuracy metrics:
- Word Error Rate (WER)
- Character Error Rate (CER)
This version resolves paths relative to project root and is tolerant to JSON key names.
"""

import jiwer
import json
from pathlib import Path
import re

# --- helper text normalization (adjust as needed) ---
def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)           # collapse whitespace
    # NOTE: do NOT always lowercase if you want case-sensitive metrics; jiwer typically lowercases anyway
    return s

def calculate_metrics(reference_text: str, predicted_text: str):
    """
    Calculate WER and CER between two text strings.
    Returns None for metrics when text is empty.
    """
    ref = _normalize_text(reference_text)
    hyp = _normalize_text(predicted_text)

    if not ref or not hyp:
        return {"WER": None, "CER": None}

    # jiwer expects plain strings
    wer = jiwer.wer(ref, hyp)
    cer = jiwer.cer(ref, hyp)
    return {"WER": round(wer, 4), "CER": round(cer, 4)}

def evaluate_json(reference_json: str, prediction_json: str, output_path: str, repo_root: str = None):
    """
    Compare two result JSONs containing model transcriptions and produce a metrics JSON.
    - reference_json: path to baseline results (openai-whisper JSON)
    - prediction_json: path to model under test results
    - output_path: where to write metrics JSON
    If paths are relative, they are resolved relative to repo_root (defaults to two levels up from this file).
    """
    # Resolve repo root (default to project root like model_comparison uses)
    if repo_root:
        BASE_DIR = Path(repo_root).resolve()
    else:
        BASE_DIR = Path(__file__).resolve().parent.parent

    ref_path = Path(reference_json)
    pred_path = Path(prediction_json)
    out_path = Path(output_path)

    # If relative, resolve relative to BASE_DIR / results / reports (common location)
    def _resolve(p: Path):
        if p.is_absolute():
            return p
        # try as-is relative to repo root
        cand = BASE_DIR / p
        if cand.exists():
            return cand
        # try results/reports
        cand2 = BASE_DIR / "results" / "reports" / p
        if cand2.exists():
            return cand2
        # fallback: resolved against BASE_DIR
        return cand

    ref_path = _resolve(ref_path)
    pred_path = _resolve(pred_path)
    out_path = _resolve(out_path)

    # Existence checks
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference JSON not found: {ref_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction JSON not found: {pred_path}")

    with open(ref_path, "r", encoding="utf-8") as f:
        references = json.load(f)

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Build mapping file -> transcript for reference and prediction
    def _build_map(items):
        m = {}
        for it in items:
            fname = Path(it.get("file", "")).name
            # accept either 'transcript' or 'text'
            text = it.get("transcript") if it.get("transcript") is not None else it.get("text", "")
            m[fname] = _normalize_text(text)
        return m

    ref_map = _build_map(references)
    pred_map = _build_map(predictions)

    results_summary = []
    for fname, ref_text in ref_map.items():
        pred_text = pred_map.get(fname)
        if pred_text is None:
            print(f"⚠️ Prediction missing for {fname}; skipping.")
            continue
        metrics = calculate_metrics(ref_text, pred_text)
        results_summary.append({
            "file": fname,
            "reference_len_chars": len(ref_text),
            "prediction_len_chars": len(pred_text),
            "WER": metrics["WER"],
            "CER": metrics["CER"]
        })

    # Save to JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Metrics report saved to: {out_path}")
    return results_summary
