# 🎧 VisionAI-Audio — Whisper Model Evaluation & Comparison

A complete research-grade benchmark suite for **evaluating Whisper-based speech-to-text models** across multiple languages, noise levels, and audio qualities.  
This phase of the project (Task 2) focuses on **model evaluation, metric analysis, and visualization** to determine the most accurate and efficient transcription system.

---

## 🧭 Overview

This module compares multiple **Whisper variants** — including open-source and optimized implementations — using diverse real-world audio datasets.  
It establishes a **standardized evaluation pipeline** for accuracy, speed, and resource efficiency.

**Models Evaluated**
- 🧩 **OpenAI Whisper (Large-v3)** — Baseline reference  
- ⚡ **Faster-Whisper** — Optimized CTranslate2 version  
- 💻 **Whisper.cpp** — Lightweight C++ implementation  
- 🔍 **WhisperX** — Alignment-enhanced variant  

---

## 🎯 Objectives
- Evaluate transcription accuracy and efficiency across different models  
- Measure performance under varied audio conditions (clean → noisy, short → long)  
- Generate unified reports with **WER**, **CER**, and performance charts  
- Identify trade-offs between accuracy, inference speed, and system load  

---

## 📊 Evaluation Metrics

| Category | Metrics |
|-----------|----------|
| **Accuracy** | Word Error Rate (WER), Character Error Rate (CER), Language-specific accuracy (English & Urdu) |
| **Speed** | Transcription time, Cold vs Warm start latency |
| **Resource Usage** | CPU %, GPU VRAM, RAM consumption |
| **Robustness** | Performance on noisy audio, multiple speakers, accent recognition |
| **Scenarios Tested** | Clean speech, street noise, office background, Urdu/English code-switching |

---

## 🧩 Repository Structure

```
VisionAI-Audio/
├── code/
│   ├── model_comparison.py              # Core batch-comparison logic
│   ├── metrics_calculator.py           # WER/CER computation
│   ├── resource_monitor.py             # CPU/GPU/RAM tracking
│   └── transcription_benchmark.py      # Automated metrics runner
│
├── notebooks/
│   └── model_comparison_analysis.ipynb # Visualization & statistical analysis
│
├── models/
│   ├── faster_whisper/
│   ├── whisper_cpp/
│   ├── whisperx/
│   └── model_configs.yaml
│
├── results/
│   ├── reports/                        # JSON metric outputs & PDF/HTML summaries
│   └── visualizations/                # Charts & plots from analysis
│
├── docs/
│   ├── model_evaluation.md            # Detailed metric and methodology guide
│   └── api_documentation.md
│
├── tests/
│   ├── test_audio_samples/            # Evaluation dataset (English & Urdu)
│   └── test_model_comparison.py
│
└── README.md
```
## ⚙️ How It Works

1. **Generate Transcripts**  
   Each model transcribes the same set of audio samples.  
   The output JSONs are saved under:  
   `results/reports/{model_name}_results.json`

2. **Evaluate Metrics**  
   `metrics_calculator.py` computes WER & CER by comparing each model to the **OpenAI Whisper baseline**.

3. **Benchmark Runner**
   ```bash
   python code/transcription_benchmark.py
   ```
   → Produces metrics JSONs for all models and saves them in `results/reports/`

---

## 📊 Visual Analysis

Open the notebook:

```bash
notebooks/model_comparison_analysis.ipynb
```

To view bar charts, line graphs, and heatmaps for accuracy vs speed vs resources.

---

## 📈 Outputs & Reports

By the end of Task 2, the system produces:

- 4 model transcription JSONs  
- 3 metric comparison JSONs  
- 1 combined summary JSON  
- 1 visualization notebook    

All outputs are stored under:

- `results/reports/`  

---

## 🧪 Testing & Validation

- ≥ 20 diverse audio samples per model  
- Mix of English & Urdu speech  
- Varied lengths (30 s → 15 min) and noise levels  
- Automated WER/CER calculation  
- Resource usage logged via `resource_monitor.py`

---

## 📚 Documentation

See `docs/model_evaluation.md` for:

- Metric definitions & formulas  
- Evaluation methodology  
- Interpretation guide for results  
- Model selection recommendations

---

## 🏁 Deliverables Summary

| Deliverable                                                                 | Status |
|-----------------------------------------------------------------------------|--------|
| 4 Whisper models tested (Faster-Whisper, Whisper.cpp, WhisperX, OpenAI Whisper) | ✅     |
| 20+ audio samples evaluated per model                                       | ✅     |
| Complete WER/CER metrics & visual reports                                   | ✅     |
| Evaluation scripts & notebook completed                                     | ✅     |
| Documentation & README updated                                              | ✅     |
| Integration ready for Flask pipeline (Task 3)                               | 🔄     |

---

## 👥 Contributors

**VisionAI Team** — R&D on Automatic Speech Recognition and Model Evaluation  
Maintained by **Talha**

---

## 🗓️ Project Timeline

**Task 2: Model Evaluation & Comparison**  
**Deadline:** 15 Oct 2025 ✅ Completed

  
