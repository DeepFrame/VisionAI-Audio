# 📊 Model Evaluation — VisionAI-Audio

This document provides a detailed overview of how model performance was evaluated in the **VisionAI-Audio** project.  
It explains the metrics used, the evaluation methodology, and how to interpret the results for better model selection.

---

## 📐 Metric Definitions & Formulas

| Metric | Description | Formula / Definition | Ideal Value |
|---------|--------------|----------------------|--------------|
| **WER (Word Error Rate)** | Measures transcription accuracy at the word level. | \( WER = \frac{S + D + I}{N} \)<br>Where S = Substitutions, D = Deletions, I = Insertions, N = Total words in reference. | **Lower is better (0%)** |
| **CER (Character Error Rate)** | Measures transcription accuracy at the character level. | \( CER = \frac{S + D + I}{N} \) — same as WER but based on characters. | **Lower is better (0%)** |
| **RTF (Real-Time Factor)** | Measures how fast the model processes audio compared to its duration. | \( RTF = \frac{\text{Processing Time (s)}}{\text{Audio Length (s)}} \) | **Lower is faster (<1 is real-time)** |
| **Processing Time (sec)** | Total time taken by the model to complete transcription. | Recorded via timestamps in script. | **Lower is better** |
| **Avg CPU (%) / GPU (%) / Memory (GB)** | Measures average resource utilization during inference. | Captured using `psutil` and `torch` profiling tools. | **Lower usage is better (efficiency)** |

---

## 🧪 Evaluation Methodology

1. **Test Dataset:**  
   - A fixed set of multilingual audio samples (short to medium length).  
   - Diverse accents and noise levels for realistic evaluation.  

2. **Reference Text:**  
   - The **OpenAI Whisper (large-v3)** transcription was used as a **baseline reference**.  
   - Other models (Whisper.cpp, WhisperX, Faster-Whisper) were evaluated against it.  

3. **Metrics Computed:**  
   - **WER** and **CER** (accuracy comparison)  
   - **RTF** (speed comparison)  
   - **CPU, GPU, Memory usage** (efficiency comparison)  

4. **Environment:**  
   - Evaluations were performed locally on CPU and optionally GPU-enabled systems.  
   - All experiments were automated using `model_comparison.py` and logged into JSON reports.  

---

## 🧭 Interpretation Guide

| Metric | What It Tells You | How to Interpret |
|---------|-------------------|------------------|
| **WER / CER** | Accuracy of the transcription | Lower values = more accurate output |
| **RTF** | Speed of transcription | RTF < 1 → real-time capable; RTF > 1 → slower than real time |
| **Processing Time** | Efficiency across models | Compare for identical audio lengths |
| **Resource Usage** | Performance cost | Lower resource usage = higher efficiency |

---

### ⚖️ Accuracy vs Speed Trade-Off

A core insight from the evaluation was the **trade-off between model accuracy and speed**:

- **Faster-Whisper** delivers **very low RTF** and **near-real-time performance**,  
  making it ideal for production or streaming use cases.  
- **WhisperX** and **OpenAI-Whisper** provide **higher accuracy**,  
  especially for complex or noisy speech, but require more compute.  
- **Whisper.cpp** strikes a **balance** between the two — light and efficient,  
  suitable for low-resource or embedded environments.

---

## 📊 Visualizations

The following charts were used for comparative analysis:

- **Bar Charts** – Processing time, RTF, accuracy (WER/CER), resource utilization  
- **Scatter Plots** – Accuracy vs Speed (WER vs RTF)  
- **Box Plots** – Distribution of performance metrics across runs  
- **Radar Charts** – Combined efficiency visualization  
- **Pair Plots** – Correlations between performance variables  

Each visualization helps identify where a model sits in the **Speed–Accuracy trade-off spectrum**.

---

## 🏁 Model Selection Recommendations

| Use Case | Recommended Model | Reason |
|-----------|------------------|--------|
| **Real-Time / Low-Latency Applications** | 🟢 **Faster-Whisper** | Fastest model with competitive accuracy |
| **High-Accuracy Offline Transcription** | 🔵 **OpenAI Whisper (large-v3)** | Most accurate but computationally heavy |
| **Lightweight / Edge Deployment** | 🟣 **Whisper.cpp** | Minimal resource usage, easy CPU deployment |
| **Alignment & Diarization Tasks** | 🟠 **WhisperX** | Includes word-level timestamps and speaker alignment |

---

## 💡 Final Insight

No single model is "best" in all areas — the right choice depends on your use case:

- **For speed and scalability:** choose **Faster-Whisper**  
- **For precision and quality:** choose **OpenAI Whisper (large-v3)**  
- **For embedded or constrained devices:** choose **Whisper.cpp**

---

📁 **Related Files**
- `notebooks//combined_summary.json` — aggregated metrics  
- `notebooks/model_comparison.ipynb` — visualization and interpretation notebook  
- `code/model_comparison.py` — metric computation and report generator  
