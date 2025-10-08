# 🎧 VisionAI-Audio — Whisper Audio Transcription (Local Faster-Whisper + Flask + Colab + CLI)

A complete project demonstrating **local audio transcription** using **Faster-Whisper**, featuring:
- 🌐 A Flask web app for browser-based uploads
- 💻 A CLI script for quick terminal transcriptions
- ☁️ A Google Colab notebook for quick setup and GPU testing

---

## 🚀 Features
- Upload and process audio directly in your browser (Flask)
- Local transcription using Faster-Whisper (no API key or cost)
- Real-time streaming via SSE (Flask)
- CLI option for quick local transcriptions
- Google Colab support with GPU auto-detection
- Secure file uploads handled by Flask
- Support for `.mp3`, `.wav`, `.m4a`, and `.mp4` files

---

## 🧰 Tech Stack
- **Python 3.9+**
- **Flask**
- **Faster-Whisper**
- **Pydub**
- **dotenv**
- **Werkzeug**
- **Torch (for GPU detection)**

---

## ⚙️ Setup Instructions (Local Flask App)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

```
### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # on Windows
# OR
source venv/bin/activate   # on macOS/Linux
````
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Flask App
```bash
python app.py
```

### 💻 Command-Line Interface (CLI)
Run transcription directly from your terminal:
```bash
python tests/transcribe.py --audio data/samples/sample_ur.mp3 --out results/transcript.txt
```
✅ This will generate transcript.txt in the results/ folder.


### ☁️ Colab Quickstart Notebook
Run transcription directly from your terminal:

Open the Colab notebook for GPU-based transcription:

    📓 notebooks/01_quickstart_colab.ipynb
    
    Upload your audio file or use the provided sample
    
    Automatically detects GPU and runs Faster-Whisper
    
    Exports transcription as plain text
