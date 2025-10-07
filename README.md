# 🎧 Whisper Audio Transcription (Local Faster-Whisper + Flask)

A simple Flask web app for **transcribing audio files** locally using **Faster-Whisper**.  
You can upload `.mp3`, `.wav`, `.m4a`, or `.mp4` files, and see real-time transcription streaming via **Server-Sent Events** (SSE).

---

## 🚀 Features
- Upload and process audio directly in your browser
- Local transcription using Faster-Whisper (no external API cost)
- Real-time streaming transcription output
- Secure file uploads handled via Flask

---

## 🧰 Tech Stack
- **Python 3.9+**
- **Flask**
- **Faster-Whisper**
- **Pydub**
- **dotenv** (for environment variables)
- **Werkzeug**

---

## ⚙️ Setup Instructions

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