import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")
from flask import Flask, request, render_template, session, Response
import os
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from faster_whisper import WhisperModel
import torch

# ---------------------- CONFIG ----------------------

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "supersecretkey"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------- GPU / CPU AUTO DETECTION ----------------------

# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Choose model size according to your GPU power
# RTX 3060 (8GB) → "small" or "medium" are both great options
model_size = "large-v3"

# Pick compute type for optimal performance
compute_type = "float16" if device == "cuda" else "int8"

print(f"🔥 Loading Faster-Whisper model '{model_size}' on {device.upper()} with {compute_type} precision...")

# Initialize model
fw_model = WhisperModel(model_size, device=device, compute_type=compute_type)

# ---------------------- HELPERS ----------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------- ROUTES ----------------------

@app.route("/", methods=["GET", "POST"])
def index():
    duration_minutes = None
    filepath = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "upload":
            if "file" not in request.files:
                return "No file part"
            file = request.files["file"]
            if file.filename == "":
                return "No selected file"
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                # Get audio duration
                audio = AudioSegment.from_file(filepath)
                duration_minutes = len(audio) / 60000

                # Save path in session
                session["filepath"] = filepath

    return render_template(
        "index.html",
        duration=duration_minutes,
        filepath=session.get("filepath")
    )


@app.route("/stream_fw")
def stream_fw():
    """Stream transcription results from Faster-Whisper (segment by segment)."""
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return "No file uploaded"

    def generate():
        segments, info = fw_model.transcribe(
            filepath,
            beam_size=15,
            vad_filter=True,
            chunk_length=30,
            without_timestamps=False,
            multilingual=True,
        )
        for segment in segments:
            yield f"data: {segment.text}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
