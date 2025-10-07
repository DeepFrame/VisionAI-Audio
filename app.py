from flask import Flask, request, render_template, session, Response
import os
from werkzeug.utils import secure_filename
import openai
from pydub import AudioSegment
from faster_whisper import WhisperModel


UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "supersecretkey"  # required for session

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Faster-Whisper model (small for faster UX)
fw_model = WhisperModel("tiny", device="cpu", compute_type="int8")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    cost_estimate = ""
    duration_minutes = None
    filepath = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "upload":
            # Step 1: upload
            if "file" not in request.files:
                return "No file part"
            file = request.files["file"]
            if file.filename == "":
                return "No selected file"
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                # Get audio duration in minutes
                audio = AudioSegment.from_file(filepath)
                duration_minutes = len(audio) / 60000

                # Calculate cost for OpenAI (just info)
                cost_estimate = f"${0.006 * duration_minutes:.4f}"

                # Store filepath in session
                session["filepath"] = filepath

    return render_template(
        "index.html",
        cost=cost_estimate,
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
            beam_size=5,
            vad_filter=True,
            chunk_length=30,
            without_timestamps=True,
            multilingual=True,

        )
        for segment in segments:
            yield f"data: {segment.text}\n\n"

        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")



@app.route("/models")
def models():
    models_page = openai.models.list()
    return "<br>".join([model.id for model in models_page.data])


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
