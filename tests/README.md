## 🧩 CLI Transcription Script

The file `tests/transcribe.py` provides a **command-line interface (CLI)** for transcribing audio files using the **Faster-Whisper** model.

It loads the `large-v3` Whisper model, processes your provided audio file, and saves the generated transcript to a text file.

### 🧠 Usage

Run the script from your terminal:

```bash
python tests/transcribe.py --audio data/samples/sample_ur.mp3 --out results/transcript.txt
