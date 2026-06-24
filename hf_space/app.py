"""
Speech Emotion Recognition — HuggingFace Spaces deployment.
Loads WavLM-large weights from HF Hub, logs predictions to Aiven MySQL.
Set Aiven credentials as Space secrets (Settings → Repository secrets).
"""

import io
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from transformers import AutoFeatureExtractor, WavLMForSequenceClassification

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_MODEL_ID = "aslhhhhj/wavlm-ser"
SR      = 16000
MAX_LEN = SR * 5
CLASSES = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad"]
EMOJI   = {
    "angry": "😠", "disgusted": "🤢", "fearful": "😨",
    "happy": "😄", "neutral":   "😐", "sad":     "😢",
}

# ---------------------------------------------------------------------------
# Aiven MySQL — prediction logging
# ---------------------------------------------------------------------------
from sqlalchemy import create_engine, text

_CONNECT_ARGS = {"ssl": {"ssl": True}}
_SSL_CA = os.getenv("MYSQL_SSL_CA", "")
if _SSL_CA:
    _CONNECT_ARGS = {"ssl": {"ca": _SSL_CA}}

_PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    ts         DATETIME DEFAULT CURRENT_TIMESTAMP,
    emotion    VARCHAR(32),
    confidence FLOAT,
    angry      FLOAT, disgusted FLOAT, fearful FLOAT,
    happy      FLOAT, neutral   FLOAT, sad     FLOAT
)
"""

db_ok  = False
engine = None
try:
    url = (
        f"mysql+pymysql://{os.getenv('MYSQL_USER','root')}:{os.getenv('MYSQL_PASSWORD','')}"
        f"@{os.getenv('MYSQL_HOST','localhost')}:{os.getenv('MYSQL_PORT','3306')}"
        f"/{os.getenv('MYSQL_DB','defaultdb')}"
    )
    engine = create_engine(url, connect_args=_CONNECT_ARGS, pool_pre_ping=True)
    with engine.begin() as conn:
        conn.execute(text(_PREDICTIONS_DDL))
    db_ok = True
    print("DB  : connected — predictions table ready")
except Exception as e:
    print(f"DB  : NOT connected ({e}) — predictions will not be logged")


def log_prediction(emotion, confidence, probs):
    if not db_ok:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO predictions
                    (emotion, confidence, angry, disgusted, fearful, happy, neutral, sad)
                VALUES
                    (:emotion, :confidence, :angry, :disgusted, :fearful, :happy, :neutral, :sad)
            """), {"emotion": emotion, "confidence": confidence, **probs})
    except Exception as e:
        print(f"DB log error: {e}")

# ---------------------------------------------------------------------------
# Load model from HF Hub
# ---------------------------------------------------------------------------
print(f"Loading model from HF Hub: {HF_MODEL_ID} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

feature_extractor = AutoFeatureExtractor.from_pretrained(HF_MODEL_ID)
model = WavLMForSequenceClassification.from_pretrained(HF_MODEL_ID)
model.to(device)
model.eval()
print("Model ready.")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

HTML_PATH = Path(__file__).parent / "emotion_detector_app.html"

@app.route("/")
def index():
    return send_file(str(HTML_PATH))


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file. Send field name: audio"}), 400
    audio_bytes = request.files["audio"].read()
    if not audio_bytes:
        return jsonify({"error": "Empty audio file"}), 400
    try:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
    except Exception as e:
        return jsonify({"error": f"Could not decode audio: {e}"}), 422

    y = y[:MAX_LEN]
    inputs = feature_extractor([y], sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_idx   = int(probs.argmax())
    emotion    = CLASSES[pred_idx].lower()
    confidence = float(probs[pred_idx])
    probs_dict = {CLASSES[i].lower(): round(float(p), 4) for i, p in enumerate(probs)}

    log_prediction(emotion, confidence, probs_dict)

    return jsonify({
        "emotion":       emotion,
        "confidence":    round(confidence, 4),
        "emoji":         EMOJI.get(emotion, "🎭"),
        "probabilities": probs_dict,
    })


@app.route("/history", methods=["GET"])
def history():
    if not db_ok:
        return jsonify({"error": "Database not connected"}), 503
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT ts, emotion, confidence FROM predictions ORDER BY id DESC LIMIT 50"
        )).fetchall()
    return jsonify([{"ts": str(r[0]), "emotion": r[1], "confidence": r[2]} for r in rows])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": HF_MODEL_ID, "device": device,
                    "db": "connected" if db_ok else "disconnected"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
