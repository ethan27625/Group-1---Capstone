"""
Flask inference server for Speech Emotion Recognition.
Connects the HTML front-end to the fine-tuned WavLM-large model
and logs every prediction to the Aiven MySQL database.

Usage:
    pip install flask flask-cors transformers torch librosa soundfile sqlalchemy pymysql python-dotenv
    python app.py

MODEL SETUP:
  Download w2v2_ser_best.zip from Colab and unzip it at:
      Group-1---Capstone/w2v2_ser_best/
  It should contain: config.json, model.safetensors,
  preprocessor_config.json, tokenizer_config.json

DB SETUP:
  Copy .env.example -> .env and fill in your Aiven credentials.
  The server creates the `predictions` log table automatically on first run.
"""

import io
import os
from pathlib import Path
from datetime import datetime

import librosa
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoFeatureExtractor, WavLMForSequenceClassification

# ---------------------------------------------------------------------------
# Load .env (Aiven credentials)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "Notebooks" / "w2v2_ser_best"
SR        = 16000
MAX_LEN   = SR * 5   # 5 s cap (matches training)
CLASSES   = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad"]

EMOJI = {
    "angry":     "😠",
    "disgusted": "🤢",
    "fearful":   "😨",
    "happy":     "😄",
    "neutral":   "😐",
    "sad":       "😢",
}

# ---------------------------------------------------------------------------
# Aiven MySQL — log table setup
# ---------------------------------------------------------------------------
from sqlalchemy import create_engine, text

_DB_USER = os.getenv("MYSQL_USER", "root")
_DB_PWD  = os.getenv("MYSQL_PASSWORD", "")
_DB_HOST = os.getenv("MYSQL_HOST", "localhost")
_DB_PORT = os.getenv("MYSQL_PORT", "3306")
_DB_NAME = os.getenv("MYSQL_DB", "defaultdb")
_SSL_CA  = os.getenv("MYSQL_SSL_CA", "")
_CONNECT_ARGS = {"ssl": {"ca": _SSL_CA}} if _SSL_CA else {"ssl": {"ssl": True}}

_PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    ts          DATETIME     DEFAULT CURRENT_TIMESTAMP,
    emotion     VARCHAR(32),
    confidence  FLOAT,
    angry       FLOAT,
    disgusted   FLOAT,
    fearful     FLOAT,
    happy       FLOAT,
    neutral     FLOAT,
    sad         FLOAT
)
"""

db_ok = False
engine = None
try:
    engine = create_engine(
        f"mysql+pymysql://{_DB_USER}:{_DB_PWD}@{_DB_HOST}:{_DB_PORT}/{_DB_NAME}",
        connect_args=_CONNECT_ARGS,
        pool_pre_ping=True,
    )
    with engine.begin() as conn:
        conn.execute(text(_PREDICTIONS_DDL))
    db_ok = True
    print(f"DB  : connected to {_DB_HOST}:{_DB_PORT}/{_DB_NAME} — predictions table ready")
except Exception as e:
    print(f"DB  : NOT connected ({type(e).__name__}: {e})")
    print("      predictions will not be logged — check .env credentials")


def log_prediction(emotion: str, confidence: float, probs: dict):
    if not db_ok:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO predictions
                    (emotion, confidence, angry, disgusted, fearful, happy, neutral, sad)
                VALUES
                    (:emotion, :confidence, :angry, :disgusted, :fearful, :happy, :neutral, :sad)
            """), {
                "emotion":    emotion,
                "confidence": confidence,
                **{k: probs[k] for k in ["angry", "disgusted", "fearful", "happy", "neutral", "sad"]},
            })
    except Exception as e:
        print(f"DB log error: {e}")


# ---------------------------------------------------------------------------
# Load model at startup (once)
# ---------------------------------------------------------------------------
print(f"Model: loading from {MODEL_DIR} ...")
if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"\n\nModel folder not found: {MODEL_DIR}\n"
        "Unzip w2v2_ser_best.zip (downloaded from Colab) into the Notebooks/ folder.\n"
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

feature_extractor = AutoFeatureExtractor.from_pretrained(str(MODEL_DIR))
model = WavLMForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()
print("Model ready.\n")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file. Send field name: audio"}), 400

    audio_bytes = request.files["audio"].read()
    if len(audio_bytes) == 0:
        return jsonify({"error": "Empty audio file received"}), 400

    try:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
    except Exception as e:
        return jsonify({"error": f"Could not decode audio: {e}"}), 422

    y = y[:MAX_LEN]

    inputs = feature_extractor([y], sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs     = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_idx  = int(probs.argmax())
    emotion   = CLASSES[pred_idx].lower()
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
    """Return the last 50 predictions from Aiven for dashboard use."""
    if not db_ok:
        return jsonify({"error": "Database not connected"}), 503
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT ts, emotion, confidence FROM predictions ORDER BY id DESC LIMIT 50"
        )).fetchall()
    return jsonify([{"ts": str(r[0]), "emotion": r[1], "confidence": r[2]} for r in rows])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model":  str(MODEL_DIR),
        "device": device,
        "db":     "connected" if db_ok else "disconnected",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
