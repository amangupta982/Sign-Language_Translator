"""
SignSpeak — Flask Application

Routes:
  /            → Main page (webcam + upload)
  /guide       → ASL gesture guide
  /metrics     → Evaluation metrics dashboard
  /predict     → POST: predict from base64 image
  /upload      → POST: predict from uploaded file
  /speak       → POST: generate TTS audio
"""

import os
import io
import json
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from gtts import gTTS

from predict import SignPredictor
from model import LETTERS

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

predictor = SignPredictor("sign_model_v2.keras")


# ── Pages ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/guide")
def guide():
    return render_template("guide.html", letters=LETTERS)


@app.route("/metrics")
def metrics():
    report = {}
    report_path = "static/plots/classification_report.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
    return render_template("metrics.html", report=report, letters=LETTERS)


# ── API ────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """Predict from base64 image (webcam frame)."""
    try:
        image_data = request.json["image"]
        result = predictor.predict_from_base64(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """Predict from uploaded image file."""
    try:
        file = request.files["file"]
        from PIL import Image
        img = Image.open(file.stream)
        result = predictor.predict(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/speak", methods=["POST"])
def speak():
    """Generate TTS audio and return as base64."""
    try:
        sentence = (request.json.get("sentence") or "").strip()
        if not sentence:
            return jsonify({"error": "Nothing to say"}), 400

        tts = gTTS(text=sentence, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode()
        return jsonify({"audio": audio_b64, "spoken": sentence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("static/plots", exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=True)