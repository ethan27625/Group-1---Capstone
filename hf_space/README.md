---
title: Sound Sense Emotion Detector
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Sound Sense — Emotion Detector

WavLM-large speech emotion recognition, deployed as a Hugging Face Space (Docker SDK).

Serves a self-contained demo page at `/` and a `POST /predict` endpoint (multipart
form field `audio`, WAV/webm/etc. — anything `soundfile`/`librosa` can decode).

Model weights are loaded from the local `clean_ser_best/` folder shipped alongside
`app.py` in this Space's repo — they are not committed to the main project's GitHub
repo due to size.
