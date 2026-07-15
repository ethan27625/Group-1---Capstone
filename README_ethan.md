# Sound Sense — Ethan's Branch

This branch contains the frontend/UI and model-integration work for **Sound Sense**, a speech emotion recognition product, plus EDA and data-cleaning contributions on the merged emotion dataset (TESS + RAVDESS + CREMA-D + SAVEE).

The live demo is served by the FastAPI app in `backend/`, exposed publicly via ngrok during the presentation. A React frontend and a Hugging Face Spaces deployment package are also included as alternative surfaces.

## Folder / file map

```
├── README_ethan.md                This file
├── README.md                      Repo root readme (project name only)
├── ethan_research.md              Research notes: what MFCCs are and why they're the standard
│                                  feature extraction method for audio classification
│
├── EDA/
│   └── ethan_eda.ipynb            Loads 12,798 files across 7 emotion folders (TESS/RAVDESS/
│                                  CREMA-D/SAVEE), audits SAVEE folder labels (84.4% mislabeled
│                                  by speaker, not emotion), audits per-corpus sample rates,
│                                  visualizes 8 acoustic features per emotion, and runs a
│                                  one-way ANOVA ranking 30 features by discrimination power
│
├── cleaning/
│   └── ethan_cleaning.ipynb       Acts on the EDA conclusions: drops SAVEE (mislabeled),
│                                  drops Surprised (severe imbalance, ~4.6% of data),
│                                  quarantines corrupted/anomalous-duration files, and writes
│                                  the cleaned 11,653-file manifest
│
├── outputs/                       CSV outputs from the cleaning notebook
│   ├── cleaned_manifest.csv       Final model-ready dataset (11,653 rows)
│   └── quarantined_files.csv      73 files removed with reasons
│
├── plots/                         14 figures from the EDA notebook (class distribution,
│                                  waveforms, spectrograms, MFCCs, ANOVA ranking, etc.)
│
├── Notebooks/
│   └── CleanModel.ipynb           Reference: teammate's WavLM-large fine-tuning notebook
│                                  (frozen conv encoder, LLRD, focal loss, cosine schedule).
│                                  Not runnable here — depends on a local SQLite build not
│                                  included in this branch
│
├── backend/                       The live demo — FastAPI + WavLM model + embedded UI
│   ├── emotion_app.py             FastAPI server: serves the single-page HTML/CSS/JS UI at /
│   │                              and POST /predict. Features: required-name gate with
│   │                              inline validation, opt-in consent checkbox for audio
│   │                              saving, pulsing-dot + AnalyserNode wave-bars during
│   │                              recording, 5-second countdown timer, results logged to
│   │                              CSV with optional audio persisted to disk
│   ├── clean_ser_best/            Model weights (gitignored, ~1.2GB). WavLMFor-
│   │                              SequenceClassification, 6 classes (Angry, Disgusted,
│   │                              Fearful, Happy, Neutral, Sad)
│   ├── recordings/                Opt-in saved audio (gitignored)
│   └── results.csv                Running prediction log (gitignored)
│
├── frontend/                      React + Vite alternative UI (not wired to the live demo)
│   └── src/                       App.jsx state machine, useRecorder hook, emotion tiles,
│                                  mic button, progress bar, results panel — Calm Clarity
│                                  design language (indigo gradient, glass mic button)
│
└── hf_space/                      Hugging Face Spaces deployment package (unused in demo,
                                   included as production-ready artifact)
    ├── app.py                     Mirror of backend/emotion_app.py with CORS open to "*"
    ├── Dockerfile                 python:3.11-slim, CPU-only torch
    ├── requirements.txt           fastapi, uvicorn, torch, transformers, librosa, etc.
    └── README.md                  HF Spaces config + deploy notes
```

## Running the demo

The demo runs from `backend/emotion_app.py`. Ngrok exposes it publicly during the presentation.

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install torch transformers librosa soundfile fastapi uvicorn python-multipart
python emotion_app.py
```

Open `http://localhost:7860`. The server picks CUDA, then MPS, then CPU automatically.

**Model weights are gitignored due to size.** Place `clean_ser_best/` (containing `model.safetensors`, `config.json`, and `preprocessor_config.json`) directly inside `backend/` before running.

## The live UI (backend/emotion_app.py)

The FastAPI app serves a single-page HTML/CSS/JS interface directly at `/`. Users:

1. Enter their name (required — recording is blocked without it, with inline validation)
2. Optionally check a consent box to allow their audio recording to be saved
3. Tap record and speak for up to 5 seconds — a pulsing dot, live sound-wave bars, and a countdown timer provide clear recording feedback
4. See the predicted emotion, a color-coded slot-machine reveal, and a full probability breakdown across the 6 classes

Every prediction appends a row to `backend/results.csv` (timestamp, name, top emotion, all 6 probabilities, optional audio filename). Audio files are saved to `backend/recordings/` only when the consent box is checked, with filenames of the form `{timestamp}_{sanitized_name}_{emotion}.wav`.

## The model (backend/clean_ser_best/)

Fine-tuned WavLM-large (`WavLMForSequenceClassification`, 300M parameters, 24 transformer layers) trained by Haojie Huang. Six emotion classes: Angry, Disgusted, Fearful, Happy, Neutral, Sad. Input is raw mono audio at 16 kHz; the app resamples and mixes to mono automatically before inference.

## Alternative surfaces

- **`frontend/`** — a standalone React + Vite web app implementing the same emotion-tile UI in a Calm Clarity design language (indigo gradient, glass mic button, 4-state animation flow). It talks to the backend via `src/lib/predictEmotion.js` (POST to `/predict`, WAV re-encoded client-side). Not wired to the live demo (missing the required-name and consent flow added to the backend UI), but included as reference and a starting point for a separate deployment.
- **`hf_space/`** — a complete Hugging Face Spaces deployment package (Dockerfile + FastAPI mirror + requirements). Not used for the presentation demo (ngrok is faster and more reliable for a live event), but included as evidence of a production-ready deploy path. Would require uploading `clean_ser_best/` directly to the Space.

## Credit

- `backend/clean_ser_best/` (WavLM-large fine-tuning), `Notebooks/CleanModel.ipynb`, and the initial `backend/emotion_app.py` scaffold — **Haojie Huang**
- Frontend, backend UX layer (required-name gate, consent-based audio saving, wave-bar recording indicator, timer), ngrok deployment, EDA notebook, cleaning notebook, and HF Spaces package — **Ethan Perez**
