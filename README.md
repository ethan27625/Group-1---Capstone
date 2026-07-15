# Sound Sense

A machine learning model that detects human emotion directly from speech.

**Live demo:** https://savanna-enhance-robbing.ngrok-free.dev

Sound Sense listens to a short voice recording and identifies which of six emotions the speaker is expressing — Angry, Disgusted, Fearful, Happy, Neutral, or Sad. It's built on a fine-tuned **WavLM-large** transformer trained on ~11,600 speech clips from the CREMA-D and RAVDESS emotional-speech corpora, and it reaches **73.1% test accuracy** on a speaker-independent evaluation.

The problem it points at: companies handle millions of customer service calls with no visibility into how those calls feel. Sound Sense turns speech into measurable emotional signal — the kind of signal that lets a system flag a frustrated caller *while they're still on the line*, not after they've churned.

## Project Structure

```
Group-1---Capstone/
├── EDA/                                # Sprint 1 — Exploratory Data Analysis
│  ├── EDA_merged.ipynb                 # Consolidated team EDA (61 cells)
│  ├── ethan_eda.ipynb                  # Ethan: SAVEE label audit + ANOVA ranking
│  ├── haojie_eda.ipynb                 # Haojie: 4-corpus audit + preprocessing plan
│  ├── aminatu_eda.ipynb                # Aminatu: waveform / spectrogram / MFCC study
│  └── plots/                           # Committed figure outputs
│     ├── ethan/                        # 14 PNGs (class dist, ANOVA, features)
│     └── aminatu/                      # 7 PNGs (waveforms, spectrograms, MFCCs)
│
├── cleaning/                           # Sprint 2 — Data cleaning pipelines
│  ├── ethan_cleaning.ipynb             # Manifest cleaner (drops SAVEE, Surprised)
│  ├── haojie_cleaning_sprint2.ipynb    # MySQL/Aiven pipeline (Sprint 2 deliverable)
│  └── haojie_cleaning_sql.ipynb        # SQLite variant — feeds WavLM training
│
├── modeling/                           # Sprint 3 — Model training
│  ├── haojie_feature_cnn.ipynb         # CNN baseline (0.629 test accuracy)
│  ├── haojie_wavlm.ipynb               # WavLM-large fine-tune (0.731 — production)
│  └── haojie_cnn_experiment_log.ipynb  # CNN's 9-run tuning arc
│
├── research/                           # Sprint 1 research notes
│  ├── aminatu_research.md              # Waveforms & spectrograms
│  ├── angie_research.md                # Sample rate & class imbalance
│  ├── ethan_research.md                # MFCCs
│  ├── haojie_research.md               # Sprint 3 model analysis writeup
│  └── feature_exploration_all_emotions.ipynb
│
├── Data/                                # Dataset manifests (small metadata only)
│  ├── cleaned_manifest.csv             # Final model-ready dataset (11,653 rows)
│  └── quarantined_files.csv            # 73 files removed with reasons
│
├── README.md                           # This file
└── .gitignore
```

## The model

The production model is a fine-tuned **WavLMForSequenceClassification** (`microsoft/wavlm-large`, 300M parameters, 24 transformer layers). Trained on CREMA-D + RAVDESS with a speaker-independent split, using layer-wise learning rate decay, focal loss, and a cosine schedule. Held-out test accuracy: **0.731 (macro F1: 0.731)**.

| Model | Test accuracy | Macro F1 |
|---|---|---|
| CNN ensemble (hand-crafted features) | 0.629 | — |
| **WavLM-large** | **0.731** | **0.731** |

The full training story — data audit, cleaning pipeline, CNN baseline hitting a feature ceiling, and the WavLM fine-tune that broke past it — lives in the `EDA/`, `cleaning/`, and `modeling/` folders in that order.

## Live demo

The public URL above serves a single-page app that lets anyone record ~5 seconds of speech and see the model's emotion prediction with a full probability breakdown. It runs on a FastAPI backend serving the WavLM model, hosted via ngrok during the presentation window.

The demo code (React frontend + FastAPI backend) lives on the **`ethan`** branch — main is documentation-focused; the runnable app is not consolidated here by design.

## Branches

Each teammate has their own branch containing their full contribution and a per-branch README:

- **`ethan`** — Frontend, backend integration, demo hosting, EDA, cleaning, MFCC research
- **`Haojie`** — Model training (CNN + WavLM), SQL cleaning pipelines, EDA
- **`aminatu-audio-project`** — Sprint 1 EDA, waveforms & spectrograms research
- **`main`** — This branch: consolidated project view

## Team

- **Ethan Perez** — Team lead, frontend / UI, model integration, live-demo hosting, EDA, data cleaning
- **Haojie Huang** — Model training and fine-tuning (CNN baseline + WavLM-large), SQL-backed data pipelines, EDA
- **Aminatu Bawa** — Sprint 1 EDA, waveform and spectrogram research
- **Angie Juarez** — Sprint 1 research (sample rate, class imbalance)

Capstone project for **The Knowledge House × Synchrony Financial Data Analytics Fellowship**.
