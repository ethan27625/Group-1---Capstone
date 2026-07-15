# Sound Sense — Haojie's Branch

This branch contains the **modeling** work for Sound Sense — the full arc from EDA through SQL-backed data cleaning, a hand-crafted-feature CNN baseline, and a fine-tuned WavLM-large transformer that becomes the production model powering the live demo.

## Folder / file map
.
├── README.md                              This file
├── ReadMeS3.md                            Sprint 3 narrative write-up covering the CNN's
│                                          9-run experiment arc and the rationale for moving
│                                          from hand-crafted features to WavLM
├── Sprint2_Presentation.pptx              Sprint 2 slide deck
├── Sprint3Presentation.pptx               Sprint 3 slide deck
│
├── Notebooks/
│   ├── Audio_Emotion_EDA.ipynb            Main EDA (Sprint 1): loads all 4 corpora
│                                          (CREMA-D / TESS / RAVDESS / SAVEE, 12,798 clips),
│                                          detects source via filename regex, decodes SAVEE's
│                                          mislabeled folders (84%+ folder-label mismatch),
│                                          audits duration + sample rate across corpora,
│                                          concludes: drop Surprised, standardize to 16 kHz,
│                                          decode SAVEE labels
│   │
│   ├── sqltransfer.ipynb                  Cloud-MySQL preprocessing pipeline (Aiven):
│                                          metadata table, corruption/duplicate flags,
│                                          CREMA-D + RAVDESS filter, 3-second window trim,
│                                          log-mel extraction, stratified 80/20 split →
│                                          pushed to MySQL table audio_dataset
│   │
│   ├── Sprint2.ipynb                      Sprint 2 deliverable: full SQL → clean →
│                                          preprocess → tf.data pipeline. 12,798 → 9,518 →
│                                          9,326 → 7,343 clips through source/emotion/duration
│                                          filters. Extracts a (257, 110, 4) tensor per clip
│                                          (perceptual spectrogram, ZCR, spectral centroid,
│                                          RMS), per-channel z-score, wraps in tf.data
│   │
│   ├── SQLCLEAN.ipynb                     Local SQLite variant of the cleaning pipeline —
│                                          feeds the WavLM training. Same filtering (CREMA-D
│                                          + RAVDESS, drop Surprised, MD5 dedup) but a proper
│                                          3-way speaker-independent split, resample to
│                                          16 kHz / ≤5s into Emotions_w2v2_16k/, writes to
│                                          audio_dataset_w2v2. Producer for CleanModel.ipynb
│   │
│   ├── FeatureModel.ipynb                 CNN baseline (Sprint 3). 16-channel feature tensor
│                                          → 7 channels selected by permutation importance →
│                                          speaker-independent split (112 speakers) → 4-model
│                                          ensemble (MultiChannelCNN with SpatialDropout2d,
│                                          AdamW, cosine warm restarts, class-weighted focal
│                                          loss). Result: 0.629 test accuracy, per-class F1
│                                          0.557–0.793. Disgusted is the persistent bottleneck
│                                          — motivates the WavLM approach
│   │
│   ├── wav2vec2f.ipynb                    WavLM-large fine-tune (Sprint 3 final model).
│                                          WavLMForSequenceClassification (300M params, 24
│                                          layers), frozen conv encoder, trainable
│                                          weighted-layer-sum + head, LLRD optimizer (0.95
│                                          decay/layer), focal loss (γ=2) + balanced class
│                                          weights, cosine schedule, bf16 on A100. Result:
│                                          0.731 test accuracy, 0.731 macro F1 — a clear jump
│                                          over the CNN's 0.629. Model saved to
│                                          w2v2_ser_best/ (gitignored, ~1.2 GB)
│   │
│   ├── Angry.ipynb                        Source-contrast research: separates "source signal"
│                                          from "emotion signal" for Angry across all 4
│                                          corpora using per-source standardization + Cohen's
│                                          d. Finds RMS, pitch, and spectral centroid/rolloff
│                                          are the source-invariant "angry signature." Ends
│                                          with a small binary CNN validating the signal is
│                                          learnable
│   │
│   ├── AngrySample.ipynb                  Librosa walkthrough on one Angry clip — waveform,
│                                          STFT/mel spectrograms, 13-coeff MFCCs,
│                                          ZCR/centroid/RMS, and a reusable extract_features()
│                                          helper. Early exploratory scratch
│   │
│   ├── Librosa.ipynb                      Extended librosa tutorial on the same clip — adds
│                                          zero-crossing visualization, chroma, delta/delta²
│                                          MFCCs, pYIN + piptrack pitch tracking. Iterative
│                                          personal-notes companion to AngrySample.ipynb
│   │
│   └── progress/
│       └── 070126note.ipynb               CNN experiment log — documents the 9-run tuning
│                                          arc: baseline (0.634) → SpecAugment (0.623) →
│                                          11-channel expansion (0.639) → SpatialDropout +
│                                          label smoothing (0.641) → CosineAnnealingLR (0.647)
│                                          → focal loss (0.638) → warm restart (0.650) →
│                                          3-model ensemble (0.654). The drop to 0.629 in
│                                          FeatureModel.ipynb reflects switching to
│                                          speaker-independent splitting (more honest)
│
└── .gitignore                             Excludes the raw/processed audio datasets
(Emotions/, Emotions_processed/, Emotions_w2v2_16k/
— ~1.8 GB), all model weight formats
(w2v2_ser_best/, *.keras, *.pt, *.npz), the local
SQLite database (ser_local.db), and env files

## Model performance (headline numbers)

| Model | Test accuracy | Macro F1 | Notes |
|---|---|---|---|
| CNN ensemble (FeatureModel.ipynb) | 0.629 | — | 4-model, hand-crafted features, speaker-independent split |
| **WavLM-large (wav2vec2f.ipynb)** | **0.731** | **0.731** | 300M-param transformer, LLRD, focal loss, bf16 on A100 |

Per-corpus breakdown for the WavLM model: CREMA-D 0.740, RAVDESS 0.653. Disgusted remains the hardest class structurally (a well-known SER pattern — Disgusted and Angry share acoustic features), but at F1 0.768 vs the CNN's 0.557, WavLM handles it far better.

## Running the modeling notebooks

The training notebooks are **Colab-dependent by design** — they mount Google Drive for model persistence and use `kagglehub.dataset_download("uldisvalainis/audio-emotions")` for the dataset:

- **wav2vec2f.ipynb** — the final WavLM-large training run. Requires an A100 GPU (bf16). Output: `w2v2_ser_best/` folder saved to Drive.
- **FeatureModel.ipynb** — CNN baseline. Reads a cached feature tensor built in earlier stages of the notebook.
- **SQLCLEAN.ipynb** — builds `ser_local.db`, the local SQLite dataset the WavLM training consumes.
- **Sprint2.ipynb / sqltransfer.ipynb** — the earlier MySQL-based pipeline (Aiven), superseded by SQLCLEAN.ipynb for the WavLM work but included as the Sprint 2 submission.

Model weights are not tracked in git (`w2v2_ser_best/` is gitignored, ~1.2 GB). The trained WavLM checkpoint is what powers the live demo on Ethan's branch.
