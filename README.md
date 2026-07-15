# Sound Sense — Haojie's Branch

This branch contains the **modeling** work for Sound Sense — the full arc from EDA through SQL-backed data cleaning, a hand-crafted-feature CNN baseline, and a fine-tuned WavLM-large transformer that becomes the production model powering the live demo.

## Folder / file map

```
.
├── README.md                    # This file
├── ReadMeS3.md                  # Sprint 3 write-up: CNN runs, WavLM rationale
├── Sprint2_Presentation.pptx    # Sprint 2 slide deck
├── Sprint3Presentation.pptx     # Sprint 3 slide deck
│
├── Notebooks/
│  ├── Audio_Emotion_EDA.ipynb   # EDA (Sprint 1): 4 corpora, SAVEE audit
│  ├── sqltransfer.ipynb         # MySQL cleaning pipeline (Aiven)
│  ├── Sprint2.ipynb             # Sprint 2: SQL→clean→preprocess→tf.data
│  ├── SQLCLEAN.ipynb            # SQLite pipeline feeding WavLM training
│  ├── FeatureModel.ipynb        # CNN baseline ensemble — 0.629 test acc
│  ├── wav2vec2f.ipynb           # WavLM-large fine-tune — 0.731 test acc
│  ├── Angry.ipynb               # Angry vs Neutral source-contrast research
│  ├── AngrySample.ipynb         # Librosa walkthrough on one Angry clip
│  ├── Librosa.ipynb             # Extended librosa tutorial (chroma, pitch)
│  └── progress/
│     └── 070126note.ipynb       # CNN experiment log: 9 tuning runs
│
└── .gitignore                   # Excludes datasets, weights, local DB
```

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
