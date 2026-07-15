# Sound Sense — Aminatu's Branch

This branch contains the **Sprint 1 exploratory data analysis** for Sound Sense — a full audit of the merged 4-corpus emotion dataset (TESS, RAVDESS, CREMA-D, SAVEE), plus research notes on waveforms and spectrograms that ground the team's Sprint 2 preprocessing decisions.

## Project Structure

```
aminatu-audio-project/
├── 01_EDA_Audio_Emotion_Detection-3.ipynb    # Sprint 1 EDA — full dataset audit
├── aminatu_research.md                       # Research notes: waveforms & spectrograms
│
├── Plots/                                    # Committed figure outputs
│  ├── 01_class_distribution.png              # Class balance across 7 emotions
│  ├── 02_duration_distribution.png           # Clip duration histogram + boxplot
│  ├── 03_sample_rates.png                    # Native sample rate breakdown
│  ├── 04_waveforms.png                       # Waveforms per emotion
│  ├── 05_mel_spectrograms.png                # Mel spectrograms per emotion
│  ├── 06_stft_spectrograms.png               # STFT spectrograms per emotion
│  └── 07_mfccs.png                           # MFCC heatmaps per emotion
│
├── README.md                                 # This file
└── .gitignore                                # Excludes datasets, audio, and CSVs
```

## The EDA (01_EDA_Audio_Emotion_Detection-3.ipynb)

A single monolithic Sprint 1 deliverable covering the full dataset audit. Loads all 12,798 clips across the 7 emotion folders and produces:

- **Class distribution** — bar and pie charts across the 7 labels. Flags **Surprised as severely imbalanced** at 592 files (4.6%) vs 1,795–2,167 for the other classes (3.7× imbalance), recommending exclusion from Sprint 2 modeling.
- **Duration distribution** — histogram and boxplot. Mean clip length **2.73 seconds**; this becomes the team's fixed padding/truncation target.
- **Sample rate audit** — finds **5 distinct native sample rates** across the corpora (16000 / 24414 / 44100 / 48000 / 96000 Hz), recommending standardization to 22,050 Hz.
- **Waveform plots** — one representative clip per emotion. Angry and Fearful show tall spiky peaks; Neutral and Sad stay flat and low-amplitude.
- **Mel spectrograms** — 128 mel bands per emotion.
- **STFT spectrograms** — linear-frequency variant per emotion.
- **MFCC heatmaps** — 40 coefficients per emotion.
- **Corrupted / missing file check** — flags 102 files as anomalously long (>3 standard deviations above mean duration); 0 failed to load, 0 silent.
- **Manual playback verification** — plays one sample per emotion to sanity-check labels by ear.
- **Written 5-finding summary** at the end.

## Research (aminatu_research.md)

Sprint 1 research notes on **waveforms and spectrograms**, written in a teaching style to walk teammates through both concepts:

- What a waveform is (amplitude over time, sample rate, why consistency matters) with a runnable `librosa.display.waveshow` snippet.
- What a spectrogram is (STFT → Mel scale, why Mel matches human hearing, why it's CNN-friendly) with a runnable `librosa.feature.melspectrogram` snippet.
- A waveform-vs-spectrogram comparison table and a 4-point "what this means for Sprint 2" plan: resample everything to a uniform rate, pad/trim to a fixed length, convert to mel/MFCC, remove corrupted files.

## How this fed the rest of the project

The three findings that shaped Sprint 2 across the whole team came from this EDA:

- **Drop Surprised** — severe class imbalance, adopted by the team's cleaning pipeline.
- **Standardize sample rate** — five different native rates ruled out any file-by-file approach.
- **Pad/truncate to a fixed window** — the 2.73s mean informed the eventual 5-second cap used by the WavLM-large model.

## Credit

All work on this branch — the EDA notebook, plots, and research notes — **Aminatu Bawa**.
