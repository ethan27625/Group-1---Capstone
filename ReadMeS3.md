# Sprint 3 — Speech Emotion Recognition Summary

**Task:** 6-class SER — Angry · Disgusted · Fearful · Happy · Neutral · Sad  
**Dataset:** CREMA-D + RAVDESS (from `uldisvalainis/audio-emotions`), ~7 300 clips after filtering  
**Setup:** 16 kHz, 1–3.5 s clips, speaker-independent 80/20 train/test split, seed=42

---

## CNN Experiment Log (new1.ipynb)

We ran 9 rounds of experiments building up from a simple 3-block CNN to a more advanced architecture. Here's what we learned:

### Run 1 — Baseline: 0.634
Started with 7 mel-based channels and a basic 3-block CNN. Big overfitting gap (train ~0.80 vs test 0.634). `ReduceLROnPlateau` cut the LR too early whenever validation accuracy wobbled, which stalled training.

### Run 2 — SpecAugment: 0.623 (worse)
Tried time and frequency masking. Hurt more than it helped — the dataset is too small (~5 900 train clips) and aggressive masking destroyed the subtle patterns the model was using to identify Neutral. Dropped it.

### Run 3 — More features (11 channels): 0.639
Added spectral bandwidth, contrast, tempogram, and voiced probability. Contrast and bandwidth helped the most. Ran permutation feature importance — chroma, pitch_conf, and tempogram came back near-zero so we cut them.

**Top features by importance (accuracy drop when shuffled):**

| Channel | Drop |
|---------|------|
| rms | +0.199 |
| delta2 | +0.167 |
| delta | +0.148 |
| f0 | +0.124 |
| mel | +0.105 |
| rolloff | +0.103 |
| contrast | +0.101 |

Final active channels: `mel, delta, delta2, rms, f0, rolloff, bandwidth, contrast` (8 total)

### Run 4 — SpatialDropout2d + Label Smoothing: 0.641
SpatialDropout2d(0.15) drops entire feature maps instead of individual pixels — stronger regularization for conv layers. Combined with label smoothing=0.1 it cut the overfitting gap from 0.161 → 0.064.

### Run 5 — CosineAnnealingLR: 0.647
Swapped `ReduceLROnPlateau` for a single cosine decay over 150 epochs. No more reactive LR cuts from noisy validation. The low-LR fine-tuning phase (epochs 125–150) kept finding small improvements — this was the key schedule change.

### Run 6 — Focal Loss: 0.638 (worse)
Tried focal loss (γ=2) to focus on the hard classes (Disgusted, Fearful). Didn't help — class weights were already doing that job. Focal loss was redundant and reducing label smoothing from 0.1→0.05 hurt regularization. Reverted.

### Run 7 — Warm Restart: 0.650
After focal loss stalled, ran 50 more epochs which reset the cosine schedule to full LR — effectively a warm restart (SGDR). The LR kick let the model escape a local minimum. Biggest gain was on Sad (+0.033).

### Run 8 — 3-Model Ensemble: **0.654 (best)**
Trained 3 models independently and averaged their softmax outputs:
- Model A: FC=128, seed=42 → 0.650
- Model B: FC=256, seed=42 → 0.649
- Model C: FC=128, seed=7  → 0.642

Averaging works because models with different seeds and FC sizes make different errors. Final per-class F1:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Angry | 0.739 | 0.769 | 0.754 |
| Neutral | 0.643 | 0.739 | 0.688 |
| Sad | 0.641 | 0.649 | 0.645 |
| Happy | 0.656 | 0.616 | 0.635 |
| Fearful | 0.631 | 0.599 | 0.614 |
| Disgusted | 0.600 | 0.555 | 0.577 |

### Run 9 — Architecture Overhaul (in progress)
Upgraded to a 4-block CNN with SE blocks, residual skips, and asymmetric kernels (3×3 + 1×5 parallel). Also switched to AdamW and a linear warmup + cosine schedule. Added 4 new channels (centroid, onset, tempogram, voiced_prob) bringing the extraction to 12 channels. Results pending.

---

## Why We Hit a Ceiling with CNN

**Disgusted is the structural bottleneck** (F1 ≈ 0.577) and it never moved regardless of what we tried.

- Disgusted is acoustically very close to Angry — both are high-arousal, negative-valence emotions
- RMS energy is the #1 most important feature, but Angry and Disgusted share similar energy levels
- No loss function, architecture change, or ensemble trick moved this class — it's a **data problem**: CREMA-D + RAVDESS don't have enough actor diversity for Disgusted to learn the subtle differences

After 9 runs, our hand-crafted mel + prosody features appear to have hit their ceiling around 0.65. The features we can extract simply don't carry enough information to separate the confusable classes.

---

## Why We Chose WavLM / wav2vec2 (wav2vec2f.ipynb)

Hand-crafted features are a lossy compression of the raw audio — we decide upfront what to measure (pitch, energy, spectral shape) and throw away everything else. The problem is we don't know what the "everything else" contains.

Pretrained speech encoders like WavLM learn directly from raw waveforms on hundreds of thousands of hours of speech. They build internal representations that capture things we can't easily hand-engineer — fine-grained voice quality, breathiness, micro-variations in timing and pitch that correlate with emotion.

**Specific reasons WavLM made sense here:**

1. **Feature ceiling** — we already tried every reasonable hand-crafted feature and squeezed from 0.634 → 0.654. A fundamentally different representation was the next logical step.

2. **Fearful / Sad / Neutral confusion** — these classes are separated by subtle cues (breathiness, tension, speaking rate) that mel spectrograms average out. WavLM's transformer layers operate on the raw waveform and can capture these frame-level patterns.

3. **WavLM-large specifically** — 24 transformer layers vs 12 in base, hidden size 1024 vs 768. Benchmarks consistently show 5–8% better performance on SER tasks than smaller variants. The weighted layer sum lets the model learn which of the 24 layers is most useful for emotion (upper layers capture semantics, lower layers capture phonetics/prosody).

4. **Speaker-independent evaluation stays fair** — we kept the same `GroupShuffleSplit(test_size=0.15, random_state=42)` so the test clips are identical to the CNN runs. The accuracy numbers are directly comparable.

5. **Built-in augmentation** — WavLM uses SpecAugment internally during training. We tried SpecAugment manually in Run 2 and it hurt; WavLM handles it more carefully as part of its pretraining regime.

**What we changed vs what stayed the same:**

| | CNN notebooks | wav2vec2f |
|--|--------------|-----------|
| Features | Hand-crafted (mel, rms, f0, …) | Raw 16 kHz waveform |
| Model | 3–4 block CNN (~200k params) | WavLM-large (300 M params) |
| Optimizer | Adam / AdamW flat LR | LLRD (lower layers get smaller LR) |
| Loss | CE + class weights | Focal loss + class weights |
| Augmentation | None (SpecAugment hurt) | Built into WavLM forward pass |
| Split | Speaker-independent 80/20 | Same — directly comparable |
| Dataset | Same CREMA-D + RAVDESS | Same |

---

## Files

| File | Description |
|------|-------------|
| `Notebooks/new1.ipynb` | 3-block CNN, 3-model ensemble, 8-channel features |
| `Notebooks/mybest.ipynb` | Single-model version of the best CNN (0.654 config, no ensemble) |
| `Notebooks/wav2vec2f.ipynb` | WavLM-large fine-tuning — speaker-independent, LLRD, focal loss |
| `Notebooks/progress/070126note.ipynb` | Full experiment log for all 9 CNN runs |
