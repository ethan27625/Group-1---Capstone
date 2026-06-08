# Sprint 1 Research Notes — Aminatu
**Topics:** Waveforms And Spectrograms


## Topic 1: What is a Waveform?

A waveform is just audio stored as a list of numbers.

When sound plays, it creates tiny changes in air pressure. Your computer measures those pressure changes thousands of times per second and saves each measurement as a number. That list of numbers is the waveform.

**Two things you need to know:**

**Amplitude** — the value of each number. A big number = loud sound. A small number = quiet sound.

**Sample rate** — how many measurements happen per second. We use 22,050 Hz, meaning the audio is measured 22,050 times every second. Every file in our dataset needs to use the same sample rate or the model will break.

**What it looks like in code:**

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("audio_clip.wav", sr=22050)
# y = the waveform (a long list of numbers)
# sr = sample rate (22,050)

librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.show()
```

**Why it matters for our project:**

When we plot waveforms for each emotion, we can actually see differences. Angry audio has tall, jagged spikes. Sad audio is flat and low. This helps us confirm our files loaded correctly and that different emotions look different before we even start modeling.

---

## Topic 2: What is a Spectrogram?

A spectrogram turns audio into a picture so a neural network can "see" it.

A waveform only shows loudness over time. A spectrogram shows **which frequencies** are in the sound at each moment — like seeing which musical notes are playing and how loud each one is, all at once.

- **X-axis** = time
- **Y-axis** = frequency (low sounds at the bottom, high sounds at the top)
- **Color** = how strong that frequency is at that moment (bright = loud, dark = quiet)

We use a **Mel spectrogram** specifically. The Mel scale adjusts the frequency axis to match how human ears actually hear — we're better at telling apart low sounds than high sounds, and the Mel scale reflects that. This makes it better for speech and emotion data.

**What it looks like in code:**
```python
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load("audio_clip.wav", sr=22050)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.show()
```

**Why it matters for our project:**

A spectrogram looks like an image, which means we can feed it into a CNN (convolutional neural network) the same way you'd feed in a photo. During EDA, we'll plot one spectrogram per emotion to visually confirm the classes look different — angry speech will have a very different color pattern than calm or sad speech.


## Summary

| | Waveform | Spectrogram |

| What it shows | Loudness over time | Frequencies over time |
| Shape | 1D (a list of numbers) | 2D (like an image) |
| Used for | Loading audio, checking files | Feeding into CNN models |
| Librosa function | `librosa.load()` | `librosa.feature.melspectrogram()` |

**Preprocessing steps this tells us we'll need in Sprint 2:**
1. Make sure all files use the same sample rate (resample to 22,050 Hz if not)
2. Pad or trim all clips to the same length so shapes match
3. Convert audio to mel spectrograms or MFCCs for model input
4. Remove any corrupted or silent files found during EDA
