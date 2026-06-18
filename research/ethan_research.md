# Research 

# What MFCCs (Mel Frequency Cepstral Coefficients) are and why they're the standard feature extraction method for audio classification tasks. 

# First what exactly are MFCCs 

MFCCs are a compact set of numbers that describe the shape of a sound's frequency content, designed to match how human hearing works. It is a standard feature extraction technique used in audio and speech processing. They act as a mathematical "fingerprint" of sound by converting raw audio into a compact set of numbers that mimic how the humna ear perceives pitch and timbre. 

# More in depth: 
MFCC (Mel Frequency Cepstral Coefficients)
Mel - Frequencies are mapped onto a scale warped to match human hearing (fine resolution at low frequencies, coarse at high)
Cepstral - A transform applied to the log frequency spectrum that isolates the overall shape of the sound from fine detail
Coefficients - The output is just a list of numbers 

What they capture is the spectral envelope - essentially the timbre (the "tone color" or "personality" of a sound. It is the quality that allows you to identify what is making a sound, even if two different sources (like a guitar and a piano) are playing the exact same note at the exact same volume) of a sound. This is why a flute and a voilin playign the same note produce different MFCCs. 

# How is it calculated? 

MFCCs are computed frame by frame from raw audio using a fixed sequence of signal processing steps. Different libraries may tweak details, but the core pipeline is the same. 

Here is the Standard step by step procedure that you can describe in your research file: 

- 1. Pre-emphasis 
Apply a high pass filter to boost higher frequencies slightly, compensating for the natural roll of in speech signals and improving signal to noise for higher formants. 

- 2. Framing 
Split the continuous audio signal into short, overlapping frames, typically 20 - 40 ms long with 25 ms frame length and 10 ms stride as common choices. 

- 3. Windowing 
Multiply each frame by a window function to reduce spectral leakage in the later Fourier transform. 

- 4. FFT and power spectrum 
Compute the Fast Fourier Transform (FFT) of each windowed frame. 
Convert to a power spectrum, which shows how energy is distributed over frequency for that frame. 

- 5. Apply mel filterbank
Pass the power spectrum through a bank of overlapping triangular filters that are spaced according to the mel scale: dense spacing at low frequencies, sparser at high frequencies, mimicking human frequency resolution. 

- 6. Logarithm of filterbank energies 
Take the logarithm of each filterbank energy to approximate the ear’s logarithmic sensitivity to loudness and to turn multiplicative spectral effects into additive ones.

- 7. Apply the DCT to the vector of log mel‑filterbank energies. 

To compute MFCCs, the audio signal is pre‑emphasized, framed, and windowed; the FFT is taken to obtain the power spectrum; a bank of mel‑spaced triangular filters is applied; the logarithm of the filterbank energies is computed; and finally a discrete cosine transform is performed, with the first few coefficients retained as the MFCCs.

# why they're the standard feature extraction method for audio classification tasks.

MFCCs are the standard feature extraction method for audio classification because they compress perceptually relevant spectral information into a small, decorrelated feature vector that traditional and modern machine‑learning models can exploit efficiently, and decades of successful use in speech and audio recognition have established them as a strong, well‑understood baseline.



