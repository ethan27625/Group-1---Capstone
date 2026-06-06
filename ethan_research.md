- Research 

- What MFCCs (Mel Frequency Cepstral Coefficients) are and why they're the standard feature extraction method for audio classification tasks. 

- First what exactly are MFCCs 

MFCCs are a compact set of numbers that describe the shape of a sound's frequency content, designed to match how human hearing works. It is a standard feature extraction technique used in audio and speech processing. They act as a mathematical "fingerprint" of sound by converting raw audio into a compact set of numbers that mimic how the humna ear perceives pitch and timbre. 

- More in depth: 
MFCC (Mel Frequency Cepstral Coefficients)
Mel - Frequencies are mapped onto a scale warped to match human hearing (fine resolution at low frequencies, coarse at high)
Cepstral - A transform applied to the log frequency spectrum that isolates the overall shape of the sound from fine detail
Coefficients - The output is just a list of numbers 

What they capture is the spectral envelope - essentially the timbre (the "tone color" or "personality" of a sound. It is the quality that allows you to identify what is making a sound, even if two different sources (like a guitar and a piano) are playing the exact same note at the exact same volume) of a sound. This is why a flute and a voilin playign the same note produce different MFCCs. 

