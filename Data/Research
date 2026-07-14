 Research


What sample rate means and why keeping it consistent across the dataset matters for preprocessing?


Sample rate means - refers to how many per second an analog signal ( sound wave) is measured or "photographed" to convert it into digital data. It is measured in Hertz (Hz) or kilohertz (kHz). The higher the number, the more accurately the digital audio captures the original sound
Keeping the sample rate consistent across a dataset is vital for preprocessing, primarily because machine learning models require uniform input dimensions and frequency ranges.


Examples:


44.1 kHz (CD Quality): Captures 44,100 samples per second. This is the global standard for music streaming and commercial audio CDs.


48 kHz (Video & Film Standard): Captures 48,000 samples per second. It is the universal standard used for movies, video, and broadcast television.


96 kHz (High-Resolution): Captures 96,000 samples per second. Often used when recording audio that requires heavy editing, pitch-shifting, or slowing down in post-production. 




When AI models analyze audio for emotions, the most common sample rate used is 16 kHz.
analyzing deeper emotional prosody, whispered consonants, and subtle tones often utilizes sampling rates of 22.05 kHz, 44.1 kHz, or higher


https://vapi.ai/blog/sampling-rate


https://www.lewitt-audio.com/blog/what-sample-rate


https://medium.com/@diego-rios/speech-emotion-recognition-with-convolutional-neural-network-ae5406a1c0f7




2. What class imbalance looks like in an audio dataset and how it can mess with training




In an audio dataset, class imbalance occurs when certain sound classes—such as "speech" or "background noise"—massively outnumber rarer events like "glass breaking" or "bird calls". This skews training because the model updates its weights to favor common sounds, ultimately failing to recognize or correctly classify the rare, minority audio events.


Extremely Skewed Class Counts: One or two common categories dominate the dataset (e.g., thousands of hours of music or speech), while highly specific target sounds (e.g., an equipment malfunction or a specific animal) only have a few dozen brief clips.


Model Bias Towards the Majority: Deep learning models optimize for overall error reduction. Because common classes appear so frequently, the model learns to simply predict the majority class and still achieve a high overall accuracy—even if it is totally blind to the minority class