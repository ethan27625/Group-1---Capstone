/**
 * ─── MODEL INTEGRATION BOUNDARY ──────────────────────────────────────────────
 *
 * This is the ONLY file that knows how emotion predictions are produced.
 * All UI code calls `predictEmotion(audioBlob)` and nothing else.
 *
 * Predictions come from the WavLM backend (backend/emotion_app.py), which must
 * be running locally: `python backend/emotion_app.py` → http://localhost:7860.
 *
 * INPUT
 *   audioBlob : Blob — raw audio captured from the browser microphone via
 *               MediaRecorder (typically audio/webm;codecs=opus, ~5 seconds — the
 *               model was trained on 1-5s clips and the backend does not trim).
 *               The backend decodes with soundfile, which can't read webm, so
 *               we re-encode to 16 kHz mono 16-bit PCM WAV before uploading.
 *
 * OUTPUT
 *   Promise resolving to an object with one integer percentage per emotion:
 *
 *   {
 *     angry: 4, disgusted: 2, fearful: 6, happy: 61,
 *     neutral: 18, sad: 9
 *   }
 *
 *   Throws Error with a user-friendly message if the backend is unreachable.
 * ─────────────────────────────────────────────────────────────────────────────
 */

const API_BASE_URL = 'http://localhost:7860';

export const EMOTION_KEYS = [
  'angry',
  'disgusted',
  'fearful',
  'happy',
  'neutral',
  'sad',
];

export async function predictEmotion(audioBlob) {
  const wavBlob = await toWavBlob(audioBlob);

  const form = new FormData();
  form.append('audio', wavBlob, 'clip.wav');

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, { method: 'POST', body: form });
  } catch {
    throw new Error(
      'Could not reach the analysis server. Start it with "python backend/emotion_app.py", then try again.'
    );
  }
  if (!response.ok) {
    throw new Error('The analysis server returned an error. Check its terminal for details.');
  }

  const data = await response.json();
  return toPercentages(data.probabilities);
}

/**
 * Backend response → UI shape: {"Angry": 0.6123, ...} fractions with
 * Capitalized keys become lowercase integer percentages.
 */
function toPercentages(probabilities) {
  const scores = {};
  for (const key of EMOTION_KEYS) {
    const capitalized = key[0].toUpperCase() + key.slice(1);
    scores[key] = Math.round((probabilities?.[capitalized] ?? 0) * 100);
  }
  return scores;
}

/**
 * Decode whatever MediaRecorder produced, resample to 16 kHz, mix to mono,
 * and encode as 16-bit PCM WAV — mirrors the backend's own test page.
 */
async function toWavBlob(audioBlob) {
  const arrayBuffer = await audioBlob.arrayBuffer();
  const ctx = new AudioContext({ sampleRate: 16000 });
  try {
    const decoded = await ctx.decodeAudioData(arrayBuffer);
    let pcm = decoded.getChannelData(0);
    if (decoded.numberOfChannels > 1) {
      const right = decoded.getChannelData(1);
      pcm = pcm.map((v, i) => (v + right[i]) / 2);
    }
    return new Blob([encodePCM(pcm, 16000)], { type: 'audio/wav' });
  } finally {
    ctx.close();
  }
}

/** Float32 samples → 16-bit PCM WAV file bytes (44-byte RIFF header). */
function encodePCM(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const str = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  str(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  str(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}
