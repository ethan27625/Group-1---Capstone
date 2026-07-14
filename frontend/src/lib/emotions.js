// The 7 emotions Sound Sense classifies, in display order.
export const EMOTIONS = [
  { key: 'angry', label: 'Angry', emoji: '😠' },
  { key: 'disgusted', label: 'Disgusted', emoji: '🤢' },
  { key: 'fearful', label: 'Fearful', emoji: '😨' },
  { key: 'happy', label: 'Happy', emoji: '😄' },
  { key: 'neutral', label: 'Neutral', emoji: '😐' },
  { key: 'sad', label: 'Sad', emoji: '😢' },
  { key: 'surprised', label: 'Surprised', emoji: '😲' },
];

/** Emotion keys ranked by score, highest first. */
export function rankEmotions(scores) {
  return Object.entries(scores)
    .sort(([, a], [, b]) => b - a)
    .map(([key]) => key);
}
