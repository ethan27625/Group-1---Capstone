/**
 * Circular glassmorphism mic control. Pulses while recording;
 * a second tap stops early. Disabled during analysis.
 */
export default function MicButton({ isRecording, disabled, onClick }) {
  return (
    <div className="mic">
      <button
        type="button"
        className={`mic__button ${isRecording ? 'mic__button--recording' : ''}`}
        onClick={onClick}
        disabled={disabled}
        aria-label={isRecording ? 'Stop recording' : 'Start recording'}
      >
        {isRecording && <span className="mic__pulse" aria-hidden="true" />}
        <MicIcon />
      </button>
      <span className="mic__hint">{isRecording ? 'Tap to stop' : 'Tap to speak'}</span>
    </div>
  );
}

function MicIcon() {
  return (
    <svg
      width="26"
      height="26"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
      <path d="M19 10v1a7 7 0 0 1-14 0v-1" />
      <line x1="12" y1="18" x2="12" y2="22" />
    </svg>
  );
}
