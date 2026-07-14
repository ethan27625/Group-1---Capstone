import { RECORD_DURATION_MS } from '../hooks/useRecorder';

/**
 * The 5-second listening window: status label, fill bar, and a live
 * mono counter ("3.2s / 5s"). Fill animates via CSS width transition.
 */
export default function ProgressBar({ isRecording, elapsedMs }) {
  const pct = Math.min((elapsedMs / RECORD_DURATION_MS) * 100, 100);
  const seconds = (elapsedMs / 1000).toFixed(1);
  const total = RECORD_DURATION_MS / 1000;

  return (
    <div className="progress">
      <div className="progress__header">
        <span className="progress__label">{isRecording ? 'Listening…' : 'Ready'}</span>
        <span className="progress__counter">
          {seconds}s / {total}s
        </span>
      </div>
      <div
        className="progress__track"
        role="progressbar"
        aria-label="Recording progress"
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className="progress__fill" style={{ transform: `scaleX(${pct / 100})` }} />
      </div>
    </div>
  );
}
