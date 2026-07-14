/**
 * One emotion in the analysis grid: small emoji accent, label, thin
 * confidence bar with a live percentage. `rank` (0/1/2) applies the
 * graded green ring; anything else keeps a neutral hairline border.
 */
export default function EmotionTile({ emoji, label, value, rank, shimmer }) {
  const ranked = rank >= 0 && rank <= 2;
  const className = [
    'tile',
    ranked ? `tile--rank-${rank + 1}` : '',
    shimmer ? 'tile--shimmer' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={className} role="listitem">
      <span className="tile__emoji" aria-hidden="true">
        {emoji}
      </span>
      <span className="tile__label">{label}</span>
      <div
        className="tile__bar"
        role="meter"
        aria-label={`${label} confidence`}
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className="tile__bar-fill" style={{ transform: `scaleX(${value / 100})` }} />
      </div>
      <span className="tile__value">{value}%</span>
    </div>
  );
}
