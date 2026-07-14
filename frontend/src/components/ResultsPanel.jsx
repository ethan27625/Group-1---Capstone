import { EMOTIONS } from '../lib/emotions';

/**
 * Ranked top-3 readout: #1 largest and boldest, #2 and #3 progressively
 * smaller and dimmer. Shows an empty-state prompt before the first run.
 */
export default function ResultsPanel({ scores, ranking }) {
  return (
    <section className="card results" aria-label="Top detected emotions">
      <h2 className="results__heading">Top detected emotions</h2>

      {!scores ? (
        <p className="results__empty">Record something to see results.</p>
      ) : (
        <ol className="results__list">
          {ranking.slice(0, 3).map((key, i) => {
            const emotion = EMOTIONS.find((e) => e.key === key);
            const value = scores[key];
            return (
              <li key={key} className={`result result--${i + 1}`}>
                <div className="result__header">
                  <span className="result__rank">#{i + 1}</span>
                  <span className="result__emoji" aria-hidden="true">
                    {emotion.emoji}
                  </span>
                  <span className="result__name">
                    {emotion.label} — <strong>{value}%</strong>
                  </span>
                </div>
                <div className="result__track">
                  <div className="result__fill" style={{ transform: `scaleX(${value / 100})` }} />
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
