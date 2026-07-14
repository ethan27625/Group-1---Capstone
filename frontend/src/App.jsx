import { useCallback, useState } from 'react';
import { EMOTIONS, rankEmotions } from './lib/emotions';
import { predictEmotion } from './lib/predictEmotion';
import { useRecorder } from './hooks/useRecorder';
import EmotionTile from './components/EmotionTile';
import MicButton from './components/MicButton';
import ProgressBar from './components/ProgressBar';
import ResultsPanel from './components/ResultsPanel';

// Phases: idle → recording → analyzing → result (then back to recording).
export default function App() {
  const [phase, setPhase] = useState('idle');
  const [scores, setScores] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);

  const handleAudio = useCallback(async (blob) => {
    setPhase('analyzing');
    setAnalysisError(null);
    try {
      const result = await predictEmotion(blob);
      setScores(result);
      setPhase('result');
    } catch (err) {
      setAnalysisError(err.message);
      setPhase('idle');
    }
  }, []);

  const { isRecording, elapsedMs, micError, start, stop } = useRecorder(handleAudio);

  const handleMicTap = async () => {
    if (isRecording) {
      stop();
      return;
    }
    if (phase === 'analyzing') return;
    if (await start()) setPhase('recording');
  };

  const hasResult = phase === 'result' && scores;
  const ranking = hasResult ? rankEmotions(scores) : [];

  return (
    <div className="app">
      <header className="nav">
        <div className="nav__inner">
          <span className="nav__wordmark" translate="no">
            Sound Sense
          </span>
          <span className="nav__badge">
            <span className="nav__badge-dot" aria-hidden="true" />
            Live demo
          </span>
        </div>
      </header>

      <main className="container">
        <section className="card analysis" aria-label="Emotion analysis">
          <div className="analysis__header">
            <h1 className="analysis__title">Emotion analysis</h1>
            <p className="analysis__subtitle">
              Speak for up to 5 seconds — confidence per emotion updates after each take.
            </p>
          </div>
          <div className="analysis__grid" role="list">
            {EMOTIONS.map(({ key, label, emoji }) => (
              <EmotionTile
                key={key}
                emoji={emoji}
                label={label}
                value={hasResult ? scores[key] : 0}
                rank={ranking.indexOf(key)}
                shimmer={phase === 'analyzing'}
              />
            ))}
          </div>
        </section>

        <section className="card controls" aria-label="Recording controls">
          <MicButton
            isRecording={isRecording}
            disabled={phase === 'analyzing'}
            onClick={handleMicTap}
          />
          <div className="controls__progress">
            <ProgressBar isRecording={isRecording} elapsedMs={elapsedMs} />
            {(micError || analysisError) && (
              <p className="controls__error" role="alert">
                {micError || analysisError}
              </p>
            )}
            {phase === 'analyzing' && (
              <p className="controls__status" role="status">
                Analyzing your recording…
              </p>
            )}
          </div>
        </section>

        <ResultsPanel scores={hasResult ? scores : null} ranking={ranking} />
      </main>
    </div>
  );
}
