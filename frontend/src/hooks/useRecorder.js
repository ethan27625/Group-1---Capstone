import { useCallback, useEffect, useRef, useState } from 'react';

export const RECORD_DURATION_MS = 5_000;

/**
 * Owns the browser microphone lifecycle: permission, MediaRecorder,
 * an elapsed-time ticker, and auto-stop at RECORD_DURATION_MS.
 * The finished audio Blob is handed to `onComplete`.
 */
export function useRecorder(onComplete) {
  const [isRecording, setIsRecording] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [micError, setMicError] = useState(null);

  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const cleanup = useCallback(() => {
    clearInterval(timerRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    recorderRef.current = null;
  }, []);

  // Release the mic if the component unmounts mid-recording.
  useEffect(() => cleanup, [cleanup]);

  const stop = useCallback(() => {
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.stop(); // fires onstop → onComplete
    }
  }, []);

  const start = useCallback(async () => {
    setMicError(null);

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setMicError(
        err.name === 'NotAllowedError'
          ? 'Microphone access is blocked. Allow it in your browser settings, then try again.'
          : 'No microphone found. Connect one and try again.'
      );
      return false;
    }

    const chunks = [];
    const recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: recorder.mimeType });
      setIsRecording(false);
      setElapsedMs(0); // reset the counter for the next take
      cleanup();
      onCompleteRef.current(blob);
    };

    streamRef.current = stream;
    recorderRef.current = recorder;
    recorder.start();
    setIsRecording(true);
    setElapsedMs(0);

    const startedAt = performance.now();
    timerRef.current = setInterval(() => {
      const elapsed = performance.now() - startedAt;
      if (elapsed >= RECORD_DURATION_MS) {
        setElapsedMs(RECORD_DURATION_MS);
        stop();
      } else {
        setElapsedMs(elapsed);
      }
    }, 100);

    return true;
  }, [cleanup, stop]);

  return { isRecording, elapsedMs, micError, start, stop };
}
