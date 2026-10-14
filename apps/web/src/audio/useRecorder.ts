import { useState, useCallback, useRef, useEffect } from "react";

interface AudioSettings {
  sampleRate: number;
  channelCount: number;
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
}

interface RecorderState {
  recording: boolean;
  duration: number;
  audioBlob: Blob | null;
  audioBuffer: Float32Array | null;
  appliedSettings: AudioSettings | null;
  error: string | null;
}

export function useRecorder() {
  const [state, setState] = useState<RecorderState>({
    recording: false,
    duration: 0,
    audioBlob: null,
    audioBuffer: null,
    appliedSettings: null,
    error: null,
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);
  const recordingRef = useRef<boolean>(false);
  const stopRecordingRef = useRef<() => void>();
  const onCompleteRef = useRef<((blob: Blob) => void) | null>(null);
  /**
   * Monotonic token identifying the current start attempt. Each call to
   * `startRecording` claims the next value; an async continuation compares the
   * token it captured against this ref and bails (releasing the mic) if a newer
   * attempt has superseded it, or if `stopRecording` bumped it.
   *
   * A plain boolean cannot express this: React StrictMode runs start #1,
   * cleanup (stop), then start #2, and resetting a shared boolean for start #2
   * would silently un-abort start #1, leaving two live mic graphs fighting.
   */
  const startGenerationRef = useRef<number>(0);
  /** True once the worklet has delivered its first buffer this run. */
  const streamedRef = useRef<boolean>(false);
  /** Fires if no buffer arrives shortly after start (suspended-context guard). */
  const noAudioTimerRef = useRef<number | null>(null);
  /** True while startRecording is between its first line and settling. */
  const startingRef = useRef<boolean>(false);
  /** Safety timer for the stop->flush->teardown handshake. */
  const flushTimerRef = useRef<number | null>(null);

  const setOnComplete = useCallback((callback: ((blob: Blob) => void) | null) => {
    onCompleteRef.current = callback;
  }, []);

  const updateDuration = useCallback(() => {
    // Read `recording` from a ref, not state: this callback is scheduled with
    // requestAnimationFrame and would otherwise capture a stale `recording`
    // value and keep rescheduling after the recording stopped.
    if (recordingRef.current) {
      const elapsed = (Date.now() - startTimeRef.current) / 1000;
      setState((s) => ({ ...s, duration: Math.min(elapsed, 5) }));
      if (elapsed < 5) {
        animationFrameRef.current = requestAnimationFrame(updateDuration);
      }
    }
  }, []);

  /**
   * Releases every live resource (nodes, mic track, context). Separate from
   * `stopRecording` so the flush handshake can happen first: closing the
   * AudioContext tears down the worklet thread, and doing that immediately
   * after asking it to flush loses the final "complete" message that carries
   * the recorded audio.
   */
  const teardown = useCallback(() => {
    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (noAudioTimerRef.current !== null) {
      window.clearTimeout(noAudioTimerRef.current);
      noAudioTimerRef.current = null;
    }
    if (flushTimerRef.current !== null) {
      window.clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
    recordingRef.current = false;
    setState((s) => ({ ...s, recording: false }));
  }, []);

  const stopRecording = useCallback(() => {
    // Supersede any in-flight startRecording() so its async continuation stops
    // touching the mic once it resumes after an await.
    startGenerationRef.current += 1;
    startingRef.current = false;

    // If a graph is live, ask the worklet to flush and let its "complete"
    // message drive the teardown, so the final buffer is not lost. A safety
    // timer tears down anyway if the message never arrives (e.g. the worklet
    // was already gone).
    if (workletNodeRef.current && mediaStreamRef.current) {
      workletNodeRef.current.port.postMessage("stop");
      if (flushTimerRef.current !== null) window.clearTimeout(flushTimerRef.current);
      flushTimerRef.current = window.setTimeout(() => {
        flushTimerRef.current = null;
        teardown();
      }, 800);
      return;
    }
    teardown();
  }, [teardown]);

  stopRecordingRef.current = stopRecording;

  const startRecording = useCallback(async () => {
    // Claim this attempt's token. Any previous in-flight start now sees a
    // newer generation and releases itself at its next checkpoint, so the two
    // never both hold a live mic graph. `startingRef` additionally suppresses
    // a *synchronous* second call before the first has claimed its token.
    if (startingRef.current) return;
    startingRef.current = true;
    const generation = (startGenerationRef.current += 1);
    streamedRef.current = false;
    setState((s) => ({ ...s, error: null, audioBlob: null }));

    // True once a newer start (or stopRecording) has superseded this attempt.
    const isStale = (): boolean => generation !== startGenerationRef.current;

    // Releases only the handles *this* attempt acquired. It must never touch
    // the shared refs, because a superseded attempt would otherwise close the
    // media stream / AudioContext the newer attempt is already running on.
    const release = (owned: {
      stream?: MediaStream | null;
      context?: AudioContext | null;
      source?: AudioNode | null;
      worklet?: AudioWorkletNode | null;
    }): void => {
      owned.source?.disconnect();
      owned.worklet?.disconnect();
      owned.stream?.getTracks().forEach((t) => t.stop());
      if (owned.context) void owned.context.close();
    };

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new DOMException("microphone unavailable", "NotSupportedError");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      });
      if (isStale()) {
        release({ stream });
        return;
      }
      mediaStreamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      // Autoplay policy: an AudioContext created outside a user gesture (the
      // record screen auto-starts on mount) begins "suspended". A suspended
      // context never runs the graph, so the worklet's process() is never
      // called and no audio is ever captured, even though getUserMedia
      // succeeded and the UI shows "recording".
      //
      // Do NOT await resume() unbounded: with the autoplay policy blocking the
      // context, resume() can return a promise that never settles (there is no
      // user gesture yet), which deadlocks the whole start and freezes the
      // screen on "0%" forever. Kick it off, wait at most a moment for it to
      // take effect, then continue: the graph is built either way, and a later
      // gesture (tapping the record button, or anywhere on the page) resumes it.
      if (audioContext.state === "suspended") {
        await Promise.race([
          audioContext.resume().catch(() => {}),
          new Promise<void>((resolve) => window.setTimeout(resolve, 250)),
        ]);
      }
      if (isStale()) {
        release({ stream, context: audioContext });
        return;
      }

      if (!audioContext.audioWorklet) {
        throw new DOMException("AudioWorklet unavailable", "NotSupportedError");
      }
      await audioContext.audioWorklet.addModule("/audio/recorder-processor.js");
      if (isStale()) {
        release({ stream, context: audioContext });
        return;
      }

      if (isStale()) {
        release({ stream, context: audioContext });
        return;
      }

      const source = audioContext.createMediaStreamSource(stream);
      sourceNodeRef.current = source;
      const workletNode = new AudioWorkletNode(audioContext, "recorder-processor");
      workletNodeRef.current = workletNode;

      source.connect(workletNode);
      // The worklet must be connected to the destination so the graph keeps
      // pulling render quanta, but connecting it directly plays the live mic
      // through the speakers (a loud feedback howl on a laptop). Route it
      // through a zero-gain node: the graph still runs, nothing is audible.
      const silentSink = audioContext.createGain();
      silentSink.gain.value = 0;
      workletNode.connect(silentSink);
      silentSink.connect(audioContext.destination);

      const track = stream.getAudioTracks()[0];
      const settings = track.getSettings();

      setState((s) => ({
        ...s,
        appliedSettings: {
          sampleRate: audioContext.sampleRate,
          channelCount: settings.channelCount ?? 1,
          echoCancellation: settings.echoCancellation ?? false,
          noiseSuppression: settings.noiseSuppression ?? false,
          autoGainControl: settings.autoGainControl ?? false,
        },
      }));

      workletNode.port.onmessage = (e) => {
        if (e.data.type === "stream") {
          // Live audio buffer for real-time visualization and cough counting.
          // Its arrival is the proof the graph is actually running.
          streamedRef.current = true;
          setState((s) => ({ ...s, audioBuffer: e.data.data }));
          return;
        }
        if (e.data.type === "complete") {
          const float32Array = e.data.data;
          const int16Array = new Int16Array(float32Array.length);
          for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 32768 : s * 32767;
          }

          const wavBuffer = encodeWAV(int16Array, audioContext.sampleRate);
          const blob = new Blob([wavBuffer], { type: "audio/wav" });

          recordingRef.current = false;
          setState((s) => ({
            ...s,
            recording: false,
            duration: 5,
            audioBlob: blob,
            audioBuffer: float32Array,
          }));
          onCompleteRef.current?.(blob);
          // The flush handshake is done: release the graph now (this also
          // clears the stopRecording safety timer).
          teardown();
        }
      };
      workletNode.port.postMessage("start");
      recordingRef.current = true;
      setState((s) => ({ ...s, recording: true, duration: 0 }));
      animationFrameRef.current = requestAnimationFrame(updateDuration);

      // Watchdog: a running graph streams a buffer every ~100ms. If nothing
      // arrives, the context stayed suspended (autoplay policy) and we would
      // otherwise sit in a fake "recording" state that captures the mic but
      // delivers no audio. Surface it as an error the UI can act on instead.
      noAudioTimerRef.current = window.setTimeout(() => {
        if (recordingRef.current && !streamedRef.current) {
          stopRecordingRef.current?.();
          setState((s) => ({ ...s, error: "audio_context", recording: false }));
        }
      }, 1500);

      startingRef.current = false;
    } catch (err) {
      // Release anything already acquired (mic track, AudioContext, worklet)
      // so a failed start does not leak a live microphone.
      stopRecordingRef.current?.();
      startingRef.current = false;
      // Map the raw DOM/engine message to a stable machine-readable code; the
      // UI (S4/S5 copy) decides how to phrase it, we do not surface engine text.
      const errorMsg = toRecorderErrorCode(err);
      setState((s) => ({ ...s, error: errorMsg, recording: false }));
    }
  }, [teardown, updateDuration]);

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  /** Resumes the live AudioContext from a user gesture; see the helper below. */
  const resume = useCallback(() => resumeAudioContext(audioContextRef.current), []);

  return { state, startRecording, stopRecording, setOnComplete, resume };
}

function encodeWAV(samples: Int16Array, sampleRate: number): ArrayBuffer {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(offset, samples[i], true);
    offset += 2;
  }

  return buffer;
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

/** Stable, UI-mappable error codes for the recorder. */
export type RecorderErrorCode =
  | "mic_permission"
  | "mic_not_found"
  | "audio_context";

/**
 * Maps a thrown error to a stable code so the UI can show a localised, curated
 * message instead of raw engine text (which is English, verbose, and leaks
 * browser internals). See `copy.record.error*`.
 */
export function toRecorderErrorCode(err: unknown): RecorderErrorCode {
  const name = err instanceof DOMException ? err.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "mic_permission";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "mic_not_found";
  }
  return "audio_context";
}

/**
 * Resumes a live AudioContext that the autoplay policy left "suspended".
 * A context created outside a user gesture (the record screen auto-starts on
 * mount) may stay suspended until resumed inside a real gesture task, so the
 * record button calls this as a fallback. Returns true when it is running.
 */
export function resumeAudioContext(ctx: AudioContext | null): boolean {
  if (!ctx) return false;
  if (ctx.state === "suspended") {
    void ctx.resume();
  }
  return ctx.state === "running";
}