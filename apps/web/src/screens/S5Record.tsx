import { useCallback, useEffect, useMemo, useRef } from "react";
import { useRecorder, type RecorderErrorCode } from "@/audio/useRecorder";
import { useLiveCoughCounter } from "@/audio/useLiveCoughCounter";
import { MobileShell, WaveformCanvas, RecordButton } from "@/components";
import { copy } from "@/copy";

interface S5RecordProps {
  onComplete: (audioBlob: Blob, coughData: { count: number; events: Array<{startMs: number, endMs: number}> }) => void;
  onBack: () => void;
}

export function S5Record({ onComplete, onBack }: S5RecordProps) {
  const { state, startRecording, stopRecording, setOnComplete, resume } = useRecorder();
  const recorderError = state.error ? copy.record.recorderErrors[state.error as RecorderErrorCode] ?? copy.record.errorAudioContext : null;
  // The mic stream runs at the AudioContext's native rate (usually 44.1k or
  // 48k), not the 16k the analysis pipeline assumes. The counter and the
  // waveform must be told the real rate, otherwise every frame/hop/duration
  // calculation is off by ~3x and real coughs are never detected.
  const sampleRate = state.appliedSettings?.sampleRate ?? 16000;

  /**
   * Set once a take has finished (or the screen is being left), so the
   * auto-start effect does not immediately begin a second take. This is a
   * *completion* guard, not a "ran once" guard: React StrictMode intentionally
   * mounts, cleans up, then mounts again in dev, and the cleanup stops the
   * recording the first mount began. A "ran once" guard would then leave the
   * screen silently dead (mic granted, nothing recorded), so the second mount
   * must be allowed to start again.
   */
  const doneRef = useRef(false);
  /** Set when the user leaves via "Kembali", to suppress the flush callback. */
  const cancelledRef = useRef(false);

  const {
    coughCount,
    events,
    reset: resetCoughCounter,
    calibrationPhase,
    calibrationProgress,
    energyDb,
    noiseFloorDb,
    thresholdHi,
    thresholdLo,
  } = useLiveCoughCounter(state.audioBuffer ?? null, { sampleRate });

  // Peak of the latest energy batch, for the dev overlay. Computed with a loop
  // rather than Math.max(...energyDb): spreading a large Float32Array (tens of
  // thousands of frames at 48 kHz) overflows the call stack with RangeError.
  const peakEnergyDb = useMemo(() => {
    if (!energyDb || energyDb.length === 0) return "-";
    let peak = -Infinity;
    for (let i = 0; i < energyDb.length; i++) {
      if (energyDb[i] > peak) peak = energyDb[i];
    }
    return Number.isFinite(peak) ? peak.toFixed(1) : "-";
  }, [energyDb]);

  useEffect(() => {
    // Start a take whenever the screen is idle. `doneRef` blocks a fresh take
    // only after one has genuinely finished; a StrictMode cleanup (which stops
    // the first take and then re-runs this effect) is therefore allowed to
    // start again instead of leaving the screen dead.
    if (doneRef.current) return;
    if (cancelledRef.current) return;
    if (state.recording || state.error) return;
    startRecording();
  }, [state.recording, state.error, startRecording]);

  // The mount-time auto-start usually runs before any gesture, so the browser
  // autoplay policy can leave the AudioContext "suspended" and silent even
  // though the mic was granted. The first gesture of any kind resumes it, so
  // the user only has to touch the screen (they already tap "allow" and the
  // record button) for capture to begin.
  useEffect(() => {
    const events: Array<keyof WindowEventMap> = ["pointerdown", "keydown", "touchstart"];
    events.forEach((e) => window.addEventListener(e, resume, { passive: true }));
    return () => events.forEach((e) => window.removeEventListener(e, resume));
  }, [resume]);

  const handleRecord = useCallback(() => {
    // This click is a real user gesture: use it to un-suspend the AudioContext
    // in case the autoplay policy blocked the mount-time auto-start (otherwise
    // the graph never runs and no audio is captured despite a granted mic).
    resume();
    if (state.recording) {
      stopRecording();
    } else {
      if (calibrationPhase === "calibrating") {
        return;
      }
      resetCoughCounter();
      startRecording();
    }
  }, [state.recording, startRecording, stopRecording, resetCoughCounter, calibrationPhase, resume]);

  const handleBack = useCallback(() => {
    // Mark the run as cancelled *before* stopping, so the worklet's flush does
    // not fire `onComplete` and bounce us forward to the quality screen.
    cancelledRef.current = true;
    stopRecording();
    onBack();
  }, [stopRecording, onBack]);

  // Auto-stop when 5 coughs detected
  useEffect(() => {
    if (state.recording && coughCount >= 5) {
      stopRecording();
    }
  }, [state.recording, coughCount, stopRecording]);

  useEffect(() => {
    setOnComplete((blob) => {
      if (cancelledRef.current) return;
      // The take is over: keep the auto-start effect from beginning another.
      doneRef.current = true;
      onComplete(blob, { count: coughCount, events });
    });
  }, [coughCount, events, onComplete, setOnComplete]);

  const isCalibrating = calibrationPhase === "calibrating";

  return (
    <MobileShell>
      <div
        className="flex flex-col flex-1"
        style={{ paddingTop: "calc(env(safe-area-inset-top, 62px) + 24px)" }}
      >
        <header className="text-center">
          <h1 className="text-xl font-medium leading-[26px] text-text-primary/80">
            {copy.record.title}
          </h1>
          <h2 className="mt-[24px] text-[22px] font-medium leading-[29px] text-text-primary/80">
            {isCalibrating ? copy.record.calibratingTitle : copy.record.subtitle}
          </h2>
          <p className="mt-[8px] text-base font-normal leading-[21px] text-text-primary/60">
            {isCalibrating ? copy.record.calibratingSubtitle : copy.record.hint}
          </p>
          {recorderError && (
            <p className="mt-[12px] text-base font-medium leading-[21px] text-danger" role="alert">
              {recorderError}
            </p>
          )}
        </header>

        {/* The Voice stage: waveform row, live counter, and the record control,
            laid out to Figma node 20:146. The group is vertically centred in the
            free space between the header and the footer (rather than pinned by
            fixed top margins) so it stays put across phone heights. */}
        <main className="relative flex-1 flex flex-col items-center justify-center gap-8">
          <WaveformCanvas
            audioData={state.audioBuffer ?? null}
            coughMarkers={events.map((e) => Math.round((e.startMs / 1000) * sampleRate))}
            windowMs={5000}
            sampleRate={sampleRate}
            className="!h-[60px]"
          />

          {/* The counter ticks many times per second; making it aria-live
              floods screen readers. Keep it silent and announce only discrete
              milestones in the dedicated live region rendered at the end of
              this screen. */}
          <p className="text-center text-base font-normal leading-[21px] text-text-primary/80">
            {isCalibrating
              ? `${copy.record.calibratingProgress} ${Math.round(calibrationProgress * 100)}%`
              : copy.record.recording.replace("{n}", String(coughCount))}
          </p>

          {isCalibrating && (
            <div className="w-full rounded-lg border border-brand/20 bg-brand-soft/40 p-3 text-center">
              <p className="text-sm font-medium text-brand">{copy.record.calibratingCoughHint}</p>
              <div className="mt-2 h-2 overflow-hidden rounded-pill bg-brand-soft">
                <div
                  className="h-full bg-brand transition-all duration-300"
                  style={{ width: `${calibrationProgress * 100}%` }}
                />
              </div>
            </div>
          )}

          {import.meta.env.DEV && !isCalibrating && (
            <div className="w-full rounded-lg border border-dashed border-black/20 bg-black/[0.03] p-2 text-left font-mono text-[11px] leading-4 text-text-primary/70">
              <div>floor {noiseFloorDb?.toFixed(1) ?? "-"} dB</div>
              <div>ambang {thresholdHi?.toFixed(1) ?? "-"} / {thresholdLo?.toFixed(1) ?? "-"} dB</div>
              <div>level {peakEnergyDb} dB</div>
              <div>batuk {coughCount}</div>
            </div>
          )}

          {/* Record button centred on the row; Refresh control to its left
              (Figma Frame 8745). */}
          <div className="relative w-full flex items-center justify-center">
            <button
              type="button"
              onClick={resetCoughCounter}
              disabled={state.recording}
              aria-label="Reset hitungan"
              className="absolute left-0 grid h-[60px] w-[60px] place-items-center rounded-full border border-black bg-white text-text-primary/80 disabled:opacity-40"
            >
              <svg width="34" height="34" viewBox="0 0 34 34" fill="none" aria-hidden="true">
                <path
                  d="M17 24.5a7.5 7.5 0 1 1 7.3-9.2l-2.1.6A5.3 5.3 0 1 0 17 22.3V24.5Z"
                  fill="currentColor"
                />
                <path d="M17 6v6l5-3-5-3Z" fill="currentColor" />
              </svg>
            </button>
            <RecordButton
              state={state.recording ? "recording" : "idle"}
              size={82}
              onClick={handleRecord}
              disabled={isCalibrating}
            >
              {state.recording
                ? copy.record.stopButton
                : isCalibrating
                ? copy.record.calibratingButton
                : copy.record.recording.replace("{n}", "0")}
            </RecordButton>
          </div>
        </main>

        <div
          className="flex gap-2"
          style={{ paddingBottom: "calc(env(safe-area-inset-bottom, 34px) + 13px)" }}
        >
          <button
            type="button"
            onClick={handleBack}
            className="flex-1 h-[57px] rounded-lg bg-brand-soft text-lg font-medium text-brand transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.questions.backButton}
          </button>
          <button
            type="button"
            onClick={state.recording ? stopRecording : undefined}
            disabled={!state.recording}
            className="flex-1 h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.questions.nextButton}
          </button>
        </div>

        {/* Discrete status announcements for assistive tech. The visible
            counters above tick continuously and would spam a live region. */}
        <p className="sr-only" aria-live="polite">
          {recorderError
            ? recorderError
            : isCalibrating
            ? copy.record.calibratingTitle
            : state.recording
            ? copy.record.subtitle
            : ""}
        </p>
      </div>
    </MobileShell>
  );
}