import { useCallback, useEffect } from "react";
import { useRecorder } from "@/audio/useRecorder";
import { useLiveCoughCounter } from "@/audio/useLiveCoughCounter";
import { Button, ProgressHeader, WaveformCanvas, LevelMeter, CoughCounter, RecordButton } from "@/components";
import { copy } from "@/copy";

interface S5RecordProps {
  onComplete: (audioBlob: Blob, coughData: { count: number; events: Array<{startMs: number, endMs: number}> }) => void;
  onBack: () => void;
}

export function S5Record({ onComplete, onBack }: S5RecordProps) {
  const { state, startRecording, stopRecording, setOnComplete } = useRecorder();

  const { coughCount, events, reset: resetCoughCounter } = useLiveCoughCounter(
    state.audioBuffer ?? null
  );

  const handleRecord = useCallback(() => {
    if (state.recording) {
      stopRecording();
    } else {
      resetCoughCounter();
      startRecording();
    }
  }, [state.recording, startRecording, stopRecording, resetCoughCounter]);

  // Auto-stop when 5 coughs detected
  useEffect(() => {
    if (state.recording && coughCount >= 5) {
      stopRecording();
    }
  }, [state.recording, coughCount, stopRecording]);

  useEffect(() => {
    setOnComplete((blob) => {
      onComplete(blob, { count: coughCount, events });
    });
  }, [coughCount, events, onComplete, setOnComplete]);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-8 px-4">
      <div className="w-full max-w-md">
        <ProgressHeader step={2} totalSteps={4} />

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h1 className="text-2xl font-semibold text-gray-900 text-center mb-1">{copy.record.title}</h1>
          <p className="text-gray-600 text-center mb-6">{copy.record.subtitle}</p>

          <CoughCounter count={coughCount} target={5} />

          <WaveformCanvas
            audioData={state.audioBuffer ?? null}
            coughMarkers={events.map((e) => e.startMs * 16)}
            maxDurationMs={30000}
          />

          <LevelMeter level="quiet" label={copy.record.qualityHint} />

          <p className="text-center text-sm text-gray-600 mt-2" aria-live="polite">
            Menunggu batuk pertama
          </p>

          <div className="flex justify-center mt-6">
            <RecordButton
              state={state.recording ? "recording" : "idle"}
              size={88}
              onClick={handleRecord}
              disabled={false}
            >
              {state.recording ? copy.record.stopButton : copy.record.recording}
            </RecordButton>
          </div>

          <div className="mt-4 space-y-3">
            <Button variant="secondary" size="md" onClick={() => {}} disabled={state.recording}>
              {copy.record.stopButton}
            </Button>
            <Button variant="ghost" size="md" onClick={onBack} disabled={state.recording}>
              {copy.common.back}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}