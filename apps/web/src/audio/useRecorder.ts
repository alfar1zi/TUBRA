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
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);
  const stopRecordingRef = useRef<() => void>();
  const onCompleteRef = useRef<((blob: Blob) => void) | null>(null);

  const setOnComplete = useCallback((callback: ((blob: Blob) => void) | null) => {
    onCompleteRef.current = callback;
  }, []);

  const updateDuration = useCallback(() => {
    if (state.recording) {
      const elapsed = (Date.now() - startTimeRef.current) / 1000;
      setState((s) => ({ ...s, duration: Math.min(elapsed, 5) }));
      if (elapsed < 5) {
        animationFrameRef.current = requestAnimationFrame(updateDuration);
      }
    }
  }, [state.recording]);

  const stopRecording = useCallback(() => {
    if (workletNodeRef.current) {
      workletNodeRef.current.port.postMessage("stop");
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    setState((s) => ({ ...s, recording: false }));
  }, []);

  stopRecordingRef.current = stopRecording;

  const startRecording = useCallback(async () => {
    setState((s) => ({ ...s, error: null, audioBlob: null }));

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      });

      mediaStreamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      await audioContext.audioWorklet.addModule("/audio/recorder-processor.js");

      const source = audioContext.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioContext, "recorder-processor");
      workletNodeRef.current = workletNode;

      source.connect(workletNode);
      workletNode.connect(audioContext.destination);

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
          // Live audio buffer for real-time visualization and cough counting
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

          setState((s) => ({
            ...s,
            recording: false,
            duration: 5,
            audioBlob: blob,
            audioBuffer: float32Array,
          }));
          onCompleteRef.current?.(blob);
          stopRecordingRef.current?.();
        }
      };
      workletNode.port.postMessage("start");
      setState((s) => ({ ...s, recording: true, duration: 0 }));
      animationFrameRef.current = requestAnimationFrame(updateDuration);

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      setState((s) => ({ ...s, error: errorMsg, recording: false }));
    }
  }, [updateDuration]);

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  return { state, startRecording, stopRecording, setOnComplete };
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