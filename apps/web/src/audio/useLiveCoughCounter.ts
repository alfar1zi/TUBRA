import { useState, useEffect, useRef, useCallback, useMemo } from "react";

interface LiveCoughCounterConfig {
  sampleRate: number;
  frameLength: number;
  hopLength: number;
  noisePercentile: number;
  noiseMarginDb: number;
  hysteresisDb: number;
  minCoughDurationMs: number;
  maxCoughDurationMs: number;
}

interface CoughEvent {
  startMs: number;
  endMs: number;
}

const DEFAULT_CONFIG: LiveCoughCounterConfig = {
  sampleRate: 16000,
  frameLength: 1024,
  hopLength: 512,
  noisePercentile: 10,
  noiseMarginDb: 6,
  hysteresisDb: 3,
  minCoughDurationMs: 100,
  maxCoughDurationMs: 2000,
};

export function useLiveCoughCounter(
  audioData: Float32Array | null,
  config: Partial<LiveCoughCounterConfig> = {}
) {
  const {
    sampleRate = DEFAULT_CONFIG.sampleRate,
    frameLength = DEFAULT_CONFIG.frameLength,
    hopLength = DEFAULT_CONFIG.hopLength,
    noisePercentile = DEFAULT_CONFIG.noisePercentile,
    noiseMarginDb = DEFAULT_CONFIG.noiseMarginDb,
    hysteresisDb = DEFAULT_CONFIG.hysteresisDb,
    minCoughDurationMs = DEFAULT_CONFIG.minCoughDurationMs,
    maxCoughDurationMs = DEFAULT_CONFIG.maxCoughDurationMs,
  } = config;

  const mergedConfig = useMemo(
    () => ({
      sampleRate,
      frameLength,
      hopLength,
      noisePercentile,
      noiseMarginDb,
      hysteresisDb,
      minCoughDurationMs,
      maxCoughDurationMs,
    }),
    [sampleRate, frameLength, hopLength, noisePercentile, noiseMarginDb, hysteresisDb, minCoughDurationMs, maxCoughDurationMs]
  );

  const [coughCount, setCoughCount] = useState(0);
  const [events, setEvents] = useState<CoughEvent[]>([]);
  const [energyDb, setEnergyDb] = useState<Float32Array | null>(null);
  const [noiseFloorDb, setNoiseFloorDb] = useState<number | null>(null);
  const [thresholdHi, setThresholdHi] = useState<number | null>(null);
  const [thresholdLo, setThresholdLo] = useState<number | null>(null);

  const processedLengthRef = useRef(0);
  const energyHistoryRef = useRef<number[]>([]);
  const inCoughRef = useRef(false);
  const coughStartFrameRef = useRef(0);
  const frameIndexRef = useRef(0);
  const processFrames = useCallback(() => {
    if (!audioData || audioData.length === 0) return;

    const {
      sampleRate: sr,
      frameLength: fl,
      hopLength: hl,
      noisePercentile: np,
      noiseMarginDb: nm,
      hysteresisDb: hys,
      minCoughDurationMs: minMs,
      maxCoughDurationMs: maxMs,
    } = mergedConfig;

    const minFrames = Math.ceil((minMs / 1000) * sr / hl);
    const maxFrames = Math.floor((maxMs / 1000) * sr / hl);

    const newFrames = Math.floor((audioData.length - processedLengthRef.current) / hl);
    if (newFrames <= 0) return;

    const energies: number[] = [];
    const energyDbValues: number[] = [];

    for (let i = 0; i < newFrames; i++) {
      const start = processedLengthRef.current + i * hl;
      const end = start + fl;
      if (end > audioData.length) break;

      let frameEnergy = 0;
      for (let j = start; j < end; j++) {
        frameEnergy += audioData[j] * audioData[j];
      }
      frameEnergy = Math.sqrt(frameEnergy / fl + 1e-12);
      energies.push(frameEnergy);
      energyDbValues.push(20 * Math.log10(frameEnergy + 1e-12));
    }

    if (energies.length === 0) return;

    // Update processed length to match audioData length so we only process new frames next time
    processedLengthRef.current = audioData.length;
    energyHistoryRef.current.push(...energyDbValues);
    const sortedEnergy = [...energyHistoryRef.current].sort((a, b) => a - b);
    const percentileIdx = Math.floor(sortedEnergy.length * np / 100);
    const noiseFloor = sortedEnergy[percentileIdx] ?? -120;

    const tauHi = noiseFloor + nm;
    const tauLo = tauHi - hys;

    const newEvents: CoughEvent[] = [];
    let localInCough = inCoughRef.current;
    let localCoughStart = coughStartFrameRef.current;

    for (let i = 0; i < energyDbValues.length; i++) {
      const eDb = energyDbValues[i];
      const globalFrameIdx = frameIndexRef.current + i;

      if (!localInCough && eDb > tauHi) {
        localInCough = true;
        localCoughStart = globalFrameIdx;
      } else if (localInCough && eDb < tauLo) {
        const durationFrames = globalFrameIdx - localCoughStart;
        if (durationFrames >= minFrames && durationFrames <= maxFrames) {
          const startMs = (localCoughStart * hl / sr) * 1000;
          const endMs = (globalFrameIdx * hl / sr) * 1000;
          newEvents.push({ startMs, endMs });
        }
        localInCough = false;
      }
    }

    inCoughRef.current = localInCough;
    coughStartFrameRef.current = localCoughStart;
    frameIndexRef.current += energyDbValues.length;

    if (newEvents.length > 0) {
      setEvents((prev) => [...prev, ...newEvents]);
      setCoughCount((prev) => prev + newEvents.length);
    }

    setEnergyDb(new Float32Array(energyDbValues));
  }, [audioData, mergedConfig]);

  useEffect(() => {
    processFrames();
  }, [processFrames]);

  const reset = useCallback(() => {
    setCoughCount(0);
    setEvents([]);
    setEnergyDb(null);
    setNoiseFloorDb(null);
    setThresholdHi(null);
    setThresholdLo(null);
    processedLengthRef.current = 0;
    energyHistoryRef.current = [];
    inCoughRef.current = false;
    coughStartFrameRef.current = 0;
    frameIndexRef.current = 0;
  }, []);
  return {
    coughCount,
    events,
    energyDb,
    noiseFloorDb,
    thresholdHi,
    thresholdLo,
    reset,
  };
}