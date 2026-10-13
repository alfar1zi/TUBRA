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
  calibrationDurationMs: number;
  /** Stage-2 spectral gate: mirror of the server's AudioConfig defaults. */
  useSpectralGate: boolean;
  minSpectralCentroidHz: number;
  minZeroCrossingRate: number;
  maxZeroCrossingRate: number;
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
  calibrationDurationMs: 900,
  // Mirror the server's AudioConfig spectral-gate defaults exactly.
  useSpectralGate: true,
  minSpectralCentroidHz: 350,
  minZeroCrossingRate: 0.03,
  maxZeroCrossingRate: 0.34,
};

export type CalibrationPhase = "calibrating" | "ready";

/** Frame-level RMS energy in dB, matching the server's audio/events.py. */
function frameEnergyDb(frame: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < frame.length; i++) {
    const s = frame[i];
    sum += s * s;
  }
  const rms = Math.sqrt(sum / frame.length + 1e-12);
  return 20 * Math.log10(rms + 1e-12);
}

/**
 * In-place iterative radix-2 FFT. `re`/`im` must have a power-of-two length.
 * Kept dependency-free so the live counter matches the server without pulling
 * a DSP library into the bundle.
 */
function fftInPlace(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curR = 1;
      let curI = 0;
      for (let k = 0; k < len / 2; k++) {
        const uR = re[i + k];
        const uI = im[i + k];
        const vR = re[i + k + len / 2] * curR - im[i + k + len / 2] * curI;
        const vI = re[i + k + len / 2] * curI + im[i + k + len / 2] * curR;
        re[i + k] = uR + vR;
        im[i + k] = uI + vI;
        re[i + k + len / 2] = uR - vR;
        im[i + k + len / 2] = uI - vI;
        const nextR = curR * wr - curI * wi;
        curI = curR * wi + curI * wr;
        curR = nextR;
      }
    }
  }
}

/**
 * Per-frame spectral features matching the server's `spectral_frame_features`:
 * spectral centroid (Hz) and zero-crossing rate. Uses a Hann window and the
 * same power spectrum, so the values track the server.
 */
export function frameSpectralFeatures(
  frame: Float32Array,
  sampleRate: number
): { centroid: number; zcr: number } {
  const n = frame.length;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const windowed = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
    windowed[i] = frame[i] * w;
    re[i] = windowed[i];
  }
  fftInPlace(re, im);

  const bins = n / 2 + 1;
  let powerSum = 0;
  let weighted = 0;
  const binHz = sampleRate / n;
  for (let k = 0; k < bins; k++) {
    const p = re[k] * re[k] + im[k] * im[k];
    powerSum += p;
    weighted += p * (k * binHz);
  }
  const centroid = powerSum > 0 ? weighted / powerSum : 0;
  // Server: mean(|diff(sign(windowed))|) / 2, i.e. one per sign change divided
  // by the number of samples (N-1), matching events.py exactly.
  let zc = 0;
  for (let i = 1; i < n; i++) {
    const a = Math.sign(windowed[i]);
    const b = Math.sign(windowed[i - 1]);
    zc += Math.abs(a - b) / 2;
  }
  return { centroid, zcr: zc / (n - 1) };
}

/** Linear-interpolation percentile over an already-sorted-free copy. */
function percentile(values: number[], p: number): number {
  if (values.length === 0) return -120;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length * p) / 100)));
  return sorted[idx];
}

/**
 * Live cough counter.
 *
 * Deliberately mirrors the server algorithm in `apps/api/tbcare_api/audio/events.py`
 * (`detect_cough_events`): stage 1 is short-term RMS energy with hysteresis
 * (`noise_margin_db` above the noise floor to trigger, `hysteresis_db` below to
 * release) and a min/max cough duration window; stage 2 is a spectral gate that
 * keeps a candidate only when its mean spectral centroid is high enough and its
 * zero-crossing rate falls inside a window. That gate rejects loud tones, music
 * and speech bursts, which the energy stage alone would count as coughs.
 *
 * Unlike the server (which sees the whole clip), the live counter streams, so
 * the noise floor is a rolling percentile and the spectral features are kept
 * per frame alongside the energy history.
 */
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
    calibrationDurationMs = DEFAULT_CONFIG.calibrationDurationMs,
    useSpectralGate = DEFAULT_CONFIG.useSpectralGate,
    minSpectralCentroidHz = DEFAULT_CONFIG.minSpectralCentroidHz,
    minZeroCrossingRate = DEFAULT_CONFIG.minZeroCrossingRate,
    maxZeroCrossingRate = DEFAULT_CONFIG.maxZeroCrossingRate,
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
      calibrationDurationMs,
      useSpectralGate,
      minSpectralCentroidHz,
      minZeroCrossingRate,
      maxZeroCrossingRate,
    }),
    [
      sampleRate,
      frameLength,
      hopLength,
      noisePercentile,
      noiseMarginDb,
      hysteresisDb,
      minCoughDurationMs,
      maxCoughDurationMs,
      calibrationDurationMs,
      useSpectralGate,
      minSpectralCentroidHz,
      minZeroCrossingRate,
      maxZeroCrossingRate,
    ]
  );

  const [coughCount, setCoughCount] = useState(0);
  const [events, setEvents] = useState<CoughEvent[]>([]);
  const [energyDb, setEnergyDb] = useState<Float32Array | null>(null);
  const [noiseFloorDb, setNoiseFloorDb] = useState<number | null>(null);
  const [thresholdHi, setThresholdHi] = useState<number | null>(null);
  const [thresholdLo, setThresholdLo] = useState<number | null>(null);
  const [calibrationPhase, setCalibrationPhase] = useState<CalibrationPhase>("calibrating");
  const [calibrationProgress, setCalibrationProgress] = useState(0);

  const inCoughRef = useRef(false);
  const coughStartFrameRef = useRef(0);
  const frameIndexRef = useRef(0);
  const calibrationFramesRef = useRef(0);
  const calibrationEnergyRef = useRef<number[]>([]);
  /** Every frame of energy seen so far (index = global frame). Detection is
   *  re-run over this whole history each update, so a cough that lands during
   *  the calibration window is still counted once the floor is known. */
  const allEnergyRef = useRef<number[]>([]);
  const recentEnergyRef = useRef<number[]>([]);
  const noiseFloorRef = useRef(-120);
  const isCalibratedRef = useRef(false);
  /** Per-frame spectral centroid (Hz) and ZCR, index-aligned with
   *  `allEnergyRef` so the stage-2 gate can score any candidate span. */
  const allCentroidRef = useRef<number[]>([]);
  const allZcrRef = useRef<number[]>([]);

  /** Samples already scored, so a resent take is not scored twice. */
  const processedLengthRef = useRef<number>(0);

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
      calibrationDurationMs: calMs,
      useSpectralGate,
      minSpectralCentroidHz,
      minZeroCrossingRate,
      maxZeroCrossingRate,
    } = mergedConfig;

    const minFrames = Math.ceil((minMs / 1000) * (sr / hl));
    const maxFrames = Math.floor((maxMs / 1000) * (sr / hl));
    const calibrationFrames = Math.ceil((calMs / 1000) * (sr / hl));
    // Keep ~2.5s of history for the rolling noise floor.
    const floorWindowFrames = Math.ceil(2.5 * (sr / hl));

    // The worklet resends the whole take on every stream, so only score what
    // follows the frames already scored here. Tracked locally rather than
    // taken from the message offset: React batches renders, and a newer
    // snapshot can replace an unseen one, which would otherwise skip the
    // frames in between. If the buffer restarts (a new take) start over.
    if (audioData.length < processedLengthRef.current) processedLengthRef.current = 0;
    if (audioData.length <= processedLengthRef.current + hl) return;
    const newFrames = Math.floor((audioData.length - processedLengthRef.current) / hl);
    if (newFrames <= 0) return;

    const newEnergyDb: number[] = [];
    const newCentroid: number[] = [];
    const newZcr: number[] = [];
    for (let i = 0; i < newFrames; i++) {
      const start = processedLengthRef.current + i * hl;
      const end = start + fl;
      if (end > audioData.length) break;
      const frame = audioData.subarray(start, end);
      newEnergyDb.push(frameEnergyDb(frame));
      if (useSpectralGate) {
        const { centroid, zcr } = frameSpectralFeatures(frame, sr);
        newCentroid.push(centroid);
        newZcr.push(zcr);
      }
    }
    if (newEnergyDb.length === 0) return;

    processedLengthRef.current += newEnergyDb.length * hl;

    allEnergyRef.current.push(...newEnergyDb);
    if (useSpectralGate) {
      allCentroidRef.current.push(...newCentroid);
      allZcrRef.current.push(...newZcr);
    }

    // --- Calibration: gather a short window and seed the noise floor. ---
    // Frames arriving here are NOT discarded; they are appended to the history
    // and replayed through the detector the moment the floor is known.
    if (!isCalibratedRef.current) {
      calibrationEnergyRef.current.push(...newEnergyDb);
      calibrationFramesRef.current += newEnergyDb.length;
      setCalibrationProgress(Math.min(1, calibrationFramesRef.current / calibrationFrames));

      if (calibrationFramesRef.current < calibrationFrames) {
        setEnergyDb(new Float32Array(newEnergyDb));
        return;
      }

      const floor = percentile(calibrationEnergyRef.current, np);
      noiseFloorRef.current = floor;
      recentEnergyRef.current = [...calibrationEnergyRef.current];
      isCalibratedRef.current = true;
      setNoiseFloorDb(floor);
      setThresholdHi(floor + nm);
      setThresholdLo(floor + nm - hys);
      setCalibrationPhase("ready");
      // fall through to detection over the full history (incl. calibration)
    }

    // --- Rolling noise floor, only refreshed between coughs. The trailing
    // window always covers the most recent audio so a rising room tone is
    // tracked, while the floor is recomputed from scratch each pass. ---
    const history = allEnergyRef.current;
    const windowStart = Math.max(0, history.length - floorWindowFrames);
    const floor = percentile(history.slice(windowStart), np);
    noiseFloorRef.current = floor;
    setNoiseFloorDb(floor);

    const tauHi = floor + nm;
    const tauLo = tauHi - hys;
    setThresholdHi(tauHi);
    setThresholdLo(tauLo);

    // --- Detect over the whole history (idempotent). ---
    const detected: CoughEvent[] = [];
    let inCough = false;
    let coughStart = 0;
    for (let idx = 0; idx < history.length; idx++) {
      const eDb = history[idx];
      if (!inCough && eDb > tauHi) {
        inCough = true;
        coughStart = idx;
      } else if (inCough && eDb < tauLo) {
        const durationFrames = idx - coughStart;
        if (durationFrames >= minFrames && durationFrames <= maxFrames) {
          // Stage 2: spectral gate. When enabled, keep the candidate only if
          // its mean centroid and ZCR fall inside the cough band, mirroring
          // the server's `detect_cough_events`.
          let keep = true;
          if (useSpectralGate) {
            let centroidSum = 0;
            let zcrSum = 0;
            let n = 0;
            const centroids = allCentroidRef.current;
            const zcrs = allZcrRef.current;
            for (let f = coughStart; f < idx && f < centroids.length; f++) {
              centroidSum += centroids[f];
              zcrSum += zcrs[f];
              n++;
            }
            if (n > 0) {
              const meanCentroid = centroidSum / n;
              const meanZcr = zcrSum / n;
              keep =
                meanCentroid >= minSpectralCentroidHz &&
                meanZcr >= minZeroCrossingRate &&
                meanZcr <= maxZeroCrossingRate;
            }
          }
          if (keep) {
            detected.push({
              startMs: (coughStart * hl * 1000) / sr,
              endMs: (idx * hl * 1000) / sr,
            });
          }
        }
        inCough = false;
      }
    }
    // A cough still open at the end of the buffer stays pending: it is left
    // open (inCough) and will close when the audio falls below tauLo.
    inCoughRef.current = inCough;
    coughStartFrameRef.current = coughStart;
    frameIndexRef.current = history.length;

    setEvents(detected);
    setCoughCount(detected.length);
    setEnergyDb(new Float32Array(newEnergyDb));
  }, [audioData, mergedConfig]);

  useEffect(() => {
    processFrames();
  }, [processFrames]);

  const reset = useCallback(() => {
    processedLengthRef.current = 0;
    inCoughRef.current = false;
    coughStartFrameRef.current = 0;
    frameIndexRef.current = 0;
    calibrationFramesRef.current = 0;
    calibrationEnergyRef.current = [];
    allEnergyRef.current = [];
    allCentroidRef.current = [];
    allZcrRef.current = [];
    noiseFloorRef.current = -120;
    isCalibratedRef.current = false;
    setCoughCount(0);
    setEvents([]);
    setEnergyDb(null);
    setNoiseFloorDb(null);
    setThresholdHi(null);
    setThresholdLo(null);
    setCalibrationPhase("calibrating");
    setCalibrationProgress(0);
  }, []);

  return {
    coughCount,
    events,
    energyDb,
    noiseFloorDb,
    thresholdHi,
    thresholdLo,
    calibrationPhase,
    calibrationProgress,
    reset,
  };
}
