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
  /**
   * Longest burst accepted as one cough, far shorter than the config ceiling.
   * Measured on the CODA-TB coughs (160 recordings) every cough burst ran
   * 12-17 frames, while speech, a held note and a snore run 39-49 frames and
   * a finger tap 2. Keeping the burst inside 6-28 frames is what separates
   * them; spectral features alone did not (a third of real coughs have a
   * zero-crossing rate as high as a hand clap).
   */
  maxBurstFrames: number;
  /**
   * Upper bound on the zero-crossing rate, shape check only. It is not the
   * separator it was once assumed to be: 31% of real cough recordings exceed
   * 0.34, so this stays loose and the burst length does the work.
   */
  maxZeroCrossingRate: number;
  /**
   * Extra margin over the calibrated floor, as a fraction of its dynamic
   * range. The take starts with ~1 s of calibration audio and keeps running
   * on the same floor, but a browser capture is not the pre-processed signal
   * the server sees: Chrome may later route audio through echo processing and
   * the room changes when the phone is brought to the mouth. A too-low floor
   * then counts every breath as a cough. Raising the trigger by this share of
   * (loudest frame - floor), capped by `maxAdaptiveMarginDb`, keeps real
   * coughs (20-30 dB over the room) while rejecting breathing.
   */
  adaptiveMarginRatio: number;
  maxAdaptiveMarginDb: number;
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
  minCoughDurationMs: 190,
  maxCoughDurationMs: 900,
  calibrationDurationMs: 900,
  // Mirror the server's AudioConfig spectral-gate defaults exactly.
  useSpectralGate: true,
  minSpectralCentroidHz: 350,
  maxBurstFrames: 28,
  maxZeroCrossingRate: 0.7,
  adaptiveMarginRatio: 0.2,
  maxAdaptiveMarginDb: 12,
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
 * Mirrors the server algorithm in `apps/api/tbcare_api/audio/events.py`
 * (`detect_cough_events`) so what the user sees while recording is what the
 * server will count: stage 1 is short-term RMS energy with hysteresis (above
 * the noise floor plus `noiseMarginDb` to trigger, `hysteresisDb` below to
 * release) inside a min/max cough duration window; stage 2 is a spectral gate
 * that keeps a candidate only when its mean spectral centroid and
 * zero-crossing rate look like a cough rather than a tone, a tap or speech.
 *
 * Two properties matter for a live count:
 *
 * - The noise floor is fixed once calibration ends, like the server's global
 *   percentile. A rolling floor moved the thresholds while the user was still
 *   coughing, so a counted cough could be "uncounted" a moment later and the
 *   counter dropped back to 0.
 * - History is scored once, incrementally: only frames after the last scored
 *   frame are examined and a closed candidate is final, so the count only ever
 *   grows within one take.
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
    maxZeroCrossingRate = DEFAULT_CONFIG.maxZeroCrossingRate,
    maxBurstFrames = DEFAULT_CONFIG.maxBurstFrames,
    adaptiveMarginRatio = DEFAULT_CONFIG.adaptiveMarginRatio,
    maxAdaptiveMarginDb = DEFAULT_CONFIG.maxAdaptiveMarginDb,
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
      maxZeroCrossingRate,
      maxBurstFrames,
      adaptiveMarginRatio,
      maxAdaptiveMarginDb,
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
      maxZeroCrossingRate,
      maxBurstFrames,
      adaptiveMarginRatio,
      maxAdaptiveMarginDb,
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
  /** Peak of the latest batch, for the take's clipping check. */
  const [peakAmplitude, setPeakAmplitude] = useState(0);

  const calibrationFramesRef = useRef(0);
  const calibrationEnergyRef = useRef<number[]>([]);
  /** Every frame of energy seen so far (index = global frame). */
  const allEnergyRef = useRef<number[]>([]);
  /** Per-frame spectral centroid (Hz) and ZCR, index-aligned with the energy. */
  const allCentroidRef = useRef<number[]>([]);
  const allZcrRef = useRef<number[]>([]);
  /** Samples already scored, so a resent take is not scored twice. */
  const processedLengthRef = useRef<number>(0);
  /** True once calibration seeded the floor; reset per take. */
  const isCalibratedRef = useRef(false);
  /** Loudest frame since calibration, used to lift the floor adaptively. */
  const maxEnergyRef = useRef(-Infinity);
  /** Thresholds fixed at the end of calibration; +-Infinity until then. */
  const tauHiRef = useRef<number>(Infinity);
  const tauLoRef = useRef<number>(Infinity);
  /** Open candidate at the tail of the audio, and the events closed so far. */
  const inCoughRef = useRef(false);
  const coughStartFrameRef = useRef(0);
  const closedEventsRef = useRef<CoughEvent[]>([]);
  /** Frames already through the state machine, so a batch only scores new ones. */
  const scoredFramesRef = useRef(0);
  /** Set after a forced cut; blocks reopening until the level falls under `tauLo`. */
  const holdingRef = useRef(false);

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
      maxZeroCrossingRate,
      maxBurstFrames,
    } = mergedConfig;

    const minFrames = Math.ceil((minMs / 1000) * (sr / hl));
    let maxFrames = Math.floor((maxMs / 1000) * (sr / hl));
    // `maxBurstFrames` is calibrated at the analysis rate the rule was
    // measured at (16 kHz). The frame length is fixed in samples, so at the
    // microphone's native 44.1/48 kHz the same cough spans three times as
    // many frames: comparing against the raw 16 kHz number rejected every
    // real cough on a phone. Scale the budget with the sample rate.
    const maxBurstAtRate = Math.round((maxBurstFrames * sr) / DEFAULT_CONFIG.sampleRate);
    // A long, loud stretch (the user coughed straight into the mic, or the
    // room got louder than the calibrated floor) can stay above `tauHi` for
    // seconds, so the release check below never fires. Close it as its own
    // event once it is too long to be one cough instead of letting the tail
    // swallow every later cough.
    if (maxFrames < minFrames) maxFrames = minFrames + 1;
    const calibrationFrames = Math.ceil((calMs / 1000) * (sr / hl));

    // The worklet resends the whole take on every stream, so only score what
    // follows the frames already scored here. If the buffer restarts (a new
    // take), start over.
    if (audioData.length < processedLengthRef.current) {
      // A new take: do not carry the old take's in-progress state over.
      processedLengthRef.current = 0;
      inCoughRef.current = false;
      holdingRef.current = false;
      coughStartFrameRef.current = 0;
      scoredFramesRef.current = 0;
    }
    if (audioData.length <= processedLengthRef.current + hl) return;
    const newFrames = Math.floor((audioData.length - processedLengthRef.current) / hl);
    if (newFrames <= 0) return;

    const newEnergyDb: number[] = [];
    const newCentroid: number[] = [];
    const newZcr: number[] = [];
    let peak = 0;
    for (let i = 0; i < newFrames; i++) {
      const start = processedLengthRef.current + i * hl;
      const end = start + fl;
      if (end > audioData.length) break;
      const frame = audioData.subarray(start, end);
      for (let k = 0; k < frame.length; k++) {
        const a = Math.abs(frame[k]);
        if (a > peak) peak = a;
      }
      newEnergyDb.push(frameEnergyDb(frame));
      if (useSpectralGate) {
        const { centroid, zcr } = frameSpectralFeatures(frame, sr);
        newCentroid.push(centroid);
        newZcr.push(zcr);
      }
    }
    if (newEnergyDb.length === 0) return;

    processedLengthRef.current += newEnergyDb.length * hl;
    if (peak > peakAmplitude) setPeakAmplitude(peak);
    allEnergyRef.current.push(...newEnergyDb);
    if (useSpectralGate) {
      allCentroidRef.current.push(...newCentroid);
      allZcrRef.current.push(...newZcr);
    }

    // --- Calibration: gather a short window and seed the noise floor. ---
    // Frames arriving here are kept: detection below replays them once the
    // floor is known, so a cough during calibration is still counted.
    if (calibrationFramesRef.current < calibrationFrames) {
      calibrationEnergyRef.current.push(...newEnergyDb);
      calibrationFramesRef.current += newEnergyDb.length;
      setCalibrationProgress(Math.min(1, calibrationFramesRef.current / calibrationFrames));
      setEnergyDb(new Float32Array(newEnergyDb));
      if (calibrationFramesRef.current < calibrationFrames) return;

      const floor = percentile(calibrationEnergyRef.current, np);
      tauHiRef.current = floor + nm;
      tauLoRef.current = floor + nm - hys;
      isCalibratedRef.current = true;
      setNoiseFloorDb(floor);
      setThresholdHi(tauHiRef.current);
      setThresholdLo(tauLoRef.current);
      setCalibrationPhase("ready");
    }

    // Lift the trigger with the loudest frame heard so far (see
    // `adaptiveMarginRatio`), never above the calibrated value plus the cap.
    const tauLo = tauLoRef.current;
    let tauHi = tauHiRef.current;
    for (const e of newEnergyDb) if (e > maxEnergyRef.current) maxEnergyRef.current = e;
    const loudest = maxEnergyRef.current;
    if (Number.isFinite(loudest)) {
      const margin =
        hysteresisDb + Math.min(maxAdaptiveMarginDb, adaptiveMarginRatio * (loudest - (tauHi - noiseMarginDb)));
      tauHi = tauHi - noiseMarginDb + margin;
    }
    const history = allEnergyRef.current;
    const centroids = allCentroidRef.current;
    const zcrs = allZcrRef.current;
    const closed = closedEventsRef.current;

    /**
     * True when the frames in [from, to) look like one cough: a burst of the
     * right length and not pitched low. The length bound carries the weight
     * (see `maxBurstFrames`); the spectral floor rejects tapping and thuds.
     */
    const isCough = (from: number, to: number): boolean => {
      const frames = to - from;
      if (frames < minFrames || frames > maxBurstAtRate) return false;
      if (!useSpectralGate) return true;
      // Measure the spectral features over the loud frames only, so speech
      // that trails into a cough (and shares the candidate window) does not
      // drag the centroid down toward the voice's fundamental.
      let peak = -Infinity;
      for (let f = from; f < to && f < history.length; f++) {
        if (history[f] > peak) peak = history[f];
      }
      let centroid = 0;
      let zcr = 0;
      let loud = 0;
      for (let f = from; f < to && f < history.length; f++) {
        if (history[f] < peak - 6) continue;
        centroid += centroids[f];
        zcr += zcrs[f];
        loud++;
      }
      if (loud === 0) return false;
      centroid /= loud;
      zcr /= loud;
      return centroid >= minSpectralCentroidHz && zcr <= maxZeroCrossingRate;
    };

    /**
     * Scores one above-threshold stretch. When the stretch is short enough to
     * be a single cough it is tested as a whole. When it is longer (speech
     * that never falls back under the release threshold, a room that got
     * loud), a cough inside it would be averaged away, so every window of
     * one-cough length is tested on its own and a passing window is kept.
     * Windows are spaced half a window apart so a burst that straddles a
     * boundary is still seen whole by the neighbouring window.
     */
    const closeCandidate = (startFrame: number, endFrame: number): void => {
      if (!isCough(startFrame, endFrame)) return;
      closed.push({
        startMs: (startFrame * hl * 1000) / sr,
        endMs: (endFrame * hl * 1000) / sr,
      });
    };

    // --- Detect, incrementally, over the frames not scored yet. ---
    // Starting at frame 0 every time replayed the state machine over the whole
    // take: a stretch still open from the previous batch was closed against an
    // `idx` from the start of the take and the same cough was pushed again on
    // every batch. One pass per frame, ever.
    const fromFrame = scoredFramesRef.current;
    scoredFramesRef.current = history.length;
    for (let idx = fromFrame; idx < history.length; idx++) {
      const eDb = history[idx];
      if (inCoughRef.current && (eDb < tauLo || idx - coughStartFrameRef.current > maxFrames)) {
        // A stretch cut at `maxFrames` was still loud when cut, so it is a
        // sustained sound, not a cough: drop it rather than scoring a fraction
        // of it. Stay held until the level really falls, or the rest of that
        // sound is picked up as a fresh burst.
        const forced = eDb >= tauLo;
        if (!forced) closeCandidate(coughStartFrameRef.current, idx);
        inCoughRef.current = false;
        holdingRef.current = forced;
      } else if (!inCoughRef.current && holdingRef.current && eDb < tauLo) {
        holdingRef.current = false;
      } else if (!inCoughRef.current && !holdingRef.current && eDb > tauHi) {
        inCoughRef.current = true;
        coughStartFrameRef.current = idx;
      }
    }

    setEvents([...closed]);
    setCoughCount(closed.length);
    setEnergyDb(new Float32Array(newEnergyDb));
  }, [
    audioData,
    mergedConfig,
    peakAmplitude,
    adaptiveMarginRatio,
    hysteresisDb,
    maxAdaptiveMarginDb,
    noiseMarginDb,
  ]);

  useEffect(() => {
    processFrames();
  }, [processFrames]);

  const reset = useCallback(() => {
    processedLengthRef.current = 0;
    inCoughRef.current = false;
    holdingRef.current = false;
    coughStartFrameRef.current = 0;
    scoredFramesRef.current = 0;
    calibrationFramesRef.current = 0;
    calibrationEnergyRef.current = [];
    allEnergyRef.current = [];
    allCentroidRef.current = [];
    allZcrRef.current = [];
    closedEventsRef.current = [];
    tauHiRef.current = Infinity;
    tauLoRef.current = Infinity;
    isCalibratedRef.current = false;
    maxEnergyRef.current = -Infinity;
    setCoughCount(0);
    setEvents([]);
    setEnergyDb(null);
    setNoiseFloorDb(null);
    setThresholdHi(null);
    setThresholdLo(null);
    setCalibrationPhase("calibrating");
    setCalibrationProgress(0);
    setPeakAmplitude(0);
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
    /** Loudest sample in the take so far, in [0, 1]. */
    peakAmplitude,
    reset,
  };
}
