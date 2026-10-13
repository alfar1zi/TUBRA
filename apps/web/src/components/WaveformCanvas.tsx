import { useRef, useEffect, useCallback, type CanvasHTMLAttributes, forwardRef } from "react";

export interface WaveformCanvasProps extends CanvasHTMLAttributes<HTMLCanvasElement> {
  audioData: Float32Array | null;
  /** Sample indices (in `audioData`) where a cough event begins. */
  coughMarkers?: number[];
  sampleRate?: number;
  /** Length of the sliding window shown, in milliseconds. */
  windowMs?: number;
}

/**
 * WaveformCanvas renders the Figma "Voice" row (node 20:146, Group 8723): 29
 * teal pill dashes, 8px wide with a 50% radius, centred on the row's midline.
 *
 * Idle dashes are 2px tall. While audio is present the dashes show a *recent*
 * amplitude envelope (a sliding window, newest audio on the right) rather than
 * the whole recording, so a short cough is actually visible instead of being
 * averaged into a flat line. Values are normalized against the window peak so
 * quiet recordings still pulse.
 */
export const WaveformCanvas = forwardRef<HTMLCanvasElement, WaveformCanvasProps>(
  ({ audioData, coughMarkers = [], sampleRate = 16000, windowMs = 5000, className = "", ...props }, _ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    /** Latching peak used to scale bars; survives frames so the row is stable. */
    const scalePeakRef = useRef(1e-4);

    const DASH_COUNT = 29;
    const DASH_WIDTH = 8;
    const IDLE_HEIGHT = 2;

    const draw = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const width = rect.width;
      const height = rect.height;
      const centerY = height / 2;
      const maxBarHeight = height;

      ctx.clearRect(0, 0, width, height);

      // A cleared buffer (fresh recording) resets the latching scale.
      if (!audioData || audioData.length === 0) scalePeakRef.current = 1e-4;

      const gap = DASH_COUNT > 1 ? (width - DASH_COUNT * DASH_WIDTH) / (DASH_COUNT - 1) : 0;
      ctx.fillStyle = "#0F766E";

      // Map dashes to *time*, not to buffer length: dash i covers the slice of
      // the window beginning at (i / DASH_COUNT) * windowMs. A recording that is
      // only a few hundred ms long therefore lights up only its left dashes and
      // grows rightwards as the user speaks, instead of filling the whole row
      // the instant any audio arrives.
      const windowSamples = Math.max(1, Math.round((windowMs / 1000) * sampleRate));
      const total = audioData ? audioData.length : 0;
      const windowStart = Math.max(0, total - windowSamples);
      const samplesPerDash = windowSamples / DASH_COUNT;

      const levels: number[] = [];
      let peak = 1e-6;
      for (let i = 0; i < DASH_COUNT; i++) {
        let level = 0;
        if (total > windowStart) {
          const start = windowStart + Math.floor(i * samplesPerDash);
          const end = windowStart + Math.floor((i + 1) * samplesPerDash);
          // Only light a dash once its slice actually contains audio, so the
          // row fills left-to-right in real time.
          if (end <= total) {
            let sum = 0;
            let n = 0;
            for (let s = start; s < end; s++) {
              sum += audioData![s] * audioData![s];
              n++;
            }
            level = n > 0 ? Math.sqrt(sum / n) : 0;
          }
        }
        levels.push(level);
        if (level > peak) peak = level;
      }

      // Latch the scale to a slowly-rising peak so the bars do not jump on every
      // frame (a per-frame floor/peak normalization made the row flicker). The
      // peak decays gently so a one-off loud noise does not flatten the rest.
      const prevPeak = scalePeakRef.current;
      const decayedPeak = prevPeak * 0.98;
      const nextPeak = Math.max(peak, decayedPeak, 1e-4);
      scalePeakRef.current = nextPeak;

      for (let i = 0; i < DASH_COUNT; i++) {
        let barHeight = IDLE_HEIGHT;
        if (levels[i] > 0) {
          const normalized = Math.min(1, levels[i] / nextPeak);
          // Mild curve so mid-level sounds are clearly visible.
          barHeight = IDLE_HEIGHT + Math.pow(normalized, 0.6) * (maxBarHeight - IDLE_HEIGHT);
        }
        const x = i * (DASH_WIDTH + gap);
        const y = centerY - barHeight / 2;
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(x, y, DASH_WIDTH, barHeight, DASH_WIDTH / 2);
        } else {
          ctx.rect(x, y, DASH_WIDTH, barHeight);
        }
        ctx.fill();
      }

      // Cough markers (crimson dots) along the bottom of the window.
      if (coughMarkers.length > 0 && total > 0) {
        ctx.fillStyle = "#C83E5C";
        const windowEnd = windowStart + windowSamples;
        for (const sampleIndex of coughMarkers) {
          if (sampleIndex < windowStart || sampleIndex > windowEnd) continue;
          const x = ((sampleIndex - windowStart) / windowSamples) * width;
          ctx.beginPath();
          ctx.arc(Math.min(x, width - 3), height - 4, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }, [audioData, coughMarkers, sampleRate, windowMs]);

    useEffect(() => {
      draw();
    }, [draw]);

    useEffect(() => {
      const handleResize = () => draw();
      window.addEventListener("resize", handleResize);
      return () => window.removeEventListener("resize", handleResize);
    }, [draw]);

    return <canvas ref={canvasRef} className={`w-full h-32 ${className}`} {...props} />;
  }
);

WaveformCanvas.displayName = "WaveformCanvas";
