/**
 * Spectral-gate parity test for the live cough counter.
 *
 * The server (`apps/api/tbcare_api/audio/events.py`) and the live counter
 * (`useLiveCoughCounter.ts`) both run a two-stage cough detector: RMS energy
 * with hysteresis, then a spectral gate on the candidate's mean spectral
 * centroid and zero-crossing rate. This test pins the browser-side feature
 * extraction to values the server also produces for identical inputs, so the
 * two cannot silently drift apart.
 *
 * The "server reference" numbers were produced by
 * `tbcare_api.audio.events.spectral_frame_features` for the same signals.
 */

import { describe, expect, it } from "vitest";
import { frameSpectralFeatures } from "../useLiveCoughCounter";

const SR = 16000;
const N = 1024;

function sine(freq: number, amp = 0.4): Float32Array {
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * i) / SR);
  return out;
}

describe("frameSpectralFeatures", () => {
  it("matches the server for a pure 2500 Hz tone (centroid 2500, zcr 0.313)", () => {
    const { centroid, zcr } = frameSpectralFeatures(sine(2500), SR);
    expect(centroid).toBeCloseTo(2500, 0);
    expect(zcr).toBeCloseTo(0.3128, 3);
  });

  it("matches the server for a pure 1000 Hz tone (centroid 1000, zcr 0.125)", () => {
    const { centroid, zcr } = frameSpectralFeatures(sine(1000), SR);
    expect(centroid).toBeCloseTo(1000, 0);
    expect(zcr).toBeCloseTo(0.1251, 3);
  });

  it("gives a low tone a low centroid", () => {
    const { centroid } = frameSpectralFeatures(sine(150), SR);
    expect(centroid).toBeGreaterThan(120);
    expect(centroid).toBeLessThan(200);
  });

  it("gives a 3 kHz tone a high centroid (so the gate would reject it via zcr)", () => {
    // A pure tone has a *high* centroid but the zero-crossing rate of a
    // 3 kHz tone exceeds the cough window (0.34), which is what rejects it.
    const { centroid, zcr } = frameSpectralFeatures(sine(3000), SR);
    expect(centroid).toBeGreaterThan(2900);
    expect(zcr).toBeGreaterThan(0.34);
  });

  it("is finite and non-negative on silence", () => {
    const { centroid, zcr } = frameSpectralFeatures(new Float32Array(N), SR);
    expect(Number.isFinite(centroid)).toBe(true);
    expect(centroid).toBeGreaterThanOrEqual(0);
    expect(zcr).toBeGreaterThanOrEqual(0);
  });

  it("produces deterministic, repeatable output", () => {
    const a = frameSpectralFeatures(sine(1234), SR);
    const b = frameSpectralFeatures(sine(1234), SR);
    expect(a.centroid).toBe(b.centroid);
    expect(a.zcr).toBe(b.zcr);
  });
});
