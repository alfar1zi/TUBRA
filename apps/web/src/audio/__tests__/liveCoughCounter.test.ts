/**
 * Behaviour tests for the live cough counter's gating rules.
 *
 * The counter used to count anything louder than the room: a held note, a
 * snore, or a finger tap all passed. These tests pin the three rules that
 * separate a cough burst from them, using synthetic signals that stand in for
 * each class (the shape, not a recorded sample, is what the rules look at).
 */

import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { createRoot } from "react-dom/client";
import { act } from "react";
import { useLiveCoughCounter } from "../useLiveCoughCounter";

// The hook is driven through a detached root; tell React this is a test so
// `act` flushes effects synchronously instead of warning.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SR = 16000;
const rnd = (): number => Math.random() * 2 - 1;

function silence(n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = rnd() * 0.0005;
  return out;
}

function concat(...parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

function cough(): Float32Array {
  const n = Math.round(0.25 * SR);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const env = Math.exp(-t * 9) * Math.min(1, t * 120);
    out[i] =
      0.6 * env * (Math.sin(2 * Math.PI * 600 * t) * 0.5 + Math.sin(2 * Math.PI * 1380 * t) * 0.3 + rnd() * 0.5);
  }
  return out;
}

function clap(): Float32Array {
  const n = Math.round(0.02 * SR);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = rnd() * 0.9;
  return out;
}

function knock(): Float32Array {
  const n = Math.round(0.06 * SR);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = 0.5 * Math.exp(-(i / SR) * 60) * Math.sin(2 * Math.PI * 220 * (i / SR));
  return out;
}

function speech(): Float32Array {
  const n = Math.round(1.2 * SR);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const env = (0.5 + 0.5 * Math.sin(2 * Math.PI * 4 * t)) * Math.min(1, t * 20);
    let v = 0;
    for (let k = 1; k <= 5; k++) v += Math.sin(2 * Math.PI * 150 * k * t) / k;
    out[i] = 0.3 * env * (v / 2);
  }
  return out;
}

function music(): Float32Array {
  const n = Math.round(1.5 * SR);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const env = 0.6 + 0.4 * Math.sin(2 * Math.PI * 1.5 * t);
    out[i] = (0.35 * env * (Math.sin(2 * Math.PI * 440 * t) + 0.5 * Math.sin(2 * Math.PI * 554 * t))) / 1.5;
  }
  return out;
}

function snore(): Float32Array {
  const n = Math.round(1.5 * SR);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    out[i] = 0.4 * Math.sin((Math.PI * t) / 1.5) * (Math.sin(2 * Math.PI * 110 * t) + 0.4 * rnd());
  }
  return out;
}

/** A take with `n` copies of `sound`, each followed by half a second of room. */
function take(sound: () => Float32Array, n: number): Float32Array {
  const parts: Float32Array[] = [silence(SR)];
  for (let i = 0; i < n; i++) parts.push(sound(), silence(SR / 2));
  parts.push(silence(SR));
  return concat(...parts);
}

/**
 * Feeds a take through the hook and returns the final cough count. The hook
 * is mounted into a detached container: only its returned state matters here.
 */
/**
 * Streams the take the way the worklet does, in ~100 ms snapshots that each
 * carry the whole take so far, instead of handing the hook the finished array.
 * This is the only shape that reproduces what the record screen sees.
 */
function countStreamed(audio: Float32Array, sr: number, stepMs = 100): number {
  const container = document.createElement("div");
  const root = createRoot(container);
  let api: ReturnType<typeof useLiveCoughCounter> | null = null;
  function Probe({ buf }: { buf: Float32Array | null }): null {
    api = useLiveCoughCounter(buf, { sampleRate: sr });
    return null;
  }
  act(() => root.render(createElement(Probe, { buf: null })));
  const step = Math.round((stepMs / 1000) * sr);
  for (let i = step; i <= audio.length; i += step) {
    const buf = audio.slice(0, i);
    act(() => root.render(createElement(Probe, { buf })));
  }
  const count = api!.coughCount;
  act(() => root.unmount());
  return count;
}

/** The same take rebuilt at `sr` instead of the 16 kHz test rate. */
function atRate(audio: Float32Array, sr: number): Float32Array {
  const out = new Float32Array(Math.round((audio.length * sr) / SR));
  for (let i = 0; i < out.length; i++) {
    const p = (i * SR) / sr;
    const a = Math.floor(p);
    const b = Math.min(a + 1, audio.length - 1);
    out[i] = audio[a] + (audio[b] - audio[a]) * (p - a);
  }
  return out;
}

function countCoughs(audio: Float32Array): number {
  const container = document.createElement("div");
  const root = createRoot(container);
  let api: ReturnType<typeof useLiveCoughCounter> | null = null;
  function Probe(): null {
    api = useLiveCoughCounter(audio);
    return null;
  }
  act(() => {
    root.render(createElement(Probe));
  });
  act(() => {
    api!.reset();
  });
  const count = api!.coughCount;
  act(() => root.unmount());
  return count;
}

describe("useLiveCoughCounter gating", () => {
  it("counts cough bursts", () => {
    expect(countCoughs(take(cough, 5))).toBe(5);
  });

  it("does not count finger taps", () => {
    expect(countCoughs(take(clap, 3))).toBe(0);
  });

  it("does not count knocks on a table", () => {
    expect(countCoughs(take(knock, 2))).toBe(0);
  });

  it("does not count speech", () => {
    expect(countCoughs(concat(silence(SR), speech(), silence(SR)))).toBe(0);
  });

  it("does not count a held musical note", () => {
    expect(countCoughs(concat(silence(SR), music(), silence(SR)))).toBe(0);
  });

  it("does not count a snore", () => {
    expect(countCoughs(concat(silence(SR), snore(), silence(SR)))).toBe(0);
  });

  it("ignores a cough buried in continuous speech of the same loudness", () => {
    // Known limit of an energy-and-spectrum detector: `speech()` never falls
    // back under the release threshold, so the whole take is one candidate and
    // a cough that is only ~4 dB above the voice is averaged away. Separating
    // those two needs a learned model; this pins the current behaviour so a
    // change to it is noticed rather than assumed.
    expect(countCoughs(concat(silence(SR), speech(), cough(), speech(), silence(SR)))).toBe(0);
  });
});

describe("at the microphone's native rate", () => {
  // The burst-length budget used to be a raw frame count calibrated at
  // 16 kHz. The frame length is fixed in samples, so on a phone (44.1/48 kHz)
  // one cough spans three times as many frames and every real cough was
  // rejected: the counter sat at 0 no matter how hard the user coughed.
  for (const sr of [44100, 48000]) {
    it(`counts coughs at ${sr} Hz`, () => {
      expect(countStreamed(atRate(take(cough, 5), sr), sr)).toBe(5);
    });

    it(`still rejects taps at ${sr} Hz`, () => {
      expect(countStreamed(atRate(take(clap, 3), sr), sr)).toBe(0);
    });

    it(`still rejects music at ${sr} Hz`, () => {
      expect(countStreamed(atRate(take(music, 3), sr), sr)).toBe(0);
    });
  }
});
