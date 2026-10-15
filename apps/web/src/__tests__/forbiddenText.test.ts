/**
 * Forbidden-text scanner (AGENTS.md section 4.1, 4.2, and task T7).
 *
 * AGENTS.md requires an automated test that scans user-facing copy for:
 *   - forbidden words ("aman", "sehat", ...)
 *   - em dashes (the project bans them in text, comments, and code)
 *   - emoji used as icons or decoration
 *   - the word "diagnosis" unless it appears as "bukan diagnosis"
 *
 * The scan covers every string in the `copy` module, which is the single
 * source of truth for user-facing text (AGENTS.md section 4.1).
 *
 * Matching uses word boundaries: a plain substring search would wrongly flag
 * "halaman" (contains "aman") and "kesehatan" (contains "sehat").
 */

import { describe, expect, it } from "vitest";
import { copy } from "../copy";

/** Collect every string reachable from the copy object. */
function collectStrings(value: unknown, path = "copy"): Array<[string, string]> {
  const out: Array<[string, string]> = [];
  if (typeof value === "string") {
    out.push([path, value]);
  } else if (Array.isArray(value)) {
    value.forEach((v, i) => out.push(...collectStrings(v, `${path}[${i}]`)));
  } else if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) {
      out.push(...collectStrings(v, `${path}.${k}`));
    }
  }
  return out;
}

const strings = collectStrings(copy);

// AGENTS.md section 4.2 forbidden words. "diagnosis" is handled separately.
const FORBIDDEN_WORDS = [
  "aman",
  "sehat",
  "negatif",
  "positif tbc",
  "terdeteksi tbc",
  "akurat",
  "canggih",
  "ai-powered",
  "seamless",
  "cutting-edge",
];

/** Build a word-boundary regex for a forbidden term. */
function wordBoundaryRe(term: string): RegExp {
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // \b is unreliable around non-ASCII; use explicit non-letter guards.
  return new RegExp(`(?<![a-z])${escaped}(?![a-z])`, "i");
}

// True emoji and pictographs only. Arrows such as "→" (U+2192) are typography,
// not emoji, and are used in flow sentences ("Isi gejala → Rekam").
const EMOJI_RE = /[\u{1F000}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/u;

describe("forbidden words in copy", () => {
  it.each(FORBIDDEN_WORDS)('does not contain "%s"', (word) => {
    const re = wordBoundaryRe(word);
    const offenders = strings.filter(([, text]) => re.test(text));
    expect(
      offenders.map(([p, t]) => `${p}: ${t}`),
      `"${word}" found as a whole word`,
    ).toEqual([]);
  });
});

describe("em dashes", () => {
  it("does not appear anywhere in copy", () => {
    const offenders = strings.filter(([, text]) => text.includes("\u2014"));
    expect(offenders.map(([p]) => p)).toEqual([]);
  });
});

// The greeting wave is part of the Figma design's consent headline ("Hai! 👋
// Sebelum mulai,"), so it is an intentional brand glyph, not decoration.
const BRAND_GLYPHS: readonly string[] = ["\u{1F44B}"];

describe("emoji", () => {
  it("does not appear in copy except for approved brand glyphs", () => {
    const offenders = strings.filter(([, text]) => {
      const withoutBrandGlyphs = BRAND_GLYPHS.reduce(
        (acc, glyph) => acc.split(glyph).join(""),
        text,
      );
      return EMOJI_RE.test(withoutBrandGlyphs);
    });
    expect(offenders.map(([p]) => p)).toEqual([]);
  });
});

describe("the word diagnosis", () => {
  it("only ever appears as 'bukan diagnosis'", () => {
    // Match only the standalone noun: "didiagnosis" ("was diagnosed") is a
    // verb from the design copy and carries no diagnostic claim.
    const standalone = /(?<![a-z])diagnosis/g;
    for (const [path, text] of strings) {
      const lower = text.toLowerCase();
      for (const match of lower.matchAll(standalone)) {
        const before = lower.slice(0, match.index);
        expect(
          /\bbukan\s*$/.test(before),
          `"diagnosis" at ${path} must be preceded by "bukan": ...${lower.slice(Math.max(0, (match.index ?? 0) - 20), (match.index ?? 0) + 9)}`,
        ).toBe(true);
      }
    }
  });
});

describe("tier C never claims the user is healthy", () => {
  it("has no reassuring wording in the lowest tier", () => {
    for (const [path, text] of strings) {
      const lower = path.toLowerCase();
      if (!lower.includes("tierc") && !lower.includes("tier_c")) {
        continue;
      }
      expect(
        /\b(aman|sehat|negatif)\b/i.test(text),
        `tier C copy at ${path} contains reassuring wording`,
      ).toBe(false);
    }
  });
});

// AGENTS.md section 4.1 says the copy module is the single source of truth for
// user-facing text. This scanner reads the screen and component sources and
// checks the literal JSX text they render, so a hardcoded string cannot sneak a
// forbidden word past the copy-only scan above.
describe("forbidden words in rendered JSX text", () => {
  it("has no forbidden wording or em dashes in component JSX text", () => {
    // Vite resolves these source files at build time, so the test needs no
    // node:fs (and therefore no @types/node in this browser-targeted project).
    const sources = import.meta.glob("../{screens,components}/*.tsx", {
      query: "?raw",
      import: "default",
      eager: true,
    }) as Record<string, string>;

    // Extract text nodes: characters between a JSX tag close ">" and the next
    // "<". Anything containing braces is code, not literal copy, so skip it.
    const jsxTextRe = />([^<>{}]+)</g;
    const failures: string[] = [];

    for (const [file, source] of Object.entries(sources)) {
      for (const match of source.matchAll(jsxTextRe)) {
        const text = match[1].trim();
        if (!text) continue;
        if (wordBoundaryRe("diagnosis").test(text) && !/bukan diagnosis/i.test(text)) {
          failures.push(`${file}: "diagnosis" outside "bukan diagnosis": ${text}`);
        }
        for (const word of FORBIDDEN_WORDS) {
          if (wordBoundaryRe(word).test(text)) {
            failures.push(`${file}: forbidden word "${word}" in JSX text: ${text}`);
          }
        }
        if (/[^\s]—|—[^\s]|\s—\s/.test(text)) {
          failures.push(`${file}: em dash in JSX text: ${text}`);
        }
      }
    }

    expect(failures, failures.join("\n")).toEqual([]);
  });
});
