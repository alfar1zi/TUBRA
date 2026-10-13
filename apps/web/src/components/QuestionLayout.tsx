import type { ReactNode } from "react";
import { copy } from "@/copy";

/**
 * ProgressBar mirrors the Figma question header: a 342x8 rounded track
 * (`#E1E1E1`) with a filled portion. The fill width is proportional to the
 * current step, as drawn in the design.
 */
export interface ProgressBarProps {
  step: number;
  total: number;
}

export function ProgressBar({ step, total }: ProgressBarProps) {
  const safeTotal = Math.max(1, total);
  // Clamp into [min, max]: an out-of-range aria-valuenow is an ARIA violation
  // and confuses assistive tech.
  const safeStep = Math.min(Math.max(1, step), safeTotal);
  const pct = Math.max(0, Math.min(1, safeStep / safeTotal)) * 100;
  return (
    <div
      className="h-2 w-full rounded-pill bg-progress-track"
      role="progressbar"
      aria-valuemin={1}
      aria-valuemax={safeTotal}
      aria-valuenow={safeStep}
    >
      <div className="h-2 rounded-pill bg-progress-fill transition-all" style={{ width: `${pct}%` }} />
    </div>
  );
}

/**
 * QuestionLayout is the shared chrome for the nine single-question screens
 * (Figma frames 7:137 .. 11:639). Every vertical value below is read from the
 * design's node coordinates, measured from the top of the 402x874 viewport:
 *   - "Tahap Pertanyaan" heading        y = 86,  centred, Geist 20/500
 *   - 342x8 progress track              y = 128
 *   - "N dari 9 Pertanyaan" caption     y = 144
 *   - question title                    y = 210 (Q1 age: 218)
 *   - helper note (Q1 only)             title bottom + 8
 *   - answer area                       note/title bottom + 16
 *   - bottom action row                 y = 769 (Q1: 766, single button)
 * Horizontal gutter comes from MobileShell (30px).
 */
export interface QuestionLayoutProps {
  step: number;
  total: number;
  question: string;
  helper?: string;
  children: ReactNode;
  /** Bottom action row (e.g. Kembali / Lanjut). */
  actions: ReactNode;
  /**
   * Gap between the caption and the title. 45px puts the title at y=210 for the
   * eight standard questions; the age step uses 53px so it lands at y=218, as
   * drawn in frame 7:137.
   */
  titleGap?: number;
  /** Bottom padding for the action row: 47px -> y=769, 50px -> y=766 (age). */
  actionsPad?: number;
}

export function QuestionLayout({
  step,
  total,
  question,
  helper,
  children,
  actions,
  titleGap = 45,
  actionsPad = 47,
}: QuestionLayoutProps) {
  return (
    <div className="relative flex flex-col flex-1">
      {/* In Figma the heading sits at y=86, i.e. 24px below the 62px status
          bar. Anchoring to the device's real safe-area top raups the mock
          status-bar space so the heading hugs the top without hitting the
          notch. */}
      <h1
        className="text-center text-xl font-medium leading-[26px] text-text-primary/80"
        style={{ paddingTop: "calc(env(safe-area-inset-top, 62px) + 24px)" }}
      >
        {copy.questions.progressLabel}
      </h1>
      <div className="mt-[16px]">
        <ProgressBar step={step} total={total} />
      </div>
      <p className="mt-[8px] text-base leading-[21px] text-text-primary/60">
        {copy.questions.progressTemplate.replace("{n}", String(step))}
      </p>

      {/* Title sits at a fixed y=210 (218 for the age step), so the answer area
          below it stays put no matter how many lines the title wraps to. */}
      <h2 className="text-[24px] font-medium leading-[31px] text-text-primary/80" style={{ marginTop: titleGap }}>
        {question}
      </h2>
      {helper && <p className="mt-[8px] text-base leading-[21px] text-text-primary/60">{helper}</p>}
      <div className="mt-[16px]">{children}</div>

      {/* Action row pinned to the bottom. Figma measures actionsPad from the
          frame bottom (874) with the 34px home indicator baked in, so subtract
          that and add the device's real safe-area bottom instead. */}
      <div
        className="mt-auto pt-6 space-y-3"
        style={{ paddingBottom: `calc(env(safe-area-inset-bottom, 34px) + ${actionsPad - 34}px)` }}
      >
        {actions}
      </div>
    </div>
  );
}
