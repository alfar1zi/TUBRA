import type { ReactNode } from "react";
import { MobileShell } from "@/components";
import { copy } from "@/copy";

interface AnalysisResult {
  tier: "A" | "B" | "C";
  probability: number;
  explanation: {
    symptom_contributions: Record<string, number>;
    gradcam_available: boolean;
    gradcam_url?: string;
  };
  next_steps: string[];
  metadata: {
    accepted_coughs: number;
    rejected_coughs: number;
    processing_time_ms: number;
  };
}

export interface S8ResultProps {
  result: AnalysisResult;
  showHemoptysisBanner: boolean;
  onFindClinic: () => void;
  onWhy: () => void;
  onRestart: () => void;
}

/**
 * S8Result reproduces Figma frame 27:658: the screening result. It shows the
 * tier card (soft warning surface), the "Kenapa hasilnya ini?" explanation
 * card with the reasons that drove the tier, the medical disclaimer, and the
 * two primary actions.
 */
export function S8Result({ result, showHemoptysisBanner, onFindClinic, onWhy, onRestart }: S8ResultProps) {
  const config = { A: copy.result.tierA, B: copy.result.tierB, C: copy.result.tierC }[result.tier];

  return (
    <MobileShell>
      <div
        className="flex flex-col flex-1"
        style={{ paddingTop: "calc(env(safe-area-inset-top, 62px) + 14px)" }}
      >
        <header className="text-center">
          <h1 className="text-2xl font-medium leading-[29px] text-text-primary/80">{copy.result.title}</h1>
          <p className="mt-[7px] text-base font-normal leading-[21px] text-text-primary/60">
            {copy.result.subtitle}
          </p>
        </header>

        {showHemoptysisBanner && (
          <div className="mt-4 rounded-lg bg-danger-soft p-4 text-sm font-medium text-danger">
            {copy.result.hemoptysisBanner}
          </div>
        )}

        {/* Tier card (Figma Frame 8749: y=162, 342x92) */}
        <div className="mt-[29px] h-[92px] rounded-lg border border-warning bg-warning-soft px-4">
          <div className="flex items-center gap-3 h-full">
            <span
              className="grid place-items-center w-[63px] h-[69px] rounded-[40px] bg-[#FFE683] flex-shrink-0"
              aria-hidden="true"
            >
              <TierIcon />
            </span>
            <div className="min-w-0">
              <p className="text-lg font-medium leading-[25px] text-text-primary">{config.text}</p>
              <p className="mt-[5px] text-sm font-normal leading-[18px] text-text-primary/80">
                {config.description}
              </p>
            </div>
          </div>
        </div>

        {/* Reasons card (Figma Frame 8753: y=267, 342x343). The whole card is the
            entry point to the "why" detail screen; Figma has no separate link. */}
        <div
          role="button"
          tabIndex={0}
          onClick={onWhy}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              onWhy();
            }
          }}
          className="mt-[13px] rounded-lg border border-black bg-white px-4 pt-[14px] pb-4 text-left w-full cursor-pointer"
        >
          <div className="flex items-baseline justify-between gap-2 h-[26px]">
            <h2 className="text-xl font-medium leading-[26px] text-text-primary">{copy.result.whyButton}</h2>
          </div>
          <p className="mt-[0px] text-base font-normal leading-[21px] text-text-primary/60">
            {copy.result.whyIntro}
          </p>
          <ul className="mt-[28px] space-y-[17px]">
            <ReasonItem
              title={copy.result.reasons.cough.title}
              body={copy.result.reasons.cough.body}
              icon={<TimeIcon />}
            />
            <ReasonItem
              title={copy.result.reasons.fever.title}
              body={copy.result.reasons.fever.body}
              icon={<TemperatureIcon />}
            />
            <ReasonItem
              title={copy.result.reasons.nightSweats.title}
              body={copy.result.reasons.nightSweats.body}
              icon={<MoonIcon />}
            />
          </ul>
        </div>

        <div className="flex-1" />
        <p className="mt-6 px-10 text-center text-xs font-normal leading-[16px] text-text-primary/60">
          {copy.result.disclaimer}
        </p>

        <div className="mt-[11px] space-y-[6px]">
          <button
            type="button"
            onClick={onFindClinic}
            className="w-full h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.result.findClinicButton}
          </button>
          <button
            type="button"
            onClick={onRestart}
            className="w-full h-[57px] rounded-lg bg-brand-soft text-lg font-medium text-brand transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.result.recheckButton}
          </button>
        </div>
        {/* Figma ends the last action 13px above the 34px home indicator. */}
        <div
          className="flex-shrink-0"
          style={{ height: "calc(env(safe-area-inset-bottom, 34px) + 13px)" }}
          aria-hidden="true"
        />
      </div>
    </MobileShell>
  );
}

/**
 * Tier + reason icons are the exact Figma vectors exported to public/design
 * (nodes 28:695 warning triangle, 29:717 clock, 29:746 thermometer, 29:752
 * moon). Loading them as images keeps the path data byte-for-byte identical
 * to Figma instead of re-typing long path strings by hand.
 */
function TierIcon() {
  return (
    <img src="/design/icon-warning.svg" alt="" aria-hidden="true" className="w-[33px] h-[29px]" />
  );
}

function TimeIcon() {
  return <img src="/design/icon-time.svg" alt="" aria-hidden="true" className="w-[38px] h-[38px]" />;
}

function TemperatureIcon() {
  return (
    <img src="/design/icon-temperature.svg" alt="" aria-hidden="true" className="w-[38px] h-[38px]" />
  );
}

function MoonIcon() {
  return <img src="/design/icon-moon.svg" alt="" aria-hidden="true" className="w-[38px] h-[38px]" />;
}

function ReasonItem({ title, body, icon }: { title: string; body: ReactNode; icon: ReactNode }) {
  return (
    <li className="flex items-start gap-3">
      <span className="mt-0.5 grid place-items-center w-[38px] h-[38px] flex-shrink-0" aria-hidden="true">
        {icon}
      </span>
      <div>
        <p className="text-base font-medium leading-6 text-text-primary">{title}</p>
        <p className="mt-0.5 text-sm font-normal leading-[21px] text-text-primary/70">{body}</p>
      </div>
    </li>
  );
}

