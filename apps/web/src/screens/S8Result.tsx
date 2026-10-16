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
 * One entry per tier. The components are the same on every tier (one card,
 * one badge, the same type scale and the same buttons); only the Figma tier
 * colour, the icon and the wording change. The emphasis comes from order and
 * font weight inside that same card, never from a different layout. Tier A and
 * B use a darker text on their soft fills for contrast, because the vivid
 * danger and warning colours stayed under 4.5:1 on their own surfaces. Tier C
 * is a neutral card: the house rules forbid green for any result status, and
 * a lower concern should not borrow a colour that reads as an all-clear.
 */
const TIER_THEME = {
  A: {
    surface: "border-danger bg-danger-soft",
    badge: "bg-[#FFC2CE] text-danger",
    accentText: "text-[#A82F4A]",
    icon: "/design/icon-tier-a.svg",
    label: "Perlu segera diperiksa",
    primaryLabel: copy.result.tierA.primaryAction,
    deadline: "1-2 hari",
  },
  B: {
    surface: "border-warning bg-warning-soft",
    badge: "bg-[#FFE683] text-[#92400E]",
    accentText: "text-[#92400E]",
    icon: "/design/icon-warning.svg",
    label: "Perlu diperiksa minggu ini",
    primaryLabel: copy.result.tierB.primaryAction,
    deadline: "Minggu ini",
  },
  C: {
    surface: "border-black/15 bg-white",
    badge: "bg-[#E3E3E3] text-text-primary",
    accentText: "text-text-primary/70",
    icon: "/design/icon-tier-c.svg",
    label: "Risiko relatif lebih rendah",
    primaryLabel: copy.result.tierC.primaryAction,
    deadline: null,
  },
} as const;

type Tier = keyof typeof TIER_THEME;
type TierConfig =
  | (typeof copy.result)["tierA"]
  | (typeof copy.result)["tierB"]
  | (typeof copy.result)["tierC"];
type TierTheme = (typeof TIER_THEME)[Tier];

/**
 * S8Result is the screen behind "Lihat hasil". It renders the tier the server
 * returned, not a fixed one: the banner, the headline, the badge icon and the
 * primary action all come from `result.tier`. Screens drive their own
 * transitions (see App), so reaching this screen is never a dead end.
 */
export function S8Result({
  result,
  showHemoptysisBanner,
  onFindClinic,
  onWhy,
  onRestart,
}: S8ResultProps) {
  const tier: Tier = result.tier;
  const config = {
    A: copy.result.tierA,
    B: copy.result.tierB,
    C: copy.result.tierC,
  }[tier];
  const theme = TIER_THEME[tier];

  return (
    <MobileShell>
      <div
        className="flex flex-col flex-1 min-h-0"
        style={{ paddingTop: "calc(env(safe-area-inset-top, 62px) + 14px)" }}
      >
        {/* Everything above the actions scrolls. A long result (banner plus an
            extra risk factor) used to push the two 58px buttons below the
            fold, out of reach; the actions must stay pinned to the bottom
            while the detail scrolls. */}
        <div className="flex-1 min-h-0 overflow-y-auto">
          <header className="text-center">
            <h1 className="text-2xl font-medium leading-[29px] text-text-primary/80">
              {copy.result.title}
            </h1>
            <p className="mt-[7px] text-base font-normal leading-[21px] text-text-primary/60">
              {copy.result.subtitle}
            </p>
          </header>

          {showHemoptysisBanner && (
            <div className="mt-4 rounded-lg bg-danger-soft p-4 text-sm font-medium text-danger">
              {copy.result.hemoptysisBanner}
            </div>
          )}

          <TierCard tier={tier} theme={theme} config={config} />

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
              <h2 className="text-xl font-medium leading-[26px] text-text-primary">
                {copy.result.whyButton}
              </h2>
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
        </div>

        {/* Pushes the actions to the bottom when the detail is short. */}
        <div className="flex-shrink-0" />
        <p className="mt-6 px-10 text-center text-xs font-normal leading-[16px] text-text-primary/60">
          {copy.result.disclaimer}
        </p>

        <div className="mt-[11px] space-y-[6px]">
          <PrimaryAction tier={tier} label={theme.primaryLabel} onClick={onFindClinic} />
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
 * The primary action keeps the brand gradient on every tier so the page still
 * reads as one product; only the wording changes with the tier.
 */
function PrimaryAction({
  tier,
  label,
  onClick,
}: {
  tier: Tier;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      data-tier-action={tier}
      className="w-full h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
    >
      {label}
    </button>
  );
}

/**
 * The tier card (Figma Frame 8749: y=162). Every tier uses this layout: a soft
 * fill with a matching outline, a badge on the left and the tier message on
 * the right. A and B add the deadline to the label line; tier A also raises
 * the heading one step in size and weight so it leads the page, and tier C
 * stays neutral.
 */
function TierCard({
  tier,
  theme,
  config,
}: {
  tier: Tier;
  theme: TierTheme;
  config: TierConfig;
}) {
  return (
    <section
      data-tier={tier}
      aria-label={theme.label}
      className={`mt-[29px] rounded-lg border px-4 py-3 ${theme.surface}`}
    >
      <div className="flex items-center gap-3">
        <span
          className={`grid place-items-center w-[63px] h-[69px] rounded-[40px] flex-shrink-0 ${theme.badge}`}
          aria-hidden="true"
        >
          <img src={theme.icon} alt="" aria-hidden="true" className="w-[33px] h-[29px]" />
        </span>
        <div className="min-w-0">
          <p className={`text-[11px] font-semibold uppercase tracking-wide ${theme.accentText}`}>
            {theme.label}
            {theme.deadline !== null && (
              <span className="ml-2 font-medium normal-case">· {theme.deadline}</span>
            )}
          </p>
          <p
            className={`mt-[2px] leading-[25px] ${
              tier === "A"
                ? `text-xl font-semibold ${theme.accentText}`
                : "text-lg font-medium text-text-primary"
            }`}
          >
            {config.text}
          </p>
          <p className="mt-[2px] text-sm font-normal leading-[18px] text-text-primary/80">
            {config.description}
          </p>
        </div>
      </div>
    </section>
  );
}

/**
 * Tier + reason icons are the exact Figma vectors exported to public/design
 * (nodes 28:695 warning triangle, 29:717 clock, 29:746 thermometer, 29:752
 * moon). Loading them as images keeps the path data byte-for-byte identical
 * to Figma instead of re-typing long path strings by hand.
 */
function TimeIcon() {
  return (
    <img
      src="/design/icon-time.svg"
      alt=""
      aria-hidden="true"
      className="w-[38px] h-[38px]"
    />
  );
}

function TemperatureIcon() {
  return (
    <img
      src="/design/icon-temperature.svg"
      alt=""
      aria-hidden="true"
      className="w-[38px] h-[38px]"
    />
  );
}

function MoonIcon() {
  return (
    <img
      src="/design/icon-moon.svg"
      alt=""
      aria-hidden="true"
      className="w-[38px] h-[38px]"
    />
  );
}

function ReasonItem({
  title,
  body,
  icon,
}: {
  title: string;
  body: ReactNode;
  icon: ReactNode;
}) {
  return (
    <li className="flex items-start gap-3">
      <span
        className="mt-0.5 grid place-items-center w-[38px] h-[38px] flex-shrink-0"
        aria-hidden="true"
      >
        {icon}
      </span>
      <div>
        <p className="text-base font-medium leading-6 text-text-primary">
          {title}
        </p>
        <p className="mt-0.5 text-sm font-normal leading-[21px] text-text-primary/70">
          {body}
        </p>
      </div>
    </li>
  );
}
