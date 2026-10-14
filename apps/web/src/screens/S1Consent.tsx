import { copy } from "@/copy";
import { MobileShell } from "@/components";

/**
 * S1Consent reproduces Figma frame 3:27: the greeting, a card listing what the
 * session involves, and two full-width actions ("Iya, aku menyetujuinya" /
 * "Aku tidak setuju"). Agreeing *is* the consent gesture, so there is no
 * separate checkbox.
 */
export interface S1ConsentProps {
  onContinue: () => void;
  onBack: () => void;
}

export function S1Consent({ onContinue, onBack }: S1ConsentProps) {
  return (
    <MobileShell>
      <div className="flex flex-col flex-1">
        {/* Match the other screens: Figma draws a 62px status bar and places the
            heading at y=127, so the real content sits 65px below it. Anchoring
            to the device safe-area top (0 on desktop/notchless) plus that 65px
            raups the mock status-bar space so there is no dead gap up top, and
            it stays put on every phone. */}
        <header style={{ paddingTop: "calc(env(safe-area-inset-top, 0px) + 65px)" }}>
          <h1 className="text-[24px] font-medium leading-[31px] text-text-primary/80">{copy.consent.title}</h1>
          <p className="mt-2 text-base font-normal leading-[21px] text-text-primary/60">{copy.consent.subtitle}</p>
        </header>

        {/* Frame 1: white card, 1px black rim, r16, 16px padding, 10px rhythm. */}
        <div className="mt-[39px] rounded-lg border border-black bg-white p-4 space-y-2.5">

          {copy.consent.body.map((text, i) => (
            <p
              key={i}
              className="flex gap-2 text-base font-normal leading-[21px] text-text-primary"
            >
              <span aria-hidden="true" className="text-text-primary">
                {i + 1}.
              </span>
              <span className="text-justify">{text}</span>
            </p>
          ))}
        </div>

        <div className="flex-1" />

        {/* Frame 2 / Frame 3: agreed y=703, declined y=767, 6px apart, 50px off
            the bottom edge. */}
        <div className="pb-[clamp(24px,5.7dvh,50px)] space-y-1.5">
          <button
            type="button"
            onClick={onContinue}
            className="w-full h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.consent.agreeButton}
          </button>
          <button
            type="button"
            onClick={onBack}
            className="w-full h-[57px] rounded-lg bg-brand-soft text-lg font-medium text-brand transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.consent.disagreeButton}
          </button>
        </div>
      </div>
    </MobileShell>
  );
}
