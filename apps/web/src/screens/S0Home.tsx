import { copy } from "@/copy";
import { MobileShell } from "@/components";

/**
 * S0Home reproduces the Figma home frame (4:101): a full-bleed hero photo with
 * white gradient scrims top and bottom, the TUBRA wordmark top-centre, the
 * headline over the top scrim, and the subtitle + primary CTA sitting on the
 * bottom scrim.
 *
 * The hero art is a single pre-composited image (base photo + top scrim +
 * subject cutout + bottom scrim, baked from the Figma nodes in that layer
 * order) rather than four live layers. Compositing in the browser produced a
 * visible horizontal seam where the base photo's edge met the scrims; one
 * image has no internal edges to misalign.
 *
 * Responsiveness: the image is anchored TOP and drawn with object-cover
 * object-top, so the subject always keeps the same crop and position and only
 * the surplus below the fold is clipped. The wordmark/headline anchor to the
 * top of the frame and the subtitle/CTA anchor to the bottom, with a flexible
 * spacer between them, so nothing sinks or floats on shorter or taller phones.
 */
export interface S0HomeProps {
  onStart: () => void;
}

export function S0Home({ onStart }: S0HomeProps) {
  return (
    <MobileShell bleed>
      <div className="absolute inset-0 overflow-hidden">
        {/* Hero art: 402x874 Figma composite, top-anchored so the subject
            never zooms or drifts. */}
        <img
          src="/design/home-bg.webp"
          alt=""
          aria-hidden="true"
          className="absolute inset-0 h-full w-full object-cover object-top"
        />
        {/* Bottom fog: a live layer anchored to the frame bottom (the top fog
            is baked into the hero art). Keeping it live guarantees the
            subtitle/CTA keep contrast at any viewport height, and because it
            is anchored to the bottom it can never drift. */}
        <div
          aria-hidden="true"
          className="absolute inset-x-0 bottom-0 h-[300px]"
          style={{
            backgroundImage:
              "linear-gradient(to top, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.65) 40%, rgba(255,255,255,0) 100%)",
          }}
        />
      </div>

      {/* Wordmark: logo glyph + TUBRA. In Figma it sits at y=78, i.e. 16px
          below the 62px status bar. Anchor to the device's real safe-area top
          instead of a fixed 78px so the row hugs the status bar on every phone
          (no dead space above) while still clearing the notch. */}
      <div
        className="relative flex items-center justify-center gap-1"
        style={{ paddingTop: "calc(env(safe-area-inset-top, 62px) + 16px)" }}
      >
        <img
          src="/design/logo-glyph.svg"
          alt=""
          aria-hidden="true"
          className="w-[33px] h-[33px]"
        />
        <span className="font-display text-[24px] font-semibold leading-[30px] text-text-primary/80">
          {copy.brand}
        </span>
      </div>

      {/* Headline over the top scrim: 312px wide, centred, Geist 24/500. */}
      <h1 className="relative mt-[16px] mx-auto w-full max-w-[312px] text-center text-[24px] font-medium leading-[31px] text-text-primary/80">
        {copy.home.title}
      </h1>

      <div className="flex-1" />

      {/* Subtitle + CTA. Figma ends the CTA 16px above the 34px home
          indicator, so anchor to the real safe-area bottom instead of a fixed
          50px to keep the same clearance on every phone. */}
      <div
        className="relative px-gutter"
        style={{ paddingBottom: "calc(env(safe-area-inset-bottom, 34px) + 16px)" }}
      >
        <p className="text-center text-base font-normal leading-[21px] text-text-primary/80">
          {copy.home.subtitle}
        </p>
        <button
          type="button"
          onClick={onStart}
          className="mt-[16px] w-full h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
        >
          {copy.home.startButton}
        </button>
      </div>
    </MobileShell>
  );
}
