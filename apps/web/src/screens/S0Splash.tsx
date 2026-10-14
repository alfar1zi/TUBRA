import { useEffect, useRef } from "react";
import { copy } from "@/copy";
import { MobileShell } from "@/components";

export interface S0SplashProps {
  onDone: () => void;
}

/**
 * S0Splash reproduces Figma frame 1:514: a faint circular dot grid behind the
 * TUBRA wordmark (Sora 40/600) and its tagline, centred vertically. It
 * advances automatically after a short beat, or immediately when tapped.
 */
export function S0Splash({ onDone }: S0SplashProps) {
  // Hold the latest callback in a ref so re-renders do not restart the timer;
  // the auto-advance must fire once, a fixed beat after mount.
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  useEffect(() => {
    const timer = window.setTimeout(() => onDoneRef.current(), 2200);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <button
      type="button"
      onClick={onDone}
      className="block w-full text-left"
      aria-label={copy.splash.subtitle}
    >
      <MobileShell bleed>
        {/* Dot grid: the exact Figma background (Group 8726) exported as SVG,
            already carrying its 0.05 opacity. It is larger than the frame
            (721x941) and anchored to the design origin, so it lives inside an
            overflow-hidden layer to keep it from widening the scroll area. */}
        <div className="absolute inset-0 overflow-hidden">
          <img
            src="/design/splash-dots.svg"
            alt=""
            aria-hidden="true"
            className="absolute left-1/2 top-0 w-[721px] max-w-none -translate-x-1/2 -translate-y-[35px]"
          />
        </div>
        {/* Logo lockup (Group 8727): the exported Figma vector (glyph + TUBRA
            + tagline) at frame y=374, i.e. 312px below the 62px status bar.
            Anchoring to the device safe-area top keeps it put on every phone
            instead of drifting with the viewport height. */}
        <div className="relative flex-1 flex justify-center">
          {/* Intrinsic 278x126 (see the SVG viewBox). Keep that ratio: forcing
              width 303 stretched the glyph ~9% horizontally. */}
          <img
            src="/design/logo-lockup.svg"
            alt=""
            aria-hidden="true"
            className="h-[126px] w-auto"
            style={{ marginTop: "calc(env(safe-area-inset-top, 62px) + 312px)" }}
          />
        </div>
        <span className="sr-only">
          {copy.splash.title} — {copy.splash.subtitle}
        </span>
      </MobileShell>
    </button>
  );
}
