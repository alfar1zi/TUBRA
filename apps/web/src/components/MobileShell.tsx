import type { ReactNode } from "react";

/**
 * MobileShell reproduces the Figma "iPhone 16 & 17 Pro" artboard: a 402px-wide
 * white surface with 35px corners, centred on a neutral page background. Every
 * screen renders inside this shell so the whole app shares one device frame,
 * matching the design file.
 *
 * The frame always fills the available viewport height (`100dvh`, so the mobile
 * browser address bar is accounted for) and clips its own bleed artwork. Long
 * content scrolls *inside* the frame rather than widening the page, which keeps
 * the layout stable on phones narrower or taller than the 402x874 reference.
 *
 * `bleed` renders children edge-to-edge (no gutter/padding) — used by the home
 * and splash screens whose background art covers the full frame.
 */
export interface MobileShellProps {
  children: ReactNode;
  /** Full-bleed content with no padding or gutter (home/splash). */
  bleed?: boolean;
  /** Optional page-level background class (outside the device frame). */
  className?: string;
}

export function MobileShell({ children, bleed = false, className = "" }: MobileShellProps) {
  return (
    <div className={`h-[100dvh] w-full bg-neutral-100 flex justify-center overflow-hidden ${className}`}>
      {/* The frame fills the viewport. We deliberately do NOT add
          env(safe-area-inset-*) padding here: every screen's vertical offsets
          are the Figma y-coordinates, which already assume the drawn 62px
          status bar as part of the 402x874 frame. Adding the live inset on top
          would push content down by the notch height a second time. With
          viewport-fit=cover the frame covers the whole screen, so the Figma
          coordinates map 1:1 to the device. Full-bleed artwork therefore also
          runs edge to edge under the notch, as in the design. */}
      <div
        className={`relative w-full max-w-device h-full bg-white overflow-y-auto sm:rounded-[35px] sm:my-0 flex flex-col ${
          bleed ? "" : "px-gutter"
        }`}
      >
        {children}
      </div>
    </div>
  );
}
