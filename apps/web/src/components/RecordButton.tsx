import { forwardRef, type ButtonHTMLAttributes } from "react";

export interface RecordButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  state: "idle" | "recording" | "stopping";
  size?: number;
}

/** The mic glyph, extracted verbatim from Figma node I20:487;20:220. */
function MicGlyph() {
  return (
    <svg
      width="29"
      height="40"
      viewBox="26 20 30 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      className="text-danger"
    >
      <path
        d="M36.5728 43.3437C35.3575 42.1285 34.7498 40.6528 34.7498 38.9167V26.4167C34.7498 24.6805 35.3575 23.2049 36.5728 21.9896C37.788 20.7743 39.2637 20.1667 40.9998 20.1667C42.7359 20.1667 44.2116 20.7743 45.4269 21.9896C46.6422 23.2049 47.2498 24.6805 47.2498 26.4167V38.9167C47.2498 40.6528 46.6422 42.1285 45.4269 43.3437C44.2116 44.559 42.7359 45.1667 40.9998 45.1667C39.2637 45.1667 37.788 44.559 36.5728 43.3437ZM38.9165 59.75V53.3437C35.3054 52.8576 32.3193 51.243 29.9582 48.5C27.5971 45.7569 26.4165 42.5625 26.4165 38.9167H30.5832C30.5832 41.7986 31.5988 44.2552 33.63 46.2864C35.6613 48.3177 38.1179 49.3333 40.9998 49.3333C43.8818 49.3333 46.3384 48.3177 48.3696 46.2864C50.4009 44.2552 51.4165 41.7986 51.4165 38.9167H55.5832C55.5832 42.5625 54.4026 45.7569 52.0415 48.5C49.6804 51.243 46.6943 52.8576 43.0832 53.3437V59.75H38.9165Z"
        fill="currentColor"
      />
    </svg>
  );
}

/**
 * The record control from Figma node I20:487;20:220: an 82px pink disc
 * (#FFD7E0) with a 1px crimson rim (#C83E5C), a white inner disc, and a crimson
 * microphone glyph. While recording, the glyph is replaced by a crimson stop
 * square so the affordance matches the design's pressed state.
 */
export const RecordButton = forwardRef<HTMLButtonElement, RecordButtonProps>(
  ({ state, size = 82, children, className = "", disabled, ...props }, ref) => {
    const isRecording = state === "recording";

    return (
      <button
        ref={ref}
        disabled={disabled || state === "stopping"}
        className={`
          relative flex items-center justify-center rounded-full border border-danger
          bg-danger-soft text-danger transition-colors duration-200
          focus:outline-none focus:ring-4 focus:ring-danger/30
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
          ${className}
        `}
        style={{ width: size, height: size, minWidth: size, minHeight: size }}
        aria-label={isRecording ? "Berhenti merekam" : "Mulai rekam"}
        aria-pressed={isRecording}
        {...props}
      >
        <span
          className="grid place-items-center rounded-full bg-white"
          style={{ width: size * 0.61, height: size * 0.61 }}
          aria-hidden="true"
        >
          {isRecording ? (
            <span
              className="rounded-[3px] bg-danger"
              style={{ width: size * 0.24, height: size * 0.24 }}
            />
          ) : (
            <MicGlyph />
          )}
        </span>
        {children && <span className="sr-only">{children}</span>}
      </button>
    );
  }
);

RecordButton.displayName = "RecordButton";
