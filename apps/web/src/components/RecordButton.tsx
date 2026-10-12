import { forwardRef, type ButtonHTMLAttributes } from "react";

export interface RecordButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  state: "idle" | "recording" | "stopping";
  size?: number;
}

export const RecordButton = forwardRef<HTMLButtonElement, RecordButtonProps>(
  ({ state, size = 88, children, className = "", disabled, ...props }, ref) => {
    const isRecording = state === "recording";

    return (
      <button
        ref={ref}
        disabled={disabled || state === "stopping"}
        className={`
          relative flex items-center justify-center rounded-full transition-all duration-200
          focus:outline-none focus:ring-4 focus:ring-blue-500/30
          ${isRecording
            ? "bg-red-600 text-white shadow-lg shadow-red-600/30 animate-pulse"
            : "bg-blue-600 text-white hover:bg-blue-700"}
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
          ${className}
        `}
        style={{ width: size, height: size, minWidth: size, minHeight: size }}
        aria-label={isRecording ? "Berhenti merekam" : "Mulai rekam"}
        aria-pressed={isRecording}
        {...props}
      >
        {isRecording ? (
          <>
            <div
              className="absolute inset-0 rounded-full bg-red-600 animate-ping opacity-75"
              aria-hidden="true"
            />
            <div className="relative w-8 h-8 bg-white rounded-sm" aria-hidden="true" />
            <span className="sr-only">Merekam...</span>
          </>
        ) : (
          <>
            <svg className="w-10 h-10" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path fill="none" d="M0 0h24v24H0z" />
            </svg>
            {children && <span className="sr-only">{children}</span>}
          </>
        )}
      </button>
    );
  }
);

RecordButton.displayName = "RecordButton";