import { forwardRef, type HTMLAttributes } from "react";

export interface ChoiceCardProps extends HTMLAttributes<HTMLDivElement> {
  selected?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
  disabled?: boolean;
}

export const ChoiceCard = forwardRef<HTMLDivElement, ChoiceCardProps>(
  ({ selected = false, onClick, children, disabled = false, className = "", ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`
          relative p-4 rounded-xl border-2 transition-all duration-150
          ${selected ? "border-blue-600 bg-blue-50" : "border-gray-200 bg-white hover:border-gray-300"}
          ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
          ${className}
        `}
        onClick={disabled ? undefined : onClick}
        role={onClick && !disabled ? "button" : undefined}
        tabIndex={onClick && !disabled ? 0 : undefined}
        onKeyDown={(e) => {
          if (onClick && !disabled && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            onClick();
          }
        }}
        {...props}
      >
        <div className="flex items-start gap-3">
          <div
            className={`
              flex-shrink-0 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-colors
              ${selected ? "border-blue-600 bg-blue-600" : "border-gray-300"}
            `}
            aria-hidden="true"
          >
            {selected && (
              <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            )}
          </div>
          <div className="flex-1 min-w-0">{children}</div>
        </div>
        {selected && (
          <div className="absolute inset-0 border-2 border-blue-600 rounded-xl pointer-events-none" aria-hidden="true" />
        )}
      </div>
    );
  }
);

ChoiceCard.displayName = "ChoiceCard";