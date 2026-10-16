import { forwardRef, type HTMLAttributes, type ReactNode } from "react";

/**
 * OptionRow is a single answer option on the question screens, styled to match
 * the Figma design: a 342x54 row, 16px corners, 1px border at 20% black, label
 * on the left and a radio indicator on the right. Selection is signalled with
 * the brand teal.
 */
export interface ChoiceCardProps extends Omit<HTMLAttributes<HTMLDivElement>, "onChange"> {
  selected?: boolean;
  onClick?: () => void;
  children: ReactNode;
  disabled?: boolean;
}

export const ChoiceCard = forwardRef<HTMLDivElement, ChoiceCardProps>(
  ({ selected = false, onClick, children, disabled = false, className = "", ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={[
          "flex items-center justify-between gap-3 h-[54px] px-4 rounded-lg border bg-transparent transition-colors",
          selected ? "border-brand bg-brand-soft/40" : "border-black/20",
          disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer hover:border-black/40",
          className,
        ].join(" ")}
        onClick={disabled ? undefined : onClick}
        role={onClick && !disabled ? "radio" : undefined}
        tabIndex={onClick && !disabled ? 0 : undefined}
        aria-checked={onClick && !disabled ? selected : undefined}
        aria-disabled={disabled || undefined}
        onKeyDown={(e) => {
          if (onClick && !disabled && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            onClick();
          }
        }}
        {...props}
      >
        <span className="text-base font-alt leading-5 text-text-primary">{children}</span>
        <span
          className={`flex-shrink-0 grid place-items-center w-5 h-5 rounded-full border ${
            selected ? "border-brand" : "border-black/30"
          }`}
          aria-hidden="true"
        >
          {selected && <span className="w-2.5 h-2.5 rounded-full bg-brand" />}
        </span>
      </div>
    );
  }
);

ChoiceCard.displayName = "ChoiceCard";
