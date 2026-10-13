import { type HTMLAttributes, forwardRef } from "react";

export interface ExpandablePanelProps extends HTMLAttributes<HTMLDetailsElement> {
  summary: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

export const ExpandablePanel = forwardRef<HTMLDetailsElement, ExpandablePanelProps>(
  ({ summary, children, defaultOpen = false, className = "", ...props }, ref) => {
    return (
      <details ref={ref} className={`group ${className}`} open={defaultOpen} {...props}>
        <summary className="flex items-center justify-between p-3 bg-surface-subtle rounded-lg cursor-pointer list-none focus:outline-none focus:ring-2 focus:ring-brand">
          <span className="text-sm font-medium text-text-primary/80">{summary}</span>
          <svg
            className="w-5 h-5 text-text-primary/50 transition-transform group-open:rotate-180"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </summary>
        <div className="mt-2 p-3 border border-black/10 rounded-lg bg-white animate-in fade-in slide-in-from-top-2 duration-150">
          {children}
        </div>
      </details>
    );
  }
);

ExpandablePanel.displayName = "ExpandablePanel";