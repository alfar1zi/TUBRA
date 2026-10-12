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
        <summary className="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer list-none focus:outline-none focus:ring-2 focus:ring-blue-500">
          <span className="text-sm font-medium text-gray-700">{summary}</span>
          <svg
            className="w-5 h-5 text-gray-500 transition-transform group-open:rotate-180"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </summary>
        <div className="mt-2 p-3 border border-gray-200 rounded-lg bg-white animate-in fade-in slide-in-from-top-2 duration-150">
          {children}
        </div>
      </details>
    );
  }
);

ExpandablePanel.displayName = "ExpandablePanel";