import { useState, type ButtonHTMLAttributes, forwardRef } from "react";

export interface CopyButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  textToCopy: string;
  successLabel?: string;
  defaultLabel?: string;
}

export const CopyButton = forwardRef<HTMLButtonElement, CopyButtonProps>(
  ({ textToCopy, successLabel = "Tersalin!", defaultLabel = "Salin", children, className = "", ...props }, ref) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
      try {
        await navigator.clipboard.writeText(textToCopy);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch {
        // Fallback for non-secure contexts
        const textarea = document.createElement("textarea");
        textarea.value = textToCopy;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
    };

    return (
      <button
        ref={ref}
        type="button"
        onClick={handleCopy}
        className={`px-4 py-2 text-sm font-medium text-brand bg-brand-soft rounded-lg hover:bg-brand-soft/80 focus:outline-none focus:ring-2 focus:ring-brand ${className}`}
        {...props}
      >
        {copied ? successLabel : children ?? defaultLabel}
      </button>
    );
  }
);

CopyButton.displayName = "CopyButton";