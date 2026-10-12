import { useEffect, type ReactNode, Fragment } from "react";
import { createPortal } from "react-dom";

export interface DialogProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm?: () => void;
  variant?: "default" | "danger";
}

export function Dialog({
  isOpen,
  onClose,
  title,
  children,
  confirmLabel = "Lanjutkan",
  cancelLabel = "Batal",
  onConfirm,
  variant = "default",
}: DialogProps) {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const confirmStyles = variant === "danger" ? "bg-red-600 hover:bg-red-700 focus:ring-red-500" : "bg-blue-600 hover:bg-blue-700 focus:ring-blue-500";

  return createPortal(
    <Fragment>
      <div
        className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
        onClick={onClose}
        role="dialog"
        aria-modal="true"
        aria-labelledby="dialog-title"
      >
        <div
          className="bg-white rounded-xl shadow-xl max-w-md w-full animate-in fade-in zoom-in-95 duration-150"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-5 border-b border-gray-200">
            <h2 id="dialog-title" className="text-lg font-semibold text-gray-900">
              {title}
            </h2>
          </div>
          <div className="p-5">{children}</div>
          <div className="p-5 border-t border-gray-200 flex justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500"
            >
              {cancelLabel}
            </button>
            <button
              type="button"
              onClick={() => {
                onConfirm?.();
                onClose();
              }}
              className={`px-4 py-2 text-sm font-medium text-white rounded-lg ${confirmStyles} focus:outline-none focus:ring-2 focus:ring-offset-2`}
            >
              {confirmLabel}
            </button>
          </div>
        </div>
      </div>
    </Fragment>,
    document.body
  );
}