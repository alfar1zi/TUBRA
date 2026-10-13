import { copy } from "@/copy";
import { MobileShell } from "@/components";

export interface S2xOutOfRangeProps {
  onFindClinic: () => void;
  onBackToStart: () => void;
}

/** Shown when the user falls outside the screening age range. */
export function S2xOutOfRange({ onFindClinic, onBackToStart }: S2xOutOfRangeProps) {
  return (
    <MobileShell>
      {/* Centred content: pad by the real safe-area insets so nothing sits
          under the notch or home indicator on a device. */}
      <div
        className="flex flex-col flex-1 justify-center text-center"
        style={{
          paddingTop: "env(safe-area-inset-top, 0px)",
          paddingBottom: "env(safe-area-inset-bottom, 0px)",
        }}
      >
        <h1 className="text-2xl font-medium text-text-primary/80">
          {copy.outOfRange.title}
        </h1>
        <p className="mt-3 text-base font-normal text-text-primary/60">
          {copy.outOfRange.body}
        </p>
        <div className="mt-10 space-y-3">
          <button
            type="button"
            onClick={onFindClinic}
            className="w-full h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.outOfRange.findClinicButton}
          </button>
          <button
            type="button"
            onClick={onBackToStart}
            className="w-full h-[57px] rounded-lg bg-brand-soft text-lg font-medium text-brand transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.outOfRange.backToStartButton}
          </button>
        </div>
      </div>
    </MobileShell>
  );
}
