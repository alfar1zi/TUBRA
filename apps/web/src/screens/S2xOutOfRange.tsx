import { copy } from "@/copy";
import { MobileShell } from "@/components";
export type OutOfRangeReason = "age" | "short_cough";

export interface S2xOutOfRangeProps {
  /** Why the user left the flow; picks the matching explanation. */
  reason: OutOfRangeReason;
  onFindClinic: () => void;
  onBackToStart: () => void;
}

/**
 * Shown when the user falls outside the screening age range or reports a
 * cough that has not lasted long enough to screen.
 *
 * The message is anchored near the top (not vertically centred) and the action
 * buttons are pinned to the bottom, so the layout matches every other screen
 * in the flow and the primary action sits under the user's thumb instead of
 * floating in the middle of the page.
 */
export function S2xOutOfRange({ reason, onFindClinic, onBackToStart }: S2xOutOfRangeProps) {
  // The screen serves two exits: too young, and coughing for under two weeks.
  // Each needs its own explanation, otherwise a 19-year-old who chose "kurang
  // dari 2 minggu" is told to bring a child to the clinic.
  const message = reason === "short_cough" ? copy.outOfRange.shortCough : copy.outOfRange;
  return (
    <MobileShell>
      <div className="flex flex-col flex-1">
        <div
          className="text-center"
          style={{ paddingTop: "calc(env(safe-area-inset-top, 62px) + 48px)" }}
        >
          <h1 className="text-2xl font-medium leading-[31px] text-text-primary/80">
            {message.title}
          </h1>
          <p className="mx-auto mt-3 max-w-[300px] text-base font-normal leading-[21px] text-text-primary/60">
            {message.body}
          </p>
        </div>
        <div
          className="mt-auto space-y-3 pt-8"
          style={{ paddingBottom: "calc(env(safe-area-inset-bottom, 34px) + 13px)" }}
        >
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
