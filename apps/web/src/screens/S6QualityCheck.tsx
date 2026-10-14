import { Banner, MobileShell } from "@/components";
import { copy } from "@/copy";

export interface S6QualityCheckProps {
  acceptedCoughs: number;
  rejectedCoughs: number;
  noiseLevel: "quiet" | "moderate" | "loud";
  clipping: boolean;
  errorCode?: string;
  snrDb?: number;
  clippingRatio?: number;
  onContinue: () => void;
  onRerecord: () => void;
  onBack: () => void;
}

/**
 * S6QualityCheck gates the analysis on recording quality. When the recording
 * passes it shows the "Batuk anda sudah terekam!" confirmation (Figma frame
 * 20:530); when it fails it explains why and offers a re-record.
 */
export function S6QualityCheck({
  acceptedCoughs,
  noiseLevel,
  clipping,
  errorCode,
  onContinue,
  onRerecord,
  onBack,
}: S6QualityCheckProps) {
  const passed = acceptedCoughs >= 3 && noiseLevel !== "loud" && !clipping;

  if (passed) {
    return (
      <MobileShell>
        <div className="flex flex-col flex-1 text-center">
        {/* Figma (frame 20:530): a solid success-green disc (215x215, #16A34A)
            with a white check, title at y=579, subtitle at 616, action row at
            769. Anchoring the disc to the real safe-area top plus a fixed
            132px keeps the column identical on every phone instead of drifting
            with dvh. */}
        <div
          className="mx-auto grid place-items-center w-[215px] h-[215px] max-w-[54vw] aspect-square rounded-full bg-success"
          style={{ marginTop: "calc(env(safe-area-inset-top, 62px) + 132px)" }}
          aria-hidden="true"
        >
          <svg
            className="w-[112px] h-[86px] text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 112 86"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={13}
              d="M14 45 L44 73 L98 13"
            />
          </svg>
        </div>
          <div className="mt-auto">
            <h1 className="text-[22px] font-medium leading-[29px] text-text-primary/80">
              {copy.recordConfirm.title}
            </h1>
            <p className="mt-2 text-base font-normal leading-[21px] text-text-primary/60">
              {copy.recordConfirm.subtitle}
            </p>
          </div>
          {/* Figma ends the action row 13px above the 34px home indicator. */}
          <div
            className="mt-8 flex gap-2"
            style={{ paddingBottom: "calc(env(safe-area-inset-bottom, 34px) + 13px)" }}
          >
            <button
              type="button"
              onClick={onRerecord}
              className="flex-1 h-[57px] rounded-lg bg-brand-soft text-lg font-medium text-brand transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
            >
              {copy.recordConfirm.rerecordButton}
            </button>
            <button
              type="button"
              onClick={onContinue}
              className="flex-1 h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
            >
              {copy.recordConfirm.seeResultButton}
            </button>
          </div>
        </div>
      </MobileShell>
    );
  }

  const errorMap: Record<string, string> = {
    E4: copy.errors.E4,
    E5: copy.errors.E5,
    E6: copy.errors.E6,
    E7: copy.errors.E7,
    E8: copy.errors.E8,
    E10: copy.errors.E10,
    E11: copy.errors.E11,
    generic: copy.errors.generic,
    network: copy.errors.network,
  };
  const errorText = errorCode
    ? errorMap[errorCode] || copy.qualityCheck.errorLowQuality
    : acceptedCoughs < 3
    ? copy.qualityCheck.errorLowQuality
    : noiseLevel === "loud"
    ? copy.qualityCheck.errorNoise
    : copy.qualityCheck.errorClipping;

  return (
    <MobileShell>
      <div
        className="flex flex-col flex-1"
        style={{ paddingTop: "calc(env(safe-area-inset-top, 62px) + 24px)" }}
      >
        <h1 className="text-center text-[22px] font-medium leading-[29px] text-text-primary/80">
          {copy.qualityCheck.title}
        </h1>
        <div className="mt-[39px]">
          <Banner variant="warning" title={copy.qualityCheck.failTitle}>
            <p>{errorText}</p>
          </Banner>
        </div>
        <div className="flex-1" />
        <div
          className="space-y-1.5"
          style={{ paddingBottom: "calc(env(safe-area-inset-bottom, 34px) + 13px)" }}
        >
          <button
            type="button"
            onClick={onRerecord}
            className="w-full h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.qualityCheck.rerecordButton}
          </button>
          <button
            type="button"
            onClick={onBack}
            className="w-full h-[57px] rounded-lg bg-brand-soft text-lg font-medium text-brand transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.common.back}
          </button>
        </div>
      </div>
    </MobileShell>
  );
}
