import { Banner, MobileShell } from "@/components";
import { copy } from "@/copy";

/** Minimum detected coughs the API requires before it will run the model. */
const MIN_ACCEPTED_COUGHS = 3;

export interface S6QualityCheckProps {
  acceptedCoughs: number;
  rejectedCoughs: number;
  noiseLevel: "quiet" | "moderate" | "loud";
  clipping: boolean;
  errorCode?: string;
  snrDb?: number;
  clippingRatio?: number;
  /** Every check the API failed, in the order it reported them. */
  qualityReasons?: string[];
  onContinue: () => void;
  onRerecord: () => void;
  onBack: () => void;
}

/**
 * The API is the authority on whether a take is usable. The on-device counter
 * is a live aid, not a gate: it runs before the final sample is in and its
 * thresholds cannot be identical to the server's, so demanding a fixed count
 * here blocked users at this screen with no way to reach their result. Only a
 * take that came back from the API with a rejection is shown as a failure.
 */
function isRecordingAcceptable(errorCode: string | undefined): boolean {
  return !errorCode;
}

/**
 * Turns the API's raw reason strings into short, actionable sentences. The
 * server reports every failed check, so all of them are shown: surfacing only
 * the first one blamed the room for takes that simply had no audible cough.
 */
function describeFailures(
  reasons: string[] | undefined,
  errorCode: string | undefined,
  acceptedCoughs: number,
  noiseLevel: "quiet" | "moderate" | "loud",
  clipping: boolean,
): string[] {
  if (reasons && reasons.length > 0) {
    const sentences: string[] = [];
    for (const reason of reasons) {
      const key = Object.keys(copy.qualityCheck.reasonLabels).find((name) => reason.includes(name));
      if (!key) continue;
      const sentence = copy.qualityCheck.reasonLabels[key as keyof typeof copy.qualityCheck.reasonLabels];
      if (!sentences.includes(sentence)) sentences.push(sentence);
    }
    if (sentences.length > 0) return sentences;
  }
  if (errorCode && errorCode !== "E6" && errorCode !== "E7" && errorCode !== "E8") {
    return [copy.qualityCheck.errorLowQuality];
  }
  if (acceptedCoughs < MIN_ACCEPTED_COUGHS) return [copy.qualityCheck.errorLowQuality];
  if (noiseLevel === "loud") return [copy.qualityCheck.errorNoise];
  if (clipping) return [copy.qualityCheck.errorClipping];
  return [copy.qualityCheck.errorLowQuality];
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
  qualityReasons,
  onContinue,
  onRerecord,
  onBack,
}: S6QualityCheckProps) {
  const passed = isRecordingAcceptable(errorCode);

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

  const isQualityCode = errorCode === "E6" || errorCode === "E7" || errorCode === "E8";
  const failures = isQualityCode || (!errorCode && acceptedCoughs < MIN_ACCEPTED_COUGHS) || noiseLevel === "loud" || clipping
    ? describeFailures(qualityReasons, errorCode, acceptedCoughs, noiseLevel, clipping)
    : [];
  const transportErrorText = errorCode && !isQualityCode ? copy.errors[errorCode as keyof typeof copy.errors] : undefined;

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
            {transportErrorText ? (
              <p>{transportErrorText}</p>
            ) : failures.length > 1 ? (
              <ul className="list-disc space-y-1 pl-5 text-left">
                {failures.map((failure) => (
                  <li key={failure}>{failure}</li>
                ))}
              </ul>
            ) : (
              <p className="text-left">{failures[0]}</p>
            )}
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
