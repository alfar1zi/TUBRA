import { useEffect, useMemo, useRef, useState } from "react";
import { Button, Banner, MobileShell } from "@/components";
import { copy } from "@/copy";
import { postScreen, ScreenResponse, ScreenApiError } from "@/api/client";
import type { ScreenPayload } from "@/api/client";
import { buildScreenPayload } from "@/state/screenState";
import type { GateData, SymptomData } from "@/state/screenState";

interface S7AnalyzeProps {
  audioBlob: Blob;
  gateData: GateData;
  symptomData: SymptomData;
  onResult: (result: ScreenResponse) => void;
  onQualityFail: (data: {
    acceptedCoughs: number;
    rejectedCoughs: number;
    noiseLevel: "quiet" | "moderate" | "loud";
    clipping: boolean;
    errorCode: string;
    snrDb?: number;
    clippingRatio?: number;
    qualityReasons?: QualityReason[];
  }) => void;
  onError: (errorCode: string) => void;
  onCancel: () => void;
}
const STEPS = copy.analyze.steps;

/**
 * Reasons the server can attach to a quality rejection. The server reports
 * every failed check, not just the first, so the screen must show all of them:
 * picking one used to blame the room ("derau terlalu besar") for a take that
 * actually contained no detected cough.
 */
export type QualityReason = "too_few_coughs" | "too_noisy" | "clipped";

function classifyQualityReasons(reasons: string[]): QualityReason[] {
  const found: QualityReason[] = [];
  if (reasons.some((r) => r.includes("too_few_coughs"))) found.push("too_few_coughs");
  if (reasons.some((r) => r.includes("too_noisy"))) found.push("too_noisy");
  if (reasons.some((r) => r.includes("clipped"))) found.push("clipped");
  return found;
}

function mapApiErrorToFrontend(error: ScreenApiError): {
  frontendCode: string;
  qualityData?: {
    acceptedCoughs: number;
    rejectedCoughs: number;
    noiseLevel: "quiet" | "moderate" | "loud";
    clipping: boolean;
    snrDb?: number;
    clippingRatio?: number;
    qualityReasons?: QualityReason[];
  };
} {
  const code = error.code;
  const details = error.details || {};
  const reasons = (details.reasons as string[]) || [];

  // Quality failed: carry every reason through so the screen can explain what
  // actually went wrong instead of guessing from a single code.
  if (code === "quality_failed") {
    const coughCount = (details.cough_count as number) || 0;
    const snrDb = (details.snr_db as number) || 0;
    const clippingRatio = (details.clipping_ratio as number) || 0;
    const qualityReasons = classifyQualityReasons(reasons);

    const frontendCode =
      qualityReasons[0] === "too_noisy"
        ? "E7"
        : qualityReasons[0] === "clipped"
        ? "E8"
        : "E6";

    return {
      frontendCode,
      qualityData: {
        acceptedCoughs: coughCount,
        rejectedCoughs: 0,
        noiseLevel: qualityReasons.includes("too_noisy") ? "loud" : "quiet",
        clipping: qualityReasons.includes("clipped"),
        snrDb,
        clippingRatio,
        qualityReasons,
      },
    };
  }
  
  // Invalid audio
  if (code === "invalid_audio") {
    const reason = (details.reason as string) || "";
    if (reason.includes("duration")) {
      return { frontendCode: "E5" };
    }
    return { frontendCode: "E4" };
  }
  
  // Invalid payload
  if (code === "payload_invalid") {
    return { frontendCode: "E11" };
  }

  // Model not ready / unavailable (transient)
  if (code === "model_unavailable") {
    return { frontendCode: "E10" };
  }

  // Inference errors
  if (code === "inference_error" || code === "explain_error") {
    return { frontendCode: "E10" };
  }
  
  // Timeout
  if (code === "timeout") {
    return { frontendCode: "E10" };
  }
  
  // Server error
  if (code === "server_error") {
    return { frontendCode: "generic" };
  }
  
  return { frontendCode: "generic" };
}

// React StrictMode mounts effects twice in development. Without this guard the
// screening POST would be sent twice (and the first one aborted on cleanup,
// surfacing as net::ERR_ABORTED). Deduplicating on the recorded Blob's identity
// keeps exactly one request per recording; the entry is weakly referenced so it
// is reclaimed once the recording is no longer held.
const inFlightScreens = new WeakMap<Blob, Promise<ScreenResponse>>();

function postScreenOnce(audioBlob: Blob, payload: ScreenPayload): Promise<ScreenResponse> {
  const existing = inFlightScreens.get(audioBlob);
  if (existing) return existing;
  const request = postScreen(audioBlob, payload).finally(() => {
    inFlightScreens.delete(audioBlob);
  });
  inFlightScreens.set(audioBlob, request);
  return request;
}

export function S7Analyze({
  audioBlob,
  gateData: _gateData,
  symptomData: _symptomData,
  onResult,
  onQualityFail,
  onError,
  onCancel,
}: S7AnalyzeProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [timedOut, setTimedOut] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // buildScreenPayload returns a fresh object each render; memoise it so the
  // effect below does not re-run (and re-POST) on every render.
  const payload = useMemo(
    () => buildScreenPayload(_gateData, _symptomData),
    [_gateData, _symptomData]
  );
  // Keep the latest callbacks in refs so the effect can list them without
  // re-running; the request must fire exactly once per mount.
  const onResultRef = useRef(onResult);
  const onQualityFailRef = useRef(onQualityFail);
  const onErrorRef = useRef(onError);
  onResultRef.current = onResult;
  onQualityFailRef.current = onQualityFail;
  onErrorRef.current = onError;

  useEffect(() => {
    let mounted = true;
    // Guarantees only ONE terminal callback (result | quality-fail | error)
    // even if the timeout and the request race.
    let settled = false;

    const finish = () => {
      if (settled) return false;
      settled = true;
      return true;
    };

    const callApi = async () => {
      try {
        const result: ScreenResponse = await postScreenOnce(audioBlob, payload);
        if (!mounted || !finish()) return;
        onResultRef.current(result);
      } catch (err) {
        if (!mounted || !finish()) return;

        if (err instanceof ScreenApiError) {
          const mapped = mapApiErrorToFrontend(err);

          if (err.code === "quality_failed" && mapped.qualityData) {
            // Quality check failed - show S6QualityCheck with real details
            onQualityFailRef.current({
              ...mapped.qualityData,
              errorCode: mapped.frontendCode,
            });
          } else {
            // Other API errors
            setApiError(mapped.frontendCode);
            onErrorRef.current(mapped.frontendCode);
          }
        } else if (err instanceof TypeError && err.message.includes("fetch")) {
          // Network error
          setApiError("network");
          onErrorRef.current("network");
        } else {
          // Unknown error
          const errorCode = err instanceof Error ? err.message : "generic";
          setApiError(errorCode);
          onErrorRef.current(errorCode);
        }
      }
    };

    callApi();

    const timer = window.setTimeout(() => {
      if (!mounted || !finish()) return;
      setTimedOut(true);
      onErrorRef.current("E10"); // timeout
    }, 20000);

    const stepTimer = window.setInterval(() => {
      setCurrentStep((prev) => Math.min(prev + 1, STEPS.length - 1));
    }, 5000);

    return () => {
      mounted = false;
      window.clearTimeout(timer);
      window.clearInterval(stepTimer);
    };
  }, [audioBlob, payload]);

  if (timedOut || apiError) {
    return (
      <MobileShell>
        <div
          className="flex flex-col flex-1 justify-center items-center text-center"
          style={{
            paddingTop: "env(safe-area-inset-top, 0px)",
            paddingBottom: "env(safe-area-inset-bottom, 0px)",
          }}
        >
          <Banner variant="warning" title={copy.analyze.errorTitle}>
            <p>{copy.analyze.errorInference}</p>
          </Banner>
          <div className="mt-6 w-full space-y-3">
            <Button size="lg" fullWidth onClick={onCancel}>
              {copy.analyze.backToRecordButton}
            </Button>
            <Button variant="secondary" size="md" fullWidth onClick={onCancel}>
              {copy.common.back}
            </Button>
          </div>
        </div>
      </MobileShell>
    );
  }

  return (
    <MobileShell>
      {/* Centred rather than pinned 289px from the top so the spinner and copy
          stay grouped on any phone height. */}
      <div
        className="flex flex-col flex-1 items-center justify-center gap-8 text-center"
        style={{
          paddingTop: "env(safe-area-inset-top, 0px)",
          paddingBottom: "env(safe-area-inset-bottom, 0px)",
        }}
      >
        {/* Figma (frame 20:610, group 27:657): a 100px ring, 14.5px thick
            (#E1E1E1 track) with a taupe (#B3A8A8) arc covering ~81% of the
            circle, rotating. */}
        <svg
          className="w-[100px] h-[100px] animate-spin"
          viewBox="0 0 100 100"
          role="status"
          aria-live="polite"
          aria-label={copy.analyze.title}
        >
          <circle cx="50" cy="50" r="42.75" fill="none" stroke="#E1E1E1" strokeWidth="14.5" />
          <path
            d="M88.5 23.5 A42.75 42.75 0 1 0 92.13 65.9"
            fill="none"
            stroke="#B3A8A8"
            strokeWidth="14.5"
            strokeLinecap="round"
          />
        </svg>
        <div>
          <h1 className="text-[22px] font-medium leading-[29px] text-text-primary/80">
            {copy.analyze.title}
          </h1>
          <p className="mt-2 text-base font-normal leading-[21px] text-text-primary/60">
            {copy.analyze.subtitle}
          </p>
        </div>
        <span className="sr-only">{STEPS[currentStep]}</span>
      </div>
    </MobileShell>
  );
}