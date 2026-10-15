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
  const [failure, setFailure] = useState<string | null>(null);
  /** Bumped by the retry button to run the request again. */
  const [attempt, setAttempt] = useState(0);

  // buildScreenPayload returns a fresh object each render; memoise it so the
  // effect below does not re-run (and re-POST) on every render.
  const payload = useMemo(
    () => buildScreenPayload(_gateData, _symptomData),
    [_gateData, _symptomData]
  );
  // Keep the latest callbacks in refs so the effect can list them without
  // re-running; the request must fire exactly once per attempt.
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
    setFailure(null);
    setCurrentStep(0);

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

        const mapped = err instanceof ScreenApiError ? mapApiErrorToFrontend(err) : null;
        if (err instanceof ScreenApiError && err.code === "quality_failed" && mapped?.qualityData) {
          // The take itself is unusable: send it back to the confirmation
          // screen, which explains every failed check and offers a re-record.
          onQualityFailRef.current({
            ...mapped.qualityData,
            errorCode: mapped.frontendCode,
          });
          return;
        }

        // Transport failures (network, CORS/PNA block, timeout, 5xx, model not
        // warm) say nothing about the take. Keep the user here with a clear
        // message and a retry: bouncing them to the confirmation screen made
        // "Lihat hasil" look broken and lost the recording they just made.
        const code = mapped ? mapped.frontendCode : "network";
        onErrorRef.current(code);
        if (!mounted) return;
        setFailure(code);
      }
    };

    callApi();

    const timer = window.setTimeout(() => {
      if (!mounted || !finish()) return;
      onErrorRef.current("E10");
      setFailure("E10"); // timeout
    }, 45000);

    const stepTimer = window.setInterval(() => {
      setCurrentStep((prev) => Math.min(prev + 1, STEPS.length - 1));
    }, 5000);

    return () => {
      mounted = false;
      window.clearTimeout(timer);
      window.clearInterval(stepTimer);
    };
  }, [audioBlob, payload, attempt]);

  if (failure) {
    const message =
      copy.analyze.failureMessages[failure as keyof typeof copy.analyze.failureMessages] ??
      copy.analyze.errorInference;
    return (
      <MobileShell>
        <div className="flex flex-col flex-1">
          <div
            className="text-center"
            style={{ paddingTop: "calc(env(safe-area-inset-top, 62px) + 48px)" }}
          >
            <h1 className="text-[22px] font-medium leading-[29px] text-text-primary/80">
              {copy.analyze.errorTitle}
            </h1>
            <div className="mt-4 text-left">
              <Banner variant="warning" title={copy.analyze.errorTitle}>
                <p>{message}</p>
              </Banner>
            </div>
          </div>
          <div
            className="mt-auto space-y-3 pt-8"
            style={{ paddingBottom: "calc(env(safe-area-inset-bottom, 34px) + 13px)" }}
          >
            <Button size="lg" fullWidth className="h-[58px]" onClick={() => setAttempt((n) => n + 1)}>
              {copy.analyze.retryButton}
            </Button>
            <Button variant="secondary" size="md" fullWidth className="h-[57px]" onClick={onCancel}>
              {copy.analyze.backToRecordButton}
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
        {/* A ring with an even track and one clearly rounded arc, rotating at a
            constant speed. The previous shape was an open path whose gap made
            the rotation read as a wobble; a dash with round caps and a longer
            drawn length gives a clean, obviously-loading spinner. */}
        <svg
          className="h-[100px] w-[100px] animate-spin text-brand-soft"
          viewBox="0 0 100 100"
          role="status"
          aria-live="polite"
          aria-label={copy.analyze.title}
          style={{ animationDuration: "900ms" }}
        >
          <circle cx="50" cy="50" r="42" fill="none" stroke="currentColor" strokeWidth="12" />
          <circle
            cx="50"
            cy="50"
            r="42"
            fill="none"
            stroke="#B3A8A8"
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray="176 264"
            transform="rotate(-90 50 50)"
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