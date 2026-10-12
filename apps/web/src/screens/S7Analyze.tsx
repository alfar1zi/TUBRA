import { useEffect, useState } from "react";
import { Button } from "@/components";
import { copy } from "@/copy";
import { postScreen, ScreenResponse, ScreenApiError } from "@/api/client";
import { buildScreenPayload } from "@/state/screenState";
import { SymptomData } from "@/state/screenState";

interface S7AnalyzeProps {
  audioBlob: Blob;
  gateData: { age: number | null; coughDurationWeeks: number | null };
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
  }) => void;
  onError: (errorCode: string) => void;
  onCancel: () => void;
}
const STEPS = [
  "Memotong suara batuk",
  "Membaca pola suara",
  "Menggabungkan dengan gejala Anda",
];

function mapApiErrorToFrontend(error: ScreenApiError): { frontendCode: string; qualityData?: { acceptedCoughs: number; rejectedCoughs: number; noiseLevel: "quiet" | "moderate" | "loud"; clipping: boolean; snrDb?: number; clippingRatio?: number } } {
  const code = error.code;
  const details = error.details || {};
  const reasons = (details.reasons as string[]) || [];
  
  // Quality failed - map based on reasons
  if (code === "quality_failed") {
    const coughCount = (details.cough_count as number) || 0;
    const snrDb = (details.snr_db as number) || 0;
    const clippingRatio = (details.clipping_ratio as number) || 0;
    
    let frontendCode = "E6"; // default: too few coughs
    let noiseLevel: "quiet" | "moderate" | "loud" = "quiet";
    let clipping = false;
    
    if (reasons.some(r => r.includes("too_noisy"))) {
      frontendCode = "E7";
      noiseLevel = "loud";
    } else if (reasons.some(r => r.includes("clipped"))) {
      frontendCode = "E8";
      clipping = true;
    } else if (reasons.some(r => r.includes("too_few_coughs"))) {
      frontendCode = "E6";
    } else if (reasons.some(r => r.includes("segments"))) {
      frontendCode = "E6"; // not enough segments extracted
    }
    
    return {
      frontendCode,
      qualityData: {
        acceptedCoughs: coughCount,
        rejectedCoughs: 0,
        noiseLevel,
        clipping,
        snrDb,
        clippingRatio,
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
  if (code === "invalid_payload") {
    return { frontendCode: "E11" };
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

  const payload = buildScreenPayload(_gateData, _symptomData);

  useEffect(() => {
    let mounted = true;

    const callApi = async () => {
      try {
        const result: ScreenResponse = await postScreen(audioBlob, payload);
        if (mounted) {
          onResult(result);
        }
      } catch (err) {
        if (!mounted) return;
        
        if (err instanceof ScreenApiError) {
          const mapped = mapApiErrorToFrontend(err);
          
          if (err.code === "quality_failed" && mapped.qualityData) {
            // Quality check failed - show S6QualityCheck with real details
            onQualityFail({
              ...mapped.qualityData,
              errorCode: mapped.frontendCode,
            });
          } else {
            // Other API errors
            setApiError(mapped.frontendCode);
            onError(mapped.frontendCode);
          }
        } else if (err instanceof TypeError && err.message.includes("fetch")) {
          // Network error
          setApiError("network");
          onError("network");
        } else {
          // Unknown error
          const errorCode = err instanceof Error ? err.message : "generic";
          setApiError(errorCode);
          onError(errorCode);
        }
      }
    };

    callApi();

    const timer = window.setTimeout(() => {
      if (mounted) {
        setTimedOut(true);
        onError("E10"); // timeout
      }
    }, 20000);

    const stepTimer = window.setInterval(() => {
      setCurrentStep((prev) => Math.min(prev + 1, STEPS.length - 1));
    }, 5000);

    return () => {
      mounted = false;
      window.clearTimeout(timer);
      window.clearInterval(stepTimer);
    };
  }, [audioBlob, payload, onResult, onQualityFail, onError]);

  if (timedOut || apiError) return null;

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 text-center">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">{copy.analyze.title}</h1>
          <p className="text-gray-600 mb-8">{copy.analyze.subtitle}</p>

          <div className="space-y-4 mb-8" role="status" aria-live="polite">
            {STEPS.map((step, i) => (
              <div
                key={i}
                className={`flex items-center gap-3 transition-colors ${i <= currentStep ? "text-gray-900" : "text-gray-400"}`}
              >
                <div
                  className={`flex-shrink-0 w-8 h-8 rounded-full border-2 flex items-center justify-center ${
                    i < currentStep ? "bg-blue-600 border-blue-600" : i === currentStep ? "border-blue-600" : "border-gray-300"
                  }`}
                >
                  {i < currentStep ? (
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <span className={i === currentStep ? "text-blue-600 font-bold" : "text-gray-400"}>{i + 1}</span>
                  )}
                </div>
                <p className="text-left">{step}</p>
              </div>
            ))}
          </div>

          <div className="space-y-3">
            <Button size="lg" fullWidth onClick={onCancel} disabled>
              {copy.common.cancel}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}