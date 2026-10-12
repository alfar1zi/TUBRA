import { useReducer, useEffect, useCallback, useState } from "react";
import { getHealth } from "@/api/client";
import { ScreenResponse } from "@/api/client";
import {
  screenReducer,
  initialContext,
  ScreenId,
} from "@/state/screenState";
import {
  S0Home,
  S1Consent,
  S2Gate,
  S2xOutOfRange,
  S2yNotTime,
  S3Symptoms,
  S4PrepRecord,
  S5Record,
  S6QualityCheck,
  S7Analyze,
  S8Result,
  S9Why,
  S10NextSteps,
  S11HowItWorks,
  S12Privacy,
} from "@/screens";
import { Banner } from "@/components";
import { copy } from "@/copy";

function App() {
  const [context, dispatch] = useReducer(screenReducer, initialContext);
  const [demoMode, setDemoMode] = useState(false);

  // Check for demo mode from URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("demo") === "1") {
      setDemoMode(true);
      dispatch({ type: "SET_DEMO_MODE", enabled: true });
    }
  }, []);

  // Warm up health check on mount
  useEffect(() => {
    getHealth().catch(() => {}); // Don't block on health check failure
  }, []);

  const navigate = useCallback((screen: ScreenId) => {
    dispatch({ type: "GO_TO", screen });
  }, []);

  const handleGateSubmit = useCallback((age: number, coughDuration: string) => {
    const weeksMap: Record<string, number> = { "2_4": 3, "4_8": 6, "8_plus": 10 };
    dispatch({ type: "SET_GATE_DATA", data: { age, coughDurationWeeks: weeksMap[coughDuration] } });
    navigate("S3_SYMPTOMS");
  }, [navigate]);

  const handleSymptomsSubmit = useCallback(
    (data: Record<string, "yes" | "no" | null>) => {
      dispatch({ type: "SET_SYMPTOM_DATA", data });
      navigate("S4_PREP_RECORD");
    },
    [navigate]
  );
  const handleRecordingComplete = useCallback((audioBlob: Blob, coughData: { count: number; events: Array<{startMs: number, endMs: number}> }) => {
    dispatch({ type: "SET_RECORDING_DATA", data: { 
      audioBlob, 
      acceptedCoughs: coughData.count,
      rejectedCoughs: 0,
      noiseLevel: "quiet",
      clipping: false,
    } });
    navigate("S6_QUALITY_CHECK");
  }, [navigate]);

  const handleAnalysisResult = useCallback((result: ScreenResponse) => {
    dispatch({ type: "SET_ANALYSIS_RESULT", result: {
      tier: result.tier,
      probability: result.metadata.probability,
      explanation: {
        symptom_contributions: result.metadata.symptom_explanations ? 
          Object.fromEntries(result.metadata.symptom_explanations.map((s: string) => [s, 0])) : {},
        gradcam_available: result.metadata.gradcam !== null,
        gradcam_url: result.gradcam_url || undefined,
      },
      next_steps: result.next_steps,
      metadata: {
        accepted_coughs: result.metadata.cough_segments_used,
        rejected_coughs: 0,
        processing_time_ms: result.metadata.processing_time_ms,
      },
    } });
    navigate("S8_RESULT");
  }, [navigate]);

  const handleQualityFail = useCallback((data: { acceptedCoughs: number; rejectedCoughs: number; noiseLevel: "quiet" | "moderate" | "loud"; clipping: boolean; errorCode: string; snrDb?: number; clippingRatio?: number }) => {
    dispatch({ type: "SET_RECORDING_DATA", data: { 
      acceptedCoughs: data.acceptedCoughs,
      rejectedCoughs: data.rejectedCoughs,
      qualityError: data.errorCode,
      noiseLevel: data.noiseLevel,
      clipping: data.clipping,
      snrDb: data.snrDb,
      clippingRatio: data.clippingRatio,
    } });
    navigate("S6_QUALITY_CHECK");
  }, [navigate]);

  const handleAnalysisError = useCallback((errorCode: string) => {
    dispatch({ type: "SET_ERROR", error: errorCode });
    // Also store in recordingData for S6QualityCheck to display
    dispatch({ type: "SET_RECORDING_DATA", data: { qualityError: errorCode } });
    navigate("S6_QUALITY_CHECK");
  }, [navigate]);

  const handleQualityCheckPass = useCallback(() => {
    navigate("S7_ANALYZE");
  }, [navigate]);
  const handleRestart = useCallback(() => {
    if (window.confirm("Mulai dari awal? Semua jawaban dan rekaman akan dihapus.")) {
      dispatch({ type: "RESET" });
    }
  }, []);
  const renderScreen = () => {
    const { currentScreen } = context;

    switch (currentScreen) {
      case "S0_HOME":
        return (
          <S0Home
            onStart={() => navigate("S1_CONSENT")}
            onHowItWorks={() => navigate("S11_HOW_IT_WORKS")}
            onPrivacy={() => navigate("S12_PRIVACY")}
          />
        );
      case "S1_CONSENT":
        return (
          <S1Consent onContinue={() => navigate("S2_GATE")} onBack={() => navigate("S0_HOME")} />
        );
      case "S2_GATE":
        return (
          <S2Gate
            onContinue={handleGateSubmit}
            onBack={() => navigate("S1_CONSENT")}
            onOutOfRange={() => navigate("S2x_OUT_OF_RANGE")}
            onNotTime={() => navigate("S2y_NOT_TIME")}
          />
        );
      case "S2x_OUT_OF_RANGE":
        return (
          <S2xOutOfRange
            onFindClinic={() => window.open("https://www.google.com/maps/search/puskesmas+terdekat", "_blank")}
            onBackToStart={() => navigate("S0_HOME")}
          />
        );
      case "S2y_NOT_TIME":
        return (
          <S2yNotTime
            onBackToStart={() => navigate("S0_HOME")}
            onFindClinic={() => window.open("https://www.google.com/maps/search/puskesmas+terdekat", "_blank")}
          />
        );
      case "S3_SYMPTOMS":
        return (
          <S3Symptoms
            onContinue={handleSymptomsSubmit}
            onBack={() => navigate("S2_GATE")}
            onHemoptysisYes={() => dispatch({ type: "SET_SYMPTOM_DATA", data: { hemoptysis: "yes" } })}
          />
        );
      case "S4_PREP_RECORD":
        return (
          <S4PrepRecord
            onContinue={() => navigate("S5_RECORD")}
            onBack={() => navigate("S3_SYMPTOMS")}
            onMicError={() => {}}
          />
        );
      case "S5_RECORD":
        return <S5Record onComplete={handleRecordingComplete} onBack={() => navigate("S4_PREP_RECORD")} />;
      case "S6_QUALITY_CHECK":
        return (
          <S6QualityCheck
            acceptedCoughs={context.recordingData.acceptedCoughs}
            rejectedCoughs={context.recordingData.rejectedCoughs}
            noiseLevel={context.recordingData.noiseLevel || "quiet"}
            clipping={context.recordingData.clipping || false}
            errorCode={context.recordingData.qualityError || undefined}
            snrDb={context.recordingData.snrDb}
            clippingRatio={context.recordingData.clippingRatio}
            onContinue={handleQualityCheckPass}
            onRerecord={() => navigate("S5_RECORD")}
            onBack={() => navigate("S4_PREP_RECORD")}
          />
        );
      case "S7_ANALYZE":
        return (
          <S7Analyze
            audioBlob={context.recordingData.audioBlob!}
            gateData={context.gateData}
            symptomData={context.symptomData}
            onResult={handleAnalysisResult}
            onQualityFail={handleQualityFail}
            onError={handleAnalysisError}
            onCancel={() => navigate("S6_QUALITY_CHECK")}
          />
        );
      case "S8_RESULT":
        return (
          <S8Result
            result={context.analysisResult!}
            showHemoptysisBanner={context.symptomData.hemoptysis === "yes"}
            onFindClinic={() => window.open("https://www.google.com/maps/search/puskesmas+terdekat", "_blank")}
            onWhy={() => navigate("S9_WHY")}
            onNextSteps={() => navigate("S10_NEXT_STEPS")}
            onShare={() => {
              const text = "Saya batuk lebih dari 2 minggu dan akan memeriksakan dahak ke puskesmas. Boleh temani saya?";
              navigator.share?.({ text }).catch(() => {
                navigator.clipboard.writeText(text);
                alert("Teks disalin ke clipboard");
              });
            }}
            onRestart={handleRestart}
          />
        );
      case "S9_WHY":
        return (
          <S9Why
            result={context.analysisResult!}
            onBack={() => navigate("S8_RESULT")}
            onHowItWorks={() => navigate("S11_HOW_IT_WORKS")}
          />
        );
      case "S10_NEXT_STEPS":
        return (
          <S10NextSteps
            onFindClinic={() => window.open("https://www.google.com/maps/search/puskesmas+terdekat", "_blank")}
            onBack={() => navigate("S8_RESULT")}
            onShare={() => {
              const text = "Saya batuk lebih dari 2 minggu dan akan memeriksakan dahak ke puskesmas. Boleh temani saya?";
              navigator.share?.({ text }).catch(() => {
                navigator.clipboard.writeText(text);
                alert("Teks disalin ke clipboard");
              });
            }}
          />
        );
      case "S11_HOW_IT_WORKS":
        return <S11HowItWorks onBack={() => navigate("S0_HOME")} modelCard={null} />;
      case "S12_PRIVACY":
        return <S12Privacy onBack={() => navigate("S0_HOME")} />;
      default:
        return <S0Home onStart={() => navigate("S1_CONSENT")} onHowItWorks={() => navigate("S11_HOW_IT_WORKS")} onPrivacy={() => navigate("S12_PRIVACY")} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {demoMode && (
        <div className="bg-yellow-100 border-b border-yellow-300 px-4 py-2 text-center text-sm font-medium text-yellow-800">
          MODE DEMO
        </div>
      )}
      {context.error && (
        <Banner variant="urgent" title={copy.errors[context.error as keyof typeof copy.errors] || copy.errors.generic}>
          <p>Silakan coba lagi atau hubungi dukungan jika masalah berlanjut.</p>
        </Banner>
      )}
      {renderScreen()}
    </div>
  );
}

export default App;