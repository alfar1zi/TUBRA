import { useReducer, useEffect, useCallback, useState } from "react";
import { getHealth, ScreenResponse } from "@/api/client";
import {
  screenReducer,
  initialContext,
  ScreenId,
  SymptomData,
  GateData,
} from "@/state/screenState";
import {
  S0Splash,
  S0Home,
  S1Consent,
  S3Questions,
  S2xOutOfRange,
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

/**
 * Copies `text` and confirms to the user, tolerating environments where the
 * Clipboard API is unavailable or blocked (insecure context, denied
 * permission). Never throws into a React event handler.
 */
async function copyToClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    alert(copy.nextSteps.copiedToClipboard);
  } catch {
    alert(text);
  }
}

function App() {
  const [context, dispatch] = useReducer(screenReducer, initialContext);
  const [demoMode, setDemoMode] = useState(false);

  // Check for demo mode from URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("demo") === "1") {
      setDemoMode(true);
    }
  }, []);

  // Warm up health check on mount
  useEffect(() => {
    void getHealth().catch(() => {}); // Don't block on health check failure
  }, []);

  const navigate = useCallback((screen: ScreenId) => {
    dispatch({ type: "GO_TO", screen });
  }, []);

  const handleGateChange = useCallback((data: Partial<GateData>) => {
    dispatch({ type: "SET_GATE_DATA", data });
  }, []);

  const handleSymptomChange = useCallback((data: Partial<SymptomData>) => {
    dispatch({ type: "SET_SYMPTOM_DATA", data });
  }, []);

  const handleQuestionStepChange = useCallback((step: number) => {
    dispatch({ type: "SET_QUESTION_STEP", step });
  }, []);

  const handleStartQuestions = useCallback(() => {
    dispatch({ type: "SET_QUESTION_STEP", step: 0 });
    navigate("S2_GATE");
  }, [navigate]);

  // The design has no separate "prep to record" screen: once the last question
  // is answered the recording stage begins directly (the mic permission prompt
  // is raised by the record screen's auto-start).
  const handleQuestionsComplete = useCallback(() => {
    navigate("S5_RECORD");
  }, [navigate]);

  const handleRecordingComplete = useCallback((audioBlob: Blob, coughData: { count: number; events: Array<{startMs: number, endMs: number}>; peakAmplitude: number }) => {
    // A sample at full scale means the input clipped: the waveform is cut flat
    // and the cough's shape is lost, which is what the server needs to score
    // it. Report it instead of always claiming the take was clean.
    dispatch({ type: "SET_RECORDING_DATA", data: {
      audioBlob,
      acceptedCoughs: coughData.count,
      rejectedCoughs: 0,
      noiseLevel: "quiet",
      clipping: coughData.peakAmplitude >= 0.99,
    } });
    navigate("S6_QUALITY_CHECK");
  }, [navigate]);

  const handleAnalysisResult = useCallback((result: ScreenResponse) => {
    dispatch({ type: "SET_ANALYSIS_RESULT", result: {
      tier: result.tier,
      probability: result.metadata.probability,
      explanation: {
        // The ONNX audio branch exposes no intermediate feature maps, so
        // Grad-CAM is never available at serving time (see routes/screen.py).
        symptom_contributions: {},
        gradcam_available: false,
        gradcam_url: result.gradcam_url ?? undefined,
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

  const handleQualityFail = useCallback(
    (data: {
      acceptedCoughs: number;
      rejectedCoughs: number;
      noiseLevel: "quiet" | "moderate" | "loud";
      clipping: boolean;
      errorCode: string;
      snrDb?: number;
      clippingRatio?: number;
      qualityReasons?: string[];
    }) => {
      dispatch({
        type: "SET_RECORDING_DATA",
        data: {
          acceptedCoughs: data.acceptedCoughs,
          rejectedCoughs: data.rejectedCoughs,
          qualityError: data.errorCode,
          noiseLevel: data.noiseLevel,
          clipping: data.clipping,
          snrDb: data.snrDb,
          clippingRatio: data.clippingRatio,
          qualityReasons: data.qualityReasons,
        },
      });
      navigate("S6_QUALITY_CHECK");
    },
    [navigate],
  );

  // A failed request is not a failed take. S7Analyze keeps the user on its own
  // screen with a retry button, so the recording is not thrown away and the
  // confirmation screen no longer shows a transport error as if the audio were
  // bad (that is what made "Lihat hasil" look like it did nothing).
  const handleAnalysisError = useCallback((errorCode: string) => {
    dispatch({ type: "SET_ERROR", error: errorCode });
  }, []);

  const handleQualityCheckPass = useCallback(() => {
    navigate("S7_ANALYZE");
  }, [navigate]);

  const handleRerecord = useCallback(() => {
    dispatch({ type: "SET_RECORDING_DATA", data: {
      audioBlob: null,
      acceptedCoughs: 0,
      rejectedCoughs: 0,
      qualityError: null,
      noiseLevel: "quiet",
      clipping: false,
      snrDb: undefined,
      clippingRatio: undefined,
    }});
    navigate("S5_RECORD");
  }, [navigate]);

  const handleQualityCheckBack = useCallback(() => {
    handleRerecord();
  }, [handleRerecord]);

  const handleRestart = useCallback(() => {
    if (window.confirm(copy.common.restartConfirm)) {
      dispatch({ type: "RESET" });
    }
  }, []);

  const renderScreen = () => {
    const { currentScreen } = context;

    switch (currentScreen) {
      case "S0_SPLASH":
        return <S0Splash onDone={() => navigate("S0_HOME")} />;
      case "S0_HOME":
        return <S0Home onStart={() => navigate("S1_CONSENT")} />;
      case "S1_CONSENT":
        return (
          <S1Consent
            onContinue={handleStartQuestions}
            onBack={() => navigate("S0_HOME")}
          />
        );
      case "S2_GATE":
        return (
          <S3Questions
            gateData={context.gateData}
            symptomData={context.symptomData}
            stepIndex={context.questionStep}
            onStepChange={handleQuestionStepChange}
            onGateChange={handleGateChange}
            onSymptomChange={handleSymptomChange}
            onComplete={handleQuestionsComplete}
            onBack={() => navigate("S1_CONSENT")}
            onOutOfRange={(reason) => {
              dispatch({ type: "SET_OUT_OF_RANGE_REASON", reason });
              navigate("S2x_OUT_OF_RANGE");
            }}
          />
        );
      case "S2x_OUT_OF_RANGE":
        return (
          <S2xOutOfRange
            reason={context.outOfRangeReason}
            onFindClinic={() => window.open("https://www.google.com/maps/search/puskesmas+terdekat", "_blank")}
            onBackToStart={() => navigate("S0_HOME")}
          />
        );
      case "S3_SYMPTOMS":
        // The gate and symptom steps are now a single nine-question flow; this
        // legacy state redirects to it so persisted state still resolves.
        return (
          <S3Questions
            gateData={context.gateData}
            symptomData={context.symptomData}
            stepIndex={context.questionStep}
            onStepChange={handleQuestionStepChange}
            onGateChange={handleGateChange}
            onSymptomChange={handleSymptomChange}
            onComplete={handleQuestionsComplete}
            onBack={() => navigate("S1_CONSENT")}
            onOutOfRange={(reason) => {
              dispatch({ type: "SET_OUT_OF_RANGE_REASON", reason });
              navigate("S2x_OUT_OF_RANGE");
            }}
          />
        );
      case "S4_PREP_RECORD":
        // Legacy state: the design folds mic permission into the record screen
        // (see handleQuestionsComplete). Resolve any persisted S4 straight into
        // the record stage instead of rendering the retired prep screen.
        return <S5Record onComplete={handleRecordingComplete} onBack={() => navigate("S2_GATE")} />;
      case "S5_RECORD":
        return <S5Record onComplete={handleRecordingComplete} onBack={() => navigate("S2_GATE")} />;
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
            qualityReasons={context.recordingData.qualityReasons}
            onContinue={handleQualityCheckPass}
            onRerecord={handleRerecord}
            onBack={handleQualityCheckBack}
          />
        );
      case "S7_ANALYZE":
        // Requires a recorded blob; if it is missing (stale deep-link/reset)
        // fall back home rather than non-null-asserting into a crash.
        if (!context.recordingData.audioBlob) return <S0Home onStart={() => navigate("S1_CONSENT")} />;
        return (
          <S7Analyze
            audioBlob={context.recordingData.audioBlob}
            gateData={context.gateData}
            symptomData={context.symptomData}
            onResult={handleAnalysisResult}
            onQualityFail={handleQualityFail}
            onError={handleAnalysisError}
            onCancel={() => navigate("S6_QUALITY_CHECK")}
          />
        );
      case "S8_RESULT":
        // Reachable only after a successful screen; if the result is missing
        // (e.g. a stale deep-link or a reset race) fall back home instead of
        // non-null-asserting into a crash.
        if (!context.analysisResult) return <S0Home onStart={() => navigate("S1_CONSENT")} />;
        return (
          <S8Result
            result={context.analysisResult}
            showHemoptysisBanner={context.symptomData.hemoptysis === "yes"}
            onFindClinic={() => window.open("https://www.google.com/maps/search/puskesmas+terdekat", "_blank")}
            onWhy={() => navigate("S9_WHY")}
            onRestart={handleRestart}
          />
        );
      case "S9_WHY":
        if (!context.analysisResult) return <S0Home onStart={() => navigate("S1_CONSENT")} />;
        return (
          <S9Why
            result={context.analysisResult}
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
              const text = copy.nextSteps.shareMessage;
              // navigator.share is undefined on desktop/older browsers; the
              // optional-call chain does not protect the following .catch(),
              // so branch explicitly and fall back to clipboard.
              if (typeof navigator.share === "function") {
                navigator.share({ text }).catch(() => copyToClipboard(text));
              } else {
                copyToClipboard(text);
              }
            }}
          />
        );
      case "S11_HOW_IT_WORKS":
        return <S11HowItWorks onBack={() => navigate("S0_HOME")} modelCard={null} />;
      case "S12_PRIVACY":
        return <S12Privacy onBack={() => navigate("S0_HOME")} />;
      default:
        return <S0Home onStart={() => navigate("S1_CONSENT")} />;
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