import type { ScreenPayload } from "@/api/client";

export type ScreenId =
  | "S0_HOME"
  | "S1_CONSENT"
  | "S2_GATE"
  | "S2x_OUT_OF_RANGE"
  | "S2y_NOT_TIME"
  | "S3_SYMPTOMS"
  | "S4_PREP_RECORD"
  | "S5_RECORD"
  | "S6_QUALITY_CHECK"
  | "S7_ANALYZE"
  | "S8_RESULT"
  | "S9_WHY"
  | "S10_NEXT_STEPS"
  | "S11_HOW_IT_WORKS"
  | "S12_PRIVACY";
export interface GateData {
  age: number | null;
  coughDurationWeeks: number | null;
}

export interface SymptomData {
  fever: "yes" | "no" | null;
  nightSweats: "yes" | "no" | null;
  weightLoss: "yes" | "no" | null;
  hemoptysis: "yes" | "no" | null;
  chestPain: "yes" | "no" | null;
  fatigue: "yes" | "no" | null;
  smoking: "yes" | "no" | null;
}

export interface RecordingData {
  audioBlob: Blob | null;
  acceptedCoughs: number;
  rejectedCoughs: number;
  qualityError: string | null;
  noiseLevel?: "quiet" | "moderate" | "loud";
  clipping?: boolean;
  snrDb?: number;
  clippingRatio?: number;
}

export interface AnalysisResult {
  tier: "A" | "B" | "C";
  probability: number;
  explanation: {
    symptom_contributions: Record<string, number>;
    gradcam_available: boolean;
    gradcam_url?: string;
  };
  next_steps: string[];
  metadata: {
    accepted_coughs: number;
    rejected_coughs: number;
    processing_time_ms: number;
  };
}

export interface ScreenContext {
  currentScreen: ScreenId;
  gateData: GateData;
  symptomData: SymptomData;
  recordingData: RecordingData;
  analysisResult: AnalysisResult | null;
  demoMode: boolean;
  error: string | null;
}

export type ScreenAction =
  | { type: "GO_TO"; screen: ScreenId }
  | { type: "SET_GATE_DATA"; data: Partial<GateData> }
  | { type: "SET_SYMPTOM_DATA"; data: Partial<SymptomData> }
  | { type: "SET_RECORDING_DATA"; data: Partial<RecordingData> }
  | { type: "SET_ANALYSIS_RESULT"; result: AnalysisResult }
  | { type: "SET_DEMO_MODE"; enabled: boolean }
  | { type: "SET_ERROR"; error: string | null }
  | { type: "RESET" };

export const initialContext: ScreenContext = {
  currentScreen: "S0_HOME",
  gateData: {
    age: null,
    coughDurationWeeks: null,
  },
  symptomData: {
    fever: null,
    nightSweats: null,
    weightLoss: null,
    hemoptysis: null,
    chestPain: null,
    fatigue: null,
    smoking: null,
  },
  recordingData: {
    audioBlob: null,
    acceptedCoughs: 0,
    rejectedCoughs: 0,
    qualityError: null,
    noiseLevel: "quiet",
    clipping: false,
    snrDb: undefined,
    clippingRatio: undefined,
  },
  analysisResult: null,
  demoMode: false,
  error: null,
};

export function screenReducer(context: ScreenContext, action: ScreenAction): ScreenContext {
  switch (action.type) {
    case "GO_TO":
      return { ...context, currentScreen: action.screen, error: null };
    case "SET_GATE_DATA":
      return { ...context, gateData: { ...context.gateData, ...action.data } };
    case "SET_SYMPTOM_DATA":
      return { ...context, symptomData: { ...context.symptomData, ...action.data } };
    case "SET_RECORDING_DATA":
      return { ...context, recordingData: { ...context.recordingData, ...action.data } };
    case "SET_ANALYSIS_RESULT":
      return { ...context, analysisResult: action.result };
    case "SET_DEMO_MODE":
      return { ...context, demoMode: action.enabled };
    case "SET_ERROR":
      return { ...context, error: action.error };
    case "RESET":
      return initialContext;
    default:
      return context;
  }
}

export function canProceedFromGate(gateData: GateData): boolean {
  return (
    gateData.age !== null &&
    gateData.age >= 18 &&
    gateData.age <= 100 &&
    gateData.coughDurationWeeks !== null &&
    gateData.coughDurationWeeks >= 2
  );
}

export function canProceedFromSymptoms(symptomData: SymptomData): boolean {
  return Object.values(symptomData).every((v) => v !== null);
}

export function buildScreenPayload(
  gateData: GateData,
  symptomData: SymptomData
): ScreenPayload {
  return {
    age: gateData.age!,
    sex: "M",
    cough_duration_weeks: gateData.coughDurationWeeks!,
    fever: symptomData.fever!,
    night_sweats: symptomData.nightSweats!,
    weight_loss: symptomData.weightLoss!,
    hemoptysis: symptomData.hemoptysis!,
    chest_pain: symptomData.chestPain!,
    fatigue: symptomData.fatigue!,
    smoking: symptomData.smoking!,
  };
}