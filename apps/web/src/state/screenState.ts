import type { CoughDuration, ScreenPayload } from "@/api/client";

export type ScreenId =
  | "S0_SPLASH"
  | "S0_HOME"
  | "S1_CONSENT"
  | "S2_GATE"
  | "S2x_OUT_OF_RANGE"
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
  /** Cough-duration bucket, exactly as the API expects (`"2_4"` | `"4_8"` | `"8_plus"`). */
  coughDuration: CoughDuration | null;
}

/** Symptom answers as captured on S3. `"yes"`/`"no"` are literal; `null` = not answered. */
export type TriState = "yes" | "no" | null;

export interface SymptomData {
  sex: TriState; // "yes" = male, "no" = female, null = skipped
  fever: TriState;
  nightSweats: TriState;
  weightLoss: TriState;
  hemoptysis: TriState;
  priorTb: TriState;
  /** "yes" = current smoker, "no" = never, "unknown" = former, null = skipped. */
  smoking: "yes" | "no" | "unknown" | null;
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
  /** Index of the question currently shown in the S3 nine-question flow. */
  questionStep: number;
  gateData: GateData;
  symptomData: SymptomData;
  recordingData: RecordingData;
  analysisResult: AnalysisResult | null;
  demoMode: boolean;
  error: string | null;
}

export type ScreenAction =
  | { type: "GO_TO"; screen: ScreenId }
  | { type: "SET_QUESTION_STEP"; step: number }
  | { type: "SET_GATE_DATA"; data: Partial<GateData> }
  | { type: "SET_SYMPTOM_DATA"; data: Partial<SymptomData> }
  | { type: "SET_RECORDING_DATA"; data: Partial<RecordingData> }
  | { type: "SET_ANALYSIS_RESULT"; result: AnalysisResult }
  | { type: "SET_DEMO_MODE"; enabled: boolean }
  | { type: "SET_ERROR"; error: string | null }
  | { type: "RESET" };

export const initialContext: ScreenContext = {
  currentScreen: "S0_SPLASH",
  questionStep: 0,
  gateData: {
    age: null,
    coughDuration: null,
  },
  symptomData: {
    sex: null,
    fever: null,
    nightSweats: null,
    weightLoss: null,
    hemoptysis: null,
    priorTb: null,
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
    case "SET_QUESTION_STEP":
      return { ...context, questionStep: action.step };
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
    gateData.coughDuration !== null
  );
}

export function canProceedFromSymptoms(symptomData: SymptomData): boolean {
  return Object.values(symptomData).every((v) => v !== null);
}

export function buildScreenPayload(
  gateData: GateData,
  symptomData: SymptomData
): ScreenPayload {
  // S2 blocks "none"/"<2 weeks", so coughDuration is always a valid bucket here.
  const coughDuration = gateData.coughDuration;
  const age = gateData.age;
  if (coughDuration === null || age === null) {
    throw new Error("buildScreenPayload: gate data incomplete");
  }

  let sex: ScreenPayload["sex"];
  if (symptomData.sex === "yes") sex = "male";
  else if (symptomData.sex === "no") sex = "female";
  else throw new Error("buildScreenPayload: sex is required");

  let smoking: ScreenPayload["smoking"];
  if (symptomData.smoking === "yes") smoking = "current";
  else if (symptomData.smoking === "no") smoking = "never";
  else if (symptomData.smoking === "unknown") smoking = "former";
  else smoking = null;

  const yes = (v: TriState): boolean | null => (v === null ? null : v === "yes");

  return {
    age,
    sex,
    cough_duration: coughDuration,
    fever: yes(symptomData.fever),
    night_sweats: yes(symptomData.nightSweats),
    weight_loss: yes(symptomData.weightLoss),
    hemoptysis: yes(symptomData.hemoptysis),
    prior_tb: yes(symptomData.priorTb),
    smoking,
  };
}