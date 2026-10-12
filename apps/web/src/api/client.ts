export interface EchoTestResponse {
  sample_rate: number;
  duration_s: number;
  peak: number;
  rms: number;
  num_samples: number;
  applied_constraints: {
    max_body_mb: number;
    required_format: string;
  };
}

export interface EchoTestError {
  error: {
    code: string;
    message: string;
  };
}

export interface ScreenPayload {
  age: number;
  sex: "M" | "F";
  cough_duration_weeks: number;
  fever: "yes" | "no";
  night_sweats: "yes" | "no";
  weight_loss: "yes" | "no";
  hemoptysis: "yes" | "no";
  chest_pain: "yes" | "no";
  fatigue: "yes" | "no";
  smoking: "yes" | "no";
}

export interface SymptomContribution {
  symptom: string;
  contribution: number;
}

export interface Explanation {
  symptom_contributions: SymptomContribution[];
  gradcam_available: boolean;
  gradcam_url: string | null;
}

export interface ScreenMetadata {
  model_version: string;
  probability: number;
  logit: number;
  thresholds: Record<string, number>;
  cough_segments_used: number;
  audio_metrics: {
    sample_rate: number;
    duration_s: number;
    peak: number;
    rms: number;
    num_samples: number;
  };
  quality_details: {
    passed: boolean;
    num_coughs: number;
    cough_segments_ms: number[][];
    snr_db: number;
    clipping_ratio: number;
    noise_floor_db: number;
    reasons: string[];
  };
  processing_time_ms: number;
  symptom_explanations: string[];
  gradcam: number[][] | null;
}

export interface ScreenResponse {
  tier: "A" | "B" | "C";
  probability: number;
  explanation: string;
  gradcam_url: string | null;
  metadata: ScreenMetadata;
  next_steps: string[];
}

export interface ScreenErrorResponse {
  error: string;
  message: string;
  details?: Record<string, unknown>;
}

export class ScreenApiError extends Error {
  constructor(
    public readonly code: string,
    public readonly message: string,
    public readonly details?: Record<string, unknown>
  ) {
    super(code);
    this.name = "ScreenApiError";
  }
}

export async function postEchoTest(audioBlob: Blob): Promise<EchoTestResponse> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");

  const response = await fetch("/api/v1/echo-test", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData: EchoTestError = await response.json();
    throw new Error(errorData.error.code);
  }

  return response.json();
}

export async function postScreen(audioBlob: Blob, payload: ScreenPayload): Promise<ScreenResponse> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");
  formData.append("payload", JSON.stringify(payload));

  const response = await fetch("/api/v1/screen", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData: ScreenErrorResponse = await response.json();
    throw new ScreenApiError(errorData.error, errorData.message, errorData.details);
  }

  return response.json();
}

export async function getHealth(): Promise<{ status: string; model_version: string }> {
  const response = await fetch("/api/v1/health");
  return response.json();
}