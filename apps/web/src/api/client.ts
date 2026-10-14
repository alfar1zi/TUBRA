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

export interface ApiErrorBody {
  code: string;
  retryable: boolean;
  message: string;
  details?: Record<string, unknown>;
}

export interface EchoTestError {
  error: ApiErrorBody;
}

export type CoughDuration = "2_4" | "4_8" | "8_plus";
export type Sex = "male" | "female";
export type SmokingStatus = "never" | "former" | "current";

export interface ScreenPayload {
  age: number;
  sex: Sex;
  cough_duration: CoughDuration;
  fever: boolean | null;
  night_sweats: boolean | null;
  weight_loss: boolean | null;
  hemoptysis: boolean | null;
  prior_tb: boolean | null;
  smoking: SmokingStatus | null;
}

export interface ScreenMetadata {
  model_version: string;
  probability: number;
  prob_audio?: number | null;
  prob_clinical?: number | null;
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
}

export interface ScreenResponse {
  tier: "A" | "B" | "C";
  tier_label?: string | null;
  explanation: string;
  next_steps: string[];
  actions?: string[] | null;
  disclaimer?: string | null;
  gradcam_url: string | null;
  metadata: ScreenMetadata;
}

export type ScreenErrorResponse = ApiErrorBody;

export class ScreenApiError extends Error {
  constructor(
    public readonly code: string,
    public readonly message: string,
    public readonly details?: Record<string, unknown>,
    public readonly retryable: boolean = false,
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

export async function postScreen(
  audioBlob: Blob,
  payload: ScreenPayload,
  signal?: AbortSignal
): Promise<ScreenResponse> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");
  formData.append("payload", JSON.stringify(payload));

  const response = await fetch("/api/v1/screen", {
    method: "POST",
    body: formData,
    signal,
  });

  if (!response.ok) {
    const { error } = (await response.json()) as { error: ScreenErrorResponse };
    throw new ScreenApiError(error.code, error.message, error.details, error.retryable);
  }

  return response.json();
}

export async function getHealth(): Promise<{ status: string; model_version: string }> {
  const response = await fetch("/api/v1/health");
  return response.json();
}