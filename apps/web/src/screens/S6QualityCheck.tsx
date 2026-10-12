import { Button, Banner } from "@/components";
import { copy } from "@/copy";

interface S6QualityCheckProps {
  acceptedCoughs: number;
  rejectedCoughs: number;
  noiseLevel: "quiet" | "moderate" | "loud";
  clipping: boolean;
  errorCode?: string;
  snrDb?: number;
  clippingRatio?: number;
  onContinue: () => void;
  onRerecord: () => void;
  onBack: () => void;
}

export function S6QualityCheck({
  acceptedCoughs,
  rejectedCoughs,
  noiseLevel,
  clipping,
  errorCode,
  snrDb: _snrDb,
  clippingRatio: _clippingRatio,
  onContinue,
  onRerecord,
  onBack,
}: S6QualityCheckProps) {
  const passed = acceptedCoughs >= 3 && noiseLevel !== "loud" && !clipping;

  if (passed) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
        <div className="w-full max-w-md">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 text-center">
            <h1 className="text-2xl font-semibold text-gray-900 mb-2">{copy.qualityCheck.title}</h1>
            <p className="text-gray-600 mb-6">{copy.qualityCheck.checking}</p>
            <div className="space-y-2 mb-6">
              <p className="text-gray-700">{copy.qualityCheck.acceptedLabel} {acceptedCoughs}</p>
              <p className="text-gray-700">{copy.qualityCheck.rejectedLabel} {rejectedCoughs}</p>
            </div>
            <div className="space-y-3">
              <Button size="lg" fullWidth onClick={onContinue}>
                {copy.qualityCheck.continueButton}
              </Button>
              <Button variant="secondary" size="md" fullWidth onClick={onRerecord}>
                {copy.qualityCheck.rerecordButton}
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  let errorText: string = copy.qualityCheck.errorLowQuality;
  if (errorCode) {
    const errorMap: Record<string, string> = {
      E4: copy.errors.E4,
      E5: copy.errors.E5,
      E6: copy.errors.E6,
      E7: copy.errors.E7,
      E8: copy.errors.E8,
      E10: copy.errors.E10,
      E11: copy.errors.E11,
      generic: copy.errors.generic,
      network: copy.errors.network,
    };
    errorText = errorMap[errorCode] || copy.qualityCheck.errorLowQuality;
  } else if (acceptedCoughs < 3) {
    errorText = copy.qualityCheck.errorLowQuality;
  } else if (noiseLevel === "loud") {
    errorText = copy.qualityCheck.errorNoise;
  } else if (clipping) {
    errorText = copy.qualityCheck.errorClipping;
  }
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 text-center">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">{copy.qualityCheck.title}</h1>
          <Banner variant="warning" title="Kualitas tidak memenuhi syarat">
            <p>{errorText}</p>
          </Banner>
          <div className="mt-4 space-y-3">
            <Button size="lg" fullWidth onClick={onRerecord}>
              {copy.qualityCheck.rerecordButton}
            </Button>
            <Button variant="secondary" size="md" fullWidth onClick={onBack}>
              {copy.common.back}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}