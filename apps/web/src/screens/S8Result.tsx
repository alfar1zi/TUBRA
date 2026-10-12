import { Button, TierCard, Banner } from "@/components";
import { copy } from "@/copy";

interface AnalysisResult {
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

interface S8ResultProps {
  result: AnalysisResult;
  showHemoptysisBanner: boolean;
  onFindClinic: () => void;
  onWhy: () => void;
  onNextSteps: () => void;
  onShare: () => void;
  onRestart: () => void;
}

export function S8Result({
  result,
  showHemoptysisBanner,
  onFindClinic,
  onWhy,
  onNextSteps,
  onShare,
  onRestart,
}: S8ResultProps) {
  const tierConfig = {
    A: copy.result.tierA,
    B: copy.result.tierB,
    C: copy.result.tierC,
  };

  const config = tierConfig[result.tier];

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        {showHemoptysisBanner && (
          <Banner variant="urgent" title="Batuk berdarah perlu diperiksa secepatnya.">
            <p>Jangan menunggu hasil dari aplikasi ini.</p>
          </Banner>
        )}

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <TierCard
            tier={result.tier}
            title={config.text}
            description={config.description}
            icon={
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {result.tier === "A" && (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                )}
                {result.tier === "B" && (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                )}
                {result.tier === "C" && (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                )}
              </svg>
            }
          />

          <p className="mt-4 text-center text-sm text-gray-600">
            Ini bukan diagnosis. Hanya pemeriksaan dahak yang bisa memastikan.
          </p>

          <div className="mt-6 space-y-3">
            <Button size="lg" fullWidth onClick={onFindClinic}>
              {copy.result.tierA.label === config.label ? "Cari puskesmas terdekat" : "Cari puskesmas terdekat"}
            </Button>
            <Button variant="secondary" size="md" fullWidth onClick={onNextSteps}>
              {copy.result.tierA.label === config.label ? "Apa yang perlu saya siapkan?" : "Apa yang perlu saya siapkan?"}
            </Button>
            <Button variant="secondary" size="md" fullWidth onClick={onWhy}>
              {copy.result.tierA.label === config.label ? "Mengapa hasilnya begini?" : "Mengapa hasilnya begini?"}
            </Button>
            <Button variant="ghost" size="md" fullWidth onClick={onShare}>
              {copy.common.back} ke keluarga
            </Button>
            <Button variant="ghost" size="sm" fullWidth onClick={onRestart}>
              {copy.common.cancel} dari awal
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}