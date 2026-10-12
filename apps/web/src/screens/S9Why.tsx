import { Button } from "@/components";
import { copy } from "@/copy";

interface S9WhyProps {
  result: {
    tier: "A" | "B" | "C";
    explanation: {
      symptom_contributions: Record<string, number>;
      gradcam_available: boolean;
      gradcam_url?: string;
    };
  };
  onBack: () => void;
  onHowItWorks: () => void;
}

export function S9Why({ result, onBack, onHowItWorks }: S9WhyProps) {
  const symptomLabels: Record<string, string> = {
    fever: "Demam",
    night_sweats: "Keringat malam",
    weight_loss: "Berat badan turun",
    hemoptysis: "Batuk darah",
    chest_pain: "Nyeri dada",
    fatigue: "Lelah berlebihan",
    smoking: "Merokok",
  };

  const contributions = Object.entries(result.explanation.symptom_contributions)
    .map(([key, value]) => ({ label: symptomLabels[key] || key, value }))
    .filter((item) => item.value > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, 3);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">{copy.why.title}</h1>
          <p className="text-gray-600 mb-8">{copy.why.subtitle}</p>

          <div className="space-y-6">
            <section>
              <h2 className="text-lg font-medium text-gray-900 mb-3">{copy.why.audioContributionTitle}</h2>
              <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                <div className="text-center p-4">
                  <p className="text-gray-500 mb-2">Spektrogram batuk</p>
                  {result.explanation.gradcam_available && (
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500" />
                      <span className="text-sm text-gray-700">Tampilkan sorotan</span>
                    </label>
                  )}
                  {result.explanation.gradcam_available && (
                    <p className="text-xs text-gray-500 mt-2 text-center max-w-xs">
                      {copy.why.gradcamNote}
                    </p>
                  )}
                </div>
              </div>
            </section>

            <section>
              <h2 className="text-lg font-medium text-gray-900 mb-3">{copy.why.symptomContributionTitle}</h2>
              {contributions.length > 0 ? (
                <div className="space-y-2">
                  {contributions.map((item) => (
                    <div key={item.label} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <span className="text-gray-700">{item.label}: menaikkan prioritas</span>
                      <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-600 rounded-full"
                          style={{ width: `${Math.min(item.value * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-600">{copy.why.symptomContributionTitle}</p>
              )}
            </section>

            <section>
              <h2 className="text-lg font-medium text-gray-900 mb-3">{copy.why.limitationsTitle}</h2>
              <ul className="space-y-2 text-sm text-gray-700 list-disc list-inside">
                {copy.why.limitations.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </section>
          </div>

          <div className="mt-8 space-y-3">
            <Button size="lg" fullWidth onClick={onBack}>
              {copy.why.backButton}
            </Button>
            <Button variant="ghost" size="md" fullWidth onClick={onHowItWorks}>
              {copy.why.howItWorksLink}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}