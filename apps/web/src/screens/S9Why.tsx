import { MobileShell } from "@/components";
import { copy } from "@/copy";

export interface S9WhyProps {
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

const SYMPTOM_LABELS: Record<string, string> = copy.why.symptomLabels;

export function S9Why({ result, onBack, onHowItWorks }: S9WhyProps) {
  const contributions = Object.entries(result.explanation.symptom_contributions)
    .map(([key, value]) => ({ label: SYMPTOM_LABELS[key] || key, value }))
    .filter((item) => item.value > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, 3);

  return (
    <MobileShell>
      <div
        className="flex flex-col flex-1"
        style={{
          paddingTop: "calc(env(safe-area-inset-top, 62px) + 24px)",
          paddingBottom: "calc(env(safe-area-inset-bottom, 34px) + 13px)",
        }}
      >
        <h1 className="text-2xl font-medium text-text-primary/80">{copy.why.title}</h1>
        <p className="mt-2 text-base font-normal text-text-primary/60">{copy.why.subtitle}</p>

        <div className="mt-8 space-y-6">
          <section>
            <h2 className="mb-3 text-lg font-medium text-text-primary">
              {copy.why.audioContributionTitle}
            </h2>
            <div className="grid aspect-video place-items-center rounded-lg bg-surface-subtle">
              <div className="p-4 text-center">
                <p className="mb-2 text-text-primary/60">Spektrogram batuk</p>
                {result.explanation.gradcam_available && (
                  <p className="mx-auto mt-2 max-w-xs text-xs text-text-primary/60">{copy.why.gradcamNote}</p>
                )}
              </div>
            </div>
          </section>

          <section>
            <h2 className="mb-3 text-lg font-medium text-text-primary">
              {copy.why.symptomContributionTitle}
            </h2>
            {contributions.length > 0 ? (
              <div className="space-y-2">
                {contributions.map((item) => (
                  <div
                    key={item.label}
                    className="flex items-center justify-between rounded-lg bg-surface-subtle p-3"
                  >
                    <span className="text-text-primary/80">{item.label}: menaikkan prioritas</span>
                    <div className="h-2 w-24 overflow-hidden rounded-pill bg-black/10">
                      <div
                        className="h-full rounded-pill bg-brand"
                        style={{ width: `${Math.min(item.value * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-text-primary/60">Belum ada gejala yang menonjol pada hasil ini.</p>
            )}
          </section>

          <section>
            <h2 className="mb-3 text-lg font-medium text-text-primary">{copy.why.limitationsTitle}</h2>
            <ul className="list-inside list-disc space-y-2 text-sm text-text-primary/70">
              {copy.why.limitations.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </section>
        </div>

        <div className="flex-1" />

        <div className="space-y-3">
          <button
            type="button"
            onClick={onBack}
            className="w-full h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.why.backButton}
          </button>
          <button
            type="button"
            onClick={onHowItWorks}
            className="w-full h-[57px] rounded-lg bg-brand-soft text-lg font-medium text-brand transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            {copy.why.howItWorksLink}
          </button>
        </div>
      </div>
    </MobileShell>
  );
}
