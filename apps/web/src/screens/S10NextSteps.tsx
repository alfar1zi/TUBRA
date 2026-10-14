import { CopyButton, MobileShell } from "@/components";
import { copy } from "@/copy";

export interface S10NextStepsProps {
  onFindClinic: () => void;
  onBack: () => void;
  onShare: () => void;
}

export function S10NextSteps({ onFindClinic, onBack, onShare }: S10NextStepsProps) {
  return (
    <MobileShell>
      <div
        className="flex flex-col flex-1"
        style={{
          paddingTop: "calc(env(safe-area-inset-top, 62px) + 24px)",
          paddingBottom: "calc(env(safe-area-inset-bottom, 34px) + 13px)",
        }}
      >
        <h1 className="text-2xl font-medium text-text-primary/80">{copy.nextSteps.title}</h1>

        <section className="mt-6">
          <h2 className="mb-3 text-lg font-medium text-text-primary">{copy.nextSteps.checklistTitle}</h2>
          <ul className="space-y-2">
            {copy.nextSteps.checklist.map((item, i) => (
              <li key={i} className="flex items-start gap-3">
                <input
                  id={`next-step-${i}`}
                  type="checkbox"
                  className="mt-0.5 h-5 w-5 rounded border-black/20 text-brand focus:ring-brand"
                />
                <label htmlFor={`next-step-${i}`} className="text-text-primary/80">{item}</label>
              </li>
            ))}
          </ul>
        </section>

        <section className="mt-6">
          <h2 className="mb-3 text-lg font-medium text-text-primary">{copy.nextSteps.clinicSentenceTitle}</h2>
          <CopyButton textToCopy={copy.nextSteps.clinicSentence} defaultLabel={copy.nextSteps.copyLabel}>
            {copy.nextSteps.copyLabel}
          </CopyButton>
          <p className="mt-2 rounded-lg bg-surface-subtle p-3 text-sm text-text-primary/70">
            "{copy.nextSteps.clinicSentence}"
          </p>
        </section>

        <section className="mt-6 space-y-3">
          <p className="text-text-primary/80">{copy.nextSteps.examInfo}</p>
          <p className="text-text-primary/80">{copy.nextSteps.supportText}</p>
          <p className="text-sm text-text-primary/60">{copy.nextSteps.costInfo}</p>
        </section>

        <div className="flex-1" />

        <div className="mt-6 space-y-3">
          <button
            type="button"
            onClick={onFindClinic}
            className="w-full h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            Cari puskesmas terdekat
          </button>
          <button
            type="button"
            onClick={onShare}
            className="w-full h-[57px] rounded-lg bg-brand-soft text-lg font-medium text-brand transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
          >
            Bagikan ke keluarga
          </button>
          <button
            type="button"
            onClick={onBack}
            className="w-full text-base font-medium text-brand underline underline-offset-2"
          >
            {copy.common.back} ke hasil
          </button>
        </div>
      </div>
    </MobileShell>
  );
}
