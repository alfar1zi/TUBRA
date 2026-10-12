import { Button, CopyButton } from "@/components";
import { copy } from "@/copy";

interface S10NextStepsProps {
  onFindClinic: () => void;
  onBack: () => void;
  onShare: () => void;
}

export function S10NextSteps({ onFindClinic, onBack, onShare }: S10NextStepsProps) {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">{copy.nextSteps.title}</h1>

          <section className="mb-6">
            <h2 className="text-lg font-medium text-gray-900 mb-3">Yang perlu disiapkan</h2>
            <ul className="space-y-2">
              {copy.nextSteps.checklist.map((item, i) => (
                <li key={i} className="flex items-start gap-3">
                  <input type="checkbox" className="w-5 h-5 mt-0.5 text-blue-600 border-gray-300 rounded focus:ring-blue-500" />
                  <span className="text-gray-700">{item}</span>
                </li>
              ))}
            </ul>
          </section>

          <section className="mb-6">
            <h2 className="text-lg font-medium text-gray-900 mb-3">Kalimat untuk petugas</h2>
            <div className="flex gap-2">
              <CopyButton textToCopy={copy.nextSteps.clinicSentence} defaultLabel="Salin">
                Salin
              </CopyButton>
            </div>
            <p className="mt-2 text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
              "{copy.nextSteps.clinicSentence}"
            </p>
          </section>

          <section className="mb-6">
            <p className="text-gray-700">{copy.nextSteps.examInfo}</p>
          </section>

          <section className="mb-6">
            <p className="text-gray-700">{copy.nextSteps.supportText}</p>
          </section>

          <section className="mb-6">
            <p className="text-sm text-gray-500">{copy.nextSteps.costInfo}</p>
          </section>

          <div className="space-y-3">
            <Button size="lg" fullWidth onClick={onFindClinic}>
              Cari puskesmas terdekat
            </Button>
            <Button variant="secondary" size="md" fullWidth onClick={onShare}>
              Bagikan ke keluarga
            </Button>
            <Button variant="ghost" size="md" fullWidth onClick={onBack}>
              {copy.common.back} ke hasil
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}