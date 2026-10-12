import { useState } from "react";
import { Button, ChoiceCard, NumberStepper, ProgressHeader } from "@/components";
import { copy } from "@/copy";

interface S2GateProps {
  onContinue: (age: number, coughDuration: string) => void;
  onBack: () => void;
  onOutOfRange: () => void;
  onNotTime: () => void;
}

export function S2Gate({ onContinue, onBack, onOutOfRange, onNotTime }: S2GateProps) {
  const [age, setAge] = useState<number>(18);
  const [coughDuration, setCoughDuration] = useState<string | null>(null);
  const [ageError, setAgeError] = useState("");
  const [step, setStep] = useState<"age" | "cough">("age");

  const handleAgeContinue = () => {
    if (age < 18) {
      onOutOfRange();
      return;
    }
    if (age > 100) {
      setAgeError("Usia maksimal 100 tahun");
      return;
    }
    setAgeError("");
    setStep("cough");
  };

  const handleCoughContinue = () => {
    if (!coughDuration) return;
    if (coughDuration === "none" || coughDuration === "lt2") {
      onNotTime();
      return;
    }
    onContinue(age, coughDuration);
  };

  if (step === "age") {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
        <div className="w-full max-w-md">
          <ProgressHeader step={1} totalSteps={4} questionNumber={1} totalQuestions={9} />
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h1 className="text-2xl font-semibold text-gray-900 mb-2">{copy.gate.title}</h1>
            <p className="text-gray-600 mb-8">{copy.gate.subtitle}</p>

            <NumberStepper
              label={copy.gate.ageLabel}
              value={age}
              onChange={(v) => {
                if (v !== null) {
                  setAge(v);
                  setAgeError("");
                }
              }}
              min={1}
              max={100}
              error={ageError}
            />

            <div className="mt-6 space-y-3">
              <Button size="lg" fullWidth onClick={handleAgeContinue}>
                Lanjut
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

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <ProgressHeader step={1} totalSteps={4} questionNumber={2} totalQuestions={9} />
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">{copy.gate.title}</h1>
          <p className="text-gray-600 mb-8">{copy.gate.coughDurationLabel}</p>

          <div className="space-y-3">
            {[
              { value: "none", label: "Saya tidak sedang batuk" },
              { value: "lt2", label: "Kurang dari 2 minggu" },
              { value: "2_4", label: "2 sampai 4 minggu" },
              { value: "4_8", label: "1 sampai 2 bulan" },
              { value: "8_plus", label: "Lebih dari 2 bulan" },
            ].map((opt) => (
              <ChoiceCard
                key={opt.value}
                selected={coughDuration === opt.value}
                onClick={() => setCoughDuration(opt.value)}
              >
                <p className="text-gray-900">{opt.label}</p>
              </ChoiceCard>
            ))}
          </div>

          <div className="mt-6 space-y-3">
            <Button size="lg" fullWidth onClick={handleCoughContinue} disabled={!coughDuration}>
              {copy.gate.continueButton}
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