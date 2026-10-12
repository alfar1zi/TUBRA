import { useState } from "react";
import { Button, ChoiceCard, ProgressHeader, Banner } from "@/components";
import { copy } from "@/copy";

type SymptomKey = "sex" | "fever" | "nightSweats" | "weightLoss" | "hemoptysis" | "chestPain" | "fatigue" | "smoking" | "tbPrior";

interface SymptomQuestion {
  key: SymptomKey;
  question: string;
  options: { value: "yes" | "no" | "unknown" | "skip"; label: string }[];
  helperText?: string;
}

const QUESTIONS: SymptomQuestion[] = [
  {
    key: "sex",
    question: "Jenis kelamin Anda?",
    options: [
      { value: "yes", label: "Laki-laki" },
      { value: "no", label: "Perempuan" },
      { value: "skip", label: "Lewati" },
    ],
  },
  {
    key: "hemoptysis",
    question: "Apakah batuk Anda pernah disertai darah?",
    options: [
      { value: "yes", label: "Ya" },
      { value: "no", label: "Tidak" },
      { value: "unknown", label: "Tidak tahu" },
    ],
  },
  {
    key: "fever",
    question: "Dalam sebulan terakhir, apakah Anda sering demam yang hilang timbul?",
    options: [
      { value: "yes", label: "Ya" },
      { value: "no", label: "Tidak" },
      { value: "unknown", label: "Tidak tahu" },
    ],
  },
  {
    key: "nightSweats",
    question: "Apakah Anda sering berkeringat banyak di malam hari sampai pakaian atau seprai basah?",
    options: [
      { value: "yes", label: "Ya" },
      { value: "no", label: "Tidak" },
      { value: "unknown", label: "Tidak tahu" },
    ],
    helperText: "Contoh: bangun tidur dengan baju basah keringat meski cuaca tidak panas.",
  },
  {
    key: "weightLoss",
    question: "Apakah berat badan Anda turun tanpa sebab yang jelas dalam sebulan terakhir?",
    options: [
      { value: "yes", label: "Ya" },
      { value: "no", label: "Tidak" },
      { value: "unknown", label: "Tidak tahu" },
    ],
    helperText: "Contoh: turun 2-3 kg tanpa diet atau sakit lain.",
  },
  {
    key: "tbPrior",
    question: "Apakah Anda pernah sakit TBC atau minum obat TBC sebelumnya?",
    options: [
      { value: "yes", label: "Ya" },
      { value: "no", label: "Tidak" },
      { value: "unknown", label: "Tidak tahu" },
    ],
  },
  {
    key: "smoking",
    question: "Apakah Anda merokok?",
    options: [
      { value: "no", label: "Tidak pernah" },
      { value: "unknown", label: "Pernah tapi sudah berhenti" },
      { value: "yes", label: "Ya masih merokok" },
      { value: "skip", label: "Lewati" },
    ],
  },
];

interface S3SymptomsProps {
  onContinue: (data: Record<SymptomKey, "yes" | "no" | null>) => void;
  onBack: () => void;
  onHemoptysisYes: () => void;
}

export function S3Symptoms({ onContinue, onBack, onHemoptysisYes }: S3SymptomsProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, "yes" | "no" | "unknown" | "skip" | null>>({});
  const [showHemoptysisBanner, setShowHemoptysisBanner] = useState(false);

  const currentQuestion = QUESTIONS[currentIndex];
  const isFirst = currentIndex === 0;
  const isLast = currentIndex === QUESTIONS.length - 1;

  const handleOptionSelect = (value: "yes" | "no" | "unknown" | "skip") => {
    const newAnswers = { ...answers, [currentQuestion.key]: value };
    setAnswers(newAnswers);

    if (currentQuestion.key === "hemoptysis" && value === "yes") {
      setShowHemoptysisBanner(true);
      onHemoptysisYes();
    }

    if (isLast) {
      const finalAnswers: Record<SymptomKey, "yes" | "no" | null> = {
        sex: (newAnswers.sex === "yes" ? "yes" : newAnswers.sex === "no" ? "no" : null) as "yes" | "no" | null,
        fever: (newAnswers.fever === "yes" ? "yes" : newAnswers.fever === "no" ? "no" : null) as "yes" | "no" | null,
        nightSweats: (newAnswers.nightSweats === "yes" ? "yes" : newAnswers.nightSweats === "no" ? "no" : null) as "yes" | "no" | null,
        weightLoss: (newAnswers.weightLoss === "yes" ? "yes" : newAnswers.weightLoss === "no" ? "no" : null) as "yes" | "no" | null,
        hemoptysis: (newAnswers.hemoptysis === "yes" ? "yes" : newAnswers.hemoptysis === "no" ? "no" : null) as "yes" | "no" | null,
        chestPain: null,
        fatigue: null,
        smoking: (newAnswers.smoking === "yes" ? "yes" : newAnswers.smoking === "no" ? "no" : null) as "yes" | "no" | null,
        tbPrior: null,
      };
      onContinue(finalAnswers);
    } else {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const handleBack = () => {
    if (isFirst) {
      onBack();
    } else {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const canContinue = answers[currentQuestion.key] !== undefined;

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <ProgressHeader step={1} totalSteps={4} questionNumber={currentIndex + 3} totalQuestions={9} onBack={handleBack} />

        {showHemoptysisBanner && (
          <Banner variant="urgent" title="Batuk berdarah perlu diperiksa secepatnya.">
            <p>Jangan menunggu hasil dari aplikasi ini.</p>
          </Banner>
        )}

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">{copy.symptoms.title}</h1>
          <p className="text-gray-600 mb-8">{copy.symptoms.subtitle}</p>

          <div className="space-y-4">
            <p className="text-lg font-medium text-gray-900">{currentQuestion.question}</p>
            {currentQuestion.helperText && (
              <p className="text-sm text-gray-500">{currentQuestion.helperText}</p>
            )}

            <div className="space-y-3 mt-4">
              {currentQuestion.options.map((opt) => (
                <ChoiceCard
                  key={opt.value}
                  selected={answers[currentQuestion.key] === opt.value}
                  onClick={() => handleOptionSelect(opt.value)}
                >
                  <p className="text-gray-900">{opt.label}</p>
                </ChoiceCard>
              ))}
            </div>
          </div>

          <div className="mt-6 space-y-3">
            <Button size="lg" fullWidth onClick={() => {}} disabled={!canContinue}>
              {isLast ? copy.symptoms.continueButton : "Lanjut"}
            </Button>
            {isFirst && (
              <Button variant="secondary" size="md" fullWidth onClick={onBack}>
                {copy.common.back}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}