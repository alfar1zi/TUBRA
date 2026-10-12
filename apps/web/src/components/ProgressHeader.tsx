export interface ProgressHeaderProps {
  step: number;
  totalSteps: number;
  questionNumber?: number;
  totalQuestions?: number;
  onBack?: () => void;
}

export function ProgressHeader({ step, totalSteps, questionNumber, totalQuestions, onBack }: ProgressHeaderProps) {
  const handleBack = onBack ?? (() => { window.history.back(); });
  return (
    <header className="flex items-center justify-between mb-6">
      <button
        type="button"
        onClick={handleBack}
        className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
        aria-label="Kembali"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>
      <div className="flex-1 text-center">
        <p className="text-sm font-medium text-gray-600">Langkah {step} dari {totalSteps}</p>
        {questionNumber !== undefined && totalQuestions !== undefined && (
          <p className="text-xs text-gray-500 mt-0.5">Pertanyaan {questionNumber} dari {totalQuestions}</p>
        )}
      </div>
      <div className="w-10" aria-hidden="true" />
    </header>
  );
}