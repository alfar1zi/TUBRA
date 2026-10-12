import { useState } from "react";
import { Button } from "@/components";
import { copy } from "@/copy";

interface S1ConsentProps {
  onContinue: () => void;
  onBack: () => void;
}

export function S1Consent({ onContinue, onBack }: S1ConsentProps) {
  const [agreed, setAgreed] = useState(false);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <header className="mb-6">
            <h1 className="text-2xl font-semibold text-gray-900">{copy.consent.title}</h1>
          </header>

          <div className="space-y-4 mb-6">
            {copy.consent.body.map((text, i) => (
              <div key={i} className="flex items-start gap-3">
                <svg className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <p className="text-gray-700">{text}</p>
              </div>
            ))}
          </div>

          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={agreed}
              onChange={(e) => setAgreed(e.target.checked)}
              className="w-5 h-5 mt-0.5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              aria-describedby="consent-desc"
            />
            <span id="consent-desc" className="text-gray-700">
              {copy.consent.body[3]}
            </span>
          </label>

          <div className="mt-6 space-y-3">
            <Button size="lg" fullWidth onClick={onContinue} disabled={!agreed}>
              {copy.consent.agreeButton}
            </Button>
            <Button variant="secondary" size="md" fullWidth onClick={onBack}>
              {copy.consent.disagreeButton}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}