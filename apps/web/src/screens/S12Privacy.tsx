import { Button } from "@/components";
import { copy } from "@/copy";

interface S12PrivacyProps {
  onBack: () => void;
}

export function S12Privacy({ onBack }: S12PrivacyProps) {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-semibold text-gray-900">{copy.privacy.title}</h1>
            <Button variant="ghost" size="sm" onClick={onBack}>
              {copy.common.back}
            </Button>
          </div>

          <div className="space-y-4">
            {copy.privacy.body.map((text, i) => (
              <p key={i} className="text-gray-700 leading-relaxed">
                {text}
              </p>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}