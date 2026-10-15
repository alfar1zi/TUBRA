import { MobileShell } from "@/components";
import { copy } from "@/copy";

export interface S12PrivacyProps {
  onBack: () => void;
}

export function S12Privacy({ onBack }: S12PrivacyProps) {
  return (
    <MobileShell>
      <div
        className="flex flex-col flex-1"
        style={{
          paddingTop: "calc(env(safe-area-inset-top, 62px) + 24px)",
          paddingBottom: "calc(env(safe-area-inset-bottom, 34px) + 13px)",
        }}
      >
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-medium text-text-primary/80">{copy.privacy.title}</h1>
          <button
            type="button"
            onClick={onBack}
            className="text-base font-medium text-brand underline underline-offset-2"
          >
            {copy.common.back}
          </button>
        </header>

        <div className="mt-6 space-y-4">
          {copy.privacy.body.map((text, i) => (
            <p key={i} className="text-base font-normal leading-relaxed text-text-primary/70">
              {text}
            </p>
          ))}
        </div>
      </div>
    </MobileShell>
  );
}
