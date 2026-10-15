import { useState } from "react";
import { ChoiceCard, MobileShell, QuestionLayout } from "@/components";
import { copy } from "@/copy";
import type { CoughDuration } from "@/api/client";
import type { GateData, SymptomData, TriState } from "@/state/screenState";

/**
 * The nine screening questions, in the order drawn in the Figma design:
 * age, cough duration, sex, hemoptysis, fever, night sweats, weight loss,
 * prior TB, smoking.
 *
 * Each question is shown on its own screen ("1 pertanyaan/layar"). The step
 * index lives here rather than in the global router so the API-facing state
 * contract (`GateData` + `SymptomData`) stays unchanged.
 */
type StepKey =
  | "age"
  | "coughDuration"
  | "sex"
  | "hemoptysis"
  | "fever"
  | "nightSweats"
  | "weightLoss"
  | "priorTb"
  | "smoking";

const STEPS: StepKey[] = [
  "age",
  "coughDuration",
  "sex",
  "hemoptysis",
  "fever",
  "nightSweats",
  "weightLoss",
  "priorTb",
  "smoking",
];

/**
 * Maps the Figma duration options onto the buckets the API expects. The two
 * remaining options are disqualifiers: "lt2w" (under two weeks) is out of the
 * screening range, and "none" (not coughing now) is not the right time.
 */
const coughDurationMap = {
  w2to4: "2_4",
  m1to2: "4_8",
  gt2m: "8_plus",
} as const satisfies Record<string, CoughDuration>;


export interface S3QuestionsProps {
  gateData: GateData;
  symptomData: SymptomData;
  /** Controlled step index, held by the router so back-navigation restores it. */
  stepIndex: number;
  onStepChange: (step: number) => void;
  onGateChange: (data: Partial<GateData>) => void;
  onSymptomChange: (data: Partial<SymptomData>) => void;
  onComplete: () => void;
  onBack: () => void;
  /** Called when the user has been coughing for less than two weeks. */
  onOutOfRange: (reason: "age" | "short_cough") => void;
}

export function S3Questions({
  gateData,
  symptomData,
  stepIndex,
  onStepChange,
  onGateChange,
  onSymptomChange,
  onComplete,
  onBack,
  onOutOfRange,
}: S3QuestionsProps) {
  /** The raw night-sweats option, so the 3-way UI choice survives selection. */
  const [nightSweatsOption, setNightSweatsOption] = useState<string | null>(null);
  const step = STEPS[stepIndex];
  const q = copy.questions.items;
  const total = STEPS.length;

  const advance = () => {
    if (stepIndex < total - 1) onStepChange(stepIndex + 1);
    else onComplete();
  };
  const goBack = () => {
    if (stepIndex > 0) onStepChange(stepIndex - 1);
    else onBack();
  };

  const ageValid = gateData.age !== null && gateData.age >= q.age.min && gateData.age <= q.age.max;

  const canAdvance = (): boolean => {
    switch (step) {
      case "age":
        return ageValid;
      case "coughDuration":
        return gateData.coughDuration !== null;
      case "sex":
        return symptomData.sex !== null;
      case "hemoptysis":
        return symptomData.hemoptysis !== null;
      case "fever":
        return symptomData.fever !== null;
      case "nightSweats":
        return symptomData.nightSweats !== null;
      case "weightLoss":
        return symptomData.weightLoss !== null;
      case "priorTb":
        return symptomData.priorTb !== null;
      case "smoking":
        return symptomData.smoking !== null;
    }
  };

  const coughOptions = Object.keys(q.coughDuration.options) as Array<
    keyof typeof q.coughDuration.options
  >;

  const isAge = step === "age";

  return (
    <MobileShell>
      <QuestionLayout
        step={stepIndex + 1}
        total={total}
        question={q[step].question}
        helper={isAge ? copy.questions.consentNote : undefined}
        titleGap={isAge ? 53 : 45}
        actionsPad={isAge ? 50 : 47}
        actions={
          isAge ? (
            // Frame 7:137 draws a single full-width "Lanjut" action.
            <button
              type="button"
              disabled={!canAdvance()}
              onClick={advance}
              className="w-full h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
            >
              {copy.questions.nextButton}
            </button>
          ) : (
            <div className="flex gap-2">
              <button
                type="button"
                onClick={goBack}
                className="flex-1 h-[57px] rounded-lg bg-brand-soft text-lg font-medium text-brand transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
              >
                {copy.questions.backButton}
              </button>
              <button
                type="button"
                disabled={!canAdvance()}
                onClick={advance}
                className="flex-1 h-[58px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-lg font-medium text-text-inverse transition-opacity hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2"
              >
                {copy.questions.nextButton}
              </button>
            </div>
          )
        }
      >
        {step === "age" && (
          <AgeInput value={gateData.age} unit={q.age.unit} onChange={(v) => onGateChange({ age: v })} />
        )}

        {step === "coughDuration" && (
          <div className="space-y-2" role="radiogroup">
            {coughOptions.map((key) => {
              const isQualifying = key in coughDurationMap;
              const selected =
                isQualifying &&
                gateData.coughDuration === coughDurationMap[key as keyof typeof coughDurationMap];
              const handleSelect = () => {
                if (key === "lt2w") {
                  onOutOfRange("short_cough");
                } else {
                  onGateChange({ coughDuration: coughDurationMap[key as keyof typeof coughDurationMap] });
                }
              };
              return (
                <ChoiceCard key={key} selected={selected} onClick={handleSelect}>
                  {q.coughDuration.options[key]}
                </ChoiceCard>
              );
            })}
          </div>
        )}

        {step === "sex" && (
          <div className="space-y-2" role="radiogroup">
            <ChoiceCard selected={symptomData.sex === "yes"} onClick={() => onSymptomChange({ sex: "yes" })}>
              {q.sex.options.male}
            </ChoiceCard>
            <ChoiceCard selected={symptomData.sex === "no"} onClick={() => onSymptomChange({ sex: "no" })}>
              {q.sex.options.female}
            </ChoiceCard>
          </div>
        )}

        {step === "hemoptysis" && (
          <BinaryQuestion
            yesLabel={q.hemoptysis.options.yes}
            noLabel={q.hemoptysis.options.no}
            value={symptomData.hemoptysis}
            onChange={(v) => onSymptomChange({ hemoptysis: v })}
          />
        )}

        {step === "fever" && (
          <BinaryQuestion
            yesLabel={q.fever.options.yes}
            noLabel={q.fever.options.no}
            value={symptomData.fever}
            onChange={(v) => onSymptomChange({ fever: v })}
          />
        )}

        {step === "nightSweats" && (
          <div className="space-y-2" role="radiogroup">
            {(
              [
                ["yes", q.nightSweats.options.yes, "yes"],
                ["sometimes", q.nightSweats.options.sometimes, "no"],
                ["no", q.nightSweats.options.no, "no"],
              ] as const
            ).map(([optionKey, label, apiValue]) => (
              <ChoiceCard
                key={optionKey}
                selected={nightSweatsOption === optionKey}
                onClick={() => {
                  setNightSweatsOption(optionKey);
                  onSymptomChange({ nightSweats: apiValue });
                }}
              >
                {label}
              </ChoiceCard>
            ))}
          </div>
        )}

        {step === "weightLoss" && (
          <BinaryQuestion
            yesLabel={q.weightLoss.options.yes}
            noLabel={q.weightLoss.options.no}
            value={symptomData.weightLoss}
            onChange={(v) => onSymptomChange({ weightLoss: v })}
          />
        )}

        {step === "priorTb" && (
          <BinaryQuestion
            yesLabel={q.priorTb.options.yes}
            noLabel={q.priorTb.options.no}
            value={symptomData.priorTb}
            onChange={(v) => onSymptomChange({ priorTb: v })}
          />
        )}

        {step === "smoking" && (
          <div className="space-y-2" role="radiogroup">
            <ChoiceCard selected={symptomData.smoking === "no"} onClick={() => onSymptomChange({ smoking: "no" })}>
              {q.smoking.options.never}
            </ChoiceCard>
            <ChoiceCard
              selected={symptomData.smoking === "unknown"}
              onClick={() => onSymptomChange({ smoking: "unknown" })}
            >
              {q.smoking.options.quit}
            </ChoiceCard>
            <ChoiceCard selected={symptomData.smoking === "yes"} onClick={() => onSymptomChange({ smoking: "yes" })}>
              {q.smoking.options.current}
            </ChoiceCard>
          </div>
        )}
      </QuestionLayout>
    </MobileShell>
  );
}

function BinaryQuestion({
  yesLabel,
  noLabel,
  value,
  onChange,
}: {
  yesLabel: string;
  noLabel: string;
  value: TriState;
  onChange: (value: "yes" | "no") => void;
}) {
  return (
    <div className="space-y-2" role="radiogroup">
      <ChoiceCard selected={value === "yes"} onClick={() => onChange("yes")}>
        {yesLabel}
      </ChoiceCard>
      <ChoiceCard selected={value === "no"} onClick={() => onChange("no")}>
        {noLabel}
      </ChoiceCard>
    </div>
  );
}

/**
 * AgeInput is the stepper shown for question 1, matching the Figma design's
 * [-][value][+] control (three 57px rounded tiles).
 */
function AgeInput({
  value,
  unit,
  onChange,
}: {
  value: number | null;
  unit: string;
  onChange: (value: number) => void;
}) {
  const current = value ?? 18;
  const clamp = (n: number) => Math.max(18, Math.min(100, n));
  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        aria-label={`Kurangi ${unit}`}
        onClick={() => onChange(clamp(current - 1))}
        className="grid place-items-center w-[57px] h-[57px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-2xl font-medium text-text-inverse"
      >
        −
      </button>
      <div className="flex-1 grid place-items-center h-[57px] rounded-lg border border-black/20 bg-white text-lg font-medium text-text-primary">
        {value ?? 18}
      </div>
      <button
        type="button"
        aria-label={`Tambah ${unit}`}
        onClick={() => onChange(clamp(current + 1))}
        className="grid place-items-center w-[57px] h-[57px] rounded-lg bg-gradient-to-b from-brand to-brand-deep text-2xl font-medium text-text-inverse"
      >
        +
      </button>
    </div>
  );
}
