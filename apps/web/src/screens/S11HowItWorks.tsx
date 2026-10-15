import { ExpandablePanel, MobileShell } from "@/components";
import { copy } from "@/copy";

interface S11HowItWorksProps {
  onBack: () => void;
  modelCard: {
    metrics: Record<string, number | null>;
    thresholds: Record<string, number>;
    dataset_info: Record<string, unknown>;
  } | null;
}

export function S11HowItWorks({ onBack, modelCard }: S11HowItWorksProps) {
  const formatMetric = (key: string, value: number | null) => {
    if (value === null) return copy.howItWorks.notEvaluated;
    if (key.includes("threshold")) return value.toFixed(2);
    if (key.includes("ece")) return value.toFixed(4);
    return value.toFixed(3);
  };

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
          <h1 className="text-2xl font-medium text-text-primary/80">{copy.howItWorks.title}</h1>
          <button
            type="button"
            onClick={onBack}
            className="text-base font-medium text-brand underline underline-offset-2"
          >
            {copy.common.back}
          </button>
        </header>

        <div className="mt-6 space-y-6">
          <ExpandablePanel summary={copy.howItWorks.sections[0].heading} defaultOpen>
            <p className="text-text-primary/80">{copy.howItWorks.sections[0].body as string}</p>
          </ExpandablePanel>

          <ExpandablePanel summary={copy.howItWorks.sections[1].heading}>
            <p className="text-text-primary/80">{copy.howItWorks.sections[1].body as string}</p>
          </ExpandablePanel>

          <ExpandablePanel summary={copy.howItWorks.sections[2].heading} defaultOpen>
            <p className="mb-4 text-text-primary/80">{copy.howItWorks.sections[2].body as string}</p>
            {modelCard && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-black/10">
                      <th className="pb-2 font-medium text-text-primary">
                        {copy.howItWorks.metricsTable.headers[0]}
                      </th>
                      <th className="pb-2 font-medium text-text-primary">
                        {copy.howItWorks.metricsTable.headers[1]}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {copy.howItWorks.metricsTable.rows.map((row: { metric: string; key: string }, i: number) => (
                      <tr key={i} className="border-b border-black/5">
                        <td className="py-2 text-text-primary/80">{row.metric}</td>
                        <td className="py-2 font-mono text-text-primary">
                          {modelCard.metrics[row.key] !== undefined
                            ? formatMetric(row.key, modelCard.metrics[row.key])
                            : copy.howItWorks.notEvaluated}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </ExpandablePanel>

          <ExpandablePanel summary={copy.howItWorks.sections[3].heading}>
            <ul className="list-inside list-disc space-y-2 text-text-primary/80">
              {(copy.howItWorks.sections[3].body as readonly string[]).map((item: string, i: number) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </ExpandablePanel>

          <ExpandablePanel summary={copy.howItWorks.sections[4].heading}>
            <p className="text-text-primary/80">{copy.howItWorks.sections[4].body as string}</p>
          </ExpandablePanel>

          <ExpandablePanel summary={copy.howItWorks.sections[5].heading}>
            <p className="text-text-primary/80">{copy.howItWorks.sections[5].body as string}</p>
          </ExpandablePanel>
        </div>
      </div>
    </MobileShell>
  );
}
