import { Button, ExpandablePanel } from "@/components";
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
    if (value === null) return "Belum dievaluasi";
    if (key.includes("threshold")) return value.toFixed(2);
    if (key.includes("ece")) return value.toFixed(4);
    return value.toFixed(3);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-semibold text-gray-900">{copy.howItWorks.title}</h1>
            <Button variant="ghost" size="sm" onClick={onBack}>
              {copy.common.back}
            </Button>
          </div>

          <div className="space-y-6">
            <ExpandablePanel summary={copy.howItWorks.sections[0].heading} defaultOpen>
              <p className="text-gray-700">{copy.howItWorks.sections[0].body as string}</p>
            </ExpandablePanel>

            <ExpandablePanel summary={copy.howItWorks.sections[1].heading}>
              <p className="text-gray-700">{copy.howItWorks.sections[1].body as string}</p>
            </ExpandablePanel>

            <ExpandablePanel summary={copy.howItWorks.sections[2].heading} defaultOpen>
              <p className="text-gray-700 mb-4">{copy.howItWorks.sections[2].body as string}</p>
              {modelCard && (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="pb-2 font-medium text-gray-900">{copy.howItWorks.metricsTable.headers[0]}</th>
                        <th className="pb-2 font-medium text-gray-900">{copy.howItWorks.metricsTable.headers[1]}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {copy.howItWorks.metricsTable.rows.map((row: { metric: string; key: string }, i: number) => (
                        <tr key={i} className="border-b border-gray-100">
                          <td className="py-2 text-gray-700">{row.metric}</td>
                          <td className="py-2 font-mono text-gray-900">
                            {modelCard.metrics[row.key] !== undefined
                              ? formatMetric(row.key, modelCard.metrics[row.key])
                              : "Belum dievaluasi"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </ExpandablePanel>

            <ExpandablePanel summary={copy.howItWorks.sections[3].heading}>
              <ul className="space-y-2 text-gray-700 list-disc list-inside">
                {(copy.howItWorks.sections[3].body as readonly string[]).map((item: string, i: number) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </ExpandablePanel>

            <ExpandablePanel summary={copy.howItWorks.sections[4].heading}>
              <p className="text-gray-700">{copy.howItWorks.sections[4].body as string}</p>
            </ExpandablePanel>

            <ExpandablePanel summary={copy.howItWorks.sections[5].heading}>
              <p className="text-gray-700">{copy.howItWorks.sections[5].body as string}</p>
            </ExpandablePanel>
          </div>
        </div>
      </div>
    </div>
  );
}