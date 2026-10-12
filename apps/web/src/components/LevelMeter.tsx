export interface LevelMeterProps {
  level: "quiet" | "moderate" | "loud";
  label?: string;
}

export function LevelMeter({ level, label }: LevelMeterProps) {
  const labels = {
    quiet: "Tenang",
    moderate: "Agak bising",
    loud: "Terlalu bising",
  };

  const colors = {
    quiet: "bg-blue-500",
    moderate: "bg-yellow-500",
    loud: "bg-red-500",
  };

  const displayLabel = label ?? labels[level];

  return (
    <div className="flex items-center gap-3" role="status" aria-live="polite">
      <div className="flex gap-1" aria-hidden="true">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className={`w-8 h-3 rounded ${i <= (level === "quiet" ? 1 : level === "moderate" ? 2 : 3) ? colors[level] : "bg-gray-200"}`}
          />
        ))}
      </div>
      <span className="text-sm font-medium text-gray-700 min-w-[100px]">{displayLabel}</span>
    </div>
  );
}