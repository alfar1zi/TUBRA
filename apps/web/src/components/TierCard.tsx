export interface TierCardProps {
  tier: "A" | "B" | "C";
  title: string;
  description: string;
  icon: React.ReactNode;
}

export function TierCard({ tier, title, description, icon }: TierCardProps) {
  const tierStyles = {
    A: "border-red-500 bg-red-50",
    B: "border-yellow-500 bg-yellow-50",
    C: "border-gray-500 bg-gray-50",
  };

  return (
    <div className={`p-5 rounded-xl border-2 ${tierStyles[tier]}`} role="status">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-white border border-gray-200 flex items-center justify-center text-gray-600" aria-hidden="true">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          <p className="mt-1 text-gray-700">{description}</p>
        </div>
        <div
          className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${
            tier === "A" ? "bg-red-600" : tier === "B" ? "bg-yellow-600" : "bg-gray-600"
          }`}
          aria-label={`Tier ${tier}`}
        >
          {tier}
        </div>
      </div>
    </div>
  );
}