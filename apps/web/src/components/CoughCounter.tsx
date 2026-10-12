export interface CoughCounterProps {
  count: number;
  target?: number;
}

export function CoughCounter({ count, target = 5 }: CoughCounterProps) {
  return (
    <div className="flex flex-col items-center gap-2" role="status" aria-live="polite">
      <div className="text-4xl font-bold text-gray-900" aria-label={`${count} dari ${target} batuk terdeteksi`}>
        {count} / {target}
      </div>
      <div className="flex gap-1.5" aria-hidden="true">
        {Array.from({ length: target }, (_, i) => (
          <div
            key={i}
            className={`w-2 h-2 rounded-full transition-colors ${
              i < count ? "bg-blue-600" : "bg-gray-300"
            }`}
          />
        ))}
      </div>
    </div>
  );
}