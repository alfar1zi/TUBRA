import { useState } from "react";
import { Button, ExpandablePanel, Banner } from "@/components";
import { copy } from "@/copy";

interface S4PrepRecordProps {
  onContinue: () => void;
  onBack: () => void;
  onMicError: (error: "denied" | "notfound" | "notsupported") => void;
}

export function S4PrepRecord({ onContinue, onBack, onMicError }: S4PrepRecordProps) {
  const [micRequested, setMicRequested] = useState(false);
  const [micError, setMicError] = useState<"denied" | "notfound" | "notsupported" | null>(null);

  const handleMicPermission = async () => {
    setMicRequested(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((t) => t.stop());
      onContinue();
    } catch (err) {
      const error = err as Error & { name?: string };
      if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
        setMicError("denied");
        onMicError("denied");
      } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
        setMicError("notfound");
        onMicError("notfound");
      } else if (error.name === "NotSupportedError") {
        setMicError("notsupported");
        onMicError("notsupported");
      } else {
        setMicError("denied");
        onMicError("denied");
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">{copy.prepRecord.title}</h1>

          <div className="space-y-6 mb-8">
            {copy.prepRecord.body.map((text, i) => (
              <div key={i} className="flex items-start gap-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                  <span className="text-blue-600 font-bold text-sm">{i + 1}</span>
                </div>
                <p className="text-gray-700 mt-0.5">{text}</p>
              </div>
            ))}
          </div>

          <ExpandablePanel summary={copy.prepRecord.warning}>
            <p className="text-sm text-gray-600">
              Mikrofon hanya aktif saat Anda menekan tombol rekam. Suara Anda tidak disimpan di server.
              Tidak ada identitas yang dikumpulkan.
            </p>
          </ExpandablePanel>

          {micError && (
            <Banner
              variant="urgent"
              title={micError === "denied" ? copy.errors.E1 : micError === "notfound" ? copy.errors.E2 : copy.errors.E3}
            >
              <p>Silakan coba lagi atau periksa pengaturan browser.</p>
            </Banner>
          )}

          <div className="mt-8 space-y-3">
            <Button size="lg" fullWidth onClick={handleMicPermission} disabled={micRequested}>
              {micRequested ? "Meminta izin..." : copy.prepRecord.startButton}
            </Button>
            <Button variant="secondary" size="md" fullWidth onClick={onBack}>
              {copy.common.back}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}