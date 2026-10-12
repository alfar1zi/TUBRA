import { Button } from "@/components";

interface S2xOutOfRangeProps {
  onFindClinic: () => void;
  onBackToStart: () => void;
}

export function S2xOutOfRange({ onFindClinic, onBackToStart }: S2xOutOfRangeProps) {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 text-center">
          <h1 className="text-2xl font-semibold text-gray-900 mb-4">Alat ini dirancang untuk orang dewasa.</h1>
          <p className="text-gray-600 mb-8">
            Untuk anak atau remaja yang batuk lebih dari 2 minggu, ajak ke puskesmas untuk diperiksa.
          </p>
          <div className="space-y-3">
            <Button size="lg" fullWidth onClick={onFindClinic}>
              Cari puskesmas terdekat
            </Button>
            <Button variant="secondary" size="md" fullWidth onClick={onBackToStart}>
              Kembali ke awal
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}