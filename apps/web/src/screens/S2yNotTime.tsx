import { Button } from "@/components";

interface S2yNotTimeProps {
  onBackToStart: () => void;
  onFindClinic: () => void;
}

export function S2yNotTime({ onBackToStart, onFindClinic }: S2yNotTimeProps) {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 text-center">
          <h1 className="text-2xl font-semibold text-gray-900 mb-4">Alat ini untuk batuk 2 minggu atau lebih.</h1>
          <p className="text-gray-600 mb-8">
            Jika batuk Anda belum 2 minggu, perhatikan dulu. Periksa jika batuk berlanjut sampai 2 minggu.
            Periksa lebih cepat jika batuk berdarah, sesak berat, atau demam tinggi.
          </p>
          <div className="space-y-3">
            <Button size="lg" fullWidth onClick={onBackToStart}>
              Kembali ke awal
            </Button>
            <Button variant="secondary" size="md" fullWidth onClick={onFindClinic}>
              Cari puskesmas terdekat
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}