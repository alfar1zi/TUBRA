import { Button } from "@/components";
import { copy } from "@/copy";

interface S0HomeProps {
  onStart: () => void;
  onHowItWorks: () => void;
  onPrivacy: () => void;
}

export function S0Home({ onStart, onHowItWorks, onPrivacy }: S0HomeProps) {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold text-gray-900">{copy.home.title}</h1>
          <p className="mt-2 text-gray-600">{copy.home.subtitle}</p>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 text-center mb-4">
            Batuk lebih dari 2 minggu? Jangan ditunda.
          </h2>
          <p className="text-gray-600 text-center mb-6">
            Rekam batuk Anda dan dapatkan saran kapan sebaiknya periksa dahak.
          </p>

          <div className="space-y-4">
            <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
              <svg className="w-6 h-6 text-blue-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <p className="font-medium text-gray-900">Sekitar 3 menit</p>
                <p className="text-sm text-gray-600">Isi gejala dan rekam 5 batuk</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
              <svg className="w-6 h-6 text-blue-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              <div>
                <p className="font-medium text-gray-900">Tanpa akun, suara tidak disimpan</p>
                <p className="text-sm text-gray-600">Data diproses sekali lalu dibuang</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
              <svg className="w-6 h-6 text-blue-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <p className="font-medium text-gray-900">Bukan diagnosis, hanya saran</p>
                <p className="text-sm text-gray-600">Hanya pemeriksaan dahak yang memastikan</p>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <Button size="lg" fullWidth onClick={onStart}>
            {copy.home.startButton}
          </Button>
          <Button variant="ghost" size="md" fullWidth onClick={onHowItWorks}>
            {copy.home.howItWorksLink}
          </Button>
          <Button variant="ghost" size="md" fullWidth onClick={onPrivacy}>
            {copy.home.privacyLink}
          </Button>
        </div>
      </div>
    </div>
  );
}