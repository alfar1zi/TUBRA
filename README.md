# TBCare

Aplikasi web skrining batuk untuk pemeriksaan dahak TBC. Pengguna menjawab gejala, merekam 5 batuk di browser, lalu mendapat tier saran (A, B, C) tentang kapan periksa dahak.

**Bukan alat diagnosis.** Tidak pernah menyatakan pengguna sehat. Hanya dokter/petugas kesehatan yang menentukan diagnosis TBC.

## Fitur

- **S0–S12**: Alur lengkap dari beranda → persetujuan → gerbang → gejala (7 pertanyaan) → persiapan → rekam 5 batuk → cek kualitas → analisis → hasil → mengapa → langkah berikutnya → cara kerja & bukti → privasi
- **Privasi**: Tidak ada data audio/jawaban disimpan di server. Audio diproses sekali lalu dibuang. Tidak ada cookie, localStorage, analytics pihak ketiga.
- **Model card driven**: Semua metrik di UI dibaca dari `model_card.json`. Nilai `null` tampil "Belum dievaluasi".
- **Mode Demo**: `?demo=1` + `DEMO_ENABLED=true` menampilkan panel teknis untuk juri.

## Stack

- **Frontend**: React + TypeScript + Vite + Tailwind → static di Vercel
- **Backend**: Flask + gunicorn + ONNX Runtime + NumPy/SciPy → Cloud Run / lokal
- **ML**: PyTorch (offline, hanya di `ml/`)

## Quick Start (Lokal)

```bash
# Prasyarat: Docker + Docker Compose
cd tbcare
./scripts/run_local.sh up

# Web:  http://localhost:5173
# API:  http://localhost:8080
# Demo: http://localhost:5173?demo=1

./scripts/run_local.sh down  # berhenti
```

## Development (Tanpa Docker)

```bash
# API
cd apps/api
pip install -r requirements.txt
python -m tbcare_api.app

# Web (terminal terpisah)
cd apps/web
npm install
npm run dev
```

## Verifikasi

```bash
# Web
cd apps/web
npm run lint && npm run typecheck && npm run build

# API
cd apps/api
python -m pytest tbcare_api/tests -v

# ML
cd ml
pytest -q && python scripts/check_parity.py
```

## Deploy Cloud Run (API)

```bash
# Butuh: gcloud auth, billing aktif, Artifact Registry repo
./scripts/deploy_cloudrun.sh <PROJECT_ID> [REGION] [SERVICE_NAME]
```

Web deploy ke Vercel: set `VITE_API_BASE=<CLOUD_RUN_URL>`.

## Struktur Repo

```
tbcare/
├── apps/
│   ├── web/          # React + Vite + Tailwind
│   └── api/          # Flask + ONNX Runtime
├── ml/               # PyTorch training (offline)
│   ├── artifacts/    # backbone.onnx, head.npz, model_card.json (gitignored)
│   └── scripts/      # prepare_data, train, evaluate, export_onnx, check_parity
├── scripts/          # run_local.sh, deploy_cloudrun.sh, measure_latency.sh
└── docs/             # PRD, ARCHITECTURE, UI_SPEC, METHOD
```

## Artefak Model

`ml/artifacts/` (tidak di-commit):
- `backbone.onnx` — embedding + feature map (ResNet18 log-Mel)
- `head.npz` — Dense head weights + calibration temperature
- `model_card.json` — metrik evaluasi, threshold, dataset info

Ganti artefak saat data CODA-TB tersedia (Synapse approval + Kaggle weights).

## Lisensi

MIT — Kode sumber terbuka. Jalankan sendiri secara lokal.