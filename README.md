<div align="center">

<img src="assets/tubra-logo.png" alt="Logo TUBRA" width="320">

# TUBRA

</div>

Aplikasi web skrining batuk untuk pemeriksaan dahak TBC. Pengguna menjawab gejala, merekam 5 batuk di browser, lalu mendapat tier saran (A, B, C) tentang kapan periksa dahak.

**Bukan alat diagnosis.** Tidak pernah menyatakan pengguna sehat. Hanya dokter/petugas kesehatan yang menentukan diagnosis TBC.

## Fitur

- **Alur lengkap**: dari splash, beranda, persetujuan, 9 pertanyaan skrining, rekam 5 batuk, cek kualitas, analisis, hasil tier, alasan, langkah berikutnya, cara kerja, sampai privasi
- **Privasi**: Tidak ada data audio/jawaban disimpan di server. Audio diproses sekali lalu dibuang. Tidak ada cookie, localStorage, analytics pihak ketiga.
- **Model card driven**: Semua metrik di UI dibaca dari `model_card.json`. Nilai `null` tampil "Belum dievaluasi".
- **Mode Demo**: `?demo=1` + `DEMO_ENABLED=true` menampilkan panel teknis untuk juri.

## Stack

- **Frontend**: React + TypeScript + Vite + Tailwind
- **Backend**: Flask + gunicorn + ONNX Runtime + NumPy/SciPy
- **ML**: PyTorch (offline, hanya di `ml/`)

## Quick Start (Lokal)

```bash
# Prasyarat: Docker + Docker Compose
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

Web memanggil API lewat path relatif `/api`, diteruskan proxy Vite ke `http://localhost:8080`. Untuk mengarahkan dev ke API yang sudah jalan di host lain, set `VITE_PROXY_TARGET`.

## Verifikasi

```bash
# Web (dari apps/web)
cd apps/web
npm run lint && npm run typecheck && npm run test && npm run build

# API + ML (dari root repo, satu perintah)
python3 -m pytest -q

# Artefak model (dari root repo)
python3 ml/scripts/verify_artifacts.py
```

`pytest` dijalankan dari root (lihat `pytest.ini`) dan mencakup `ml/tests` serta
`apps/api/tbcare_api/tests`. Runtime API tidak boleh mengimpor `torch`, `librosa`,
atau `numba`; dependensi itu hanya dipakai untuk pelatihan di `ml/`.

## Artefak dan Hasil Model

`ml/artifacts/` dihasilkan oleh `ml/scripts/export_artifacts.py`. Yang di-commit:
`model_manifest.json`, `cv_results.json`, `subgroup_metrics.json`, `model_card.json`,
`head.npz`, dan `backbone.onnx.data`. Bobot penuh (`backbone.onnx`), model klinis
(`*.joblib`), dan prediksi (`oof_predictions.npz`) tidak di-commit.

| Berkas | Isi |
| --- | --- |
| `backbone.onnx` (+ `.data`) | Residual-spectrogram CNN audio branch, output 2 logits `(B, 1, 64, 64)` |
| `head.npz` | Kepala klasifikasi (identitas 2x2 untuk branch ini) |
| `clinical_model.joblib` | Logistic regression terkalibrasi atas 26 fitur klinis |
| `supervisor_model.joblib` | Fusion terkalibrasi atas `[p_audio, p_clinical, 26 fitur]` |
| `clinical_preprocessor.joblib` | Imputer + scaler (paritas latih/serving) |
| `model_manifest.json` | Metadata + status; `CANDIDATE` memblokir inference produksi |
| `model_card.json` | ROC-AUC, pAUC@90, sens/spec @ Tier A, ECE, threshold B/A |
| `cv_results.json` | Metrik out-of-fold lengkap + CI |
| `oof_predictions.npz` | Prediksi out-of-fold (labels, audio, klinis, fusion, groups) |

### Hasil latih (CODA-TB + TBscreen/Nairobi, 1163 partisipan, 5-fold out-of-fold, CPU)

> Angka di bawah dibaca langsung dari `ml/artifacts/cv_results.json` dan
> `ml/artifacts/model_card.json`. Jangan mengedit manual; jalankan ulang
> `ml/scripts/export_artifacts.py`.

| Metrik | Nilai | Sumber |
| --- | --- | --- |
| AUROC fusion (out-of-fold) | **0.8345** (CI 95%: 0.8113 sampai 0.8580) | `cv_results.fusion` |
| AUROC audio-only | 0.6675 | `cv_results.audio_only` |
| AUROC clinical-only | 0.8235 | `cv_results.clinical_only` |
| pAUC@90% Sens | 0.33 | `model_card.metrics.pauc_90` |
| Sensitivitas @ Tier A | 0.4865 | `model_card.metrics.sens_at_A` |
| Spesifisitas @ Tier A | 0.9004 | `model_card.metrics.spec_at_A` |
| ECE | 0.0352 | `model_card.metrics.ece` |
| Threshold Tier A / B | 0.5552 / 0.1937 | `model_card.metrics.threshold_A/B` |
| Paritas torch dan ONNX | max\|selisih logit\| = 1.8e-07 | `ml/tests/test_tta_parity.py` |

> **Catatan kejujuran.** AUROC fusion (0.8345) hampir seluruhnya berasal dari
> cabang klinis (0.8235); cabang audio hanya menyumbang ~0.011 AUROC. Ini hasil
> validasi silang pada 4 epoch CPU, **bukan** uji klinis. Jangan menaikkan status
> model menjadi `VALIDATED` tanpa evaluasi eksternal prospektif.

Regenerasi (butuh data CODA-TB + TBscreen, CPU-only):

```bash
PYTHONPATH=.:apps/api python3 ml/scripts/export_artifacts.py \
  --config ml/configs/train.yaml \
  --coda-root ml/data/coda_tb \
  --tbscreen-root /path/to/TBscreen_Dataset \
  --output ml/artifacts --epochs 4 --max-clips 5 --outer-folds 5
```

Verifikasi artefak: `python3 ml/scripts/verify_artifacts.py`.

**Status model.** Artefak diekspor sebagai `CANDIDATE`, sehingga inference produksi
tertutup secara default. Untuk uji staging:

```bash
DEPLOYMENT_ENV=staging ALLOW_BLOCKED_CANDIDATE=true python3 -m tbcare_api.app
```

Jangan set `VALIDATED` sampai evaluasi eksternal (kohort prospektif) tersedia:
metrik pada `model_card.json` berasal dari validasi silang out-of-fold, bukan uji klinis.

### Paritas log-Mel training dan serving

`ml/tests/test_logmel_parity.py` membandingkan pipeline serving
(`apps/api/tbcare_api/audio/logmel.py`) dengan training
(`ml/src/features/audio.py::extract_logmel`, recipe `energy_v2`) pada sinyal
sintetis yang sama. Keduanya memakai window energi, normalisasi `[0, 1]`, dan
resampler yang sama, sehingga `max|diff|` selalu di bawah toleransi `1e-4`.
Test ini wajib lulus sebelum artefak apa pun diekspor.

## Reproduksi

```bash
python3 -m pytest -q                      # test API dan ML
python3 ml/scripts/verify_artifacts.py    # kontrak artefak
cd apps/web && npm run build              # build frontend
```

Metrik tidak dihitung ulang saat build. UI membaca angka dari
`apps/web/public/model_card.json`, jadi pastikan file itu hasil ekspor terakhir.

## Lisensi

MIT. Kode sumber terbuka. Jalankan sendiri secara lokal.
