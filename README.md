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

## Deteksi batuk YOHO

Server tidak lagi memotong batas batuk dari ambang energi saja. Setiap rekaman
melewati `ml/src/yoho/`:

1. **Nucleus.** Frame RMS yang cukup keras dan cukup lebar dicari dulu
   (`find_nuclei`). Batuk selalu punya satu ledakan broadband, jadi kandidat ini
   yang menentukan berapa jendela yang disaring.
2. **Regresi batas.** Untuk tiap nucleus dipotong konteks 8 detik dan
   dikirim ke MobileNet kecil. Kepala jaringan memberi tiga bidang per kelas
   pada 40 langkah 0.2 detik: ada atau tidaknya event, pergeseran awal, dan
   pergeseran akhir. Batas event dibaca dari regresi ini, bukan dari ambang
   energi.
3. **Vote.** Tiap langkah yang lolos ambang presence memberi satu suara
   `(awal, akhir)` dengan skor presence kali porsi langkah yang tertutup. Suara
   dari jendela yang tumpang tindih digabung, lalu event hanya dihitung kalau ada
   nucleus dalam toleransi 0.3 detik dari batasnya. Nucleus yang belum terjawab
   tetap jadi segmen sendiri (nucleus + 0.05 detik sampai + 0.25 detik), karena
   batuk yang terlewat jaringan tetap batuk. Event di atas 2 detik dibuang
   karena itu gabungan beberapa batuk. Jendela dengan beberapa batuk otomatis
   jadi beberapa segmen.

Referensi: Venkatesh, Moffat & Miranda, *YOHO: You Only Hear Once*, 2022
(arXiv 2109.00962). Implementasi di repo ini bukan salinan kode mereka:
tugasnya satu kelas (batuk), jendela dan resolusi langkahnya dipilih sendiri,
dan penilaian loss-nya ditulis ulang.

### Batas yang perlu diketahui

- **Label lemah.** Batas batuk pada data latih berasal dari trim envelope dan
  segmen batuk yang dicatat per partisipan, bukan anotasi sampel per sampel.
  Skor batas karena itu tetap perkiraan.
- **Hanya batuk yang dilatih.** Kelas negatif (diam, suara keras non-batuk)
  hanya dipakai sebagai penekan bias dengan porsi kecil, tidak sebagai kelas
  kedua.
- **Training di CPU** dengan 4 thread, batch 32. Epoch penuh makan waktu belasan
  menit.
- **Audio latih** dari CODA-TB dan ESC-50. ESC-50 berlisensi CC BY-NC 3.0, jadi
  hasil latihnya tidak untuk penggunaan komersial.
- **Counter live di web** masih memakai energi dan spectral gate, jadi angkanya
  bisa berbeda sedikit dari hitungan akhir server.
- **Nucleus dibatasi 6 per rekaman** (`max_nuclei`). Rekaman yang punya lebih
  dari enam suara keras memproses enam kandidat terkuat saja.
- **Klasifikasi batuk vs suara lain belum dilatih.** Jaringan hanya dilatih
  untuk menempatkan batas batuk. Suara di telinga yang mirip batuk, misalnya
  bersin, bisa kehilangan batasnya dan tertahan sebagian oleh nucleus; tidak
  ada kelas kedua yang memfilter suara seperti itu.
- Angka evaluasi dilaporkan apa adanya dari `ml/scripts/evaluate_yoho.py` dan
  log training. Jangan menaikkan klaim sebelum ada evaluasi di audio lapangan.

### Hasil pada fold 0

Dua sumber, keduanya fold 0 dan tidak saling menggantikan:

- **Streaming** (480 window validasi berisi batuk, plus batuk yang ditempel ke
  rekaman pasif): jalur deploy yang sama, batas dihitung dari suara batuk,
  window dinilai per segmen.
- **Evaluasi resmi** (`evaluate_yoho.py`): 200 clip CODA-TB yang tidak dilatih
  dan suara negatif. Recall di sini berarti "ada satu segmen yang mengenai
  batuk", bukan penempatan batas per batuk.

| Uji | Hasil | Sumber |
| --- | --- | --- |
| Streaming, semua window | P 0.803 / R 0.957 / F1 0.873 | `streaming_predictions` |
| Streaming, batuk tersusun | P 0.760 / R 0.991 / F1 0.860 | `streaming-compose` |
| Streaming, batuk ditempel | P 0.900 / R 0.900 / F1 0.900 | `streaming-pastefree` |
| CODA-TB holdout, 200 clip | R 0.990 / P 0.990, 2 false alarm (4.5 per jam) | `eval_fold0.coda_holdout` |
| Baseline energi, 200 clip yang sama | R 0.850, 24 false alarm | `eval_fold0.energy_baseline_coda` |
| Baseline energi, 720 suara negatif | 3855 event | `eval_fold0.energy_baseline_negatives` |
| ESC-50 hard negative, 320 clip | 331 event, 465.5 per jam | `eval_fold0.esc50_hard` |
| ESC-50 bersin, 40 clip | 58 event | `eval_fold0.esc50_sneeze` |
| ESC-50 batuk, 40 clip | 0 segmen yang mengenai batuk | `eval_fold0.esc50_cough` |

Baris ESC-50 jangan dibaca sebagai keunggulan. YOHO lebih sedikit memotong
suara non-batuk daripada baseline energi, tapi event per jam di hard negative
masih jauh dari nol: hampir setiap clip punya satu segmen di sekitar suara
utamanya. Baris batuk ESC-50 juga kegagalan, karena satu crop 5 detik yang
memuat batuk tidak menghasilkan segmen yang mengenai batuk. Keduanya bukti
bahwa penempatan batas belum sama dengan klasifikasi batuk.

### Menjalankan

```bash
python3 ml/scripts/train_yoho.py --fold 0 --epochs 10 --steps-per-epoch 150 --batch-size 32
python3 ml/scripts/export_yoho.py --checkpoint ml/artifacts/yoho/yoho_fold0.pt --out ml/artifacts/yoho
python3 ml/scripts/evaluate_yoho.py --checkpoint ml/artifacts/yoho/yoho_fold0.pt
```

Ekspor menulis `yoho.onnx` (bobot di file `.data` di sebelahnya) dan `yoho.json`
yang mencatat bentuk input/output, stride waktu, selisih torch-ONNX, dan
konstanta segmentasi yang dipakai setelah jaringan. Kalau `yoho.onnx` tidak ada atau
gagal dimuat, server kembali ke pemotongan energi lama, jadi deployment tanpa
artefak ini tetap bisa menerima rekaman.

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
