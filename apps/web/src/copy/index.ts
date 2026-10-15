/**
 * All user-facing text lives here. The forbidden-word test enforces that UI
 * components reference `copy` instead of hardcoding strings.
 *
 * Content is authored to match the Figma design "Joints UGM 2026 UI Section"
 * (product name: TUBRA). Where the design and the clinical constraints diverge,
 * the clinical wording wins (the app must never claim a diagnosis).
 */
export const copy = {
  /** Product name shown in the UI and metadata. */
  brand: "TUBRA",
  home: {
    title: "Batuk lebih dari 2 minggu? Jangan ditunda",
    subtitle:
      "Rekam suara batukmu dan dapatkan saran langkah selanjutnya untuk skrining TBC.",
    startButton: "Mulai cek batuk",
  },
  splash: {
    title: "TUBRA",
    subtitle: "Listen and Care earlier.",
  },
  consent: {
    title: "Hai! 👋 Sebelum mulai,",
    subtitle: "Yuk, kenali dulu cara TUBRA membantu Anda memahami kondisi pernapasan.",
    body: [
      "Kamu akan menjawab beberapa pertanyaan tentang gejala yang kamu rasakan.",
      "Kamu akan diminta merekam batuk sebanyak 5 kali.",
      "Setelah selesai, kamu bisa melihat hasil skrining awal dari TUBRA. Hasil ini bukan diagnosis dan belum bisa memastikan apakah kamu terkena TBC atau tidak.",
      "Rekaman dan jawabanmu tidak disimpan di server, dan kamu tidak perlu memasukkan data diri.",
    ],
    agreeButton: "Iya, aku menyetujuinya",
    disagreeButton: "Aku tidak setuju",
  },
  /**
   * The question flow is 9 single-question screens, matching the Figma design.
   * Order and wording are taken verbatim from the design.
   */
  questions: {
    progressLabel: "Tahap Pertanyaan",
    progressTemplate: "{n} dari 9 Pertanyaan",
    backButton: "Kembali",
    nextButton: "Lanjut",
    consentNote:
      "Peruntukan screening untuk orang dewasa yang sudah batuk selama 2 minggu atau lebih",
    items: {
      age: {
        question: "Berapa umur kamu?",
        unit: "tahun",
        min: 18,
        max: 100,
        error: "Usia minimal 18 tahun, maksimal 100 tahun.",
      },
      coughDuration: {
        question: "Sudah berapa lama kamu batuk?",
        options: {
          lt2w: "Kurang dari 2 minggu",
          w2to4: "2–4 minggu",
          m1to2: "1–2 bulan",
          gt2m: "Lebih dari 2 bulan",
        },
      },
      sex: {
        question: "Apa jenis kelamin kamu?",
        options: { male: "Laki-laki", female: "Perempuan" },
      },
      hemoptysis: {
        question: "Apakah ada darah saat kamu batuk?",
        options: { yes: "Iya, berdarah", no: "Tidak, tidak ada darah" },
      },
      fever: {
        question:
          "Dalam sebulan terakhir, apakah kamu mengalami demam yang datang dan pergi?",
        options: { yes: "Iya, sempat mengalami", no: "Tidak sama sekali" },
      },
      nightSweats: {
        question:
          "Apakah kamu sering berkeringat banyak saat tidur malam, sampai pakaian atau seprai basah?",
        options: {
          yes: "Iya, sangat berkeringat",
          sometimes: "Hanya berkeringat karena panas",
          no: "Tidak sama sekali",
        },
      },
      weightLoss: {
        question:
          "Apakah berat badan kamu turun tanpa alasan yang jelas dalam sebulan terakhir?",
        options: { yes: "Iya, sempat turun", no: "Tidak turun sama sekali" },
      },
      priorTb: {
        question:
          "Apakah kamu pernah didiagnosis TBC atau menjalani pengobatan TBC sebelumnya?",
        options: { yes: "Iya, pernah", no: "Tidak" },
      },
      smoking: {
        question: "Apakah kamu merokok?",
        options: {
          never: "Tidak pernah",
          quit: "Pernah, tapi sudah berhenti",
          current: "Iya, saya merokok",
        },
      },
    },
  },
  prepRecord: {
    title: "Tahap Rekam Batuk",
    subtitle: "Sekarang, Rekam Batukmu",
    body: "Batuk sebanyak 5 kali dengan klik tombol rekam. Suara batuk harus terdengar jelas.",
    warning: "Pastikan izin mikrofon diizinkan di browser.",
    startButton: "Mulai Rekam 5 Batuk",
  },
  outOfRange: {
    title: "Alat ini dirancang untuk orang dewasa.",
    body: "Untuk anak atau remaja yang batuk lebih dari 2 minggu, ajak ke puskesmas untuk diperiksa.",
    findClinicButton: "Cari puskesmas terdekat",
    backToStartButton: "Kembali ke awal",
    /** Shown when the user has coughed for less than two weeks. */
    shortCough: {
      title: "Batuk baru beberapa hari.",
      body:
        "Pemeriksaan ini untuk batuk yang sudah 2 minggu atau lebih. Kalau batukmu belum sembuh dalam 2 minggu, batuk berdarah, atau bertambah berat, periksakan ke puskesmas terdekat.",
    },
  },
  record: {
    title: "Tahap Rekam Batuk",
    subtitle: "Sekarang, Rekam Batukmu",
    hint: "Batuk sebanyak 5 kali dengan klik tombol rekam. Suara batuk harus terdengar jelas.",
    recording: "Sedang merekam {n} dari 5 batuk",
    stopButton: "Hentikan & Kirim",
    qualityHint: "Batuk terlalu pelan atau terlalu dekat akan ditolak.",
    calibratingTitle: "Kalibrasi Mikrofon",
    calibratingSubtitle: "Diamkan sebentar, lalu coba batuk 1-2 kali untuk latihan.",
    calibratingProgress: "Mempelajari suara lingkungan...",
    calibratingCoughHint: "Coba batuk sekarang",
    calibratingSlow: "Kalau persentase tidak bergerak, cek izin mikrofon lalu rekam ulang.",
    retryButton: "Coba nyalakan mikrofon lagi",
    errorMicPermission: "Izin mikrofon ditolak. Buka pengaturan browser.",
    errorMicNotFound: "Mikrofon tidak ditemukan.",
    errorAudioContext: "Gagal memulai audio. Coba refresh halaman.",
    /** Maps the stable codes from useRecorder.toRecorderErrorCode to UI text. */
    recorderErrors: {
      mic_permission: "Izin mikrofon ditolak. Buka pengaturan browser.",
      mic_not_found: "Mikrofon tidak ditemukan.",
      audio_context: "Gagal memulai audio. Coba refresh halaman.",
      no_audio:
        "Mikrofon tidak mengirim suara. Pastikan mikrofon tidak dipakai aplikasi lain, lalu rekam ulang.",
    },
    errorDuration: "Rekaman terlalu pendek. Minimal 3 detik.",
    errorNoCough: "Tidak ada batuk terdeteksi. Coba lagi.",
    calibratingButton: "Kalibrasi...",
  },
  recordConfirm: {
    title: "Batuk anda sudah terekam!",
    subtitle: "Kami bisa mendengar anda dengan baik. Ingin melihat hasilnya?",
    rerecordButton: "Rekam ulang",
    seeResultButton: "Lihat hasil",
  },
  qualityCheck: {
    title: "Memeriksa Kualitas Rekaman",
    checking: "Menganalisis kualitas audio...",
    acceptedLabel: "Batuk diterima:",
    rejectedLabel: "Batuk ditolak:",
    minCoughs: "Minimal 3 batuk jelas diperlukan.",
    errorLowQuality:
      "Batuk kamu belum terdengar jelas. Batuk lebih keras dan dekatkan sedikit ke mikrofon.",
    errorClipping: "Suara batuk terlalu keras sampai terpotong. Jauhkan mikrofon sedikit.",
    errorNoise: "Suara di sekitar terlalu berisik. Cari tempat lebih tenang, lalu coba lagi.",
    failTitle: "Rekaman belum bisa diproses",
    reasonLabels: {
      too_few_coughs: "Batuk belum terdengar jelas oleh sistem.",
      too_noisy: "Suara di sekitar terlalu berisik.",
      clipped: "Suara batuk terlalu keras sampai terpotong.",
    },
    rerecordButton: "Rekam Ulang",
    continueButton: "Lanjut Analisis",
  },
  analyze: {
    title: "Tunggu sebentar, ya.",
    subtitle: "Kami sedang memproses hasil dari tes batuk kamu untuk mendapatkan data.",
    processing: "Mohon tunggu, ini hanya beberapa detik.",
    errorModelUnavailable: "Model tidak tersedia. Coba lagi nanti.",
    errorInference: "Gagal menganalisis. Coba rekam ulang.",
    errorNetwork: "Gagal terhubung ke server. Periksa koneksi.",
    errorValidation: "Data tidak valid. Coba mulai dari awal.",
    steps: [
      "Memotong suara batuk",
      "Membaca pola suara",
      "Menggabungkan dengan gejala Anda",
    ],
    errorTitle: "Gagal Menganalisis",
    backToRecordButton: "Kembali ke Rekam",
    retryButton: "Coba lagi",
    /** Per-failure explanation. Only a broken take needs a re-record. */
    failureMessages: {
      network: "Koneksi ke server terputus. Pastikan internetmu jalan, lalu coba lagi.",
      E4: "File rekaman tidak bisa dibaca server. Coba rekam ulang.",
      E5: "Durasi rekaman terlalu pendek atau terlalu panjang. Coba rekam ulang.",
      E10: "Server sedang sibuk atau butuh waktu lebih lama. Coba lagi sebentar.",
      E11: "Ada jawaban yang tidak lengkap. Coba ulangi dari awal.",
      generic: "Server sedang gangguan. Coba lagi sebentar.",
    },
  },
  result: {
    title: "Hasil Tes Batuk",
    subtitle: "Lihat hasil tes batuk anda",
    findClinicButton: "Cek puskesmas terdekat",
    hemoptysisBanner: "Batuk berdarah perlu diperiksa secepatnya.",
    recheckButton: "Cek ulang",
    whyButton: "Kenapa hasilnya ini?",
    tierA: {
      label: "Tier A",
      text: "Segera periksa dahak!",
      description:
        "Berdasarkan pola batuk dan gejala anda, disarankan segera memeriksa dahak ke fasilitas kesehatan.",
    },
    tierB: {
      label: "Tier B",
      text: "Cek dahak minggu ini!",
      description:
        "Berdasarkan pola batuk anda, disarankan tes dahak minggu ini.",
    },
    tierC: {
      label: "Tier C",
      text: "Tetap periksa dahak",
      description:
        "Meski risiko relatif lebih rendah, pemeriksaan dahak tetap disarankan untuk memastikan.",
    },
    disclaimer: "Hasil ini merupakan skrining awal, bukan diagnosis medis.",
    whyIntro: "Alasan dari hasil batuk dan gejala",
    reasons: {
      cough: {
        title: "Batuk lebih dari 2 minggu",
        body: "Batuk yang berlangsung lama perlu untuk diperiksa lebih lanjut.",
      },
      fever: {
        title: "Demam",
        body: "Demam yang disertai batuk bisa menjadi tanda adanya infeksi.",
      },
      nightSweats: {
        title: "Berkeringat pada malam hari",
        body: "Keringat berlebihan saat tidur, meskipun udara tidak panas, merupakan gejala yang perlu diperhatikan.",
      },
    },
  },
  why: {
    title: "Kenapa hasilnya ini?",
    subtitle: "Alasan dari hasil batuk dan gejala.",
    symptomLabels: {
      fever: "Demam",
      night_sweats: "Keringat malam",
      weight_loss: "Berat badan turun",
      hemoptysis: "Batuk darah",
      chest_pain: "Nyeri dada",
      fatigue: "Lelah berlebihan",
      smoking: "Merokok",
    },
    symptomContributionTitle: "Kontribusi Gejala",
    audioContributionTitle: "Kontribusi Pola Batuk",
    gradcamTitle: "Area Paling Mempengaruhi Model",
    gradcamNote:
      "Area ini menunjukkan bagian spektrogram yang paling memengaruhi keluaran model, bukan tanda penyakit.",
    limitationsTitle: "Batasan",
    limitations: [
      "Model ini bukan diagnosis medis.",
      "Akurasi terbatas pada data pelatihan yang tersedia.",
      "Hasil dapat berbeda pada populasi yang tidak terwakili di data latih.",
    ],
    howItWorksLink: "Lihat cara kerja & bukti",
    backButton: "Kembali ke Hasil",
  },
  nextSteps: {
    title: "Langkah Berikutnya",
    checklistTitle: "Yang perlu disiapkan",
    checklist: [
      "Datang ke puskesmas atau fasilitas kesehatan yang menyediakan pemeriksaan dahak (TCM / mikroskopis).",
      "Bawa hasil skrining ini sebagai referensi (bisa screenshot).",
      "Petugas akan meminta Anda mengeluarkan dahak untuk diperiksa.",
      "Pemeriksaan dahak yang menentukan hasilnya. Hasil aplikasi ini bukan diagnosis.",
    ],
    clinicSentenceTitle: "Kalimat untuk petugas",
    copyLabel: "Salin",
    clinicSentence: "Saya batuk lebih dari 2 minggu dan ingin memeriksa dahak untuk TBC.",
    shareMessage: "Saya batuk lebih dari 2 minggu dan akan memeriksakan dahak ke puskesmas. Boleh temani saya?",
    copiedToClipboard: "Teks disalin ke clipboard",
    examInfo: "Pemeriksaan dahak (TCM) gratis di puskesmas dan rumah sakit pemerintah.",
    supportText: "Butuh bantuan mencari fasilitas? Hubungi Dinas Kesehatan setempat.",
    costInfo: "Pemeriksaan dahak dan pengobatan TBC gratis di fasilitas BPJS Kesehatan.",
  },
  howItWorks: {
    notEvaluated: "Belum dievaluasi",
    title: "Cara Kerja & Bukti",
    sections: [
      {
        heading: "Alur Singkat",
        body: "Isi gejala → Rekam 5 batuk → Model memproses audio + gejala → Tier saran (A/B/C).",
      },
      {
        heading: "Model",
        body: "Backbone: ResNet18 pada log-Mel spektrogram. Head: Dense + kalibrasi suhu. Dilatih pada data batuk CODA-TB dengan pemisahan per partisipan (GroupKFold).",
      },
      {
        heading: "Evaluasi (Model Card)",
        body: "Semua metrik dibaca dari model_card.json. Nilai null = Belum dievaluasi.",
      },
      {
        heading: "Batasan",
        body: [
          "Model ini bukan diagnosis medis.",
          "Akurasi terbatas pada data pelatihan yang tersedia.",
          "Hasil dapat berbeda pada populasi yang tidak terwakili di data latih.",
        ],
      },
      {
        heading: "Privasi",
        body: "Tidak ada data audio, jawaban gejala, atau identitas yang dikirim ke server untuk disimpan. Audio diproses sekali di server saat analisis, lalu dibuang.",
      },
      {
        heading: "Sumber",
        body: "Kode sumber: https://github.com/alfar1zi/TUBRA. Dataset CODA-TB: DREAM Challenge. Referensi model: model_card.json.",
      },
    ],
    metricsTable: {
      headers: ["Metrik", "Nilai"],
      rows: [
        { metric: "ROC-AUC", key: "roc_auc" },
        { metric: "pAUC@90% Sens", key: "pauc_90" },
        { metric: "Sensitivity @ Tier A", key: "sens_at_A" },
        { metric: "Specificity @ Tier A", key: "spec_at_A" },
        { metric: "ECE (Expected Calibration Error)", key: "ece" },
        { metric: "Threshold Tier A", key: "threshold_A" },
        { metric: "Threshold Tier B", key: "threshold_B" },
      ],
    },
    backButton: "Kembali",
  },
  privacy: {
    title: "Privasi",
    body: [
      "Tidak ada data audio, jawaban gejala, atau identitas yang dikirim ke server untuk disimpan.",
      "Audio diproses sekali di server saat analisis, lalu dibuang.",
      "Tidak ada cookie, localStorage, sessionStorage, atau analytics pihak ketiga.",
      "Log server tidak memuat payload, audio, atau IP mentah.",
      "Kode sumber terbuka. Anda bisa menjalankan sendiri secara lokal.",
    ],
    backButton: "Kembali",
  },
  errors: {
    E1: "Izin mikrofon ditolak. Aktifkan izin mikrofon di browser untuk melanjutkan.",
    E2: "Mikrofon tidak ditemukan. Pastikan perangkat mikrofon terhubung.",
    E3: "Mikrofon tidak didukung di browser ini.",
    E4: "Format audio tidak didukung. Gunakan WAV PCM 16-bit mono.",
    E5: "Durasi audio tidak valid. Maksimal 30 detik.",
    E6: "Kami cuma mendengar 3 batuk atau kurang. Batuk lebih keras, dekatkan sedikit ke mikrofon, lalu rekam ulang.",
    E7: "Suara di sekitar terlalu berisik. Cari tempat lebih tenang, lalu rekam ulang.",
    E8: "Suara batuk terlalu keras sampai terpotong. Jauhkan mikrofon sedikit, lalu rekam ulang.",
    E9: "Model tidak tersedia saat ini.",
    E10: "Gagal menganalisis. Coba rekam ulang.",
    E11: "Data permintaan tidak valid.",
    generic: "Terjadi kesalahan. Silakan coba lagi.",
    network: "Gagal terhubung ke server. Periksa koneksi internet.",
  },
  common: {
    loading: "Memproses...",
    back: "Kembali",
    close: "Tutup",
    retry: "Coba Lagi",
    cancel: "Batal",
    restartConfirm: "Mulai dari awal? Semua jawaban dan rekaman akan dihapus.",
  },
} as const;
