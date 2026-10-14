# Dashboard Streamlit K-Means Tasikmalaya

Dashboard ini mengikuti alur penelitian K-Means 2023–2026 pada notebook KP.

## Struktur folder

Letakkan file seperti ini:

```text
streamlit_kmeans_tasikmalaya/
│
├── app.py
├── requirements.txt
│
├── assets/
│   └── logo_kota_tasikmalaya.png
│
├── dataset/
│   ├── kependudukan_2023.csv
│   ├── kependudukan_2024.csv
│   ├── kependudukan_2025.csv
│   └── kependudukan_2026.csv
│
└── peta/
    ├── ADMINISTRASI_AR_KECAMATAN.shp
    ├── ADMINISTRASI_AR_KECAMATAN.dbf
    ├── ADMINISTRASI_AR_KECAMATAN.shx
    ├── ADMINISTRASI_AR_KECAMATAN.prj
    └── ADMINISTRASI_AR_KECAMATAN.cpg
```

## Menjalankan

Buka Terminal / Command Prompt pada folder project:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sidebar

1. Dashboard
2. Upload CSV
3. Evaluasi K
4. Hasil Clustering
5. Peta Cluster
6. Perbandingan Tahun

## Metode

- X1 = Jumlah Penduduk
- X2 = Laju Pertumbuhan Penduduk
- X3 = Kepadatan Penduduk
- X4 = Rasio Jenis Kelamin
- Persentase Penduduk hanya ditampilkan sebagai informasi deskriptif.
- Elbow / WCSS menggunakan K = 1–8.
- Silhouette dan Davies-Bouldin menggunakan K = 2–8.
- Analisis dilakukan terpisah untuk setiap tahun 2023–2026.
