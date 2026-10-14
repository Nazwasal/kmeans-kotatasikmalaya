# Dashboard Klasterisasi Kecamatan Kota Tasikmalaya Menggunakan K-Means

Dashboard berbasis **Streamlit** untuk menampilkan hasil klasterisasi kecamatan di Kota Tasikmalaya menggunakan algoritma **K-Means** berdasarkan karakteristik kependudukan tahun **2023–2026**.

Aplikasi ini dikembangkan sebagai bagian dari proyek **Kerja Praktik** untuk membantu melihat pola kemiripan karakteristik kependudukan antar-kecamatan secara lebih ringkas, interaktif, dan mudah dipahami.

---

## Tujuan

Dashboard ini bertujuan untuk:

- Mengelompokkan kecamatan di Kota Tasikmalaya berdasarkan karakteristik kependudukan.
- Menentukan jumlah cluster optimal pada setiap tahun.
- Menampilkan hasil clustering secara interaktif.
- Mengetahui anggota kecamatan pada masing-masing cluster.
- Membandingkan pola cluster antar tahun 2023–2026.
- Menyajikan hasil clustering dalam bentuk tabel, grafik, dan peta.

---

## Dataset

Dataset yang digunakan dalam penelitian ini bersumber dari **Badan Pusat Statistik (BPS) Kota Tasikmalaya**. Data yang digunakan merupakan data kependudukan menurut kecamatan di Kota Tasikmalaya untuk periode **2023–2026**.

Tabel yang digunakan adalah:

> **Penduduk, Laju Pertumbuhan Penduduk, Distribusi Persentase Penduduk, Kepadatan Penduduk, dan Rasio Jenis Kelamin Penduduk Menurut Kecamatan di Kota Tasikmalaya**

Dataset mencakup **10 kecamatan di Kota Tasikmalaya**, yaitu:

- Kawalu
- Tamansari
- Cibeureum
- Purbaratu
- Tawang
- Cihideung
- Mangkubumi
- Indihiang
- Bungursari
- Cipedes

Indikator yang tersedia pada dataset meliputi:

| Indikator | Penggunaan |
|---|---|
| Jumlah Penduduk | Variabel K-Means (X1) |
| Laju Pertumbuhan Penduduk | Variabel K-Means (X2) |
| Distribusi Persentase Penduduk | Data deskriptif |
| Kepadatan Penduduk | Variabel K-Means (X3) |
| Rasio Jenis Kelamin | Variabel K-Means (X4) |

**Sumber Data:** Badan Pusat Statistik (BPS) Kota Tasikmalaya, tabel *Penduduk, Laju Pertumbuhan Penduduk, Distribusi Persentase Penduduk, Kepadatan Penduduk, dan Rasio Jenis Kelamin Penduduk Menurut Kecamatan di Kota Tasikmalaya*, periode 2023–2026.

---

## Variabel Penelitian

Variabel yang digunakan dalam proses K-Means adalah:

| Kode | Variabel |
|---|---|
| X1 | Jumlah Penduduk |
| X2 | Laju Pertumbuhan Penduduk |
| X3 | Kepadatan Penduduk |
| X4 | Rasio Jenis Kelamin |

**Persentase Penduduk** tetap ditampilkan sebagai informasi deskriptif, tetapi tidak digunakan sebagai fitur dalam proses K-Means.

Sebelum proses clustering, variabel X1–X4 distandardisasi menggunakan **Z-score** agar perbedaan skala antarvariabel tidak memengaruhi hasil clustering.

---

## Metode

Tahapan analisis yang digunakan pada aplikasi ini meliputi:

1. Membaca dataset kependudukan per tahun.
2. Melakukan preprocessing data.
3. Memilih variabel X1–X4.
4. Melakukan standardisasi data.
5. Menguji jumlah cluster dari **K = 2 sampai K = 8**.
6. Menentukan jumlah cluster optimal menggunakan:
   - Elbow Method
   - Silhouette Score
   - Davies-Bouldin Index
7. Menjalankan K-Means menggunakan K optimal.
8. Menginterpretasikan karakteristik cluster berdasarkan centroid Z-score.
9. Menampilkan hasil dalam dashboard dan peta.

Untuk grafik Elbow, nilai WCSS juga dihitung mulai dari **K = 1 sampai K = 8**.

---

## Hasil K Optimal

Jumlah cluster optimal diperoleh secara terpisah untuk setiap tahun.

| Tahun | K Optimal |
|---|---:|
| 2023 | 3 |
| 2024 | 2 |
| 2025 | 2 |
| 2026 | 2 |

Perbedaan jumlah cluster antar tahun menunjukkan bahwa struktur kemiripan karakteristik kependudukan antar-kecamatan dapat berubah dari tahun ke tahun.

Label cluster seperti **C1, C2, dan C3 tidak boleh dianggap memiliki makna yang sama antar tahun**. Interpretasi cluster dilakukan berdasarkan centroid pada tahun masing-masing.

---

## Fitur Dashboard

Aplikasi terdiri dari beberapa halaman:

### 1. Dashboard
Menampilkan ringkasan hasil clustering berdasarkan tahun yang dipilih, meliputi:

- Total kecamatan
- Total variabel
- K optimal
- Ringkasan cluster
- Jumlah kecamatan pada masing-masing cluster
- Distribusi anggota cluster
- Daftar kecamatan pada setiap cluster

### 2. Upload CSV (Opsional)
Digunakan untuk melakukan pemeriksaan kualitas file CSV tanpa mengubah dataset penelitian utama.

Pemeriksaan meliputi:

- Jumlah baris dan kolom
- Missing value
- Data duplikat
- Tipe data
- Nilai unik
- Struktur dataset
- Validasi dataset kependudukan

### 3. Evaluasi K
Menampilkan hasil evaluasi jumlah cluster menggunakan:

- Elbow / WCSS
- Silhouette Score
- Davies-Bouldin Index
- Tabel hasil evaluasi K

### 4. Hasil Clustering
Menampilkan:

- Kecamatan dan cluster
- Jumlah anggota cluster
- Centroid hasil clustering
- Profil cluster
- Visualisasi hasil clustering

### 5. Peta Cluster
Menampilkan persebaran cluster kecamatan pada peta Kota Tasikmalaya.

Fitur peta meliputi:

- Warna berbeda untuk setiap cluster
- Nama kecamatan
- Informasi cluster saat diarahkan atau diklik
- Pencarian kecamatan
- Fullscreen
- Export peta ke PNG/JPEG

### 6. Perbandingan Tahun
Menampilkan perbandingan hasil clustering tahun 2023–2026.

---

## Interpretasi Cluster

Keterangan **tinggi**, **sedang**, dan **rendah** pada setiap cluster bersifat **relatif terhadap kecamatan lain pada tahun yang sama** berdasarkan centroid Z-score.

Contoh:

- Nilai centroid relatif tinggi → karakteristik cenderung tinggi dibanding kecamatan lain pada tahun tersebut.
- Nilai centroid mendekati rata-rata → karakteristik relatif sedang.
- Nilai centroid relatif rendah → karakteristik cenderung rendah.

Karena proses analisis dilakukan secara terpisah setiap tahun, label C1 pada satu tahun tidak selalu memiliki karakteristik yang sama dengan C1 pada tahun lainnya.

---

## Struktur Folder

```text
project/
│
├── app.py
├── requirements.txt
├── README.md
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

---

## Instalasi

Pastikan Python sudah terinstal.

Clone repository:

```bash
git clone https://github.com/USERNAME/NAMA-REPOSITORY.git
```

Masuk ke folder project:

```bash
cd NAMA-REPOSITORY
```

Install dependency:

```bash
python -m pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
python -m streamlit run app.py
```

Setelah itu buka alamat lokal yang muncul, biasanya:

```text
http://localhost:8501
```

---

## Library Utama

Aplikasi menggunakan beberapa library Python, antara lain:

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Kneed
- GeoPandas
- Folium
- Streamlit-Folium
- Matplotlib
- Pillow

---

## Deployment

Aplikasi dapat dideploy menggunakan **Streamlit Community Cloud**.

Langkah singkat:

1. Upload seluruh project ke GitHub.
2. Pastikan `app.py` dan `requirements.txt` tersedia di repository.
3. Buka Streamlit Community Cloud.
4. Hubungkan akun GitHub.
5. Pilih repository project.
6. Pilih `app.py` sebagai main file.
7. Klik **Deploy**.

Setelah proses deployment selesai, aplikasi dapat diakses melalui URL publik Streamlit.

---

## Catatan

- Dataset utama tahun 2023–2026 digunakan sebagai basis analisis.
- Fitur Upload CSV hanya digunakan sebagai pemeriksa kualitas data dan tidak menggantikan dataset penelitian utama.
- Jumlah cluster optimal dapat berbeda antar tahun.
- Interpretasi cluster harus dilakukan berdasarkan karakteristik centroid pada masing-masing tahun.

---

## Author

Project Kerja Praktik  
**Klasterisasi Kecamatan Kota Tasikmalaya Menggunakan K-Means Berdasarkan Karakteristik Kependudukan Tahun 2023–2026**

