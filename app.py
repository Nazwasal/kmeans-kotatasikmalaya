import re
import io
import base64
from html import escape as html_escape
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import streamlit as st

from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

from kneed import KneeLocator

import geopandas as gpd
import folium
from folium.plugins import Fullscreen, Search
from streamlit_folium import folium_static


# ============================================================
# 1. KONFIGURASI APLIKASI
# ============================================================
st.set_page_config(
    page_title="K-Means Tasikmalaya",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "dataset"
PETA_DIR = APP_DIR / "peta"
ASSET_DIR = APP_DIR / "assets"

LOGO_PATH = ASSET_DIR / "logo_kota_tasikmalaya.png"

FILE_PATHS = {
    2024: DATA_DIR / "kependudukan_2024.csv",
    2025: DATA_DIR / "kependudukan_2025.csv",
    2026: DATA_DIR / "kependudukan_2026.csv",
}

SHP_PATH = PETA_DIR / "ADMINISTRASI_AR_KECAMATAN.shp"

TAHUN_LIST = [2024, 2025, 2026]

RANDOM_STATE = 42
N_INIT = 20
MAX_ITER = 300
K_MAX = 5

KECAMATAN_TARGET = [
    "Kawalu",
    "Tamansari",
    "Cibeureum",
    "Purbaratu",
    "Tawang",
    "Cihideung",
    "Mangkubumi",
    "Indihiang",
    "Bungursari",
    "Cipedes",
]

INDIKATOR_EDA = [
    "Jumlah_Penduduk",
    "Laju_Pertumbuhan",
    "Persentase_Penduduk",
    "Kepadatan_Penduduk",
    "Rasio_Jenis_Kelamin",
]

FITUR_KMEANS = [
    "Jumlah_Penduduk",
    "Laju_Pertumbuhan",
    "Kepadatan_Penduduk",
    "Rasio_Jenis_Kelamin",
]

NAMA_X = ["X1", "X2", "X3", "X4"]

KETERANGAN_X = {
    "X1": "Jumlah Penduduk",
    "X2": "Laju Pertumbuhan Penduduk",
    "X3": "Kepadatan Penduduk",
    "X4": "Rasio Jenis Kelamin",
}

DISPLAY_NAME = {
    "Jumlah_Penduduk": "Jumlah Penduduk",
    "Laju_Pertumbuhan": "Laju Pertumbuhan",
    "Persentase_Penduduk": "Persentase Penduduk",
    "Kepadatan_Penduduk": "Kepadatan Penduduk",
    "Rasio_Jenis_Kelamin": "Rasio Jenis Kelamin",
}

# Warna cluster terpusat untuk grafik, peta, dan kartu (semua tabel polos).
# C1 = BIRU (pertumbuhan relatif tinggi, kepadatan relatif rendah)
# C2 = HIJAU (kepadatan relatif tinggi, pertumbuhan relatif rendah)
CLUSTER_COLORS = {
    "C1": "#4E79A7",
    "C2": "#59A14F",
    "C3": "#F28E2B",
    "C4": "#E15759",
    "C5": "#B07AA1",
    "C6": "#76B7B2",
    "C7": "#EDC948",
    "C8": "#9C755F",
}


# Versi latar dan teks khusus kartu (tidak digunakan pada tabel).
CLUSTER_LIGHT_COLORS = {
    "C1": "#E7EFF7",   # Biru muda
    "C2": "#EAF5E8",   # Hijau muda
}
CLUSTER_TEXT_COLORS = {
    "C1": "#365F85",   # Biru gelap
    "C2": "#3B7837",   # Hijau gelap
}


def warna_tabel_cluster(data):
    """Tabel polos: data tidak diwarnai dan kode cluster tetap teks C1/C2."""
    return data


DATA_ADMIN_KECAMATAN = {
    "Cihideung": {
        "kode_kemendagri": "32.78.01",
        "kode_wilayah": "327801",
        "kode_bps": "3278050",
        "jumlah_kelurahan": 6,
    },
    "Cipedes": {
        "kode_kemendagri": "32.78.02",
        "kode_wilayah": "327802",
        "kode_bps": "3278080",
        "jumlah_kelurahan": 4,
    },
    "Tawang": {
        "kode_kemendagri": "32.78.03",
        "kode_wilayah": "327803",
        "kode_bps": "3278040",
        "jumlah_kelurahan": 5,
    },
    "Indihiang": {
        "kode_kemendagri": "32.78.04",
        "kode_wilayah": "327804",
        "kode_bps": "Bervariasi",
        "jumlah_kelurahan": 6,
    },
    "Kawalu": {
        "kode_kemendagri": "32.78.05",
        "kode_wilayah": "327805",
        "kode_bps": "Bervariasi",
        "jumlah_kelurahan": 10,
    },
    "Cibeureum": {
        "kode_kemendagri": "32.78.06",
        "kode_wilayah": "327806",
        "kode_bps": "3278030",
        "jumlah_kelurahan": 9,
    },
    "Tamansari": {
        "kode_kemendagri": "32.78.07",
        "kode_wilayah": "327807",
        "kode_bps": "3278020",
        "jumlah_kelurahan": 8,
    },
    "Mangkubumi": {
        "kode_kemendagri": "32.78.08",
        "kode_wilayah": "327808",
        "kode_bps": "Bervariasi",
        "jumlah_kelurahan": 8,
    },
    "Bungursari": {
        "kode_kemendagri": "32.78.09",
        "kode_wilayah": "327809",
        "kode_bps": "3278071",
        "jumlah_kelurahan": 7,
    },
    "Purbaratu": {
        "kode_kemendagri": "32.78.10",
        "kode_wilayah": "327810",
        "kode_bps": "3278031",
        "jumlah_kelurahan": 6,
    },
}

# ============================================================
# PENAMAAN WILAYAH BERDASARKAN PROFIL CENTROID AKTUAL
# ============================================================
# Nama tidak dikunci pada tahun atau nomor cluster.
# Untuk K=2, peringkat X2 (pertumbuhan) dan X3 (kepadatan)
# digunakan untuk memberikan nama kawasan yang mudah dipahami.
# Kata tinggi/rendah/kurang padat selalu RELATIF antarklaster.

def nama_profil_cluster(hasil_tahun, tahun=None, pendek=False):
    """Berikan nama kawasan dari centroid untuk model tahun yang dipilih.

    Tidak mengubah K-Means atau label C1/C2. Nama muncul berdasarkan
    urutan centroid X2 & X3, bukan asumsi semua kecamatan padat secara absolut.
    """
    centroid = hasil_tahun["result"]["centroid_z"].copy()
    k = int(hasil_tahun["k_optimal"])
    result = {}

    # Catatan khusus untuk kasus nilai yang benar-benar sama/nyaris sama.
    def label_peringkat(kolom, kecil, besar):
        data = centroid[kolom].to_numpy(dtype=float)
        if len(data) == 0 or np.all(np.isclose(data, data[0], atol=1e-10, rtol=0)):
            return ["setara"] * len(data)
        if k == 2:
            return [kecil if v < max(data) else besar for v in data]
        if k == 3:
            # Jika dua centroid sama, jangan berpura-pura memiliki urutan berbeda.
            if len(np.unique(np.round(data, 10))) < k:
                return ["setara" if np.sum(np.isclose(data, v, atol=1e-10, rtol=0)) > 1
                        else (kecil if v == min(data) else besar) for v in data]
            posisi = np.argsort(np.argsort(data, kind="stable"), kind="stable")
            tingkat = [kecil, "sedang", besar]
            return [tingkat[int(i)] for i in posisi]
        return ["relatif beragam"] * len(data)

    tingkat_pertumbuhan = label_peringkat("X2", "rendah", "tinggi")
    tingkat_kepadatan = label_peringkat("X3", "kurang padat", "padat")
    for i, row in centroid.reset_index(drop=True).iterrows():
        cluster = str(row["Cluster"])
        kepadatan = tingkat_kepadatan[i]
        pertumbuhan = tingkat_pertumbuhan[i]
        if kepadatan == "setara" or pertumbuhan == "setara":
            nama = f"Kawasan dengan Kepadatan/Pertumbuhan Setara ({cluster})"
        elif k in (2, 3):
            if kepadatan == "sedang":
                nama = f"Kawasan berkepadatan sedang dengan pertumbuhan {pertumbuhan}"
            else:
                nama = f"Kawasan {kepadatan} dengan pertumbuhan {pertumbuhan}"
        else:
            nama = f"Profil {cluster} (lihat centroid X2–X3)"
        result[cluster] = nama
    return result


def judul_subjudul_cluster(hasil_tahun, cluster):
    """Judul tebal dan subjudul deskriptif untuk kartu Profil Cluster."""
    kategori = kategori_variabel_cluster(hasil_tahun, cluster) if "kategori_variabel_cluster" in globals() else {}
    x2 = str(kategori.get("X2", "-")).lower()
    x3 = str(kategori.get("X3", "-")).lower()
    k = int(hasil_tahun.get("k_optimal", 0))

    if k == 2:
        if x2 == "tinggi" and x3 == "rendah":
            return "Kawasan Berkembang", "Pertumbuhan Tinggi dan Kepadatan Rendah"
        if x2 == "rendah" and x3 == "tinggi":
            return "Kawasan Perkotaan Padat", "Kepadatan Tinggi dan Pertumbuhan Rendah"
        if x2 == "tinggi" and x3 == "tinggi":
            return "Kawasan Dinamis", "Pertumbuhan Tinggi dan Kepadatan Tinggi"
        if x2 == "rendah" and x3 == "rendah":
            return "Kawasan Relatif Longgar", "Pertumbuhan Rendah dan Kepadatan Rendah"

    nama_baku = nama_profil_cluster(hasil_tahun).get(cluster, cluster)
    return nama_baku, nama_baku


def rincian_profil(hasil_tahun):
    """Teks X1–X4 dari interpretasi relatif, untuk tampilan kartu."""
    rows = hasil_tahun["result"]["interpretasi"]
    return dict(zip(rows["Cluster"], rows["Interpretasi_Relatif"]))

# ============================================================
# 2. STYLE
# ============================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #0f2942;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e7edf4;
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(15, 41, 66, 0.06);
    }

    .page-subtitle {
        color: #64748b;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    .info-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 15px 18px;
        margin: 8px 0 16px 0;
    }

    .success-box {
        background: #effcf3;
        border: 1px solid #b7ebc4;
        border-radius: 12px;
        padding: 15px 18px;
        margin: 8px 0 16px 0;
    }

    .cluster-box {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: white;
    }


    /* Kartu Profil Cluster pada halaman Hasil Clustering */
    .kp-profile-card {
        --kp-accent: #4E79A7;
        --kp-soft: #E7EFF7;
        --kp-text: #365F85;
        border: 1px solid #dce5ed;
        border-top: 5px solid var(--kp-accent);
        border-radius: 16px;
        background: #fff;
        padding: 22px 22px 19px;
        box-shadow: 0 3px 12px rgba(15, 41, 66, 0.055);
        min-height: 365px;
        display: flex;
        flex-direction: column;
        gap: 0;
        margin-bottom: 12px;
    }
    .kp-profile-card .kp-label {
        width: fit-content;
        border-radius: 999px;
        padding: 4px 13px;
        background: var(--kp-soft);
        color: var(--kp-text);
        font-weight: 800;
        font-size: 14px;
        letter-spacing: .15px;
    }
    .kp-profile-card .kp-card-title {
        font-size: 19px;
        font-weight: 800;
        color: #142535;
        line-height: 1.28;
        margin: 11px 0 4px;
        min-height: 0;
    }
    .kp-profile-card .kp-card-subtitle {
        font-size: 14px;
        color: #64748b;
        line-height: 1.4;
        margin: 0 0 18px;
        min-height: 20px;
    }
    .kp-profile-card .kp-var-list { display: grid; gap: 7px; }
    .kp-profile-card .kp-var-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }
    .kp-profile-card .kp-var-name {
        font-size: 14px;
        line-height: 1.35;
        color: #596c7b;
    }
    .kp-profile-card .kp-value-pill {
        flex-shrink: 0;
        min-width: 88px;
        padding: 4px 11px;
        border: 1px solid #dbe3ea;
        border-radius: 10px;
        background: #f8fafc;
        color: #233443;
        text-align: center;
        font-size: 13px;
        font-weight: 700;
        text-transform: capitalize;
    }
    .kp-profile-card .kp-value-pill.is-high {
        background: #e8f1ff;
        border: 1.5px solid #234a76;
        color: #0f172a;
        font-weight: 800;
        box-shadow: inset 0 0 0 1px rgba(35,74,118,.08);
    }
    .kp-profile-card .kp-value-pill.is-low {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #334155;
        font-weight: 700;
    }
    .kp-profile-card .kp-value-pill.is-mid {
        background: #fff7ed;
        border: 1px solid #fdba74;
        color: #9a3412;
        font-weight: 700;
    }
    .kp-profile-card .kp-card-members {
        border-top: 1px solid #dce5ec;
        margin-top: auto;
        padding-top: 17px;
        font-size: 13px;
        color: #253546;
        line-height: 1.55;
    }
    .kp-profile-card .kp-card-note {
        background: #fff8e8;
        border-radius: 9px;
        color: #9b5b00;
        padding: 9px 10px;
        font-size: 12px;
        margin-top: 12px;
    }
    @media(max-width: 900px) {
        .kp-profile-card { padding: 17px 15px; min-height: 330px; }
        .kp-profile-card .kp-card-title { font-size: 17px; min-height: 0; }
        .kp-profile-card .kp-var-name { font-size: 12px; }
    }

    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 2px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 9px;
        padding: 4px 8px;
        margin-bottom: 0px;
        transition: all .18s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,.08);
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: rgba(255,255,255,.12);
    }

    [data-testid="stSidebar"] [role="radiogroup"] p {
        font-size: 14px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. FUNGSI PEMBERSIHAN DATA
# ============================================================
def normalize_text(value):
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_key(value):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        normalize_text(value).lower()
    )


def read_csv_smart(source):
    last_error = None

    for encoding in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            if hasattr(source, "seek"):
                source.seek(0)

            return pd.read_csv(
                source,
                sep=None,
                engine="python",
                encoding=encoding,
            )

        except Exception as exc:
            last_error = exc

    raise last_error


def parse_numeric_value(value, integerish=False):
    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):
        return float(value)

    s = str(value).strip()
    s = s.replace("\xa0", "")
    s = s.replace(" ", "")
    s = s.replace("%", "")
    s = s.replace("*", "")

    if s in {"", "-", "nan", "None"}:
        return np.nan

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    elif "," in s:
        if (
            integerish
            and
            re.fullmatch(r"-?\d{1,3}(,\d{3})+", s)
        ):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")

    elif "." in s:
        if (
            integerish
            and
            re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s)
        ):
            s = s.replace(".", "")

    try:
        return float(s)
    except ValueError:
        return np.nan


def to_numeric_safe(series, integerish=False):
    return series.map(
        lambda value: parse_numeric_value(
            value,
            integerish=integerish
        )
    )


def rename_columns_auto(df):
    mapping = {}

    for col in df.columns:
        c = normalize_text(col).lower()

        if c == "kecamatan" or "nama kecamatan" in c:
            mapping[col] = "Kecamatan"

        elif (
            "jumlah penduduk" in c
            and
            "persentase" not in c
        ):
            mapping[col] = "Jumlah_Penduduk"

        elif "laju pertumbuhan" in c:
            mapping[col] = "Laju_Pertumbuhan"

        elif "persentase penduduk" in c:
            mapping[col] = "Persentase_Penduduk"

        elif "kepadatan penduduk" in c:
            mapping[col] = "Kepadatan_Penduduk"

        elif "rasio jenis kelamin" in c:
            mapping[col] = "Rasio_Jenis_Kelamin"

    return df.rename(columns=mapping)


def preprocessing(df_raw, tahun):
    df = df_raw.copy()

    df.columns = [
        normalize_text(
            str(col).replace("\n", " ")
        )
        for col in df.columns
    ]

    df = rename_columns_auto(df)

    kolom_wajib = [
        "Kecamatan"
    ] + INDIKATOR_EDA

    kolom_hilang = [
        col
        for col in kolom_wajib
        if col not in df.columns
    ]

    if kolom_hilang:
        raise ValueError(
            "Kolom tidak ditemukan: "
            + ", ".join(kolom_hilang)
        )

    df["Kecamatan"] = (
        df["Kecamatan"]
        .astype(str)
        .map(normalize_text)
    )

    target_map = {
        normalize_key(kecamatan): kecamatan
        for kecamatan in KECAMATAN_TARGET
    }

    df["_key"] = (
        df["Kecamatan"]
        .map(normalize_key)
    )

    df = df[
        df["_key"].isin(target_map)
    ].copy()

    df["Kecamatan"] = (
        df["_key"]
        .map(target_map)
    )

    df = df[
        ["Kecamatan"] + INDIKATOR_EDA
    ].copy()

    for col in INDIKATOR_EDA:
        integerish = col in [
            "Jumlah_Penduduk",
            "Kepadatan_Penduduk",
        ]

        df[col] = to_numeric_safe(
            df[col],
            integerish=integerish
        )

    df.insert(
        1,
        "Tahun",
        tahun
    )

    # Dataset penelitian harus satu baris per kecamatan.
    df.drop_duplicates(
        subset=["Kecamatan"],
        keep="first",
        inplace=True
    )

    urutan = {
        kecamatan: i
        for i, kecamatan
        in enumerate(KECAMATAN_TARGET)
    }

    df["_urutan"] = (
        df["Kecamatan"]
        .map(urutan)
    )

    df = (
        df
        .sort_values("_urutan")
        .drop(columns="_urutan")
        .reset_index(drop=True)
    )

    return df


def validasi_dataframe(df):
    hasil = {
        "Jumlah kecamatan = 10":
            df["Kecamatan"].nunique() == 10,

        "Semua kolom ditemukan":
            all(
                col in df.columns
                for col in (
                    ["Kecamatan", "Tahun"]
                    +
                    INDIKATOR_EDA
                )
            ),

        "Tidak ada missing value":
            df[INDIKATOR_EDA]
            .isna()
            .sum()
            .sum()
            == 0,

        "Tidak ada duplikasi kecamatan":
            df.duplicated(
                subset=["Kecamatan"]
            ).sum()
            == 0,

        "Semua variabel numerik":
            all(
                pd.api.types.is_numeric_dtype(
                    df[col]
                )
                for col in INDIKATOR_EDA
            ),
    }

    return hasil


# ============================================================
# 4. ENGINE K-MEANS
# ============================================================
def kmeans_trace(
    X,
    k,
    seed=42,
    max_iter=300,
    tol=1e-10
):
    X = np.asarray(
        X,
        dtype=float
    )

    centroid, indeks_awal = kmeans_plusplus(
        X,
        n_clusters=k,
        random_state=seed
    )

    centroid_awal = centroid.copy()
    labels_lama = None
    riwayat = []

    for iterasi in range(
        1,
        max_iter + 1
    ):
        jarak = np.linalg.norm(
            X[:, None, :]
            -
            centroid[None, :, :],
            axis=2
        )

        labels = np.argmin(
            jarak,
            axis=1
        )

        centroid_baru = np.empty_like(
            centroid
        )

        for j in range(k):
            anggota = X[
                labels == j
            ]

            if len(anggota) == 0:
                return None

            centroid_baru[j] = (
                anggota.mean(axis=0)
            )

        riwayat.append({
            "iterasi": iterasi,
            "centroid_sebelum":
                centroid.copy(),
            "jarak":
                jarak.copy(),
            "cluster":
                labels.copy(),
            "centroid_baru":
                centroid_baru.copy(),
        })

        labels_stabil = (
            labels_lama is not None
            and
            np.array_equal(
                labels,
                labels_lama
            )
        )

        centroid_stabil = np.allclose(
            centroid_baru,
            centroid,
            atol=tol,
            rtol=0
        )

        centroid = centroid_baru.copy()
        labels_lama = labels.copy()

        if (
            labels_stabil
            or
            centroid_stabil
        ):
            break

    jarak_final = np.linalg.norm(
        X[:, None, :]
        -
        centroid[None, :, :],
        axis=2
    )

    labels_final = np.argmin(
        jarak_final,
        axis=1
    )

    if (
        len(np.unique(labels_final))
        != k
    ):
        return None

    inertia = float(
        np.sum(
            (
                X
                -
                centroid[
                    labels_final
                ]
            ) ** 2
        )
    )

    ukuran_cluster = [
        int(
            np.sum(
                labels_final == j
            )
        )
        for j in range(k)
    ]

    return {
        "centroid_awal":
            centroid_awal,
        "indeks_awal":
            indeks_awal,
        "centroid_final":
            centroid,
        "cluster_final":
            labels_final,
        "jarak_final":
            jarak_final,
        "riwayat":
            riwayat,
        "inertia":
            inertia,
        "jumlah_iterasi":
            len(riwayat),
        "seed":
            seed,
        "ukuran_cluster":
            ukuran_cluster,
    }


def best_kmeans_trace(
    X,
    k,
    n_init=20
):
    runs_valid = []

    for seed in range(
        RANDOM_STATE,
        RANDOM_STATE + 200
    ):
        run = kmeans_trace(
            X,
            k,
            seed=seed,
            max_iter=MAX_ITER
        )

        if run is not None:
            runs_valid.append(run)

        if len(runs_valid) >= n_init:
            break

    if not runs_valid:
        raise RuntimeError(
            f"Tidak ditemukan run K-Means "
            f"valid untuk K={k}."
        )

    return min(
        runs_valid,
        key=lambda result:
            result["inertia"]
    )


def hitung_wcss_elbow(X_scaled):
    rows = []

    for k in range(
        1,
        K_MAX + 1
    ):
        hasil = best_kmeans_trace(
            X_scaled,
            k,
            n_init=N_INIT
        )

        rows.append({
            "K": k,
            "Inertia_WCSS":
                hasil["inertia"],
        })

    return pd.DataFrame(rows)


def hitung_evaluasi_k(X_scaled):
    """Skor Silhouette untuk K=2..5; DBI hanya pada model final."""
    rows, hasil_per_k = [], {}
    for k in range(2, K_MAX + 1):
        hasil = best_kmeans_trace(X_scaled, k, n_init=N_INIT)
        labels = np.asarray(hasil['cluster_final'])
        hasil_per_k[k] = hasil
        rows.append({
            'K': k,
            'Inertia_WCSS': hasil['inertia'],
            'Silhouette_Score': float(silhouette_score(X_scaled, labels)),
            'Ukuran_Cluster': str(hasil['ukuran_cluster']),
            'Ukuran_Min': min(hasil['ukuran_cluster']),
        })
    return pd.DataFrame(rows), hasil_per_k


def deteksi_k_elbow(df_elbow):
    knee = KneeLocator(
        df_elbow["K"],
        df_elbow["Inertia_WCSS"],
        curve="convex",
        direction="decreasing"
    )

    if knee.elbow is not None:
        return int(knee.elbow)

    x = (
        df_elbow["K"]
        .to_numpy(dtype=float)
    )

    y = (
        df_elbow[
            "Inertia_WCSS"
        ]
        .to_numpy(dtype=float)
    )

    x_norm = (
        (x - x.min())
        /
        (x.max() - x.min())
    )

    y_norm = (
        (y - y.min())
        /
        (y.max() - y.min())
    )

    p1 = np.array([
        x_norm[0],
        y_norm[0]
    ])

    p2 = np.array([
        x_norm[-1],
        y_norm[-1]
    ])

    line = p2 - p1
    line_norm = np.linalg.norm(line)

    distances = []

    for xi, yi in zip(
        x_norm,
        y_norm
    ):
        p = np.array([
            xi,
            yi
        ])

        distance = abs(
            line[0]
            *
            (p1[1] - p[1])
            -
            line[1]
            *
            (p1[0] - p[0])
        ) / line_norm

        distances.append(
            distance
        )

    idx = int(
        np.argmax(distances)
    )

    return int(
        df_elbow.iloc[
            idx
        ]["K"]
    )


def tentukan_k_optimal(k_elbow, df_evaluasi):
    """Pilih K dengan Silhouette tertinggi; Elbow hanya konfirmasi."""
    kandidat = df_evaluasi.dropna(subset=['Silhouette_Score']).sort_values(
        ['Silhouette_Score', 'K'], ascending=[False, True]
    )
    if kandidat.empty:
        raise ValueError('Tidak ada skor Silhouette yang valid.')
    row = kandidat.iloc[0]
    k = int(row['K'])
    note = (f'Elbow searah pada K={k}.' if k_elbow == k else
            f'Elbow menunjukkan K={k_elbow}, tetapi K final mengikuti Silhouette.')
    if int(row['Ukuran_Min']) == 1:
        note += ' Ada cluster tunggal yang perlu dijelaskan pada interpretasi.'
    return k, f'Silhouette tertinggi={row["Silhouette_Score"]:.4f}, K={k}. {note} DBI hanya evaluasi akhir.'


def finalisasi_kmeans(
    df,
    X_scaled,
    scaler,
    k_optimal,
    tahun,
    hasil_kmeans
):
    hasil = hasil_kmeans
    labels = hasil[
        "cluster_final"
    ]

    df_final = df.copy()

    df_final["Cluster"] = [
        f"C{label + 1}"
        for label in labels
    ]

    centroid_z = pd.DataFrame(
        hasil["centroid_final"],
        columns=NAMA_X
    )

    centroid_z.insert(
        0,
        "Cluster",
        [
            f"C{i + 1}"
            for i
            in range(k_optimal)
        ]
    )

    centroid_asli = pd.DataFrame(
        scaler.inverse_transform(
            hasil["centroid_final"]
        ),
        columns=FITUR_KMEANS
    )

    centroid_asli.insert(
        0,
        "Cluster",
        [
            f"C{i + 1}"
            for i
            in range(k_optimal)
        ]
    )

    profil_asli = (
        df_final
        .groupby(
            "Cluster"
        )[FITUR_KMEANS]
        .mean()
        .reset_index()
    )

    # Interpretasi berdasarkan peringkat centroid antarcluster untuk SETIAP variabel.
    # K=2: Rendah/Tinggi; K=3: Rendah/Sedang/Tinggi.
    # Nilai sama diberikan label Setara (tidak dipaksa berbeda).
    interpretasi_rows = []
    kategori_per_x = {}
    for x in NAMA_X:
        nilai = centroid_z[x].to_numpy(dtype=float)
        urutan_unik = np.unique(nilai)
        if k_optimal == 2:
            tingkat = ["rendah", "tinggi"]
        elif k_optimal == 3:
            tingkat = ["rendah", "sedang", "tinggi"]
        else:
            tingkat = [f"urutan {i+1} dari {k_optimal}" for i in range(k_optimal)]

        if len(urutan_unik) != k_optimal:
            # Jika centroid sama pada indikator ini, bedakan hanya nilai yang
            # memang berbeda dan beri label Setara untuk nilai yang sama.
            peringkat = np.argsort(np.argsort(urutan_unik))
            if len(urutan_unik) == 1:
                kategori_per_x[x] = ["setara"] * k_optimal
                continue
            posisi = {float(v): int(round(i * (k_optimal-1) / (len(urutan_unik)-1)))
                      for i, v in enumerate(urutan_unik)}
            hitungan = {float(v): int(np.sum(np.isclose(nilai, v, atol=1e-10, rtol=0)))
                        for v in urutan_unik}
            kategori_per_x[x] = ["setara" if hitungan[float(v)] > 1
                                  else tingkat[posisi[float(v)]] for v in nilai]
        else:
            peringkat = np.argsort(np.argsort(nilai, kind="stable"), kind="stable")
            kategori_per_x[x] = [tingkat[int(r)] for r in peringkat]

    for indeks, row in centroid_z.reset_index(drop=True).iterrows():
        bagian = [f"{x} ({KETERANGAN_X[x]}) {kategori_per_x[x][indeks]}"
                  for x in NAMA_X]
        interpretasi_rows.append({
            "Cluster": row["Cluster"],
            "Interpretasi_Relatif": "; ".join(bagian),
        })

    interpretasi = pd.DataFrame(interpretasi_rows)

    pca = PCA(
        n_components=2
    )

    xy = pca.fit_transform(
        X_scaled
    )

    pca_df = pd.DataFrame({
        "Kecamatan":
            df["Kecamatan"].values,

        "PC1":
            xy[:, 0],

        "PC2":
            xy[:, 1],

        "Cluster":
            df_final[
                "Cluster"
            ].values,
    })

    evaluasi_final = {
        "Tahun":
            tahun,
        "K_Optimal":
            k_optimal,
        "WCSS_Final":
            hasil["inertia"],
        "Silhouette_Final":
            silhouette_score(
                X_scaled,
                labels
            ),
        "Davies_Bouldin_Final":
            davies_bouldin_score(
                X_scaled,
                labels
            ),
        "Jumlah_Iterasi":
            hasil[
                "jumlah_iterasi"
            ],
        "Seed_Best_Run":
            hasil["seed"],
        "Ukuran_Cluster":
            hasil[
                "ukuran_cluster"
            ],
    }

    return {
        "hasil":
            hasil,
        "df_final":
            df_final,
        "centroid_z":
            centroid_z,
        "centroid_asli":
            centroid_asli,
        "profil_asli":
            profil_asli,
        "interpretasi":
            interpretasi,
        "pca":
            pca_df,
        "evaluasi_final":
            evaluasi_final,
    }


# ============================================================
# 5. ANALISIS PER TAHUN
# ============================================================
@st.cache_data(show_spinner=False)
def analisis_tahun(tahun):
    path = FILE_PATHS[
        tahun
    ]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset belum ditemukan: {path}"
        )

    raw = read_csv_smart(
        path
    )

    df = preprocessing(
        raw,
        tahun
    )

    validasi = validasi_dataframe(
        df
    )

    if not all(
        validasi.values()
    ):
        raise ValueError(
            f"Dataset {tahun} belum lolos validasi."
        )

    scaler = StandardScaler()

    X_scaled = (
        scaler.fit_transform(
            df[FITUR_KMEANS]
        )
    )

    df_scaled = pd.DataFrame(
        X_scaled,
        columns=NAMA_X
    )

    df_scaled.insert(
        0,
        "Kecamatan",
        df[
            "Kecamatan"
        ].values
    )

    elbow = hitung_wcss_elbow(
        X_scaled
    )

    evaluasi, hasil_per_k = (
        hitung_evaluasi_k(
            X_scaled
        )
    )

    k_elbow = deteksi_k_elbow(
        elbow
    )

    k_silhouette = int(
        evaluasi.loc[
            evaluasi[
                "Silhouette_Score"
            ].idxmax(),
            "K",
        ]
    )

    k_optimal, alasan = tentukan_k_optimal(k_elbow, evaluasi)

    result = finalisasi_kmeans(
        df,
        X_scaled,
        scaler,
        k_optimal,
        tahun,
        hasil_kmeans=
            hasil_per_k[
                k_optimal
            ],
    )

    return {
        "raw":
            raw,
        "df":
            df,
        "validasi":
            validasi,
        "X_scaled":
            X_scaled,
        "df_scaled":
            df_scaled,
        "elbow":
            elbow,
        "evaluasi":
            evaluasi,
        "k_elbow":
            k_elbow,
        "k_silhouette":
            k_silhouette,
        "k_optimal":
            k_optimal,
        "alasan":
            alasan,
        "result":
            result,
    }


def get_all_results():
    results = {}

    for tahun in TAHUN_LIST:
        results[tahun] = (
            analisis_tahun(
                tahun
            )
        )

    return results


# ============================================================
# 6. SHAPEFILE DAN PETA
# ============================================================
def deteksi_kolom_kecamatan_peta(gdf):
    kandidat = [
        "WADMKC",
        "KECAMATAN",
        "Kecamatan",
        "kecamatan",
    ]

    for col in kandidat:
        if col in gdf.columns:
            return col

    kandidat_otomatis = [
        col
        for col in gdf.columns
        if "kec" in str(
            col
        ).lower()
    ]

    if kandidat_otomatis:
        return (
            kandidat_otomatis[0]
        )

    raise ValueError(
        "Kolom nama kecamatan "
        "tidak ditemukan."
    )


@st.cache_data(show_spinner=False)
def load_boundary():
    if not SHP_PATH.exists():
        raise FileNotFoundError(
            f"Shapefile belum ditemukan: {SHP_PATH}"
        )

    gdf = gpd.read_file(
        SHP_PATH
    )

    kolom_kecamatan = (
        deteksi_kolom_kecamatan_peta(
            gdf
        )
    )

    gdf["_key"] = (
        gdf[
            kolom_kecamatan
        ]
        .map(normalize_key)
    )

    target = {
        normalize_key(k)
        for k in KECAMATAN_TARGET
    }

    if "WADMKK" in gdf.columns:
        kota = gdf[
            gdf["WADMKK"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(
                "kota tasikmalaya"
            )
        ].copy()

        if (
            kota["_key"]
            .isin(target)
            .sum()
            == 10
        ):
            gdf = kota

    gdf = gdf[
        gdf["_key"]
        .isin(target)
    ].copy()

    gdf = gdf.dissolve(
        by="_key",
        as_index=False,
        aggfunc="first"
    )

    if len(gdf) != 10:
        raise ValueError(
            "Boundary harus berisi "
            "10 kecamatan."
        )

    return (
        gdf,
        kolom_kecamatan
    )


def build_map(
    hasil_tahun,
    tahun
):
    """
    Peta cluster Kota Tasikmalaya:
    - basemap Google Maps,
    - warna polygon menunjukkan cluster,
    - nama kecamatan diletakkan di tengah wilayah,
    - hover langsung menampilkan detail X1-X4,
    - klik menampilkan popup detail dengan isi yang sama,
    - search bar berada di samping tombol zoom (+/-).
    """
    gdf, kolom_nama = load_boundary()

    data = (
        hasil_tahun[
            "result"
        ][
            "df_final"
        ]
        .copy()
    )

    data["_key"] = (
        data["Kecamatan"]
        .map(normalize_key)
    )

    gdf = gdf.merge(
        data,
        on="_key",
        how="left"
    )

    web = gdf.to_crs(
        epsg=4326
    ).copy()

    minx, miny, maxx, maxy = (
        web.total_bounds
    )

    center_lat = (
        miny + maxy
    ) / 2

    center_lon = (
        minx + maxx
    ) / 2

    def format_id_number(value, max_decimals=2):
        if pd.isna(value):
            return "-"

        try:
            value = float(value)
        except Exception:
            return str(value)

        if abs(value - round(value)) < 1e-10:
            s = f"{int(round(value)):,}"
        else:
            s = f"{value:,.{max_decimals}f}"
            s = s.rstrip("0").rstrip(".")

        return (
            s.replace(",", "_")
            .replace(".", ",")
            .replace("_", ".")
        )

    # --------------------------------------------------------
    # BASEMAP GOOGLE MAPS
    # --------------------------------------------------------
    m = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=12,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
        zoom_control=True
    )

    folium.TileLayer(
        tiles=(
            "https://mt1.google.com/vt/lyrs=r"
            "&x={x}&y={y}&z={z}"
        ),
        attr="Google Maps",
        name="Google Maps",
        overlay=False,
        control=False,
        max_zoom=20
    ).add_to(m)

    # --------------------------------------------------------
    # CSS MAP
    # Search bar di samping tombol zoom.
    # --------------------------------------------------------
    map_css = """
    <style>
    /* Tombol zoom tetap vertikal di kiri.
       Search bar diletakkan sejajar tepat di samping tombol +. */
    .leaflet-top.leaflet-left {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: auto !important;
        height: auto !important;
    }

    .leaflet-top.leaflet-left .leaflet-control-zoom {
        position: relative !important;
        margin: 12px 0 0 12px !important;
        z-index: 1003 !important;
    }

    .leaflet-top.leaflet-left .leaflet-control-search {
        position: absolute !important;
        top: 12px !important;
        left: 66px !important;
        margin: 0 !important;
        width: 375px !important;
        height: 44px !important;
        background: #10283f !important;
        border: 1px solid rgba(56,189,248,.45) !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 18px rgba(15,23,42,.24) !important;
        padding: 5px 9px !important;
        z-index: 1004 !important;
    }

    .leaflet-control-search .search-input {
        display: inline-block !important;
        background: transparent !important;
        color: #ffffff !important;
        border: none !important;
        outline: none !important;
        width: 310px !important;
        height: 32px !important;
        padding: 0 7px !important;
        font-size: 13px !important;
        vertical-align: middle !important;
    }

    .leaflet-control-search .search-input::placeholder {
        color: #a7b6c5 !important;
    }

    .leaflet-control-search .search-button {
        display: inline-block !important;
        float: right !important;
        background-color: transparent !important;
        margin-top: 1px !important;
    }

    .leaflet-control-search .search-tooltip {
        margin-top: 8px !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    .leaflet-popup-content-wrapper {
        border-radius: 14px !important;
        padding: 0 !important;
        overflow: hidden !important;
        box-shadow: 0 10px 30px rgba(15,23,42,.25) !important;
    }

    .leaflet-popup-content {
        margin: 0 !important;
        width: 340px !important;
    }

    .leaflet-popup-tip {
        box-shadow: none !important;
    }

    .custom-kecamatan-tooltip {
        background: rgba(255,255,255,.98) !important;
        border: 1px solid #dbe3ec !important;
        border-radius: 10px !important;
        box-shadow: 0 6px 20px rgba(15,23,42,.18) !important;
        color: #0f172a !important;
        padding: 0 !important;
    }
    </style>
    """

    m.get_root().html.add_child(
        folium.Element(
            map_css
        )
    )

    # --------------------------------------------------------
    # LOGO POPUP
    # --------------------------------------------------------
    logo_data_uri = ""

    if LOGO_PATH.exists():
        logo_base64 = base64.b64encode(
            LOGO_PATH.read_bytes()
        ).decode("utf-8")

        logo_data_uri = (
            "data:image/png;base64,"
            + logo_base64
        )

    # --------------------------------------------------------
    # SEARCH LAYER
    # --------------------------------------------------------
    search_layer = folium.GeoJson(
        web,
        name="Pencarian Kecamatan",
        style_function=lambda feature: {
            "fillOpacity": 0,
            "opacity": 0,
            "weight": 0,
        }
    ).add_to(m)

    Search(
        layer=search_layer,
        search_label=kolom_nama,
        placeholder="Cari Kecamatan...",
        collapsed=False,
        position="topleft",
        geom_type="Polygon",
        weight=4
    ).add_to(m)

    # --------------------------------------------------------
    # TITIK TENGAH POLYGON
    # --------------------------------------------------------
    try:
        projected_crs = gdf.estimate_utm_crs()
        projected = gdf.to_crs(
            projected_crs
        ).copy()

        center_points = (
            projected.geometry.centroid
        )

        for idx in projected.index:
            if not projected.loc[
                idx,
                "geometry"
            ].contains(
                center_points.loc[idx]
            ):
                center_points.loc[idx] = (
                    projected.loc[
                        idx,
                        "geometry"
                    ].representative_point()
                )

        center_gs = gpd.GeoSeries(
            center_points,
            index=projected.index,
            crs=projected_crs
        ).to_crs(
            epsg=4326
        )

        label_points = {
            idx: point
            for idx, point
            in center_gs.items()
        }

    except Exception:
        label_points = {
            idx: row.geometry.representative_point()
            for idx, row
            in web.iterrows()
        }

    # --------------------------------------------------------
    # POLYGON + HOVER + CLICK POPUP
    # --------------------------------------------------------
    for idx, row in web.iterrows():
        nama_kecamatan = str(
            row[kolom_nama]
        )

        cluster = str(
            row["Cluster"]
        )

        warna_cluster = (
            CLUSTER_COLORS.get(
                cluster,
                "#CBD5E1"
            )
        )

        jumlah_penduduk = format_id_number(
            row["Jumlah_Penduduk"],
            2
        )

        laju_pertumbuhan = format_id_number(
            row["Laju_Pertumbuhan"],
            2
        )

        kepadatan = format_id_number(
            row["Kepadatan_Penduduk"],
            2
        )

        rasio_jk = format_id_number(
            row["Rasio_Jenis_Kelamin"],
            2
        )

        # HOVER: langsung muncul saat pointer diarahkan.
        hover_html = f"""
        <div style="
            font-family:Arial,sans-serif;
            width:265px;
            padding:11px 13px;
        ">
            <div style="
                font-size:14px;
                font-weight:700;
                color:#0f172a;
                margin-bottom:8px;
            ">
                Kecamatan {nama_kecamatan}
            </div>

            <div style="
                display:grid;
                grid-template-columns: 1fr auto;
                column-gap:14px;
                row-gap:7px;
                font-size:12px;
                color:#334155;
            ">
                <div>Tahun</div>
                <div style="font-weight:700;">{tahun}</div>

                <div>Cluster</div>
                <div style="font-weight:700;">{cluster}</div>

                <div>X1 - Jumlah Penduduk</div>
                <div style="font-weight:700;">{jumlah_penduduk}</div>

                <div>X2 - Laju Pertumbuhan</div>
                <div style="font-weight:700;">{laju_pertumbuhan}</div>

                <div>X3 - Kepadatan Penduduk</div>
                <div style="font-weight:700;">{kepadatan}</div>

                <div>X4 - Rasio Jenis Kelamin</div>
                <div style="font-weight:700;">{rasio_jk}</div>
            </div>
        </div>
        """

        tooltip = folium.Tooltip(
            hover_html,
            sticky=True,
            class_name="custom-kecamatan-tooltip"
        )

        # POPUP KLIK: style seperti kartu, isi X1-X4.
        logo_html = ""

        if logo_data_uri:
            logo_html = f"""
            <img
                src="{logo_data_uri}"
                style="
                    width:42px;
                    height:42px;
                    object-fit:contain;
                    background:white;
                    border-radius:50%;
                    padding:3px;
                    margin-right:10px;
                "
            >
            """

        popup_html = f"""
        <div style="
            font-family:Arial,sans-serif;
            width:340px;
            background:#ffffff;
        ">
            <div style="
                background:{warna_cluster};
                color:white;
                padding:13px 15px;
                display:flex;
                align-items:center;
            ">
                {logo_html}

                <div>
                    <div style="
                        font-size:15px;
                        font-weight:700;
                        line-height:1.25;
                    ">
                        Kecamatan {nama_kecamatan}
                    </div>

                    <div style="
                        font-size:11px;
                        opacity:.92;
                        margin-top:3px;
                    ">
                        Kota Tasikmalaya • Tahun {tahun} • {cluster}
                    </div>
                </div>
            </div>

            <div style="
                padding:14px 16px 16px 16px;
                color:#334155;
            ">
                <div style="
                    font-size:11px;
                    font-weight:700;
                    letter-spacing:.7px;
                    color:#64748b;
                    margin-bottom:9px;
                ">
                    DATA KEPENDUDUKAN
                </div>

                <div style="
                    display:grid;
                    grid-template-columns: 1fr auto;
                    gap:0;
                    font-size:12px;
                ">
                    <div style="padding:8px 0;border-bottom:1px solid #edf2f7;">
                        X1 - Jumlah Penduduk
                    </div>
                    <div style="padding:8px 0;border-bottom:1px solid #edf2f7;font-weight:700;color:#0f172a;">
                        {jumlah_penduduk}
                    </div>

                    <div style="padding:8px 0;border-bottom:1px solid #edf2f7;">
                        X2 - Laju Pertumbuhan
                    </div>
                    <div style="padding:8px 0;border-bottom:1px solid #edf2f7;font-weight:700;color:#0f172a;">
                        {laju_pertumbuhan}
                    </div>

                    <div style="padding:8px 0;border-bottom:1px solid #edf2f7;">
                        X3 - Kepadatan Penduduk
                    </div>
                    <div style="padding:8px 0;border-bottom:1px solid #edf2f7;font-weight:700;color:#0f172a;">
                        {kepadatan}
                    </div>

                    <div style="padding:8px 0;border-bottom:1px solid #edf2f7;">
                        X4 - Rasio Jenis Kelamin
                    </div>
                    <div style="padding:8px 0;border-bottom:1px solid #edf2f7;font-weight:700;color:#0f172a;">
                        {rasio_jk}
                    </div>
                </div>

                <div style="
                    margin-top:11px;
                    padding-top:9px;
                    border-top:1px solid #e2e8f0;
                    font-size:10.5px;
                    color:#64748b;
                    line-height:1.45;
                ">
                    Warna wilayah menunjukkan kelompok cluster
                    hasil K-Means tahun {tahun}.
                </div>
            </div>
        </div>
        """

        polygon_geojson = folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda feature, color=warna_cluster: {
                "fillColor": color,
                "color": "#FFFFFF",
                "weight": 2.4,
                "fillOpacity": 0.72,
            },
            highlight_function=lambda feature: {
                "weight": 4,
                "color": "#111827",
                "fillOpacity": 0.86,
            },
            tooltip=tooltip,
            popup=folium.Popup(
                popup_html,
                max_width=370
            )
        )

        polygon_geojson.add_to(m)

        # LABEL KECAMATAN DI TENGAH WILAYAH
        point = label_points[idx]

        label_html = f"""
        <div style="
            transform:translate(-50%,-50%);
            white-space:nowrap;
            background:rgba(15,23,42,.90);
            color:white;
            border:1px solid rgba(255,255,255,.30);
            border-radius:16px;
            padding:5px 10px;
            font-family:Arial,sans-serif;
            font-size:11px;
            font-weight:700;
            box-shadow:0 4px 12px rgba(0,0,0,.30);
            pointer-events:none;
            text-align:center;
        ">
            <span style="
                display:block;
                font-size:8px;
                letter-spacing:.65px;
                line-height:9px;
                opacity:.72;
                margin-bottom:1px;
            ">
                KECAMATAN
            </span>
            {nama_kecamatan}
        </div>
        """

        folium.Marker(
            location=[
                point.y,
                point.x
            ],
            icon=folium.DivIcon(
                html=label_html,
                icon_size=(135, 40),
                icon_anchor=(67, 20)
            )
        ).add_to(m)

    # Fullscreen di kanan atas agar search tidak terganggu.
    Fullscreen(
        position="topright",
        title="Fullscreen",
        title_cancel="Keluar Fullscreen",
        force_separate_button=True
    ).add_to(m)

    m.fit_bounds(
        [
            [miny, minx],
            [maxy, maxx]
        ],
        padding=[
            24,
            24
        ]
    )

    return m


@st.cache_data(show_spinner=False)
def buat_export_peta(tahun, format_file="png"):
    """
    Membuat file gambar peta cluster yang bisa diunduh.
    File export memuat polygon cluster, nama kecamatan,
    legenda cluster, dan judul tahun.
    """
    hasil_tahun = analisis_tahun(tahun)

    gdf, kolom_nama = load_boundary()

    data = (
        hasil_tahun[
            "result"
        ][
            "df_final"
        ]
        .copy()
    )

    data["_key"] = (
        data["Kecamatan"]
        .map(normalize_key)
    )

    gdf = gdf.merge(
        data,
        on="_key",
        how="left"
    )

    # CRS proyeksi agar bentuk dan posisi label lebih stabil.
    try:
        crs_plot = gdf.estimate_utm_crs()
        plot_gdf = gdf.to_crs(crs_plot).copy()
    except Exception:
        plot_gdf = gdf.copy()

    fig, ax = plt.subplots(
        figsize=(11, 10),
        dpi=180
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Gambar tiap cluster dengan warna yang sama seperti dashboard.
    for cluster in sorted(
        plot_gdf["Cluster"]
        .dropna()
        .unique()
    ):
        bagian = plot_gdf[
            plot_gdf["Cluster"] == cluster
        ]

        bagian.plot(
            ax=ax,
            facecolor=CLUSTER_COLORS.get(
                cluster,
                "#CBD5E1"
            ),
            edgecolor="white",
            linewidth=2.0,
            alpha=0.88
        )

    # Label kecamatan di tengah polygon.
    for _, row in plot_gdf.iterrows():
        geom = row.geometry

        try:
            point = geom.centroid

            if not geom.contains(point):
                point = geom.representative_point()

        except Exception:
            point = geom.representative_point()

        ax.text(
            point.x,
            point.y,
            str(row[kolom_nama]),
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.38",
                facecolor="#0f172a",
                edgecolor="white",
                linewidth=0.8,
                alpha=0.90
            )
        )

    # Legend
    cluster_aktif = sorted(
        plot_gdf["Cluster"]
        .dropna()
        .unique()
    )

    legend_handles = [
        Patch(
            facecolor=CLUSTER_COLORS.get(
                cluster,
                "#CBD5E1"
            ),
            edgecolor="none",
            label=(
                f"{cluster} - "
                f"{nama_profil_cluster(hasil_tahun, tahun, pendek=True).get(cluster, cluster)}"
            )
        )
        for cluster in cluster_aktif
    ]

    ax.legend(
        handles=legend_handles,
        title="Cluster",
        loc="upper right",
        frameon=True,
        fontsize=9,
        title_fontsize=9
    )

    ax.set_title(
        f"Peta Cluster Kecamatan Kota Tasikmalaya\nTahun {tahun}",
        fontsize=15,
        fontweight="bold",
        pad=16
    )

    ax.set_axis_off()

    # Batas peta dibuat sedikit longgar.
    minx, miny, maxx, maxy = plot_gdf.total_bounds
    dx = maxx - minx
    dy = maxy - miny

    ax.set_xlim(
        minx - (0.04 * dx),
        maxx + (0.04 * dx)
    )

    ax.set_ylim(
        miny - (0.04 * dy),
        maxy + (0.04 * dy)
    )

    fig.tight_layout()

    buffer_png = io.BytesIO()

    fig.savefig(
        buffer_png,
        format="png",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    buffer_png.seek(0)

    if format_file.lower() == "png":
        return buffer_png.getvalue()

    # Konversi ke JPEG.
    from PIL import Image

    image = Image.open(
        buffer_png
    ).convert("RGB")

    buffer_jpg = io.BytesIO()

    image.save(
        buffer_jpg,
        format="JPEG",
        quality=95,
        optimize=True
    )

    buffer_jpg.seek(0)

    return buffer_jpg.getvalue()


# ============================================================
# 7. SIDEBAR
# ============================================================
with st.sidebar:
    # Logo Kota Tasikmalaya: center dan sedikit lebih ke atas.
    if LOGO_PATH.exists():
        logo_sidebar_base64 = base64.b64encode(
            LOGO_PATH.read_bytes()
        ).decode("utf-8")

        logo_html = (
            f'<div style="width:100%;display:flex;justify-content:center;'
            f'align-items:center;margin:-14px 0 16px 0;">'
            f'<img src="data:image/png;base64,{logo_sidebar_base64}" '
            f'style="width:112px;height:auto;display:block;">'
            f'</div>'
        )

        st.markdown(
            logo_html,
            unsafe_allow_html=True
        )

    # Judul utama.
    title_html = (
        '<div style="text-align:center;padding:0 8px;margin:0 0 4px 0;'
        'color:#ffffff;font-size:16px;font-weight:700;line-height:1.38;">'
        'Klasterisasi Kecamatan<br>Kota Tasikmalaya'
        '</div>'
    )

    st.markdown(
        title_html,
        unsafe_allow_html=True
    )

    # Subjudul dibuat seperti tulisan biasa, tipis, tanpa italic.
    subtitle_html = (
        '<div style="text-align:center;padding:0 6px;margin:0 0 14px 0;'
        'color:#b8c5d1;font-weight:400;line-height:1.45;">'
        '<div style="font-size:11.4px;">'
        'Berdasarkan Karakteristik Kependudukan'
        '</div>'
        '<div style="font-size:11.2px;margin-top:2px;">'
        'Tahun 2024–2026'
        '</div>'
        '</div>'
    )

    st.markdown(
        subtitle_html,
        unsafe_allow_html=True
    )

    # Pemisah halus antara identitas dashboard dan menu navigasi.
    st.markdown(
        '<div style="height:1px;background:rgba(255,255,255,.14);'
        'margin:0 10px 16px 10px;"></div>',
        unsafe_allow_html=True
    )

    menu_labels = {
        "Dashboard":
            "🏠  Dashboard",

        "Data Kependudukan":
            "📋  Data Kependudukan",

        "Evaluasi K":
            "📊  Evaluasi K",

        "Hasil Clustering":
            "🧩  Hasil Clustering",

        "Peta Cluster":
            "🗺️  Peta Cluster",

        "Perbandingan Tahun":
            "🔄  Perbandingan Tahun",
    }

    menu = st.radio(
        "Navigasi",
        list(menu_labels.keys()),
        format_func=lambda item:
            menu_labels[item],
        label_visibility="collapsed"
    )


# ============================================================
# 8. HELPER UI
# ============================================================
def tahun_selector(
    key
):
    return st.selectbox(
        "Pilih Tahun",
        TAHUN_LIST,
        key=key
    )


def format_dataframe_data(
    df
):
    result = df.copy()

    rename = {
        "Jumlah_Penduduk":
            "Jumlah Penduduk",
        "Laju_Pertumbuhan":
            "Laju Pertumbuhan",
        "Persentase_Penduduk":
            "Persentase Penduduk",
        "Kepadatan_Penduduk":
            "Kepadatan Penduduk",
        "Rasio_Jenis_Kelamin":
            "Rasio Jenis Kelamin",
    }

    return result.rename(
        columns=rename
    )


def buat_deskripsi_cluster(hasil, tahun=None):
    """Nama dan rincian profil untuk legenda peta, konsisten dengan dashboard."""
    nama = nama_profil_cluster(hasil, tahun)
    rincian = rincian_profil(hasil)
    return {
        cluster: {
            "title": label,
            "short_title": label,
            "desc": rincian.get(cluster, "-"),
        }
        for cluster, label in nama.items()
    }


def analisis_kualitas_csv(df):
    """
    Pemeriksaan kualitas yang dapat digunakan untuk CSV apa pun.
    Tidak mengubah isi dataframe.
    """
    jumlah_baris = len(df)
    jumlah_kolom = len(df.columns)
    total_sel = jumlah_baris * jumlah_kolom

    missing_per_col = df.isna().sum()
    total_missing = int(missing_per_col.sum())

    duplicate_rows = int(
        df.duplicated(
            keep=False
        ).sum()
    )

    duplicate_rows_exact = int(
        df.duplicated().sum()
    )

    duplicate_column_names = (
        len(df.columns)
        -
        len(pd.Index(df.columns).unique())
    )

    unnamed_columns = [
        str(col)
        for col in df.columns
        if str(col).strip().lower().startswith(
            "unnamed:"
        )
        or str(col).strip() == ""
    ]

    ringkasan_kolom = []

    for col in df.columns:
        series = df[col]

        ringkasan_kolom.append({
            "Kolom":
                str(col),

            "Tipe Data":
                str(series.dtype),

            "Missing":
                int(series.isna().sum()),

            "Missing (%)":
                round(
                    (
                        series.isna().mean()
                        * 100
                    ),
                    2
                ),

            "Nilai Unik":
                int(
                    series.nunique(
                        dropna=True
                    )
                ),
        })

    return {
        "jumlah_baris":
            jumlah_baris,

        "jumlah_kolom":
            jumlah_kolom,

        "total_sel":
            total_sel,

        "total_missing":
            total_missing,

        "duplicate_rows":
            duplicate_rows,

        "duplicate_rows_exact":
            duplicate_rows_exact,

        "duplicate_column_names":
            duplicate_column_names,

        "unnamed_columns":
            unnamed_columns,

        "ringkasan_kolom":
            pd.DataFrame(
                ringkasan_kolom
            ),
    }


def cek_dataset_penelitian_upload(df_raw):
    """
    Mendeteksi apakah CSV yang diunggah sesuai struktur
    dataset kependudukan penelitian K-Means.
    """
    df = df_raw.copy()

    df.columns = [
        normalize_text(
            str(col).replace(
                "\n",
                " "
            )
        )
        for col in df.columns
    ]

    df = rename_columns_auto(
        df
    )

    kolom_wajib = [
        "Kecamatan"
    ] + INDIKATOR_EDA

    kolom_hilang = [
        col
        for col in kolom_wajib
        if col not in df.columns
    ]

    if kolom_hilang:
        return {
            "cocok":
                False,

            "kolom_hilang":
                kolom_hilang,

            "df":
                df,

            "validasi":
                {},
        }

    df_check = df[
        kolom_wajib
    ].copy()

    df_check["Kecamatan"] = (
        df_check[
            "Kecamatan"
        ]
        .astype(str)
        .map(normalize_text)
    )

    # Cek duplikasi sebelum data dibersihkan.
    jumlah_duplikat_kecamatan = int(
        df_check.duplicated(
            subset=["Kecamatan"],
            keep=False
        ).sum()
    )

    # Konversi semua indikator agar kualitas numeriknya bisa diperiksa.
    for col in INDIKATOR_EDA:
        integerish = col in [
            "Jumlah_Penduduk",
            "Kepadatan_Penduduk",
        ]

        df_check[col] = (
            to_numeric_safe(
                df_check[col],
                integerish=integerish
            )
        )

    kecamatan_upload = {
        normalize_key(k)
        for k in df_check[
            "Kecamatan"
        ]
    }

    kecamatan_target = {
        normalize_key(k)
        for k in KECAMATAN_TARGET
    }

    kecamatan_hilang = sorted(
        [
            k
            for k in KECAMATAN_TARGET
            if normalize_key(k)
            not in kecamatan_upload
        ]
    )

    kecamatan_tambahan = sorted(
        [
            str(k)
            for k in df_check[
                "Kecamatan"
            ].dropna().unique()
            if normalize_key(k)
            not in kecamatan_target
        ]
    )

    validasi = {
        "Struktur kolom sesuai dataset penelitian":
            len(
                kolom_hilang
            ) == 0,

        "Jumlah kecamatan = 10":
            df_check[
                "Kecamatan"
            ].nunique()
            == 10,

        "Tidak ada duplikasi kecamatan":
            jumlah_duplikat_kecamatan
            == 0,

        "Seluruh 10 kecamatan Kota Tasikmalaya tersedia":
            len(
                kecamatan_hilang
            )
            == 0,

        "Tidak ada missing value pada indikator":
            df_check[
                INDIKATOR_EDA
            ]
            .isna()
            .sum()
            .sum()
            == 0,

        "X1–X4 dapat dibaca sebagai numerik":
            all(
                pd.api.types.is_numeric_dtype(
                    df_check[col]
                )
                for col in FITUR_KMEANS
            )
            and
            df_check[
                FITUR_KMEANS
            ].isna().sum().sum()
            == 0,
    }

    siap_kmeans = all(
        validasi.values()
    )

    return {
        "cocok":
            True,

        "kolom_hilang":
            [],

        "df":
            df_check,

        "validasi":
            validasi,

        "siap_kmeans":
            siap_kmeans,

        "jumlah_duplikat_kecamatan":
            jumlah_duplikat_kecamatan,

        "kecamatan_hilang":
            kecamatan_hilang,

        "kecamatan_tambahan":
            kecamatan_tambahan,
    }


def buat_tabel_perbandingan_antar_tahun(semua_hasil):
    """Bandingkan label yang diselaraskan ke profil centroid tahun 2024.

    K yang berbeda tidak diberi status Stabil/Berubah karena nomor cluster
    tidak sebanding. Penyelarasan hanya mengubah label untuk komparasi,
    bukan hasil model K-Means pada setiap tahun.
    """
    tahun_dasar = 2024
    dasar = semua_hasil[tahun_dasar]
    centroid_dasar = dasar["result"]["centroid_z"]
    matriks_dasar = centroid_dasar[NAMA_X].to_numpy(dtype=float)
    k_dasar = int(dasar["k_optimal"])
    tabel = pd.DataFrame(index=sorted(KECAMATAN_TARGET))
    k_berbeda = False

    for tahun_kolom in TAHUN_LIST:
        data_tahun = semua_hasil[tahun_kolom]
        k_tahun = int(data_tahun["k_optimal"])
        data_cluster = data_tahun["result"]["df_final"][
            ["Kecamatan", "Cluster"]
        ].copy()
        if tahun_kolom != tahun_dasar and k_tahun == k_dasar:
            centroid = data_tahun["result"]["centroid_z"]
            matriks = centroid[NAMA_X].to_numpy(dtype=float)
            jarak = np.linalg.norm(
                matriks[:, None, :] - matriks_dasar[None, :, :], axis=2
            )
            baris, kolom = linear_sum_assignment(jarak)
            peta = {
                str(centroid.iloc[i]["Cluster"]):
                    str(centroid_dasar.iloc[j]["Cluster"])
                for i, j in zip(baris, kolom)
            }
            data_cluster["Cluster"] = data_cluster["Cluster"].map(peta)
        elif tahun_kolom != tahun_dasar and k_tahun != k_dasar:
            k_berbeda = True
        tabel[str(tahun_kolom)] = (
            data_cluster.set_index("Kecamatan")["Cluster"]
        )

    tabel.index.name = "Kecamatan"
    tabel = tabel.reset_index()
    kolom_tahun = [str(t) for t in TAHUN_LIST]
    if k_berbeda:
        tabel["Status"] = "Perlu interpretasi"
    else:
        tabel["Status"] = tabel[kolom_tahun].apply(
            lambda baris: "Stabil" if baris.nunique() == 1 else "Berubah",
            axis=1
        )
    return tabel, k_berbeda


def buat_tabel_perbandingan_visual(tabel, tahun_acuan="2024"):
    """Tabel perbandingan netral: tanpa badge, warna C1/C2, atau penanda tahun."""
    kolom_tahun = [str(t) for t in TAHUN_LIST]
    header = ['<th scope="col">Kecamatan</th>']
    for tahun in kolom_tahun:
        header.append(f'<th scope="col">{html_escape(tahun)}</th>')
    header.append('<th scope="col">Status</th>')

    baris_html = []
    for _, baris in tabel.iterrows():
        kolom = [
            f'<th scope="row">{html_escape(str(baris.get("Kecamatan", "")))}</th>'
        ]
        for tahun in kolom_tahun:
            kode = html_escape(str(baris.get(tahun, "-")))
            kolom.append(f'<td>{kode}</td>')
        status = html_escape(str(baris.get("Status", "-")))
        kolom.append(f'<td>{status}</td>')
        baris_html.append('<tr>' + ''.join(kolom) + '</tr>')

    return '''
    <style>
    .kp-compare-scroll {overflow-x:auto;border:1px solid #e0e5eb;
        border-radius:12px;background:#fff;margin:8px 0 17px;}
    .kp-compare-table {width:100%;border-collapse:collapse;min-width:580px;
        table-layout:fixed;font-size:14px;color:#263746;}
    .kp-compare-table th, .kp-compare-table td {padding:12px 14px;
        border-bottom:1px solid #e0e5eb;}
    .kp-compare-table thead th {background:#f7f8fa;color:#475569;
        font-weight:600;text-align:center;}
    .kp-compare-table thead th:first-child {text-align:left;}
    .kp-compare-table tbody th {text-align:left;font-weight:500;width:29%;}
    .kp-compare-table tbody td {text-align:center;font-weight:400;}
    .kp-compare-table tbody tr:last-child th,
    .kp-compare-table tbody tr:last-child td {border-bottom:0;}
    @media(max-width:700px) {
        .kp-compare-table {font-size:12px;}
        .kp-compare-table th,.kp-compare-table td {padding:10px 8px;}
    }
    </style>
    <div class="kp-compare-scroll">
      <table class="kp-compare-table" aria-label="Perbandingan cluster kecamatan 2024 sampai 2026">
       <thead><tr>''' + ''.join(header) + '''</tr></thead>
       <tbody>''' + ''.join(baris_html) + '''</tbody>
      </table>
    </div>
    '''


def kategori_variabel_cluster(hasil_tahun, cluster):
    """Ambil X1-X4 dari peringkat centroid yang sama dengan Profil Cluster."""
    rincian = rincian_profil(hasil_tahun).get(cluster, "")
    kategori = {}
    for nama_x in NAMA_X:
        match = re.search(rf"{nama_x}\s*\([^)]*\)\s*([^;]+)", rincian)
        kategori[nama_x] = match.group(1).strip() if match else "-"
    return kategori


def catatan_x1_antar_tahun(semua_hasil, tahun):
    """Catat jika pola X1 berubah dari 2024 pada tipe kawasan yang sama."""
    if semua_hasil is None or tahun == 2024:
        return None
    if int(semua_hasil[2024]["k_optimal"]) != int(semua_hasil[tahun]["k_optimal"]):
        return None
    dasar, terkini = semua_hasil[2024], semua_hasil[tahun]
    def x1_per_nama(h):
        nama = nama_profil_cluster(h)
        return {label: kategori_variabel_cluster(h, cluster).get("X1")
                for cluster, label in nama.items()}
    dulu, sekarang = x1_per_nama(dasar), x1_per_nama(terkini)
    if set(dulu) == set(sekarang) and any(dulu[n] != sekarang[n] for n in dulu):
        return ("Pola X1 berubah dibandingkan 2024. Penamaan kelompok "
                "tetap didasarkan pada pertumbuhan (X2) dan kepadatan (X3).")
    return None


def kartu_cluster_dashboard(hasil_tahun, tahun, cluster, catatan=None):
    """Kartu HTML presentasi: nama, X1-X4 dan anggota, semuanya dari model aktif."""
    hasil = hasil_tahun["result"]
    data_anggota = hasil["df_final"]
    anggota = sorted(
        data_anggota.loc[
            data_anggota["Cluster"] == cluster, "Kecamatan"
        ].astype(str).tolist()
    )
    jumlah = len(anggota)
    kategori = kategori_variabel_cluster(hasil_tahun, cluster)
    nama = nama_profil_cluster(hasil_tahun, tahun).get(cluster, cluster)
    judul_kartu, subjudul_kartu = judul_subjudul_cluster(hasil_tahun, cluster)
    warna = CLUSTER_COLORS.get(cluster, "#62748A")
    warna_lunak = CLUSTER_LIGHT_COLORS.get(cluster, "#FFF0E0")
    text_warna = CLUSTER_TEXT_COLORS.get(cluster, "#98541D")

    rows = ""
    for x in NAMA_X:
        keterangan = html_escape(KETERANGAN_X[x])
        nilai = html_escape(str(kategori.get(x, "-")))
        baris_kelas = "is-high" if nilai.lower() == "tinggi" else "is-low" if nilai.lower() == "rendah" else "is-mid"
        rows += (
            '<div class="kp-var-row">'
            f'<span class="kp-var-name">{x} {keterangan}</span>'
            f'<span class="kp-value-pill {baris_kelas}">{nilai}</span>'
            '</div>'
        )
    anggota_aman = ", ".join(html_escape(kecamatan) for kecamatan in anggota)
    catatan_html = (
        f'<div class="kp-card-note">{html_escape(str(catatan))}</div>'
        if catatan else ""
    )
    return (
        f'<div class="kp-profile-card" style="--kp-accent:{warna};'
        f'--kp-soft:{warna_lunak};--kp-text:{text_warna};">'
        f'<div class="kp-label">{html_escape(cluster)}</div>'
        f'<div class="kp-card-title">{html_escape(str(judul_kartu))}</div>'
        f'<div class="kp-card-subtitle">{html_escape(str(subjudul_kartu))}</div>'
        f'<div class="kp-var-list">{rows}</div>'
        f'{catatan_html}'
        f'<div class="kp-card-members"><strong>{jumlah} kecamatan:</strong> '
        f'{anggota_aman}</div>'
        '</div>'
    )


# ============================================================
# 9. DASHBOARD UTAMA — RINGKASAN PENELITIAN 2024–2026
# ============================================================
# Dashboard menampilkan ringkasan saja. Rincian X1–X4, centroid,
# anggota kecamatan, dan peta tersedia di halaman khusus.

def buat_kartu_ikhtisar_dashboard(angka, judul, satuan, ikon):
    """Kartu informasi umum penelitian; bukan nilai hasil clustering."""
    return (
        '<div class="kp-overview-kpi">'
        f'<div class="kp-overview-icon">{html_escape(ikon)}</div>'
        '<div class="kp-overview-detail">'
        f'<div class="kp-overview-num">{html_escape(str(angka))}</div>'
        f'<div class="kp-overview-title">{html_escape(judul)}</div>'
        f'<div class="kp-overview-unit">{html_escape(satuan)}</div>'
        '</div></div>'
    )


def buat_kartu_ringkasan_cluster(hasil_tahun, tahun, cluster):
    """Ringkasan singkat berdasarkan model dan centroid tahun aktif."""
    df = hasil_tahun['result']['df_final']
    jumlah = int((df['Cluster'] == cluster).sum())
    nama = nama_profil_cluster(hasil_tahun, tahun).get(cluster, cluster)
    kategori = kategori_variabel_cluster(hasil_tahun, cluster)
    if kategori.get('X2') == 'tinggi' and kategori.get('X3') == 'rendah':
        judul = 'Pertumbuhan Tinggi'
    elif kategori.get('X3') == 'tinggi' and kategori.get('X2') == 'rendah':
        judul = 'Kepadatan Tinggi'
    else:
        judul = 'Profil Kependudukan'
    warna = CLUSTER_COLORS.get(cluster, '#6B7280')
    warna_lunak = CLUSTER_LIGHT_COLORS.get(cluster, '#F8FAFC')
    warna_teks = CLUSTER_TEXT_COLORS.get(cluster, '#475569')
    return (
        f'<div class="kp-cluster-summary" style="--kp-color:{warna};'
        f'--kp-soft:{warna_lunak};--kp-text:{warna_teks}">'
        f'<div class="kp-summary-code">{html_escape(cluster)}</div>'
        f'<div class="kp-summary-heading">{html_escape(cluster)} — {html_escape(judul)}</div>'
        f'<div class="kp-summary-countval">{jumlah}</div>'
        '<div class="kp-summary-unit">Kecamatan</div>'
        f'<div class="kp-summary-description">{html_escape(nama)}</div>'
        '</div>'
    )


@st.cache_data(show_spinner=False)
def data_kependudukan_tahun(tahun):
    """Baca data penelitian asli: 10 kecamatan; abaikan total kota/catatan kaki."""
    path = FILE_PATHS[int(tahun)]
    if not path.exists():
        raise FileNotFoundError(
            f"File {path.name} tidak ditemukan di folder dataset/."
        )
    raw = read_csv_smart(path)
    bersih = preprocessing(raw, int(tahun))
    checks = validasi_dataframe(bersih)
    if not all(checks.values()):
        gagal = [nama for nama, oke in checks.items() if not oke]
        raise ValueError(
            f"Data tahun {tahun} belum valid: {', '.join(gagal)}"
        )
    return bersih


def tabel_data_kependudukan(tahun_filter="Semua Tahun", pencarian=""):
    """Data tabel: 2024–2026 atau satu tahun, tanpa mengubah nilai CSV."""
    tahun_dipakai = (
        TAHUN_LIST if tahun_filter == "Semua Tahun"
        else [int(tahun_filter)]
    )
    data = pd.concat(
        [data_kependudukan_tahun(t) for t in tahun_dipakai],
        ignore_index=True,
    )
    cari = str(pencarian).strip()
    if cari:
        data = data.loc[
            data["Kecamatan"].str.contains(cari, case=False, regex=False, na=False)
        ].copy()

    # Jumlah Penduduk di CSV satuannya RIBU JIWA, tidak dikalikan 1.000.
    return data.rename(columns={
        "Tahun": "Tahun",
        "Kecamatan": "Kecamatan",
        "Jumlah_Penduduk": "Jumlah Penduduk (Ribu Jiwa)",
        "Laju_Pertumbuhan": "Laju Pertumbuhan (%/Tahun)",
        "Persentase_Penduduk": "Persentase Penduduk (%)",
        "Kepadatan_Penduduk": "Kepadatan Penduduk (Jiwa/km²)",
        "Rasio_Jenis_Kelamin": "Rasio Jenis Kelamin",
    })[[
        "Tahun", "Kecamatan", "Jumlah Penduduk (Ribu Jiwa)",
        "Laju Pertumbuhan (%/Tahun)", "Persentase Penduduk (%)",
        "Kepadatan Penduduk (Jiwa/km²)", "Rasio Jenis Kelamin",
    ]].reset_index(drop=True)

def hitung_total_anggota_cluster(semua_hasil):
    """Hitung semua kecamatan-tahun (bukan jumlah kecamatan unik)."""
    total = {}
    for tahun in TAHUN_LIST:
        df = semua_hasil[tahun]["result"]["df_final"]
        jumlah = df["Cluster"].astype(str).value_counts()
        for label, n in jumlah.items():
            total[label] = total.get(label, 0) + int(n)
    urut = sorted(total, key=lambda c: int(c[1:]) if c.startswith("C") and c[1:].isdigit() else 999)
    output = {cluster: total[cluster] for cluster in urut}
    output["Total"] = int(sum(total.values()))
    return output


def tampilkan_total_anggota_cluster(semua_hasil):
    """Kartu total anggota seluruh tahun, dihitung otomatis dari cluster final.

    Banyaknya anggota adalah observasi kecamatan-tahun, bukan kecamatan unik.
    Semua kartu C1, C2, C3, dan seterusnya mengikuti hasil model aktif.
    """
    akumulasi = hitung_total_anggota_cluster(semua_hasil)
    total = int(akumulasi.get("Total", 0))
    kode_cluster = [kode for kode in akumulasi if kode != "Total"]

    st.subheader("Anggota Cluster dari Keseluruhan Data (2024–2026)")
    st.markdown(
        '<p class="kp-total-intro">Akumulasi jumlah keanggotaan cluster dari '
        '10 kecamatan pada masing-masing tahun penelitian.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <style>
    .kp-total-intro {
        font-size: 14px;
        line-height: 1.55;
        color: #64748b;
        margin: 0 0 14px;
    }
    .kp-total-card {
        --card-accent: #4e79a7;
        --card-soft: #e7eff7;
        --card-ink: #365f85;
        border: 1px solid #e2e8f0;
        border-top: 5px solid var(--card-accent);
        border-radius: 16px;
        background: #ffffff;
        padding: 19px 20px 17px;
        min-height: 200px;
        height: 100%;
        box-shadow: 0 5px 20px rgba(15,41,66,.06);
        display: flex;
        flex-direction: column;
        gap: 0;
        margin: 0 0 10px;
    }
    .kp-total-topline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
    }
    .kp-total-cluster-name {
        display: inline-flex;
        padding: 5px 12px;
        background: var(--card-soft);
        border-radius: 999px;
        color: var(--card-ink);
        font-weight: 800;
        font-size: 14px;
        line-height: 1.3;
    }
    .kp-total-share {
        font-size: 13px;
        font-weight: 700;
        color: #64748b;
    }
    .kp-total-number {
        font-size: clamp(30px, 3.5vw, 43px);
        line-height: 1.12;
        font-weight: 800;
        color: #122b3c;
        letter-spacing: -1px;
        margin: 18px 0 3px;
    }
    .kp-total-description {
        color: #596c7b;
        font-size: 13px;
        margin: 0 0 19px;
    }
    .kp-total-meter {
        height: 7px;
        background: #e9eef3;
        border-radius: 99px;
        overflow: hidden;
        margin-top: auto;
    }
    .kp-total-meter span {
        display: block;
        height: 100%;
        border-radius: 99px;
        background: var(--card-accent);
    }
    .kp-total-bottom {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 9px;
    }
    </style>
    """, unsafe_allow_html=True)

    kartu = [(kode, int(akumulasi[kode])) for kode in kode_cluster]
    kartu.append(("Total", total))

    # Maksimal empat kartu per baris agar tetap nyaman dibaca pada laptop.
    for awal in range(0, len(kartu), 4):
        kelompok = kartu[awal:awal + 4]
        kolom_kartu = st.columns(len(kelompok), gap="medium")
        for kol, (kode, jumlah) in zip(kolom_kartu, kelompok):
            if kode == "Total":
                warna, lunak, teks = "#0F2942", "#EDF2F7", "#0F2942"
                label = "Total Data"
                keterangan = "Observasi seluruh tahun"
            else:
                warna = CLUSTER_COLORS.get(kode, "#6B7280")
                lunak = CLUSTER_LIGHT_COLORS.get(kode, "#F1F5F9")
                teks = CLUSTER_TEXT_COLORS.get(kode, warna)
                label = kode
                keterangan = "Observasi kecamatan-tahun"

            persentase = 100.0 * jumlah / total if total > 0 else 0.0
            bagian_persen = f"{persentase:.1f}".replace(".", ",")
            bagian_total = "100%" if kode == "Total" else f"{bagian_persen}%"
            card = (
                f'<div class="kp-total-card" style="--card-accent:{html_escape(warna)};'
                f'--card-soft:{html_escape(lunak)};--card-ink:{html_escape(teks)};">'
                '<div class="kp-total-topline">'
                f'<span class="kp-total-cluster-name">{html_escape(label)}</span>'
                f'<span class="kp-total-share">{bagian_total}</span>'
                '</div>'
                f'<div class="kp-total-number">{jumlah}</div>'
                f'<div class="kp-total-description">{keterangan}</div>'
                '<div class="kp-total-meter">'
                f'<span style="width:{persentase:.2f}%;"></span>'
                '</div>'
                '<div class="kp-total-bottom">Periode 2024–2026</div>'
                '</div>'
            )
            with kol:
                st.markdown(card, unsafe_allow_html=True)

    st.caption(
        "Satu kecamatan dihitung sekali pada setiap tahun. "
        f"Total {total} observasi kecamatan-tahun, bukan {total} kecamatan berbeda."
    )
    if len({int(semua_hasil[t]["k_optimal"]) for t in TAHUN_LIST}) > 1:
        st.warning(
            "Jumlah K optimal berbeda antartahun. Akumulasi C1/C2/dst. "
            "merupakan rekap label; makna cluster harus dibandingkan "
            "berdasarkan profil dan penyelarasan antartahun."
        )


if menu == "Dashboard":
    st.title("Dashboard Kependudukan Kota Tasikmalaya")
    st.markdown(
        '<p class="page-subtitle">Ringkasan karakteristik kependudukan dan '
        'hasil klasterisasi K-Means tahun 2024–2026</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <style>
    .kp-overview-kpi {
      display:flex; gap:14px; align-items:center; min-height:112px;
      padding:19px 20px; background:#fff; border:1px solid #e0e7ed;
      border-radius:15px; box-shadow:0 3px 12px rgba(15,35,45,.05);
      margin-bottom:8px;
    }
    .kp-overview-icon { display:grid; place-items:center; width:48px; height:48px;
      flex-shrink:0; border-radius:13px; background:#eef2f7; font-size:24px; }
    .kp-overview-num { font-size:29px; line-height:1.0; font-weight:800; color:#102b3b; }
    .kp-overview-title { margin-top:6px; font-size:13px; color:#344d60; font-weight:700; }
    .kp-overview-unit { font-size:12px; color:#738696; }
    .kp-cluster-summary {
      background:linear-gradient(180deg,var(--kp-soft) 0%,#fff 48%);
      border:1px solid #dfe6ec; border-top:5px solid var(--kp-color);
      border-radius:15px; padding:22px 23px; min-height:218px;
      display:flex; flex-direction:column; margin-bottom:10px;
      box-shadow:0 3px 12px rgba(15,35,45,.05);
    }
    .kp-summary-code { font-size:13px; font-weight:800; color:var(--kp-color); }
    .kp-summary-heading { font-size:17px; line-height:1.35; font-weight:750;
      color:var(--kp-text); margin:7px 0 11px; }
    .kp-summary-countval { font-size:38px; line-height:1; font-weight:800; color:var(--kp-text); }
    .kp-summary-unit { color:#5e7181; font-size:12px; margin-top:3px; }
    .kp-summary-description { margin-top:auto; padding-top:13px; font-size:13px;
      line-height:1.4; color:#445d6b; }
    @media(max-width:760px) {
      .kp-overview-kpi { min-height:96px; padding:13px; }
      .kp-overview-num { font-size:24px; }
      .kp-cluster-summary { min-height:185px; padding:16px; }
    }
    </style>
    """, unsafe_allow_html=True)

    # 1. Angka ringkas tentang cakupan penelitian, tidak dipengaruhi nilai cluster.
    a, b, c = st.columns(3, gap='medium')
    with a:
        st.markdown(buat_kartu_ikhtisar_dashboard(
            len(KECAMATAN_TARGET), 'Wilayah', 'Kecamatan', '📍'), unsafe_allow_html=True)
    with b:
        st.markdown(buat_kartu_ikhtisar_dashboard(
            len(TAHUN_LIST), 'Periode Analisis', 'Tahun', '📅'), unsafe_allow_html=True)
    with c:
        st.markdown(buat_kartu_ikhtisar_dashboard(
            len(FITUR_KMEANS), 'Indikator', 'Variabel', '📊'), unsafe_allow_html=True)

    try:
        # Seluruh nilai visual dihitung dari CSV 2024, 2025, 2026.
        semua_hasil = get_all_results()

        # 2. Perbandingan jumlah anggota cluster dalam setiap tahun.
        st.subheader('Distribusi Cluster Tahun 2024–2026')
        data_grafik = []
        for tahun_grafik in TAHUN_LIST:
            ht = semua_hasil[tahun_grafik]
            jumlah_per_cluster = ht['result']['df_final']['Cluster'].value_counts()
            for kode in [f'C{i + 1}' for i in range(int(ht['k_optimal']))]:
                data_grafik.append({
                    'Tahun': str(tahun_grafik),
                    'Cluster': kode,
                    'Jumlah Kecamatan': int(jumlah_per_cluster.get(kode, 0)),
                })
        df_distribusi = pd.DataFrame(data_grafik)
        fig = px.bar(
            df_distribusi, x='Tahun', y='Jumlah Kecamatan', color='Cluster',
            barmode='group', text='Jumlah Kecamatan',
            color_discrete_map=CLUSTER_COLORS,
            category_orders={
                'Tahun': [str(t) for t in TAHUN_LIST],
                'Cluster': [f'C{i + 1}' for i in range(K_MAX)],
            },
        )
        fig.update_traces(textposition='outside', cliponaxis=False)
        fig.update_layout(
            height=335, margin=dict(l=5, r=8, t=15, b=5),
            legend_title_text='Cluster',
            yaxis=dict(range=[0, 10.7], dtick=2, title='Jumlah Kecamatan'),
            xaxis_title=None,
            bargap=0.28,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Grafik menunjukkan banyaknya kecamatan pada setiap cluster, bukan jumlah penduduk.')

        # 3. Pemilih tahun dan kartu C1/C2 dari hasil perhitungan tahun tersebut.
        st.subheader('Ringkasan Cluster')
        if hasattr(st, 'segmented_control'):
            tahun = st.segmented_control(
                'Pilih Tahun', options=TAHUN_LIST, default=2024,
                selection_mode='single', label_visibility='collapsed',
                key='ringkasan_cluster_tahun',
            )
            if tahun is None:
                tahun = 2024
        else:
            tahun = st.radio(
                'Pilih Tahun', options=TAHUN_LIST, horizontal=True,
                label_visibility='collapsed', key='ringkasan_cluster_tahun',
            )
        h_aktif = semua_hasil[tahun]
        k_final = int(h_aktif['k_optimal'])
        st.caption(f'Tahun {tahun} · K optimal = {k_final} (Silhouette tertinggi; Elbow sebagai konfirmasi).')
        kode_cluster = [f'C{i + 1}' for i in range(k_final)]
        for i in range(0, len(kode_cluster), 2):
            kolom = st.columns(2, gap='medium')
            for j, kode in enumerate(kode_cluster[i:i + 2]):
                with kolom[j]:
                    st.markdown(
                        buat_kartu_ringkasan_cluster(h_aktif, tahun, kode),
                        unsafe_allow_html=True,
                    )

        # 4. Ringkasan keseluruhan C1/C2, bukan status stabilitas.
        tampilkan_total_anggota_cluster(semua_hasil)
        st.caption(
            'Nama kawasan dan kategori tinggi/rendah bersifat relatif terhadap '
            'centroid antarklaster, bukan klasifikasi resmi BPS. Detail X1–X4, '
            'centroid, dan daftar kecamatan tersedia di menu Hasil Clustering.'
        )
    except Exception as exc:
        st.error(f'Data belum bisa dimuat untuk dashboard: {exc}')
        st.info('Pastikan ketiga file kependudukan_2024.csv, '
                'kependudukan_2025.csv, dan kependudukan_2026.csv '
                'tersedia dalam folder dataset dan memiliki kolom yang sesuai.')

# ============================================================
# 10. DATA KEPENDUDUKAN — TABEL TAHUNAN DARI CSV ASLI
# ============================================================
elif menu == "Data Kependudukan":
    st.title("Data Kependudukan Kota Tasikmalaya")
    st.markdown(
        '<p class="page-subtitle">Data penduduk 10 kecamatan Kota Tasikmalaya '
        'untuk tahun 2024–2026 berdasarkan CSV penelitian. '
        'Pilih tahun untuk melihat rincian datanya.</p>',
        unsafe_allow_html=True,
    )

    kiri, kanan = st.columns([1, 2], gap="medium")
    with kiri:
        filter_tahun = st.selectbox(
            "Filter Tahun",
            options=["Semua Tahun", "2024", "2025", "2026"],
            index=1,
            key="data_kependudukan_filter_tahun",
        )
    with kanan:
        cari_kecamatan = st.text_input(
            "Cari Kecamatan",
            placeholder="Contoh: Cihideung (opsional)",
            key="data_kependudukan_cari",
        )

    try:
        data_tampil = tabel_data_kependudukan(filter_tahun, cari_kecamatan)
        st.caption(
            f"Menampilkan {len(data_tampil)} baris untuk "
            f"{filter_tahun.lower() if filter_tahun == 'Semua Tahun' else 'tahun ' + filter_tahun}. "
            "Kecamatan yang ditampilkan berasal dari dataset penelitian."
        )
        if data_tampil.empty:
            st.info("Tidak ada kecamatan yang cocok dengan pencarian.")
        else:
            st.dataframe(
                data_tampil,
                use_container_width=True,
                hide_index=True,
                height=min(520, 48 + 35 * len(data_tampil)),
                column_config={
                    "Tahun": st.column_config.NumberColumn("Tahun", format="%d"),
                    "Kecamatan": st.column_config.TextColumn("Kecamatan", width="medium"),
                    "Jumlah Penduduk (Ribu Jiwa)": st.column_config.NumberColumn(
                        "Jumlah Penduduk (Ribu Jiwa)", format="%.2f"
                    ),
                    "Laju Pertumbuhan (%/Tahun)": st.column_config.NumberColumn(
                        "Laju Pertumbuhan (%/Tahun)", format="%.2f"
                    ),
                    "Persentase Penduduk (%)": st.column_config.NumberColumn(
                        "Persentase Penduduk (%)", format="%.2f"
                    ),
                    "Kepadatan Penduduk (Jiwa/km²)": st.column_config.NumberColumn(
                        "Kepadatan Penduduk (Jiwa/km²)", format="%.0f"
                    ),
                    "Rasio Jenis Kelamin": st.column_config.NumberColumn(
                        "Rasio Jenis Kelamin", format="%.2f"
                    ),
                },
            )

            file_tahun = ("2024_2026" if filter_tahun == "Semua Tahun"
                          else filter_tahun)
            csv_output = data_tampil.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Unduh Data yang Ditampilkan (CSV)",
                data=csv_output,
                file_name=f"data_kependudukan_tasikmalaya_{file_tahun}.csv",
                mime="text/csv",
                key="download_data_kependudukan",
            )

    except Exception as exc:
        st.error(f"Data kependudukan tidak dapat ditampilkan: {exc}")
        st.info(
            "Pastikan file kependudukan_2024.csv, kependudukan_2025.csv, "
            "dan kependudukan_2026.csv berada di folder dataset/ "
            "bersama proyek Streamlit."
        )

# ============================================================
# 11. EVALUASI K
# ============================================================
elif menu == "Evaluasi K":
    st.title(
        "Evaluasi Jumlah Cluster (K)"
    )

    st.markdown(
        '<p class="page-subtitle">'
        'Penentuan jumlah cluster optimal '
        'menggunakan Silhouette (utama), Elbow (konfirmasi); '
        'DBI untuk evaluasi model final.'
        '</p>',
        unsafe_allow_html=True
    )

    tahun = tahun_selector(
        "evaluasi_tahun"
    )

    try:
        hasil = analisis_tahun(
            tahun
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("K Optimal (Silhouette)", hasil['k_optimal'])
        col2.metric("K Elbow (konfirmasi)", hasil['k_elbow'])
        col3.metric("DBI model final", f"{hasil['result']['evaluasi_final']['Davies_Bouldin_Final']:.4f}")

        st.divider()

        g1, g2 = st.columns(2)

        # Elbow
        with g1:
            fig = px.line(
                hasil["elbow"],
                x="K",
                y="Inertia_WCSS",
                markers=True,
                title="Elbow / WCSS"
            )

            fig.add_vline(
                x=hasil["k_elbow"],
                line_dash="dash"
            )

            fig.update_xaxes(
                dtick=1
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Silhouette
        with g2:
            fig = px.line(
                hasil["evaluasi"],
                x="K",
                y="Silhouette_Score",
                markers=True,
                title="Silhouette Score"
            )

            fig.add_vline(
                x=hasil[
                    "k_silhouette"
                ],
                line_dash="dash"
            )

            fig.update_xaxes(
                dtick=1
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader(
            "Tabel Evaluasi K = 2–5"
        )

        tabel_eval = (
            hasil[
                "evaluasi"
            ]
            .copy()
            .rename(
                columns={
                    "Inertia_WCSS":
                        "WCSS",
                    "Silhouette_Score":
                        "Silhouette Score",
                    "Ukuran_Cluster":
                        "Ukuran Cluster",
                }
            )
        )

        st.dataframe(
            tabel_eval.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Kesimpulan"
        )

        st.info(
            f"K optimal tahun {tahun} "
            f"adalah K = "
            f"{hasil['k_optimal']}. "
            f"{hasil['alasan']}"
        )

    except Exception as exc:
        st.error(
            f"Evaluasi K tidak dapat "
            f"ditampilkan: {exc}"
        )


# ============================================================
# 12. HASIL CLUSTERING
# ============================================================
elif menu == "Hasil Clustering":
    st.title(
        "Hasil Clustering"
    )

    tahun = tahun_selector(
        "hasil_tahun"
    )

    try:
        hasil = analisis_tahun(
            tahun
        )

        result = hasil[
            "result"
        ]

        st.metric(
            f"K Optimal Tahun {tahun}",
            hasil[
                "k_optimal"
            ]
        )

        st.subheader(
            "Kecamatan dan Cluster"
        )

        cluster_table = (
            result[
                "df_final"
            ][
                ["Kecamatan", "Cluster"]
            ]
            .copy()
        )

        st.dataframe(
            warna_tabel_cluster(cluster_table),
            use_container_width=True,
            hide_index=True
        )

        c1, c2 = st.columns(2)

        with c1:
            counts = (
                cluster_table[
                    "Cluster"
                ]
                .value_counts()
                .sort_index()
                .rename_axis(
                    "Cluster"
                )
                .reset_index(
                    name="Jumlah Kecamatan"
                )
            )

            fig = px.bar(
                counts,
                x="Cluster",
                y="Jumlah Kecamatan",
                text_auto=True,
                title="Jumlah Anggota Cluster",
                color="Cluster",
                color_discrete_map=
                    CLUSTER_COLORS
            )

            fig.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with c2:
            centroid_long = (
                result[
                    "centroid_z"
                ]
                .melt(
                    id_vars=[
                        "Cluster"
                    ],
                    value_vars=NAMA_X,
                    var_name="Variabel",
                    value_name="Z-Score"
                )
            )

            centroid_long[
                "Variabel"
            ] = (
                centroid_long[
                    "Variabel"
                ]
                .map(
                    KETERANGAN_X
                )
            )

            fig = px.bar(
                centroid_long,
                x="Variabel",
                y="Z-Score",
                color="Cluster",
                barmode="group",
                title="Final Cluster Centers (Z-Score)",
                color_discrete_map=
                    CLUSTER_COLORS
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        t1, t2 = st.tabs(
            [
                "Centroid Z-Score",
                "Centroid Skala Asli",
            ]
        )

        with t1:
            st.dataframe(
                warna_tabel_cluster(result[
                    "centroid_z"
                ].round(4)),
                use_container_width=True,
                hide_index=True
            )

        with t2:
            centroid_asli = (
                result[
                    "centroid_asli"
                ]
                .rename(
                    columns=DISPLAY_NAME
                )
            )

            st.dataframe(
                warna_tabel_cluster(centroid_asli.round(4)),
                use_container_width=True,
                hide_index=True
            )

        st.subheader("Profil Cluster")
        st.caption(
            "Penamaan kawasan berdasarkan perbandingan relatif centroid "
            "laju pertumbuhan (X2) dan kepadatan penduduk (X3) "
            "pada tahun yang dipilih."
        )

        # Tampilkan C1 dan C2 berdampingan, seperti kartu contoh.
        # Pada K > 2, kartu berikutnya otomatis membuat baris baru.
        cluster_aktif = [
            f"C{i + 1}" for i in range(int(hasil["k_optimal"]))
        ]
        for awal in range(0, len(cluster_aktif), 2):
            kolom_profil = st.columns(2, gap="medium")
            for kolom, cluster in zip(
                kolom_profil, cluster_aktif[awal:awal + 2]
            ):
                with kolom:
                    st.markdown(
                        kartu_cluster_dashboard(hasil, tahun, cluster),
                        unsafe_allow_html=True,
                    )

        with st.expander(
            "Lihat Visualisasi PCA"
        ):
            pca_df = result[
                "pca"
            ]

            fig = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color="Cluster",
                text="Kecamatan",
                color_discrete_map=
                    CLUSTER_COLORS,
                title=(
                    "PCA Hasil Clustering "
                    f"Tahun {tahun}"
                )
            )

            fig.update_traces(
                textposition="top center",
                marker_size=12
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "PCA hanya digunakan untuk "
                "visualisasi hasil clustering."
            )

    except Exception as exc:
        st.error(
            f"Hasil clustering tidak "
            f"dapat ditampilkan: {exc}"
        )


# ============================================================
# 13. PETA CLUSTER
# ============================================================
elif menu == "Peta Cluster":
    st.title(
        "Peta Cluster Kecamatan"
    )

    tahun = tahun_selector(
        "peta_tahun"
    )

    try:
        hasil = analisis_tahun(
            tahun
        )

        st.caption(
            "Arahkan kursor ke wilayah untuk langsung melihat detail X1–X4. "
            "Klik wilayah untuk membuka kartu detail kecamatan."
        )

        # ----------------------------------------------------
        # LEGENDA CLUSTER DI ATAS PETA
        # ----------------------------------------------------
        st.subheader(
            "Legenda Cluster"
        )

        deskripsi_cluster = (
            buat_deskripsi_cluster(
                hasil,
                tahun
            )
        )

        cluster_aktif = [
            f"C{i + 1}"
            for i in range(
                hasil["k_optimal"]
            )
        ]

        legend_cols = st.columns(
            len(cluster_aktif)
        )

        for cluster, col in zip(
            cluster_aktif,
            legend_cols
        ):
            warna = (
                CLUSTER_COLORS[
                    cluster
                ]
            )

            info_cluster = (
                deskripsi_cluster.get(
                    cluster,
                    {
                        "title": cluster,
                        "short_title": cluster,
                        "desc": "-"
                    }
                )
            )

            judul_cluster = info_cluster["title"]
            deskripsi = info_cluster["desc"]

            # String HTML dibuat tanpa indentasi awal agar
            # Markdown tidak membacanya sebagai code block.
            card_html = (
                f'<div style="min-height:175px;'
                f'background:#ffffff;'
                f'border:1px solid #e2e8f0;'
                f'border-top:5px solid {warna};'
                f'border-radius:12px;'
                f'padding:13px 14px;'
                f'box-shadow:0 3px 12px rgba(15,23,42,.06);">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                f'<span style="width:14px;height:14px;border-radius:50%;'
                f'background:{warna};display:inline-block;"></span>'
                f'<b style="font-size:15px;color:{warna};">{cluster}</b>'
                f'</div>'
                f'<div style="color:#0f172a;font-size:13px;font-weight:700;line-height:1.45;margin-bottom:7px;">'
                f'{judul_cluster}'
                f'</div>'
                f'<div style="color:#475569;font-size:12px;line-height:1.55;">'
                f'{deskripsi}'
                f'</div>'
                f'</div>'
            )

            with col:
                st.markdown(
                    card_html,
                    unsafe_allow_html=True
                )

        cluster_caption = ", ".join(
            cluster_aktif[:-1]
        )

        if len(
            cluster_aktif
        ) > 1:
            if len(
                cluster_aktif
            ) == 2:
                nama_cluster_caption = (
                    f"{cluster_aktif[0]} dan {cluster_aktif[1]}"
                )
            else:
                nama_cluster_caption = (
                    f"{cluster_caption}, dan {cluster_aktif[-1]}"
                )
        else:
            nama_cluster_caption = (
                cluster_aktif[0]
            )

        st.caption(
            f"Warna {nama_cluster_caption} menunjukkan kelompok hasil "
            f"K-Means tahun {tahun}. "
            "Nama kawasan mencerminkan kepadatan (X3) dan pertumbuhan (X2) "
            "yang relatif antarcluster pada tahun tersebut, bukan kategori resmi BPS."
        )

        # ----------------------------------------------------
        # HEADER PETA + EXPORT DI POJOK KANAN
        # ----------------------------------------------------
        judul_col, export_col = st.columns(
            [5.5, 1.4],
            vertical_alignment="center"
        )

        with judul_col:
            st.markdown(
                "#### Peta Persebaran Cluster"
            )

        with export_col:
            with st.popover(
                "⬇ Export Map",
                use_container_width=True
            ):
                png_data = buat_export_peta(
                    tahun,
                    "png"
                )

                jpeg_data = buat_export_peta(
                    tahun,
                    "jpeg"
                )

                st.download_button(
                    "🖼 Export PNG (.png)",
                    data=png_data,
                    file_name=(
                        f"peta_cluster_tasikmalaya_{tahun}.png"
                    ),
                    mime="image/png",
                    use_container_width=True
                )

                st.download_button(
                    "🖼 Export JPEG (.jpg)",
                    data=jpeg_data,
                    file_name=(
                        f"peta_cluster_tasikmalaya_{tahun}.jpg"
                    ),
                    mime="image/jpeg",
                    use_container_width=True
                )

        peta = build_map(
            hasil,
            tahun
        )

        folium_static(
            peta,
            width=1200,
            height=700
        )

    except Exception as exc:
        st.error(
            f"Peta belum dapat ditampilkan: {exc}"
        )

        st.info(
            "Pastikan file shapefile "
            "ADMINISTRASI_AR_KECAMATAN.shp "
            "beserta .dbf, .shx, .prj, dan "
            ".cpg berada di folder peta/."
        )

# ============================================================
# 14. PERBANDINGAN TAHUN
# ============================================================
elif menu == "Perbandingan Tahun":
    st.title(
        "Perbandingan Hasil 2024–2026"
    )

    try:
        all_results = (
            get_all_results()
        )

        ringkasan = []

        for tahun in TAHUN_LIST:
            h = all_results[
                tahun
            ]

            ev_final = (
                h[
                    "result"
                ][
                    "evaluasi_final"
                ]
            )

            ringkasan.append({
                "Tahun":
                    tahun,
                "K Optimal":
                    h[
                        "k_optimal"
                    ],
                "Silhouette Score":
                    ev_final[
                        "Silhouette_Final"
                    ],
                "Davies-Bouldin Index":
                    ev_final[
                        "Davies_Bouldin_Final"
                    ],
                "WCSS":
                    ev_final[
                        "WCSS_Final"
                    ],
            })

        ringkasan_df = (
            pd.DataFrame(
                ringkasan
            )
        )

        st.subheader(
            "Ringkasan Evaluasi"
        )

        st.dataframe(
            ringkasan_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            ringkasan_df,
            x="Tahun",
            y="K Optimal",
            text_auto=True,
            title="K Optimal per Tahun"
        )

        fig.update_xaxes(
            type="category"
        )

        fig.update_yaxes(
            dtick=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Gunakan fungsi perbandingan yang sama dengan halaman Dashboard.
        tabel, k_tidak_sama = buat_tabel_perbandingan_antar_tahun(all_results)

        st.subheader("Perbandingan Cluster Kecamatan (2024–2026)")
        st.caption(
            "Cluster tiap tahun dihitung dari dataset masing-masing. "
            "Jika K sama, label 2025/2026 diselaraskan ke centroid "
            "tahun 2024 agar nomor C1, C2, dan seterusnya tidak tertukar."
        )
        # Tabel perbandingan polos: C1/C2 dan status berupa teks biasa.
        st.markdown(
            buat_tabel_perbandingan_visual(
                tabel[["Kecamatan", "2024", "2025", "2026", "Status"]],
                tahun_acuan="2024",
            ),
            unsafe_allow_html=True,
        )

        if k_tidak_sama:
            st.warning(
                "Jumlah K optimal berbeda antartahun. Tabel status "
                "Perlu interpretasi harus dibaca bersama profil centroid."
            )

        # Ganti kartu 6 stabil/4 berubah dengan total anggota 2024–2026.
        tampilkan_total_anggota_cluster(all_results)

        st.download_button(
            "Unduh perbandingan cluster (CSV)",
            data=tabel.to_csv(index=False).encode("utf-8-sig"),
            file_name="perbandingan_cluster_kecamatan_2024_2026.csv",
            mime="text/csv"
        )

    except Exception as exc:
        st.error(
            f"Perbandingan tahun tidak "
            f"dapat ditampilkan: {exc}"
        )
