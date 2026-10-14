import re
import io
import base64
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import streamlit as st

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
    2023: DATA_DIR / "kependudukan_2023.csv",
    2024: DATA_DIR / "kependudukan_2024.csv",
    2025: DATA_DIR / "kependudukan_2025.csv",
    2026: DATA_DIR / "kependudukan_2026.csv",
}

SHP_PATH = PETA_DIR / "ADMINISTRASI_AR_KECAMATAN.shp"

TAHUN_LIST = [2023, 2024, 2025, 2026]

RANDOM_STATE = 42
N_INIT = 20
MAX_ITER = 300
K_MAX = 8

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

CLUSTER_TITLES_BY_YEAR = {
    2023: {
        "C1": "Klaster Penduduk Tinggi dengan Pertumbuhan Rendah",
        "C2": "Klaster Pertumbuhan Tinggi dengan Kepadatan Rendah",
        "C3": "Klaster Kepadatan Tinggi dengan Karakteristik Sedang",
    },

    2024: {
        "C1": "Klaster Pertumbuhan Tinggi dengan Kepadatan Rendah",
        "C2": "Klaster Kepadatan Tinggi dengan Pertumbuhan dan Rasio Jenis Kelamin Rendah",
    },

    2025: {
        "C1": "Klaster Pertumbuhan dan Rasio Jenis Kelamin Tinggi dengan Kepadatan Rendah",
        "C2": "Klaster Kepadatan Tinggi dengan Rasio Jenis Kelamin Rendah",
    },

    2026: {
        "C1": "Klaster Karakteristik Kependudukan Relatif Sedang",
        "C2": "Klaster Kepadatan Tinggi dengan Penduduk, Pertumbuhan, dan Rasio Jenis Kelamin Rendah",
    },
}

CLUSTER_TITLES_SHORT_BY_YEAR = {
    2023: {
        "C1": "Penduduk Tinggi, Pertumbuhan Rendah",
        "C2": "Pertumbuhan Tinggi, Kepadatan Rendah",
        "C3": "Kepadatan Tinggi, Karakteristik Sedang",
    },

    2024: {
        "C1": "Pertumbuhan Tinggi, Kepadatan Rendah",
        "C2": "Kepadatan Tinggi, Pertumbuhan & Rasio JK Rendah",
    },

    2025: {
        "C1": "Pertumbuhan & Rasio JK Tinggi, Kepadatan Rendah",
        "C2": "Kepadatan Tinggi, Rasio JK Rendah",
    },

    2026: {
        "C1": "Karakteristik Kependudukan Relatif Sedang",
        "C2": "Kepadatan Tinggi, Penduduk/Pertumbuhan/Rasio JK Rendah",
    },
}

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
    rows = []
    hasil_per_k = {}

    for k in range(
        2,
        K_MAX + 1
    ):
        hasil = best_kmeans_trace(
            X_scaled,
            k,
            n_init=N_INIT
        )

        labels = (
            hasil[
                "cluster_final"
            ]
        )

        hasil_per_k[k] = hasil

        rows.append({
            "K":
                k,

            "Inertia_WCSS":
                hasil["inertia"],

            "Silhouette_Score":
                silhouette_score(
                    X_scaled,
                    labels
                ),

            "Davies_Bouldin":
                davies_bouldin_score(
                    X_scaled,
                    labels
                ),

            "Ukuran_Cluster":
                str(
                    hasil[
                        "ukuran_cluster"
                    ]
                ),
        })

    return (
        pd.DataFrame(rows),
        hasil_per_k
    )


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


def tentukan_k_optimal(
    k_elbow,
    k_silhouette,
    k_dbi
):
    kandidat = [
        int(k)
        for k in [
            k_elbow,
            k_silhouette,
            k_dbi,
        ]
        if (
            k is not None
            and
            2 <= int(k) <= K_MAX
        )
    ]

    vote = Counter(
        kandidat
    )

    if vote:
        k_vote, jumlah_vote = (
            vote.most_common(1)[0]
        )

        if jumlah_vote >= 2:
            return (
                int(k_vote),
                f"{jumlah_vote} dari 3 metode "
                f"memilih K={int(k_vote)}."
            )

    return (
        int(k_silhouette),
        "Tidak ada mayoritas yang sama, "
        "sehingga dipilih K dengan "
        "Silhouette Score tertinggi."
    )


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

    interpretasi_rows = []

    for _, row in centroid_z.iterrows():
        bagian = []

        for x in NAMA_X:
            if row[x] >= 0.5:
                kategori = "tinggi"

            elif row[x] <= -0.5:
                kategori = "rendah"

            else:
                kategori = "sedang"

            bagian.append(
                f"{x} "
                f"({KETERANGAN_X[x]}) "
                f"{kategori}"
            )

        interpretasi_rows.append({
            "Cluster":
                row["Cluster"],

            "Interpretasi_Relatif":
                "; ".join(bagian),
        })

    interpretasi = pd.DataFrame(
        interpretasi_rows
    )

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

    k_dbi = int(
        evaluasi.loc[
            evaluasi[
                "Davies_Bouldin"
            ].idxmin(),
            "K",
        ]
    )

    k_optimal, alasan = (
        tentukan_k_optimal(
            k_elbow,
            k_silhouette,
            k_dbi,
        )
    )

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
        "k_dbi":
            k_dbi,
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
                f"{CLUSTER_TITLES_SHORT_BY_YEAR.get(tahun, {}).get(cluster, cluster)}"
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
        'Tahun 2023–2026'
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

        "Upload CSV":
            "📤  Upload CSV (Opsional)",

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
    """
    Menghasilkan nama dan deskripsi cluster berdasarkan centroid Z-score.
    Nama cluster dibuat spesifik per tahun agar tidak memakai label 2023
    untuk hasil 2024–2026.
    """
    centroid = (
        hasil[
            "result"
        ][
            "centroid_z"
        ]
        .copy()
    )

    # Ambil tahun dari argumen atau dari hasil evaluasi final
    if tahun is None:
        tahun = (
            hasil[
                "result"
            ][
                "evaluasi_final"
            ]
            .get(
                "Tahun"
            )
        )

    titles = (
        CLUSTER_TITLES_BY_YEAR.get(
            int(tahun),
            {}
        )
        if tahun is not None
        else {}
    )

    short_titles = (
        CLUSTER_TITLES_SHORT_BY_YEAR.get(
            int(tahun),
            {}
        )
        if tahun is not None
        else {}
    )

    descriptions = {}

    for _, row in centroid.iterrows():
        cluster = row["Cluster"]

        parts = []

        for x in NAMA_X:
            value = float(row[x])

            if value >= 0.5:
                kategori = "tinggi"
            elif value <= -0.5:
                kategori = "rendah"
            else:
                kategori = "sedang"

            parts.append(
                f"{KETERANGAN_X[x]} {kategori}"
            )

        descriptions[cluster] = {
            "title":
                titles.get(
                    cluster,
                    f"Klaster {cluster}"
                ),

            "short_title":
                short_titles.get(
                    cluster,
                    f"Klaster {cluster}"
                ),

            "desc":
                ", ".join(parts)
        }

    return descriptions


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


# ============================================================
# 9. DASHBOARD
# ============================================================
if menu == "Dashboard":
    st.title(
        "Dashboard Klasterisasi Kecamatan Kota Tasikmalaya Menggunakan K-Means"
    )

    st.markdown(
        '<p class="page-subtitle">'
        'Berdasarkan Karakteristik Kependudukan Tahun 2023–2026'
        '</p>',
        unsafe_allow_html=True
    )

    tahun = tahun_selector(
        "dashboard_tahun"
    )

    try:
        hasil = analisis_tahun(
            tahun
        )

        result = hasil[
            "result"
        ]

        df_cluster = (
            result[
                "df_final"
            ]
            .copy()
        )

        # ====================================================
        # KPI UTAMA
        # ====================================================
        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:
            st.metric(
                "Total Kecamatan",
                df_cluster[
                    "Kecamatan"
                ].nunique()
            )

        with col2:
            st.metric(
                "Total Variabel",
                len(
                    FITUR_KMEANS
                )
            )

        with col3:
            st.metric(
                "Total Tahun",
                len(
                    TAHUN_LIST
                )
            )

        with col4:
            st.metric(
                f"K Optimal {tahun}",
                hasil[
                    "k_optimal"
                ]
            )

        # ====================================================
        # BAGIAN 2 — RINGKASAN HASIL CLUSTER
        # ====================================================
        st.markdown(
            f"### Ringkasan Cluster Tahun {tahun}"
        )

        cluster_aktif = [
            f"C{i + 1}"
            for i in range(
                hasil[
                    "k_optimal"
                ]
            )
        ]

        jumlah_per_cluster = (
            df_cluster[
                "Cluster"
            ]
            .value_counts()
            .reindex(
                cluster_aktif,
                fill_value=0
            )
        )

        nama_cluster_tahun = (
            CLUSTER_TITLES_BY_YEAR.get(
                tahun,
                {}
            )
        )

        cluster_cols = st.columns(
            len(
                cluster_aktif
            )
        )

        for cluster, col in zip(
            cluster_aktif,
            cluster_cols
        ):
            warna = (
                CLUSTER_COLORS.get(
                    cluster,
                    "#94A3B8"
                )
            )

            judul_cluster = (
                nama_cluster_tahun.get(
                    cluster,
                    f"Klaster {cluster}"
                )
            )

            jumlah_anggota = int(
                jumlah_per_cluster[
                    cluster
                ]
            )

            with col:
                card_html = (
                    f'<div style="'
                    f'background:#ffffff;'
                    f'border:1px solid #e2e8f0;'
                    f'border-top:5px solid {warna};'
                    f'border-radius:14px;'
                    f'padding:15px 16px;'
                    f'min-height:155px;'
                    f'box-shadow:0 4px 14px rgba(15,41,66,.06);'
                    f'">'
                    f'<div style="display:flex;align-items:center;gap:8px;'
                    f'margin-bottom:9px;">'
                    f'<span style="width:13px;height:13px;border-radius:50%;'
                    f'background:{warna};display:inline-block;"></span>'
                    f'<b style="font-size:16px;color:#0f172a;">{cluster}</b>'
                    f'</div>'
                    f'<div style="font-size:13px;font-weight:700;'
                    f'line-height:1.45;color:#1e293b;min-height:58px;">'
                    f'{judul_cluster}'
                    f'</div>'
                    f'<div style="margin-top:12px;font-size:12px;color:#64748b;">'
                    f'<b style="font-size:20px;color:#0f172a;">{jumlah_anggota}</b> '
                    f'Kecamatan'
                    f'</div>'
                    f'</div>'
                )

                st.markdown(
                    card_html,
                    unsafe_allow_html=True
                )

        # ====================================================
        # BAGIAN 3 — DISTRIBUSI ANGGOTA CLUSTER
        # ====================================================
        st.markdown(
            "### Distribusi Anggota Cluster"
        )

        grafik_col, anggota_col = (
            st.columns(
                [1.15, 1]
            )
        )

        # ----------------------------------------------------
        # KIRI: GRAFIK JUMLAH KECAMATAN PER CLUSTER
        # ----------------------------------------------------
        with grafik_col:
            chart_df = pd.DataFrame({
                "Cluster":
                    cluster_aktif,

                "Jumlah Kecamatan":
                    [
                        int(
                            jumlah_per_cluster[
                                cluster
                            ]
                        )
                        for cluster
                        in cluster_aktif
                    ],
            })

            fig = px.bar(
                chart_df,
                x="Cluster",
                y="Jumlah Kecamatan",
                color="Cluster",
                text="Jumlah Kecamatan",
                color_discrete_map=
                    CLUSTER_COLORS,
                title=(
                    "Jumlah Kecamatan per Cluster"
                )
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                showlegend=False,
                margin=dict(
                    l=20,
                    r=20,
                    t=55,
                    b=20
                ),
                height=355,
                yaxis=dict(
                    dtick=1,
                    rangemode="tozero"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # KANAN: DAFTAR ANGGOTA MASING-MASING CLUSTER
        # ----------------------------------------------------
        with anggota_col:
            st.markdown(
                "#### Anggota Cluster"
            )

            for cluster in cluster_aktif:
                warna = (
                    CLUSTER_COLORS.get(
                        cluster,
                        "#94A3B8"
                    )
                )

                anggota = (
                    df_cluster.loc[
                        df_cluster[
                            "Cluster"
                        ] == cluster,
                        "Kecamatan"
                    ]
                    .astype(str)
                    .tolist()
                )

                nama_cluster = (
                    CLUSTER_TITLES_SHORT_BY_YEAR
                    .get(
                        tahun,
                        {}
                    )
                    .get(
                        cluster,
                        cluster
                    )
                )

                daftar_anggota = (
                    ", ".join(
                        anggota
                    )
                    if anggota
                    else "-"
                )

                anggota_html = (
                    f'<div style="'
                    f'background:#ffffff;'
                    f'border:1px solid #e2e8f0;'
                    f'border-left:5px solid {warna};'
                    f'border-radius:10px;'
                    f'padding:10px 12px;'
                    f'margin-bottom:8px;'
                    f'">'
                    f'<div style="font-size:13px;font-weight:700;'
                    f'color:#0f172a;margin-bottom:3px;">'
                    f'{cluster} — {nama_cluster}'
                    f'</div>'
                    f'<div style="font-size:12px;line-height:1.5;'
                    f'color:#475569;">'
                    f'{daftar_anggota}'
                    f'</div>'
                    f'</div>'
                )

                st.markdown(
                    anggota_html,
                    unsafe_allow_html=True
                )

        st.caption(
            "Jumlah dan anggota cluster berubah otomatis "
            "sesuai tahun yang dipilih dan hasil K optimal "
            "pada tahun tersebut."
        )

    except Exception as exc:
        st.error(
            f"Data tahun {tahun} "
            f"belum dapat dibaca: {exc}"
        )

# ============================================================
# 10. UPLOAD CSV
# ============================================================
elif menu == "Upload CSV":
    st.title(
        "Upload & Validasi CSV"
    )

    st.markdown(
        '<p class="page-subtitle">'
        'Pemeriksaan kualitas data untuk file CSV. '
        'File yang diunggah hanya diperiksa dan tidak '
        'mengubah dataset penelitian utama.'
        '</p>',
        unsafe_allow_html=True
    )

    uploaded_file = (
        st.file_uploader(
            "Upload File CSV",
            type=["csv"],
            key="csv_upload",
            help=(
                "CSV dapat berupa dataset penelitian "
                "maupun file CSV lain yang ingin diperiksa."
            )
        )
    )

    st.caption(
        "Menu ini bersifat opsional dan dapat digunakan untuk "
        "mengecek struktur, missing value, duplikasi, tipe data, "
        "serta kesesuaian dataset kependudukan."
    )

    if uploaded_file is None:
        st.info(
            "Unggah file CSV untuk menampilkan preview "
            "dan hasil pemeriksaan kualitas data."
        )

        with st.expander(
            "Format dataset penelitian K-Means"
        ):
            variabel_info = pd.DataFrame({
                "Kode": [
                    "X1",
                    "X2",
                    "X3",
                    "X4",
                ],
                "Variabel": [
                    "Jumlah Penduduk",
                    "Laju Pertumbuhan Penduduk",
                    "Kepadatan Penduduk",
                    "Rasio Jenis Kelamin",
                ],
            })

            st.dataframe(
                variabel_info,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Persentase Penduduk digunakan sebagai "
                "informasi deskriptif dan tidak menjadi "
                "fitur K-Means."
            )

    else:
        try:
            raw_upload = (
                read_csv_smart(
                    uploaded_file
                )
            )

            kualitas = (
                analisis_kualitas_csv(
                    raw_upload
                )
            )

            # ------------------------------------------------
            # RINGKASAN DATASET
            # ------------------------------------------------
            st.subheader(
                "Ringkasan Dataset"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Jumlah Baris",
                    kualitas[
                        "jumlah_baris"
                    ]
                )

            with c2:
                st.metric(
                    "Jumlah Kolom",
                    kualitas[
                        "jumlah_kolom"
                    ]
                )

            with c3:
                st.metric(
                    "Missing Value",
                    kualitas[
                        "total_missing"
                    ]
                )

            with c4:
                st.metric(
                    "Baris Duplikat",
                    kualitas[
                        "duplicate_rows_exact"
                    ]
                )

            # ------------------------------------------------
            # PREVIEW DATA
            # ------------------------------------------------
            st.subheader(
                "Preview Data"
            )

            st.dataframe(
                raw_upload.head(20),
                use_container_width=True,
                hide_index=True
            )

            if len(
                raw_upload
            ) > 20:
                st.caption(
                    "Preview menampilkan 20 baris pertama."
                )

            # ------------------------------------------------
            # VALIDASI UMUM
            # ------------------------------------------------
            st.subheader(
                "Pemeriksaan Kualitas Data"
            )

            validasi_umum = {
                "File berhasil dibaca":
                    True,

                "Nama kolom tidak duplikat":
                    kualitas[
                        "duplicate_column_names"
                    ]
                    == 0,

                "Tidak ada kolom kosong / Unnamed":
                    len(
                        kualitas[
                            "unnamed_columns"
                        ]
                    )
                    == 0,

                "Tidak ada missing value":
                    kualitas[
                        "total_missing"
                    ]
                    == 0,

                "Tidak ada baris duplikat":
                    kualitas[
                        "duplicate_rows_exact"
                    ]
                    == 0,
            }

            v1, v2 = st.columns(2)

            for i, (
                keterangan,
                status
            ) in enumerate(
                validasi_umum.items()
            ):
                target = (
                    v1
                    if i % 2 == 0
                    else v2
                )

                with target:
                    if status:
                        st.markdown(
                            f"✅ {keterangan}"
                        )
                    else:
                        st.markdown(
                            f"⚠️ {keterangan}"
                        )

            # ------------------------------------------------
            # DETAIL PER KOLOM
            # ------------------------------------------------
            st.markdown(
                "#### Ringkasan per Kolom"
            )

            st.dataframe(
                kualitas[
                    "ringkasan_kolom"
                ],
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # MISSING VALUE
            # ------------------------------------------------
            missing_table = (
                kualitas[
                    "ringkasan_kolom"
                ]
                .loc[
                    lambda x:
                    x["Missing"] > 0
                ][
                    [
                        "Kolom",
                        "Missing",
                        "Missing (%)",
                    ]
                ]
            )

            if not missing_table.empty:
                with st.expander(
                    "Lihat Detail Missing Value",
                    expanded=True
                ):
                    st.warning(
                        "Terdapat missing value pada dataset."
                    )

                    st.dataframe(
                        missing_table,
                        use_container_width=True,
                        hide_index=True
                    )

            # ------------------------------------------------
            # DUPLIKAT
            # ------------------------------------------------
            if (
                kualitas[
                    "duplicate_rows_exact"
                ]
                > 0
            ):
                with st.expander(
                    "Lihat Baris Duplikat",
                    expanded=True
                ):
                    duplicate_preview = (
                        raw_upload[
                            raw_upload.duplicated(
                                keep=False
                            )
                        ]
                        .copy()
                    )

                    st.warning(
                        f"Ditemukan "
                        f"{kualitas['duplicate_rows_exact']} "
                        f"baris duplikat tambahan."
                    )

                    st.dataframe(
                        duplicate_preview.head(50),
                        use_container_width=True,
                        hide_index=True
                    )

            # ------------------------------------------------
            # STATISTIK NUMERIK
            # ------------------------------------------------
            numeric_cols = (
                raw_upload
                .select_dtypes(
                    include=np.number
                )
                .columns
                .tolist()
            )

            if numeric_cols:
                with st.expander(
                    "Statistik Data Numerik"
                ):
                    statistik = (
                        raw_upload[
                            numeric_cols
                        ]
                        .describe()
                        .T
                        .reset_index()
                        .rename(
                            columns={
                                "index":
                                    "Kolom"
                            }
                        )
                    )

                    st.dataframe(
                        statistik.round(3),
                        use_container_width=True,
                        hide_index=True
                    )

            # ------------------------------------------------
            # VALIDASI KHUSUS DATASET PENELITIAN
            # ------------------------------------------------
            st.divider()

            st.subheader(
                "Validasi Dataset Kependudukan"
            )

            cek_penelitian = (
                cek_dataset_penelitian_upload(
                    raw_upload
                )
            )

            if not cek_penelitian[
                "cocok"
            ]:
                st.info(
                    "File ini tetap dapat diperiksa sebagai CSV umum, "
                    "tetapi strukturnya tidak sesuai dengan dataset "
                    "kependudukan penelitian."
                )

                if cek_penelitian[
                    "kolom_hilang"
                ]:
                    st.caption(
                        "Kolom penelitian yang belum ditemukan: "
                        + ", ".join(
                            cek_penelitian[
                                "kolom_hilang"
                            ]
                        )
                    )

            else:
                penelitian_validasi = (
                    cek_penelitian[
                        "validasi"
                    ]
                )

                if cek_penelitian[
                    "siap_kmeans"
                ]:
                    st.success(
                        "Struktur file sesuai dataset penelitian "
                        "dan siap digunakan untuk proses K-Means."
                    )
                else:
                    st.warning(
                        "Struktur file sesuai dataset penelitian, "
                        "tetapi masih terdapat masalah kualitas data."
                    )

                p1, p2 = st.columns(2)

                for i, (
                    keterangan,
                    status
                ) in enumerate(
                    penelitian_validasi.items()
                ):
                    target = (
                        p1
                        if i % 2 == 0
                        else p2
                    )

                    with target:
                        if status:
                            st.markdown(
                                f"✅ {keterangan}"
                            )
                        else:
                            st.markdown(
                                f"❌ {keterangan}"
                            )

                if cek_penelitian[
                    "kecamatan_hilang"
                ]:
                    st.error(
                        "Kecamatan yang belum ditemukan: "
                        + ", ".join(
                            cek_penelitian[
                                "kecamatan_hilang"
                            ]
                        )
                    )

                if cek_penelitian[
                    "kecamatan_tambahan"
                ]:
                    st.warning(
                        "Nama kecamatan di luar daftar penelitian: "
                        + ", ".join(
                            cek_penelitian[
                                "kecamatan_tambahan"
                            ]
                        )
                    )

                with st.expander(
                    "Lihat Variabel K-Means"
                ):
                    variabel_info = pd.DataFrame({
                        "Kode": [
                            "X1",
                            "X2",
                            "X3",
                            "X4",
                        ],
                        "Variabel": [
                            "Jumlah Penduduk",
                            "Laju Pertumbuhan Penduduk",
                            "Kepadatan Penduduk",
                            "Rasio Jenis Kelamin",
                        ],
                    })

                    st.dataframe(
                        variabel_info,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Persentase Penduduk tetap digunakan "
                        "sebagai informasi deskriptif, tetapi "
                        "tidak menjadi fitur K-Means."
                    )

        except Exception as exc:
            st.error(
                f"File CSV tidak dapat dibaca atau diperiksa: {exc}"
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
        'menggunakan Elbow, Silhouette, '
        'dan Davies-Bouldin.'
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

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:
            st.metric(
                "K Optimal",
                hasil[
                    "k_optimal"
                ]
            )

        with col2:
            st.metric(
                "K Elbow",
                hasil[
                    "k_elbow"
                ]
            )

        with col3:
            st.metric(
                "K Silhouette",
                hasil[
                    "k_silhouette"
                ]
            )

        with col4:
            st.metric(
                "K DBI",
                hasil[
                    "k_dbi"
                ]
            )

        st.divider()

        g1, g2, g3 = st.columns(3)

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

        # DBI
        with g3:
            fig = px.line(
                hasil["evaluasi"],
                x="K",
                y="Davies_Bouldin",
                markers=True,
                title="Davies-Bouldin Index"
            )

            fig.add_vline(
                x=hasil[
                    "k_dbi"
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
            "Tabel Evaluasi K = 2–8"
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
                    "Davies_Bouldin":
                        "Davies-Bouldin Index",
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
            cluster_table,
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
                result[
                    "centroid_z"
                ].round(4),
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
                centroid_asli.round(4),
                use_container_width=True,
                hide_index=True
            )

        st.subheader(
            "Profil Cluster"
        )

        for _, row in (
            result[
                "interpretasi"
            ]
            .iterrows()
        ):
            cluster = row[
                "Cluster"
            ]

            st.markdown(
                f"""
                <div class="cluster-box">
                    <b>{cluster}</b><br>
                    {row['Interpretasi_Relatif']}
                </div>
                """,
                unsafe_allow_html=True
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
                f'<b style="font-size:15px;color:#0f172a;">{cluster}</b>'
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
            "Keterangan tinggi, sedang, dan rendah bersifat relatif terhadap "
            "kecamatan lain pada tahun yang sama berdasarkan centroid Z-score."
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
        "Perbandingan Hasil 2023–2026"
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

        rows = []

        for tahun in TAHUN_LIST:
            temp = (
                all_results[
                    tahun
                ][
                    "result"
                ][
                    "df_final"
                ][
                    [
                        "Kecamatan",
                        "Cluster"
                    ]
                ]
                .copy()
            )

            temp[
                "Tahun"
            ] = tahun

            rows.append(
                temp
            )

        rekap = pd.concat(
            rows,
            ignore_index=True
        )

        tabel_perbandingan = (
            rekap
            .pivot(
                index="Kecamatan",
                columns="Tahun",
                values="Cluster"
            )
            .reset_index()
        )

        st.subheader(
            "Cluster Kecamatan per Tahun"
        )

        st.dataframe(
            tabel_perbandingan,
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "Nomor cluster antar-tahun "
            "tidak dibandingkan secara "
            "langsung karena K optimal dapat "
            "berbeda. Interpretasi antar-tahun "
            "sebaiknya melihat karakteristik "
            "centroid masing-masing cluster."
        )

    except Exception as exc:
        st.error(
            f"Perbandingan tahun tidak "
            f"dapat ditampilkan: {exc}"
        )
