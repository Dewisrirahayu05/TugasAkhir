import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# ==============================================================================
# KONFIGURASI HALAMAN DASHBOARD
# ==============================================================================
st.set_page_config(
    page_title="Dashboard SIPROKEM (Sistem Profiling Kemiskinan)", 
    layout="wide",
    initial_sidebar_state="expanded")

# Menampilkan judul utama dashboard.
st.title("Pengelompokan Profil Kemiskinan Provinsi di Indonesia Menggunakan K-Means")

# ==============================================================================
# STYLE SEDERHANA
# ==============================================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .input-note {
        font-size: 0.86rem;
        color: #666;
        margin-top: -0.35rem;
        margin-bottom: 0.5rem;
    }

    .result-title {
        font-size: 1.45rem;
        font-weight: 700;
    }

    .small-muted {
        color: #666;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# KONSTANTA
# ==============================================================================
KOLOM_MODEL = ["IPM", "P0", "P1", "P2", "GK", "GR"]

NAMA_VARIABEL = {
    "IPM": "Indeks Pembangunan Manusia",
    "P0": "Persentase Penduduk Miskin",
    "P1": "Indeks Kedalaman Kemiskinan",
    "P2": "Indeks Keparahan Kemiskinan",
    "GK": "Garis Kemiskinan",
    "GR": "Gini Ratio",
}

PENJELASAN_VARIABEL = {
    "IPM": "Menggambarkan capaian pembangunan manusia melalui aspek kesehatan, pendidikan, dan standar hidup.",
    "P0": "Persentase penduduk yang berada di bawah garis kemiskinan.",
    "P1": "Menggambarkan rata-rata kesenjangan pengeluaran penduduk miskin terhadap garis kemiskinan.",
    "P2": "Menggambarkan tingkat ketimpangan di antara penduduk miskin.",
    "GK": "Nilai pengeluaran minimum yang digunakan sebagai batas garis kemiskinan.",
    "GR": "Menggambarkan tingkat ketimpangan distribusi pendapatan/pengeluaran.",
}

SATUAN_VARIABEL = {
    "IPM": "Skala 0–100",
    "P0": "Persen (%)",
    "P1": "Indeks",
    "P2": "Indeks",
    "GK": "Rupiah/orang/bulan",
    "GR": "Skala 0–1",
}

# Profil dan rekomendasi
PROFIL_CLUSTER = {
    0: {
        "judul": "Kemiskinan Menengah & Biaya Hidup Rendah",
        "ringkas": "Karakteristik utama kelompok ini adalah tingkat kemiskinan menengah dengan ketimpangan yang relatif tinggi.",
        "provinsi": "BENGKULU, KEPULAUAN RIAU, DKI JAKARTA, JAWA BARAT, JAWA TENGAH, DI YOGYAKARTA, JAWA TIMUR, NUSA TENGGARA BARAT, NUSA TENGGARA TIMUR, SULAWESI UTARA, SULAWESI SELATAN, SULAWESI TENGGARA, GORONTALO, SULAWESI BARAT, PAPUA BARAT, PAPUA BARAT DAYA, PAPUA, PAPUA SELATAN",
        "rekomendasi": [
            ("Pemberdayaan ekonomi & pemerataan", "Akses modal usaha bagi UMKM."),
            ("Peningkatan lapangan kerja", "Penyediaan proyek padat karya secara lokal."),
            ("Regulasi upah", "Penataan standar upah minimum wilayah untuk membantu menekan ketimpangan pendapatan.")
        ]
    },
    1: {
        "judul": "Kemiskinan Ekstrem & Biaya Hidup Sangat Tinggi",
        "ringkas": "Kelompok ini memiliki P0, P1, P2, dan GK yang tinggi serta IPM yang rendah.",
        "provinsi": "PAPUA TENGAH, PAPUA PEGUNUNGAN",
        "rekomendasi": [
            ("Jaring pengaman sosial", "Bantuan sosial langsung seperti BLT, PKH, dan Sembako untuk menjaga daya beli."),
            ("Intervensi gizi", "Program pemenuhan gizi dan pangan untuk membantu menekan kedalaman kemiskinan."),
            ("Infrastruktur dasar", "Percepatan fasilitas kesehatan dasar, sekolah, serta dukungan terhadap kebutuhan pokok.")
        ]
    },
    2: {
        "judul": "Wilayah Relatif Sejahtera",
        "ringkas": "Kelompok ini memiliki IPM relatif tinggi, P0 lebih rendah, dan GK relatif rendah/aman.",
        "provinsi": "ACEH, SUMATERA UTARA, SUMATERA BARAT, RIAU, JAMBI, SUMATERA SELATAN, LAMPUNG, KEPULAUAN BANGKA BELITUNG, BANTEN, BALI, KALIMANTAN BARAT, KALIMANTAN TENGAH, KALIMANTAN SELATAN, KALIMANTAN TIMUR, KALIMANTAN UTARA, SULAWESI TENGAH, MALUKU, MALUKU UTARA",
        "rekomendasi": [
            ("Inkubasi bisnis", "Stimulus finansial dan pendampingan untuk perluasan pasar UMKM."),
            ("Stabilitas pasar", "Menjaga kestabilan harga kebutuhan pokok agar garis kemiskinan tetap terkendali."),
            ("Pemeliharaan tren ekonomi", "Menjaga iklim investasi lokal agar kondisi kesejahteraan tetap positif.")
        ]
    }
}

# ==============================================================================
# LOAD DATA DAN MODEL
# ==============================================================================
# Load data final (sudah memiliki kolom 'Klaster' untuk K=3)
@st.cache_data
def load_data():
    df = pd.read_csv("data_final_dengan_klaster.csv")
    # Menghapus kolom eksperimen K3 dan K4 agar tidak muncul di dashboard Streamlit
    df = df.drop(columns=["Klaster_K3", "Klaster_K4"], errors="ignore")

    # Memastikan kolom Klaster ada, jika tidak ada gunakan 'Cluster'
    if "Klaster" in df.columns:
        df = df.rename(columns={"Klaster": "Cluster"})

    return df

# Data untuk scatter plot (PCA) - pastikan file data_final_klaster.csv ada
@st.cache_data
def load_pca():
    # Membaca data yang mengandung koordinat PCA1 & PCA2 khusus untuk Scatter Plot
    df_pca = pd.read_csv("data_final_klaster.csv")

    if "Klaster" in df_pca.columns:
        df_pca = df_pca.rename(columns={"Klaster": "Cluster"})

    return df_pca

# Menyimpan model di cache agar tidak perlu membaca/load ulang setiap kali 
# Streamlit melakukan rerun pada halaman
@st.cache_resource
def load_models():
    # Memuat objek scaler yang sudah dilatih sebelumnya
    # Digunakan untuk melakukan normalisasi data input
    with open("scaler.pkl", "rb") as f:
        scaler_obj = pickle.load(f)

    # Memuat objek PCA yang sudah dilatih sebelumnya
    # Digunakan untuk mengubah data yang sudah dinormalisasi
    # menjadi data berdimensi lebih rendah
    with open("pca.pkl", "rb") as f:
        pca_obj = pickle.load(f)

    # Memuat model K-Means yang sudah dilatih sebelumnya
    # Model ini digunakan untuk menentukan cluster/profil kemiskinan
    with open("kmeans_model.pkl", "rb") as f:
        kmeans_obj = pickle.load(f)

    # Mengembalikan ketiga objek model agar dapat digunakan
    # pada proses prediksi/klasifikasi cluster di aplikasi
    return scaler_obj, pca_obj, kmeans_obj

# Memuat data dan seluruh model yang dibutuhkan oleh aplikasi
try:
    df = load_data() # Memuat data kemiskinan
    df_pca = load_pca() # Memuat data hasil PCA
    scaler, pca, kmeans = load_models() # Memuat scaler, PCA, dan model K-Means
except Exception as e:
    # Jika salah satu file/data tidak ditemukan atau gagal dimuat,
    # tampilkan pesan error di halaman Streamlit
    st.error("File data atau model belum lengkap.") 
    st.code(str(e)) # Menampilkan detail error untuk membantu proses debugging
    st.stop() # Menghentikan aplikasi agar tidak lanjut ke proses berikutnya

# ==============================================================================
# FUNGSI BANTUAN
# ==============================================================================
# Fungsi untuk mengubah angka menjadi format mata uang Rupiah.
# Contoh: 1500000 -> Rp 1.500.000
def format_rupiah(value):
    return f"Rp {value:,.0f}".replace(",", ".")

# Fungsi untuk menampilkan informasi/profil dari suatu klaster.
# cluster_id = nomor cluster yang ingin ditampilkan.
# tampilkan_provinsi = menentukan apakah daftar provinsi ditampilkan atau tidak.
def tampilkan_profil_cluster(cluster_id, tampilkan_provinsi=True):
    # Mengambil informasi cluster berdasarkan ID dari dictionary PROFIL_CLUSTER.
    # int() digunakan untuk memastikan cluster_id berupa angka/integer.
    info = PROFIL_CLUSTER.get(int(cluster_id))

    # Jika informasi cluster tidak ditemukan, tampilkan peringatan
    # dan hentikan fungsi.
    if info is None:
        st.warning(f"Profil untuk Cluster {cluster_id} belum tersedia.")
        return

    # Menampilkan nomor dan nama/judul cluster.
    st.markdown(f"### Cluster {cluster_id} — {info['judul']}")
    # Menampilkan ringkasan karakteristik cluster.
    st.write(info["ringkas"])

    # Jika tampilkan_provinsi=True, daftar provinsi ditampilkan
    # dalam bentuk expander agar halaman tidak terlalu penuh.
    if tampilkan_provinsi:
        with st.expander("Lihat provinsi dalam cluster ini"):
            st.write(info["provinsi"])

    # Menampilkan rekomendasi berdasarkan karakteristik cluster.
    st.markdown("**Rekomendasi yang terkait dengan karakteristik kelompok:**")
    # Melakukan perulangan untuk setiap rekomendasi yang tersedia.
    for nama, penjelasan in info["rekomendasi"]:
        st.markdown(f"- **{nama}:** {penjelasan}")

# Fungsi untuk menampilkan informasi tambahan mengenai
# variabel yang digunakan pada input.
# Contoh informasi: nama variabel, satuan, dan penjelasannya.
def tampilkan_input_info(kode):
    st.caption(
        f"{NAMA_VARIABEL[kode]} • {SATUAN_VARIABEL[kode]}  \n"
        f"{PENJELASAN_VARIABEL[kode]}"
    )

# ==============================================================================
# SIDEBAR
# ==============================================================================
# Membuat seluruh komponen berikut berada di bagian sidebar Streamlit.
with st.sidebar:
    st.image("siprokem.png", use_container_width=True) # Menampilkan logo SIPROKEM
    st.markdown("---")

    # --------------------------------------------------------------------------
    # MENU NAVIGASI
    # --------------------------------------------------------------------------
    # Menampilkan judul bagian menu.
    st.markdown("### Pilih Halaman Menu:")
    # Membuat pilihan halaman menggunakan selectbox button.
    # Nilai yang dipilih disimpan ke dalam variabel 'menu'.
    menu = st.selectbox(
        "Pilih halaman:",
        # Daftar halaman yang tersedia pada dashboard.
        ["Visualisasi & Hasil Klaster Wilayah", "Simulasi Prediksi"],
        # Mengembunyikan label "Pilih halaman:" agar tampilan lebih ringkas.
        label_visibility="collapsed"
    )

    # --------------------------------------------------------------------------
    # Ringkasan Data
    # --------------------------------------------------------------------------
    # Menampilkan judul informasi mengenai sistem.
    st.markdown("---")
    st.markdown("Ringkasan Data: ")
    # Menampilkan informasi singkat mengenai sumber data,
    # jumlah wilayah, indikator yang digunakan, dan metode clustering.
    st.info("""
    * **Sumber Data:** Badan Pusat Statistik (BPS)
    * **Jumlah Wilayah:** 38 Provinsi di Indonesia
    * **Indikator Utama:** IPM, P0, P1, P2, GK, GR
    * **Metode:** K-Means Clustering 
    """) 

    st.markdown("---")
    st.caption("© 2026 | Dashboard Profil Kemiskinan Provinsi di Indonesia")

# ==============================================================================
# MENU 1 — Visualisasi & HASIL KLASTER
# ==============================================================================
# Menampilkan halaman "Profil & Hasil Klaster"
# ketika pengguna memilih menu tersebut pada sidebar.
if menu == "Visualisasi & Hasil Klaster Wilayah":
    # --------------------------------------------------------------------------
    # JUDUL HALAMAN
    # --------------------------------------------------------------------------
    # Menampilkan deskripsi singkat mengenai tujuan halaman.
    st.markdown("**Tujuan Penelitian:** Menerapkan algoritma K-Means clustering untuk mengelompokkan 38 provinsi di Indonesia berdasarkan enam variabel kemiskinan (IPM, P0, P1, P2, GK, dan Gini Ratio), sehingga menghasilkan suatu sistem berbasis dashboard interaktif yang dapat memberikan rekomendasi jenis bantuan yang berbeda sesuai karakteristik masing-masing kelompok (cluster), guna mendukung penyaluran bantuan/dana yang lebih tepat sasaran.")
    st.markdown("**Berikut Hasil Pengelompokan 38 Provinsi di Indonesia:**")

    # ==========================================================================
    # TAB INFORMASI
    # ==========================================================================
    # Membuat 3 tab utama:
    # 1. Peta & Profil Klaster: menampilkan peta sekaligus penjelasan klaster
    # 2. Scatter Plot PCA: menampilkan visualisasi hasil PCA
    # 3. Tabel Detail Data: menampilkan data lengkap setiap provinsi
    tab_peta, tab_pca, tab_tabel = st.tabs(
        ["Peta Interaktif", "Scatter Plot PCA", "Tabel Detail"]
    )

    # --------------------------------------------------------------------------
    # TAB PETA & PROFIL KLASTER
    # --------------------------------------------------------------------------
    # Tab ini menggabungkan visualisasi peta dengan penjelasan
    # karakteristik setiap klaster agar pengguna dapat:
    # - melihat persebaran klaster pada peta
    # - memahami karakteristik dari masing-masing klaster
    # - melihat provinsi yang termasuk ke dalam setiap klaster
    # - melihat rekomendasi berdasarkan karakteristik klaster
    with tab_peta:
        st.success("### Peta Sebaran Profil Kemiskinan Provinsi di Indonesia")
        st.write(
            "Peta menunjukkan persebaran 38 provinsi berdasarkan hasil "
            "pengelompokan K-Means. Setiap titik mewakili satu provinsi, "
            "warna menunjukkan klaster, dan ukuran titik mengikuti nilai P0."
        )

        # Koordinat Provinsi,
        # Dictionary yang berisi koordinat latitude dan longitude
        # sebagai titik representasi masing-masing provinsi.
        koordinat = {
            'ACEH': (4.6951,96.7494), 'SUMATERA UTARA': (2.1154,99.5451),
            'SUMATERA BARAT': (-0.7399,100.8000), 'RIAU': (0.2933,101.7068),
            'JAMBI': (-1.6101,103.6131), 'SUMATERA SELATAN': (-3.3194,103.9144),
            'BENGKULU': (-3.7928,102.2608), 'LAMPUNG': (-4.5585,105.4068),
            'KEPULAUAN BANGKA BELITUNG': (-2.7410,106.4406), 'KEPULAUAN RIAU': (0.9000,104.4500),
            'DKI JAKARTA': (-6.2088,106.8456), 'JAWA BARAT': (-6.9175,107.6191),
            'JAWA TENGAH': (-7.1500,110.1403), 'DI YOGYAKARTA': (-7.7956,110.3695),
            'JAWA TIMUR': (-7.5361,112.2384), 'BANTEN': (-6.4058,106.0640),
            'BALI': (-8.3405,115.0920), 'NUSA TENGGARA BARAT': (-8.6529,117.3616),
            'NUSA TENGGARA TIMUR': (-8.6574,121.0794), 'KALIMANTAN BARAT': (-0.2788,111.4753),
            'KALIMANTAN TENGAH': (-1.6815,113.3824), 'KALIMANTAN SELATAN': (-3.0926,115.2838),
            'KALIMANTAN TIMUR': (0.5387,116.4194), 'KALIMANTAN UTARA': (3.0731,116.0414),
            'SULAWESI UTARA': (1.4303,124.7752), 'SULAWESI TENGAH': (-0.9000,121.8000),
            'SULAWESI SELATAN': (-3.6688,119.9741), 'SULAWESI TENGGARA': (-3.9596,122.5907),
            'GORONTALO': (0.5435,123.0568), 'SULAWESI BARAT': (-2.4972,119.4015),
            'MALUKU': (-3.2385,130.1453), 'MALUKU UTARA': (1.5700,127.8088),
            'PAPUA BARAT': (-1.3367,133.1747), 'PAPUA BARAT DAYA': (-0.9000,131.3000),
            'PAPUA': (-4.2699,138.0804), 'PAPUA SELATAN': (-7.5000,140.0000),
            'PAPUA TENGAH': (-3.0000,136.0000), 'PAPUA PEGUNUNGAN': (-4.0000,139.0000)
        }
        
        # ----------------------------------------------------------------------
        # MENYIAPKAN DATA UNTUK PETA
        # ----------------------------------------------------------------------
        # Membuat salinan dataframe agar data asli (df) tidak berubah.
        df_map = df.copy()
        # Menambahkan kolom latitude berdasarkan nama provinsi.
        # Jika nama provinsi tidak ditemukan pada dictionary,
        # koordinat default yang digunakan adalah (0, 0).
        df_map["lat"] = df_map["38 Provinsi"].map(lambda x: koordinat.get(x, (0, 0))[0])
        # Menambahkan kolom longitude berdasarkan nama provinsi.
        df_map["lon"] = df_map["38 Provinsi"].map(lambda x: koordinat.get(x, (0, 0))[1])
        # Mengubah nomor cluster menjadi string agar dapat digunakan
        # sebagai kategori warna pada visualisasi Plotly.
        df_map["Cluster"] = df_map["Cluster"].astype(str)

        # ----------------------------------------------------------------------
        # FILTER PROVINSI
        # ----------------------------------------------------------------------
        # Membuat daftar pilihan provinsi untuk fitur pencarian.
        # Pilihan pertama adalah "Semua provinsi".
        pilihan = ["TAMPILKAN SEMUA PROVINSI"] + sorted(df_map["38 Provinsi"].tolist())
        # Membuat dropdown untuk memilih provinsi.
        provinsi_pilihan = st.selectbox(
            "🔍Cari provinsi:",
            pilihan,
            help="Pilih satu provinsi untuk memfokuskan tampilan peta."
        )
        # Jika pengguna memilih satu provinsi,
        # peta hanya menampilkan provinsi tersebut.
        if provinsi_pilihan != "TAMPILKAN SEMUA PROVINSI":
            df_fokus = df_map[df_map["38 Provinsi"] == provinsi_pilihan]
        else:
            df_fokus = df_map

        # ----------------------------------------------------------------------
        # MEMBUAT PETA SEBARAN KLASTER
        # ----------------------------------------------------------------------
        # Membuat scatter map menggunakan Plotly.
        # Setiap titik mewakili satu provinsi.
        fig = px.scatter_geo(
            df_fokus,
            # Menentukan posisi titik berdasarkan latitude dan longitude.
            lat="lat",
            lon="lon",
            # Warna titik menunjukkan klaster.
            color="Cluster",
            # Nama provinsi ditampilkan ketika titik diarahkan/diklik.
            hover_name="38 Provinsi",
            # Ukuran titik mengikuti nilai P0.
            size="P0",
            hover_data=["IPM", "P0", "P1", "P2", "GK", "GR"],
            # Membatasi ukuran maksimum titik.
            size_max=25,
            # Menggunakan proyeksi peta Natural Earth.
            projection="natural earth", 
            title=' Berdasarkan Kluster (K=3)',
            color_discrete_map={'0': '#D32F2F', '1': '#1976D2', '2': '#388E3C'},
            # Mengubah nama label agar lebih mudah dipahami pengguna.
            labels={
                "Cluster": "Cluster",
                "P0": "P0 (%)",
                "IPM": "IPM",
                "GK": "GK"
            }
        )

        # ----------------------------------------------------------------------
        # PENGATURAN TAMPILAN PETA
        # ----------------------------------------------------------------------
        # Mengatur elemen geografis pada peta.
        fig.update_geos(
            showcountries=True, # Menampilkan batas negara
            showcoastlines=True, # Menampilkan garis pantai
            showland=True, # Menampilkan daratan
            visible=False, # Menyembunyikan layer dasar tertentu
            landcolor='lightgray',
            fitbounds="locations" # Menyesuaikan tampilan dengan lokasi data
        )
        # Mengatur ukuran dan margin visualisasi.
        fig.update_layout(
            height=550,
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        # Menampilkan peta pada dashboard Streamlit.
        st.plotly_chart(fig, use_container_width=True)
        # Catatan bahwa koordinat yang digunakan hanya berupa
        # titik representasi provinsi, bukan batas administratif.
        st.caption(
            "Catatan: posisi titik merupakan titik representasi provinsi, bukan batas wilayah administratif."
        )

        # --------------------------------------------------------------------------
        # RINGKASAN HASIL KLASTER
        # --------------------------------------------------------------------------
        # Menghitung jumlah provinsi pada setiap klaster.
        # sort_index() digunakan agar urutan klaster tetap 0, 1, 2.
        jumlah_cluster = df["Cluster"].value_counts().sort_index()
        # Menghitung jumlah seluruh provinsi yang terdapat pada dataset.
        total_provinsi = len(df)

        # Menampilkan judul bagian ringkasan.
        st.markdown("#### Ringkasan Hasil Pengelompokan")
        # Membuat 4 kolom untuk menampilkan informasi ringkas.
        c1, c2, c3, c4 = st.columns(4)

        # Menampilkan jumlah seluruh provinsi.
        c1.metric("Jumlah Provinsi", total_provinsi)
        # Menampilkan jumlah provinsi pada Klaster 0.
        # get(0, 0) berarti jika Klaster 0 tidak ditemukan,
        # nilainya dianggap 0.
        c2.metric("🔴 Cluster 0", int(jumlah_cluster.get(0, 0)))
        c3.metric("🔵 Cluster 1", int(jumlah_cluster.get(1, 0)))
        c4.metric("🟢 Cluster 2", int(jumlah_cluster.get(2, 0)))

        st.markdown("---")

    # --------------------------------------------------------------------------
    # PROFIL DAN KARAKTERISTIK KLASTER
    # --------------------------------------------------------------------------
        st.info("### Keterangan Karakteristik Setiap Cluster (Interpretasi Profil)")
        st.write(
            "Bagian ini menjelaskan karakteristik setiap kelompok provinsi "
            "berdasarkan hasil K-Means. Informasi disajikan dalam bahasa "
            "yang lebih sederhana agar hasil pengelompokan mudah dipahami."
        )
        st.markdown("---")

        # ----------------------------------------------------------------------
        # MENAMPILKAN PROFIL SETIAP CLUSTER DALAM BENTUK KOLOM
        # ----------------------------------------------------------------------
        # Membuat 3 kolom untuk Cluster 0, Cluster 1, dan Cluster 2.
        col1, col2, col3 = st.columns(3)

        # Membuat daftar kolom agar setiap cluster dapat ditempatkan
        # pada kolom sesuai urutannya.
        kolom_cluster = [col1, col2, col3]

        # Melakukan perulangan untuk setiap cluster.
        for kolom, cluster_id in zip(
            kolom_cluster,
            sorted(PROFIL_CLUSTER.keys())
        ):
            # Menghitung jumlah provinsi yang termasuk dalam klaster.
            jumlah = int(jumlah_cluster.get(cluster_id, 0))
            # Menampilkan informasi di dalam kolom.
            with kolom: 
                # Menampilkan detail profil cluster,
                # termasuk ringkasan, daftar provinsi, dan rekomendasi.
                tampilkan_profil_cluster(cluster_id)

    # --------------------------------------------------------------------------
    # TAB PCA
    # --------------------------------------------------------------------------
    with tab_pca:
        # Penjelasan sederhana mengenai tujuan halaman PCA.
        st.subheader("Scatter Plot Sebaran Klaster Berdasarkan Transformasi PCA")
        st.write(
            "Halaman ini digunakan untuk melihat pola kemiripan karakteristik "
            "antarprovinsi berdasarkan indikator yang digunakan dalam penelitian. "
            "Karena data memiliki beberapa indikator, PCA digunakan untuk "
            "menyederhanakannya menjadi dua komponen agar dapat ditampilkan "
            "dalam bentuk grafik."
        )
        # ----------------------------------------------------------------------
        # PENJELASAN SINGKAT: APA ITU PCA?
        # ----------------------------------------------------------------------
        st.info(
            "**Apa itu PCA?**\n"
            "PCA (Principal Component Analysis) adalah metode yang digunakan "
            "untuk menyederhanakan beberapa variabel menjadi beberapa komponen "
            "utama tanpa harus menampilkan seluruh variabel dalam satu grafik.\n\n"
            "Pada halaman ini, hasil PCA ditampilkan dalam dua komponen, yaitu "
            "**Komponen Utama 1 (PCA1)** dan **Komponen Utama 2 (PCA2)**. "
            "Keduanya digunakan sebagai sumbu pada grafik."
        )
            # ----------------------------------------------------------------------
            # CARA MEMBACA GRAFIK
            # ----------------------------------------------------------------------
        with st.expander("📌 Panduan Membaca Grafik"):
            st.markdown("""
                * **Titik yang berdekatan** menunjukkan provinsi dengan karakteristik indikator kemiskinan yang relatif mirip. Misalnya, provinsi dengan IPM tinggi dan P0 rendah cenderung berada di dekat satu sama lain.
                * **Titik yang berjauhan** menunjukkan provinsi dengan karakteristik indikator kemiskinan yang berbeda. Misalnya, provinsi dengan IPM rendah dan P0 tinggi akan berada jauh dari provinsi dengan kondisi sebaliknya.
                * **Warna titik** menunjukkan cluster yang telah ditentukan oleh algoritma K-Means. Setiap warna mewakili satu kelompok provinsi dengan karakteristik kemiskinan yang serupa.
                * **Hover pada titik** akan menampilkan nama provinsi dan nilai indikator terkait, sehingga pengguna dapat melihat detail setiap provinsi secara interaktif.
                * **Interpretasi pola**: Dengan melihat sebaran titik pada grafik, pengguna dapat memahami bagaimana provinsi dikelompokkan berdasarkan kemiripan karakteristik indikator kemiskinan, serta mengidentifikasi provinsi yang memiliki kondisi ekstrem atau berbeda dari kelompok lainnya.
            """)

        # ----------------------------------------------------------------------
        # PENGECEKAN DATA PCA
        # ----------------------------------------------------------------------
        # Memastikan data PCA memiliki dua kolom yang dibutuhkan, yaitu 
        # PCA1 = komponen utama pertama
        # PCA2 = komponen utama kedua
        # Kedua komponen ini digunakan sebagai sumbu X dan Y pada scatter plot.
        if "PCA1" in df_pca.columns and "PCA2" in df_pca.columns:
            # Membuat salinan data PCA agar dataframe asli tidak berubah.
            df_plot = df_pca.copy()
            # Mengubah Cluster menjadi string agar dianggap sebagai kategori
            # saat digunakan sebagai pembeda warna pada grafik.
            df_plot["Cluster"] = df_plot["Cluster"].astype(str)
            
            # ------------------------------------------------------------------
            # MEMBUAT SCATTER PLOT PCA
            # ------------------------------------------------------------------
            # Membuat grafik scatter menggunakan hasil PCA.
            # Setiap titik mewakili satu provinsi.
            fig_pca = px.scatter(
                df_plot,
                x="PCA1", # Sumbu X menggunakan komponen utama 1.
                y="PCA2", # Sumbu Y menggunakan komponen utama 2.
                color="Cluster", # Warna titik menunjukkan klaster hasil K-Means.
                hover_name="38 Provinsi", # Nama provinsi muncul ketika titik diarahkan/diklik.
                title="Visualisasi Pemisahan Kelompok Klaster dengan Reduksi Dimensi (Principal Component Analysis)",
                color_discrete_map={'0': '#D32F2F', '1': '#1976D2', '2': '#388E3C'},
                # Mengubah nama label agar lebih mudah dipahami pengguna.
                labels={
                    "PCA1": "Komponen Utama 1",
                    "PCA2": "Komponen Utama 2",
                    "Cluster": "Cluster"
                }
            )

            # ------------------------------------------------------------------
            # PENGATURAN TAMPILAN GRAFIK
            # ------------------------------------------------------------------
            # Mengatur ukuran titik pada scatter plot agar lebih mudah dilihat.
            fig_pca.update_traces(marker=dict(size=12))
            # Mengatur tinggi grafik.
            fig_pca.update_layout(height=580)

            # Menampilkan grafik PCA pada halaman Streamlit.
            st.plotly_chart(fig_pca, use_container_width=True)

            # ------------------------------------------------------------------
            # PENJELASAN CARA MEMBACA GRAFIK
            # ------------------------------------------------------------------
            # Narasi Analisis 
            st.write("### Interpretasi & Analisis Sebaran Spasial PCA")
            st.markdown("Apa yang dapat dilihat dari grafik tersebut?")
            c1, c2 = st.columns(2)
            # Penjelasan fungsi grafik tersebut.
            with c1:
                st.info("""
                **Fungsi grafik di atas:**
                * Grafik ini membuktikan secara matematis bahwa algoritma *K-Means* berhasil memisahkan 38 provinsi ke dalam 3 kelompok secara optimal. Jarak antar titik mencerminkan kemiripan karakteristik indikator kemiskinan BPS.
                * Setiap titik mewakili satu provinsi, dan warnanya menunjukkan klaster yang telah ditentukan. Semakin jauh jarak antar titik, semakin berbeda karakteristik kemiskinan antar provinsi tersebut.""")
                
            #  Penjelasan analisis pola cluster.
            with c2:
                st.success("""
                **Analisis Pola Klaster:**
                * **Cluster 1 (Biru):** Terlihat memisahkan diri sangat kontras di koordinat kanan bawah, mengonfirmasi statusnya sebagai wilayah dengan beban kemiskinan paling ekstrem.
                * **Cluster 0 & 2 (Merah & Hijau):** Membentuk dua kelompok bersisian terpisah, menunjukkan batas perbedaan tingkat kesejahteraan dan ketimpangan (*Gini Ratio*) yang tegas.
                """)

            # ------------------------------------------------------------------
            # CATATAN PENTING
            # ------------------------------------------------------------------
            st.markdown("##### Catatan")
            st.info(
                "PCA pada halaman ini digunakan untuk **membantu visualisasi**, "
                "bukan untuk menentukan klaster. Pengelompokan provinsi tetap "
                "dilakukan menggunakan algoritma **K-Means** berdasarkan "
                "variabel yang digunakan dalam model."
            )
        # Jika PCA1 atau PCA2 tidak tersedia, tampilkan pesan peringatan.
        else:
            st.warning("Kolom PCA1 dan PCA2 tidak ditemukan pada data PCA.")

    # --------------------------------------------------------------------------
    # TAB TABEL DETAIL DATA
    # --------------------------------------------------------------------------
    with tab_tabel:
        st.markdown("### Tabel Detail Hasil Pengelompokan")

        # Penjelasan umum mengenai isi tabel agar pengguna awam
        # memahami tujuan dari data yang ditampilkan.
        st.write(
            "Tabel di bawah ini menampilkan data asli dari Badan Pusat Statistik (BPS) tahun 2025 "
            "yang telah digabungkan dengan label kelompok (Cluster) hasil perhitungan algoritma K-Means. "
            "Anda dapat menyortir data dengan mengklik pada nama kolom tabel."
        )

        # ----------------------------------------------------------------------
        # PENJELASAN INDIKATOR
        # ----------------------------------------------------------------------
        with st.expander("📌 Panduan Membaca Satuan dan Definisi Indikator"):
            st.markdown("""
                * **IPM (Indeks Pembangunan Manusia)**: Mengukur capaian pembangunan manusia berdasarkan aspek kesehatan, pendidikan, dan standar hidup layak (Skala 0-100).
                * **P0 (Persentase Penduduk Miskin)**: Persentase penduduk yang pengeluaran per kapita per bulannya berada di bawah garis kemiskinan (GK).
                * **P1 (Indeks Kedalaman Kemiskinan)**: Rata-rata kesenjangan pengeluaran penduduk miskin terhadap Garis Kemiskinan. Semakin tinggi nilai indeks (0-100), semakin jauh rata-rata pengeluaran mereka dari GK.
                * **P2 (Indeks Keparahan Kemiskinan)**: Mengukur penyebaran atau ketimpangan pengeluaran di antara penduduk miskin itu sendiri (Skala 0-5).
                * **GK (Garis Kemiskinan)**: GK merepresentasikan pengeluaran minimum rata-rata per kapita per bulan pada suatu provinsi untuk memenuhi kebutuhan dasar makanan dan non-makanan (Rupiah/Kapita/Bulan).
                * **Gini Ratio (GR)**: Mengukur ketimpangan pengeluaran penduduk secara menyeluruh. Semakin tinggi nilainya, semakin tinggi tingkat ketimpangan (Skala 0 hingga 1).
            """)

        # ----------------------------------------------------------------------
        # DATA HASIL CLUSTERING
        # ----------------------------------------------------------------------
        st.markdown("#### Data Hasil Clustering")
        # Mengambil kolom yang diperlukan untuk ditampilkan pada tabel.
        df_tabel = df[
            ["38 Provinsi", "IPM", "P0", "P1", "P2", "GK", "GR", "Cluster"]
        ].copy()

        # Menggunakan st.column_config untuk mempercantik tampilan ribuan rupiah pada kolom GK secara native
        st.dataframe(
            df_tabel,
            # Membuat lebar tabel mengikuti lebar halaman.
            use_container_width=True,
            # Menyembunyikan nomor index dataframe agar tampilan lebih rapi.
            hide_index=True,
            column_config={
                "38 Provinsi": st.column_config.TextColumn("Provinsi"),
                "IPM": st.column_config.NumberColumn("IPM", format="%.2f"),
                "P0": st.column_config.NumberColumn("P0 (Persentase)", format="%.2f%%"), # Tampil degan lambang %
                "P1": st.column_config.NumberColumn("P1 (Kedalaman)", format="%.2f"),
                "P2": st.column_config.NumberColumn("P2 (Keparahan)", format="%.2f"),
                "GK": st.column_config.NumberColumn("Garis Kemiskinan (GK)", format="Rp %,.0f"), # Mengubah 715103 menjadi Rp 715.103 [1]
                "GR": st.column_config.NumberColumn("Gini Ratio (GR)", format="%.3f"),
                "Cluster": st.column_config.TextColumn("Cluster Hasil K-Means")
            }
        )
        st.markdown("---")

        # ----------------------------------------------------------------------
        # RINGKASAN PROFIL KARAKTERISTIK CLUSTER
        # ----------------------------------------------------------------------
        st.markdown("#### Ringkasan Karakteristik Profil Cluster")
        st.write(
            "Berdasarkan hasil analisis rata-rata variabel pada masing-masing kelompok, "
            "berikut adalah interpretasi profil kemiskinan untuk setiap cluster:"
        )
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            st.error("""
            **🔴 CLUSTER 0**
            *   **Kategori:** Kemiskinan Menengah, Biaya Hidup Rendah.
            *   **Karakteristik:** IPM tinggi, GR (Ketimpangan Pengeluaran Tinggi), persentase kemiskinan (P0) serta indeks kesenjangan (P1, P2) relatif rendah, didukung oleh nilai Garis Kemiskinan (biaya hidup dasar) yang paling murah (rata-rata Rp648.365,22).
            """)

        with c_col2:
            st.info("""
            **🔵 CLUSTER 1**
            *   **Kategori:** Kemiskinan Ekstrem, Biaya Hidup Tinggi.
            *   **Karakteristik:** Kualitas manusia masih rendah (IPM terendah), persentase kemiskinan sangat tinggi, kesenjangan pengeluaran sangat jomplang, diperparah oleh biaya hidup dasar yang sangat mahal (rata-rata Rp1.054.585,50).
            """)

        with c_col3:
            st.success("""
            **🟢 CLUSTER 2**
            *   **Kategori:** Wilayah Relatif Sejahtera.
            *   **Karakteristik:** Memiliki kualitas pembangunan manusia (IPM) tertinggi, persentase penduduk miskin terendah, kesenjangan ekonomi sangat minim, dengan biaya hidup standar menengah (rata-rata Rp737.281,56).
            """)
        st.markdown("---")

        # ----------------------------------------------------------------------
        # DOWNLOAD DATA & Laporan
        # ----------------------------------------------------------------------
        st.markdown("### Unduh Hasil Analisis")
        col_download1, col_download2 = st.columns(2)

        # Mengubah dataframe menjadi format PDF agar dapat diunduh.
        csv_data = df_tabel.to_csv(index=False).encode("utf-8")
        
        with col_download1:

            # ----------------------------------------------------------------------
            # DOWNLOAD DATA CSV
            # ----------------------------------------------------------------------
            st.download_button(
                "Unduh Data Hasil Clustering (CSV)",
                data=csv_data,
                file_name="hasil_clustering_provinsi_indonesia.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ------------------------------------------------------------------
        # Fungsi PDF untuk Tabel Detail
        # ------------------------------------------------------------------
        def buat_laporan_clustering_pdf(df_tabel, profil_cluster):
            from io import BytesIO
            from datetime import datetime

            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
                PageBreak
            )

            buffer = BytesIO()

            # Karena tabel 38 provinsi cukup lebar,
            # gunakan A4 landscape.
            doc = SimpleDocTemplate(
                buffer,
                pagesize=landscape(A4),
                rightMargin=1.2 * cm,
                leftMargin=1.2 * cm,
                topMargin=1.2 * cm,
                bottomMargin=1.2 * cm,
                title="Laporan Hasil Pengelompokan Profil Kemiskinan",
                author="SIPROKEM"
            )

            styles = getSampleStyleSheet()

            # ------------------------------------------------------------------
            # STYLE
            # ------------------------------------------------------------------
            style_judul = ParagraphStyle(
                "JudulClustering",
                parent=styles["Title"],
                fontName="Helvetica-Bold",
                fontSize=15,
                leading=19,
                alignment=TA_CENTER,
                spaceAfter=4
            )

            style_subjudul = ParagraphStyle(
                "SubjudulClustering",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=10,
                leading=14,
                alignment=TA_CENTER,
                spaceAfter=10
            )

            style_bagian = ParagraphStyle(
                "BagianClustering",
                parent=styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=11,
                leading=14,
                spaceBefore=6,
                spaceAfter=5
            )

            style_normal = ParagraphStyle(
                "NormalClustering",
                parent=styles["BodyText"],
                fontName="Helvetica",
                fontSize=8.5,
                leading=12,
                alignment=TA_LEFT,
                spaceAfter=4
            )

            style_kecil = ParagraphStyle(
                "KecilClustering",
                parent=styles["BodyText"],
                fontName="Helvetica",
                fontSize=7.5,
                leading=10,
                alignment=TA_LEFT
            )

            # ------------------------------------------------------------------
            # HELPER
            # ------------------------------------------------------------------
            def rupiah(value):
                return f"Rp {float(value):,.0f}".replace(",", ".")

            def angka(value, digit=2):
                return f"{float(value):.{digit}f}"

            waktu_laporan = datetime.now().strftime("%d-%m-%Y %H:%M")

            story = []

            # ==================================================================
            # HEADER / KOP
            # ==================================================================
            story.append(
                Paragraph(
                    "LAPORAN HASIL PENGELOMPOKAN PROFIL KEMISKINAN",
                    style_judul
                )
            )

            story.append(
                Paragraph(
                    "Sistem Profiling Kemiskinan Provinsi di Indonesia Menggunakan K-Means",
                    style_subjudul
                )
            )

            # Waktu di sebelah kanan
            waktu_table = Table(
                [
                    [
                        "",
                        Paragraph(
                            f"<b>Waktu Laporan:</b> {waktu_laporan}",
                            style_normal
                        )
                    ]
                ],
                colWidths=[18 * cm, 8 * cm]
            )

            waktu_table.setStyle(
                TableStyle([
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ])
            )

            story.append(waktu_table)

            # Garis kop
            garis1 = Table(
                [[""]],
                colWidths=[26 * cm],
                rowHeights=[0.05 * cm]
            )

            garis1.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.black),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ])
            )

            story.append(garis1)
            story.append(Spacer(1, 0.05 * cm))

            garis2 = Table(
                [[""]],
                colWidths=[26 * cm],
                rowHeights=[0.025 * cm]
            )

            garis2.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.black),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ])
            )

            story.append(garis2)

            # Jarak setelah kop
            story.append(Spacer(1, 0.4 * cm))
            # ==================================================================
            # I. INFORMASI DATA
            # ==================================================================
            story.append(
                Paragraph(
                    "I. INFORMASI DATA",
                    style_bagian
                )
            )

            story.append(
                Paragraph(
                    "Laporan ini menyajikan hasil pengelompokan 38 provinsi "
                    "di Indonesia berdasarkan enam indikator kemiskinan, yaitu "
                    "Indeks Pembangunan Manusia (IPM), Persentase Penduduk Miskin (P0), "
                    "Indeks Kedalaman Kemiskinan (P1), Indeks Keparahan Kemiskinan (P2), "
                    "Garis Kemiskinan (GK), dan Gini Ratio (GR). "
                    "Pengelompokan dilakukan menggunakan algoritma K-Means.",
                    style_normal
                )
            )

            info_data = [
                ["Sumber Data", "Badan Pusat Statistik (BPS)"],
                ["Jumlah Wilayah", "38 Provinsi"],
                ["Jumlah Indikator", "6 Indikator"],
                ["Metode", "K-Means Clustering"],
                ["Jumlah Cluster", "3 Cluster"],
            ]

            tabel_info = Table(
                info_data,
                colWidths=[5 * cm, 21 * cm]
            )

            tabel_info.setStyle(
                TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E9ECEF")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ])
            )

            story.append(tabel_info)
            story.append(Spacer(1, 0.35 * cm))

            # ==================================================================
            # II. HASIL PENGELOMPOKAN
            # ==================================================================
            story.append(
                Paragraph(
                    "II. HASIL PENGELOMPOKAN",
                    style_bagian
                )
            )

            story.append(
                Paragraph(
                    "Tabel berikut menampilkan nilai indikator dari setiap provinsi "
                    "beserta cluster hasil pengelompokan menggunakan algoritma K-Means.",
                    style_normal
                )
            )

            # Header tabel
            data_tabel = [
                [
                    "No",
                    "Provinsi",
                    "IPM",
                    "P0 (%)",
                    "P1",
                    "P2",
                    "GK",
                    "GR",
                    "Cluster"
                ]
            ]

            # Isi tabel
            for nomor, (_, row) in enumerate(df_tabel.iterrows(), start=1):

                data_tabel.append([
                    str(nomor),
                    Paragraph(
                        str(row["38 Provinsi"]),
                        style_kecil
                    ),
                    angka(row["IPM"]),
                    angka(row["P0"]),
                    angka(row["P1"]),
                    angka(row["P2"]),
                    rupiah(row["GK"]),
                    angka(row["GR"], 3),
                    str(row["Cluster"])
                ])

            tabel_data = Table(
                data_tabel,
                colWidths=[
                    0.8 * cm,   # No
                    5.2 * cm,   # Provinsi
                    2.2 * cm,   # IPM
                    2.2 * cm,   # P0
                    2.0 * cm,   # P1
                    2.0 * cm,   # P2
                    3.3 * cm,   # GK
                    2.0 * cm,   # GR
                    2.0 * cm    # Cluster
                ],
                repeatRows=1
            )

            tabel_data.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343A40")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ])
            )

            story.append(tabel_data)
            story.append(Spacer(1, 0.35 * cm))

            # ==================================================================
            # III. RINGKASAN HASIL CLUSTER
            # ==================================================================
            story.append(
                Paragraph(
                    "III. RINGKASAN HASIL PENGELOMPOKAN",
                    style_bagian
                )
            )

            jumlah_cluster = (
                df_tabel["Cluster"]
                .value_counts()
                .sort_index()
            )

            ringkasan_cluster = [
                ["Cluster", "Jumlah Provinsi"]
            ]

            for cluster_id in sorted(jumlah_cluster.index):
                ringkasan_cluster.append([
                    f"Cluster {cluster_id}",
                    str(int(jumlah_cluster[cluster_id]))
                ])

            tabel_ringkasan = Table(
                ringkasan_cluster,
                colWidths=[10 * cm, 10 * cm]
            )

            tabel_ringkasan.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECEF")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("ALIGN", (1, 1), (1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ])
            )

            story.append(tabel_ringkasan)
            story.append(Spacer(1, 0.35 * cm))

            # ==================================================================
            # IV. KARAKTERISTIK PROFIL CLUSTER
            # ==================================================================
            story.append(
                Paragraph(
                    "IV. KARAKTERISTIK PROFIL CLUSTER",
                    style_bagian
                )
            )

            for cluster_id in sorted(profil_cluster.keys()):

                info = profil_cluster[cluster_id]

                jumlah = int(
                    jumlah_cluster.get(cluster_id, 0)
                )

                story.append(
                    Paragraph(
                        f"<b>Cluster {cluster_id} — {info['judul']}</b>",
                        style_normal
                    )
                )

                story.append(
                    Paragraph(
                        f"{info['ringkas']} "
                        f"Jumlah provinsi dalam cluster ini sebanyak "
                        f"<b>{jumlah} provinsi</b>.",
                        style_normal
                    )
                )

                story.append(
                    Paragraph(
                        "<b>Provinsi:</b> " + str(info["provinsi"]),
                        style_kecil
                    )
                )

                story.append(Spacer(1, 0.12 * cm))

            # ==================================================================
            # V. REKOMENDASI KEBIJAKAN
            # ==================================================================
            story.append(
                Paragraph(
                    "V. REKOMENDASI KEBIJAKAN",
                    style_bagian
                )
            )

            story.append(
                Paragraph(
                    "Rekomendasi berikut disusun berdasarkan karakteristik "
                    "masing-masing cluster sebagaimana digunakan pada sistem SIPROKEM.",
                    style_normal
                )
            )

            for cluster_id in sorted(profil_cluster.keys()):
                info = profil_cluster[cluster_id]

                story.append(
                    Paragraph(
                        f"<b>Cluster {cluster_id} — {info['judul']}</b>",
                        style_normal
                    )
                )

                for nomor, (program, penjelasan) in enumerate(
                    info["rekomendasi"],
                    start=1
                ):
                    story.append(
                        Paragraph(
                            f"{nomor}. <b>{program}</b><br/>{penjelasan}",
                            style_normal
                        )
                    )

                story.append(Spacer(1, 0.08 * cm))

            # ==================================================================
            # VI. CATATAN
            # ==================================================================
            story.append(
                Paragraph(
                    "VI. CATATAN",
                    style_bagian
                )
            )

            story.append(
                Paragraph(
                    "Laporan ini diterbitkan secara otomatis oleh sistem SIPROKEM "
                    "berdasarkan hasil pengolahan data indikator kemiskinan dan "
                    "pengelompokan menggunakan algoritma K-Means.",
                    style_normal
                )
            )

            story.append(
                Paragraph(
                    "Hasil clustering digunakan untuk membantu memahami kesamaan "
                    "karakteristik antarwilayah dan mendukung analisis kebijakan. "
                    "Hasil ini bukan merupakan keputusan kebijakan final.",
                    style_normal
                )
            )

            story.append(Spacer(1, 0.25 * cm))
            story.append(
                Paragraph(
                    "<i>SIPROKEM — Dashboard Pengelompokan Profil Kemiskinan "
                    "Provinsi di Indonesia Menggunakan K-Means</i>",
                    style_kecil
                )
            )

            # Bangun PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        # ------------------------------------------------------------------
        # DOWNLOAD PDF
        # ------------------------------------------------------------------
        try:
            pdf_clustering = buat_laporan_clustering_pdf(
                df_tabel=df_tabel,
                profil_cluster=PROFIL_CLUSTER
            )

            with col_download2:
                st.download_button(
                    "Unduh Laporan Hasil Clustering (PDF)",
                    data=pdf_clustering,
                    file_name="laporan_hasil_pengelompokan_kemiskinan.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        except ImportError:
            st.error(
                "Library ReportLab belum terpasang. "
            )
        except Exception as error_pdf:
            st.error(
                f"Laporan PDF gagal dibuat: {error_pdf}"
            )

# ==============================================================================
# MENU 2 — SIMULASI WILAYAH
# ==============================================================================
elif menu == "Simulasi Prediksi":
    st.markdown("### Simulasi Prediksi Klaster & Rekomendasi Kebijakan Otomatis")
    st.markdown(
        "Fitur ini memungkinkan Anda melakukan simulasi penentuan kelompok (cluster) "
        "terhadap data indikator wilayah baru atau data target daerah proyeksi masa depan "
        "menggunakan kecerdasan buatan secara *real-time*."
    )

    # ==========================================================================
    # FUNGSI PEMBUAT PDF
    # ==========================================================================
    def buat_laporan_pdf(
        waktu_simulasi,
        input_data,
        cluster_prediksi,
        pca1,
        pca2,
        rekomendasi
    ):
        """
        Membuat laporan hasil simulasi dalam format PDF menggunakan ReportLab.
        Struktur laporan mengikuti contoh laporan yang diberikan pengguna,
        tetapi disesuaikan dengan sistem SIPROKEM.
        """
        from io import BytesIO
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            KeepTogether
        )

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=1.7 * cm,
            leftMargin=1.7 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
            title="Laporan Hasil Simulasi Prediksi SIPROKEM",
            author="SIPROKEM"
        )

        styles = getSampleStyleSheet()

        style_judul = ParagraphStyle(
            "Judul",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            alignment=TA_CENTER,
            spaceAfter=4
        )

        style_subjudul = ParagraphStyle(
            "Subjudul",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=12
        )

        style_bagian = ParagraphStyle(
            "Bagian",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=8,
            spaceAfter=6
        )

        style_normal = ParagraphStyle(
            "NormalCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=5
        )

        style_kecil = ParagraphStyle(
            "Kecil",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            alignment=TA_LEFT
        )

        style_cluster = ParagraphStyle(
            "Cluster",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            alignment=TA_CENTER
        )

        def rupiah(value):
            return f"Rp {value:,.0f}".replace(",", ".")

        def angka(value, digit=2):
            return f"{float(value):.{digit}f}"

        story = []

        # ----------------------------------------------------------------------
        # HEADER
        # ----------------------------------------------------------------------
        story.append(
            Paragraph(
                "LAPORAN HASIL SIMULASI PREDIKSI KLASTER",
                style_judul
            )
        )
        story.append(
            Paragraph(
                "Sistem Profiling Kemiskinan Provinsi di Indonesia Menggunakan K-Means",
                style_subjudul
            )
        )

        waktu_table = Table(
            [
                [
                    "",
                    Paragraph(
                        f"<b>Waktu Simulasi:</b> {waktu_simulasi}",
                    style_normal
                    )
                ]
            ],
            colWidths=[10.5 * cm, 6.3 * cm]    
        )

        waktu_table.setStyle(
            TableStyle([
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ])
        )

        story.append(waktu_table)

        # ------------------------------------------------------------------
        # GARIS PEMBATAS SEPERTI KOP SURAT
        # ------------------------------------------------------------------

        garis_header = Table(
            [[""]],
            colWidths=[16.8 * cm],
            rowHeights=[0.08 * cm]
        )

        garis_header.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.black),
                ("LINEBELOW", (0, 0), (-1, -1), 1, colors.black),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ])
        )

        story.append(garis_header)

        # ------------------------------------------------------------------
        # JARAK SEBELUM BAGIAN I
        # ------------------------------------------------------------------
        story.append(Spacer(1, 0.35 * cm))

        # ----------------------------------------------------------------------
        # I. DATA INPUT SIMULASI
        # ----------------------------------------------------------------------
        story.append(
            Paragraph("I. DATA INPUT SIMULASI", style_bagian)
        )

        data_input = [
            ["Indikator", "Nilai"],
            ["Indeks Pembangunan Manusia (IPM)", angka(input_data["IPM"])],
            ["Persentase Penduduk Miskin (P0)", f"{angka(input_data['P0'])}%"],
            ["Indeks Kedalaman Kemiskinan (P1)", angka(input_data["P1"])],
            ["Indeks Keparahan Kemiskinan (P2)", angka(input_data["P2"])],
            ["Garis Kemiskinan (GK)", rupiah(input_data["GK"])],
            ["Gini Ratio (GR)", angka(input_data["GR"], 3)],
        ]

        tabel_input = Table(
            data_input,
            colWidths=[11.8 * cm, 5.0 * cm],
            repeatRows=1
        )

        tabel_input.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECEF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )

        story.append(tabel_input)
        story.append(Spacer(1, 0.15 * cm))

        # ----------------------------------------------------------------------
        # II. HASIL PREDIKSI
        # ----------------------------------------------------------------------
        story.append(
            Paragraph("II. HASIL PREDIKSI", style_bagian)
        )

        hasil_cluster = Table(
            [
                [Paragraph(
                    f"Hasil Prediksi: CLUSTER {cluster_prediksi}",
                    style_cluster
                )],
                [Paragraph(
                    rekomendasi["judul"],
                    style_normal
                )]
            ],
            colWidths=[16.8 * cm]
        )

        hasil_cluster.setStyle(
            TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.8, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F3F5")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ])
        )

        story.append(hasil_cluster)
        story.append(Spacer(1, 8))

        story.append(
            Paragraph(
                f"<b>Koordinat PCA 1:</b> {pca1:.4f}<br/>"
                f"<b>Koordinat PCA 2:</b> {pca2:.4f}",
                style_normal
            )
        )

        # Jarak setelah judul bagian agar tidak terlalu dempet
        story.append(Spacer(1, 0.15 * cm))

        # ----------------------------------------------------------------------
        # III. KARAKTERISTIK KELOMPOK
        # ----------------------------------------------------------------------
        story.append(
            Paragraph("III. KARAKTERISTIK KELOMPOK", style_bagian)
        )

        story.append(
            Paragraph(
                rekomendasi["ringkas"],
                style_normal
            )
        )

        story.append(
            Paragraph(
                "<b>Provinsi dalam kelompok yang sama:</b>",
                style_normal
            )
        )

        # PROFIL_CLUSTER saat ini menyimpan provinsi sebagai string.
        # Kita pecah berdasarkan koma agar dapat dibuat daftar di PDF.
        provinsi_raw = rekomendasi.get("provinsi", "")

        if isinstance(provinsi_raw, list):
            daftar_provinsi = provinsi_raw
        else:
            daftar_provinsi = [
                p.strip()
                for p in str(provinsi_raw).split(",")
                if p.strip()
            ]

        # Tampilkan provinsi dalam 2 kolom supaya laporan tidak terlalu panjang.
        jumlah_baris = (len(daftar_provinsi) + 1) // 2
        data_provinsi = []

        for i in range(jumlah_baris):
            kiri = daftar_provinsi[i] if i < len(daftar_provinsi) else ""
            j = i + jumlah_baris
            kanan = daftar_provinsi[j] if j < len(daftar_provinsi) else ""

            data_provinsi.append([
                Paragraph(f"• {kiri}", style_kecil) if kiri else "",
                Paragraph(f"• {kanan}", style_kecil) if kanan else ""
            ])

        if data_provinsi:
            tabel_provinsi = Table(
                data_provinsi,
                colWidths=[8.3 * cm, 8.3 * cm]
            )

            tabel_provinsi.setStyle(
                TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ])
            )

            story.append(tabel_provinsi)

        # Jarak setelah judul bagian agar tidak terlalu dempet
        story.append(Spacer(1, 0.15 * cm))

        # ----------------------------------------------------------------------
        # IV. REKOMENDASI KEBIJAKAN
        # ----------------------------------------------------------------------
        story.append(
            Paragraph("IV. REKOMENDASI KEBIJAKAN", style_bagian)
        )

        story.append(
            Paragraph(
                "Berikut merupakan rekomendasi program yang dapat "
                "dipertimbangkan berdasarkan karakteristik kelompok wilayah.",
                style_normal
            )
        )

        for nomor, (program, penjelasan) in enumerate(
            rekomendasi.get("rekomendasi", []),
            start=1
        ):
            story.append(
                Paragraph(
                    f"<b>{nomor}. {program}</b><br/>{penjelasan}",
                    style_normal
                )
            )

        # Jarak setelah judul bagian agar tidak terlalu dempet
        story.append(Spacer(1, 0.15 * cm))

        # ----------------------------------------------------------------------
        # V. CATATAN
        # ----------------------------------------------------------------------
        story.append(
            Paragraph("V. CATATAN", style_bagian)
        )

        story.append(
            Paragraph(
                "Laporan ini diterbitkan secara otomatis oleh sistem SIPROKEM "
                "berdasarkan enam indikator yang dimasukkan pada fitur simulasi "
                "prediksi, yaitu IPM, P0, P1, P2, Garis Kemiskinan (GK), dan "
                "Gini Ratio (GR).",
                style_normal
            )
        )

        story.append(
            Paragraph(
                "Hasil simulasi merupakan hasil pengelompokan menggunakan model "
                "K-Means yang telah dilatih sebelumnya. Informasi ini digunakan "
                "sebagai pendukung analisis karakteristik wilayah dan bukan "
                "merupakan keputusan kebijakan final.",
                style_normal
            )
        )

        story.append(Spacer(1, 10))

        story.append(
            Paragraph(
                "<i>SIPROKEM — Dashboard Pengelompokan Profil Kemiskinan "
                "Provinsi di Indonesia Menggunakan K-Means</i>",
                style_kecil
            )
        )

        doc.build(story)

        buffer.seek(0)
        return buffer.getvalue()

    # ==========================================================================
    # KARTU PANDUAN ACUAN CEPAT (SOLUSI AGAR PENGGUNA AWAM TIDAK BINGUNG)
    # ==========================================================================
    with st.expander("PANDUAN ACUAN: Bagaimana cara menentukan angka input?", expanded=True):
        st.markdown(
                "Sebelum Anda memasukkan angka pada form di bawah, gunakan standar acuan resmi BPS "
                "dan data riil hasil penelitian ini sebagai panduan pengisian:"
        )
        
    # Membuat layout kolom di dalam expander panduan
        p_col1, p_col2, p_col3 = st.columns(3)
    
        with p_col1:
            st.markdown("""
                ##### **Kualitas Manusia & Kemiskinan**
                *   **IPM (Skala 0 - 100)**
                    *   *Sangat Tinggi:* $\ge$ 80 | *Tinggi:* 70 - 79.99
                    *   *Sedang:* 60 - 69.99 | *Rendah:* < 60
                *   **P0 (Persentase Penduduk Miskin)**
                    *   *Rendah:* < 5% | *Sedang:* 5% - 15% | *Tinggi:* > 15%
                """)

        with p_col2:
            st.markdown("""
                ##### **Indeks Kedalaman & Keparahan**
                *   **P1 (Kedalaman Kemiskinan)**
                    *   *Semakin mendekati 0:* Kondisi semakin baik.
                    *   *Rentang normal:* 0.5 s.d. 4.0
                *   **P2 (Keparahan Kemiskinan)**
                    *   *Semakin mendekati 0:* Ketimpangan semakin rendah.
                    *   *Rentang normal:* 0.1 s.d. 1.5
                """)
        
        with p_col3:
            st.markdown("""
                ##### **Standar Biaya Hidup & Ketimpangan**
                *   **Garis Kemiskinan (Rp/Kapita/Bulan)**
                    *   *Daerah Murah:* Rp500rb - Rp650rb
                    *   *Daerah Sedang:* Rp700rb - Rp750rb
                    *   *Daerah Mahal (Papua):* Rp900rb - Rp1.05 juta
                *   **Gini Ratio (Skala 0 - 1)**
                    *   *Rendah:* < 0.3 | *Sedang:* 0.3 - 0.5 | *Tinggi:* > 0.5
                """)

    st.markdown("---")

    # ==========================================================================
    # FORM INPUT SIMULASI
    # ==========================================================================
    st.markdown("#### Masukkan Nilai Indikator Wilayah Baru untuk Disimulasikan:")

    col1, col2 = st.columns(2)
    with col1:
        # 1. IPM
        ipm = st.number_input(
            "Masukkan Indeks Pembangunan Manusia (IPM):",
            min_value=0.0, max_value=100.0,
            value=None,
            placeholder="Contoh: 76.23", # Default menggunakan data BPS Jambi/Aceh sebagai contoh
            step=0.01,
            help="Skala 0-100 (Gunakan angka di rentang 60-85 untuk simulasi wilayah Indonesia)." 
        )

        # 2. P0 
        p0 = st.number_input(
            "Masukkan Persentase Penduduk Miskin (P0) %:",
            min_value=0.0, max_value=100.0,
            value=None,
            placeholder="Contoh: 12.20",
            step=0.01,
            help="Batas input dikunci 0% s.d 100%. Umumnya daerah di Indonesia berkisar antara 3% s.d 30%."
        )
        
        # 3. P1 
        p1 = st.number_input(
            "Masukkan Indeks Kedalaman Kemiskinan (P1):",
            min_value=0.0, max_value=10.0,
            value=None,
            placeholder="Contoh: 2.06",
            step=0.01,
            help="Kesenjangan pengeluaran penduduk miskin (Skala 0 - 10). Rata-rata Indonesia berkisar antara 0.5 s.d 4.0."
        )

    with col2:
        # 4. P2 
        p2 = st.number_input(
            "Masukkan Indeks Keparahan Kemiskinan (P2):",
            min_value=0.0, max_value=5.0,
            value=None,
            placeholder="Contoh: 0.52",
            step=0.01,
            help="Ketimpangan pengeluaran di antara penduduk miskin (Skala 0 - 5). Rata-rata Indonesia berkisar antara 0.1 s.d 1.5."
        )
        
        # 5. GK
        gk = st.number_input(
            "Masukkan Garis Kemiskinan (GK) - Rp/Kapita/Bulan:",
            min_value=100000.0, max_value=3000000.0,
            value=None,
            placeholder="Contoh: 715103",
            step=1.0,
            help=(
                "Garis Kemiskinan (GK) Provinsi.\n\n"
                "• GK merepresentasikan pengeluaran minimum rata-rata per kapita per bulan pada suatu provinsi.\n"
                "• Rumus Resmi BPS: GK = GKM + GKNM.\n"
                "• Berdasarkan data riil BPS tahun 2025, rentang umum berkisar antara Rp500.000 s.d. Rp1.054.000."
            )
        )
        
        # 6. Gini Ratio
        gr = st.number_input(
            "Masukkan Gini Ratio (GR) - Jangkauan 0 s.d 1:",
            min_value=0.0, max_value=1.0,
            value=None,
            placeholder="Contoh: 0.27",
            step=0.01,
            help="Ketimpangan distribusi pendapatan (Skala 0 - 1). Di Indonesia berkisar antara 0.25 s.d 0.45."
        )
        
    # --------------------------------------------------------------------------
    # PROSES SIMULASI
    # --------------------------------------------------------------------------
     # BACKEND: PROSES PREDIKSI MENGGUNAKAN FILE BINARY PKL
    if st.button("Jalankan Proses Analisis Sistem", use_container_width=True):
        if any(
            value is None
            for value in [ipm, p0, p1, p2, gk, gr]
        ):
            st.warning("**Mohon lengkapi semua indikator wajib diisi terlebih dahulu sebelum menjalankan simulasi prediksi!**")
        elif (
            not (0.0 <= ipm <= 100.0) 
            or not (0.0 <= p0 <= 100.0) 
            or not (0.0 <= p1 <= 10.0)
            or not (0.0 <= p2 <= 5.0)
            or not (100000.0 <= gk <= 3000000.0)
            or not (0.0 <= gr <= 1.0)
        ):
            st.error("**Nilai input berada di luar batas logis!** Pastikan Anda mengisi data sesuai panduan rujukan.")
        else:
            try:
                # 1. Memasukkan data input pengguna ke DataFrame dengan kolom yang sesuai
                input_data = pd.DataFrame(
                    [[ipm, p0, p1, p2, gk, gr]], 
                    columns=['IPM', 'P0', 'P1', 'P2', 'GK', 'GR']
                )
                    
                # 2. Normalisasi menggunakan scaler.pkl
                input_scaled = scaler.transform(input_data)
                            
                # 3. Melakukan Prediksi Cluster menggunakan kmeans_model.pkl 
                cluster_prediksi = int(kmeans.predict(input_scaled)[0])
                    
                # 4. Reduksi Dimensi menggunakan pca.pkl untuk mendapatkan PC1 & PC2
                input_pca = pca.transform(input_scaled)

                pca1 = input_pca[0, 0]
                pca2 = input_pca[0, 1]
                    
                # 5. Mengambil informasi profil & rekomendasi kebijakan
                rekomendasi = PROFIL_CLUSTER[cluster_prediksi]

                # 6. Waktu Simulasi
                from datetime import datetime

                waktu_simulasi = datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                )

                # Simpan hasil ke session_state agar tetap tersedia
                # setelah Streamlit melakukan rerun.
                st.session_state["hasil_simulasi"] = {
                    "waktu_simulasi": waktu_simulasi,
                    "input_data": {
                        "IPM": float(ipm),
                        "P0": float(p0),
                        "P1": float(p1),
                        "P2": float(p2),
                        "GK": float(gk),
                        "GR": float(gr)
                    },
                    "cluster": cluster_prediksi,
                    "pca1": pca1,
                    "pca2": pca2,
                    "rekomendasi": rekomendasi
                }

            except Exception as error_sistem:
                st.error(
                    f"Sistem gagal melakukan pemrosesan prediksi: "
                    f"{str(error_sistem)}"
                )

    # ==========================================================================
    # TAMPILKAN HASIL SIMULASI
    # ==========================================================================
    if "hasil_simulasi" in st.session_state:

        hasil = st.session_state["hasil_simulasi"]

        cluster_prediksi = hasil["cluster"]
        pca1 = hasil["pca1"]
        pca2 = hasil["pca2"]
        rekomendasi = hasil["rekomendasi"]
        waktu_simulasi = hasil["waktu_simulasi"]
        input_hasil = hasil["input_data"]

        st.markdown("---")
        # Menampilkan hasil prediksi secara visual dan meyakinkan
        st.success(f"### Hasil Simulasi Prediksi: Wilayah dikelompokkan ke **Cluster {cluster_prediksi}**")
                    
        # ------------------------------------------------------------------
        # RINGKASAN HASIL
        # ------------------------------------------------------------------
        st.markdown("#### Ringkasan Hasil Prediksi")
        # Layout hasil analisis prediksi
        col1, col2, col3 = st.columns(3)
                    
        with col1:
            st.metric(
                label="Hasil Cluster",
                value=f"Cluster {cluster_prediksi}"
            )
        with col2:
            st.metric(
                label="Koordinat PCA 1",
                value=f"{pca1:.4f}"
            )
        with col3:
            st.metric(
                label="Koordinat PCA 2",
                value=f"{pca2:.4f}"
            )

        # ------------------------------------------------------------------
        # DATA INPUT YANG DIPREDIKSI
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Data Input yang Dianalisis")

        tabel_hasil = pd.DataFrame({
            "Indikator": [
                "IPM",
                "P0",
                "P1",
                "P2",
                "Garis Kemiskinan (GK)",
                "Gini Ratio (GR)"
            ],
             "Nilai": [
                f"{input_hasil['IPM']:.2f}",
                f"{input_hasil['P0']:.2f}%",
                f"{input_hasil['P1']:.2f}",
                f"{input_hasil['P2']:.2f}",
                format_rupiah(input_hasil["GK"]),
                f"{input_hasil['GR']:.3f}"
            ]
        })

        st.dataframe(
            tabel_hasil,
            use_container_width=True,
            hide_index=True
        )
        
        # ------------------------------------------------------------------
        # KARAKTERISTIK CLUSTER
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Karakteristik Kelompok")

        st.info(
            f"""
            **{rekomendasi['judul']}**

            {rekomendasi['ringkas']}
            """
        )

        # ------------------------------------------------------------------
        # PROVINSI DALAM CLUSTER
        # ------------------------------------------------------------------
        st.markdown("### Provinsi dalam Kelompok yang Sama")

        provinsi = rekomendasi['provinsi']

        # Jika provinsi berupa list
        if isinstance(provinsi, list):
            jumlah_provinsi = len(provinsi)

            st.caption(
                f"Terdapat **{jumlah_provinsi} provinsi** "
                f"yang berada dalam kelompok Cluster {cluster_prediksi}."
            )

            # Tampilkan dalam beberapa kolom
            cols = st.columns(3)

            for i, nama_provinsi in enumerate(provinsi):
                with cols[i % 3]:
                    st.write(f"• {nama_provinsi}")

        else:
            daftar_provinsi = [
                p.strip()
                for p in str(provinsi).split(",")
                if p.strip()
            ]

            st.caption(
                f"Terdapat **{len(daftar_provinsi)} provinsi** "
                f"yang berada dalam kelompok Cluster {cluster_prediksi}."
            )

            cols = st.columns(3)

            for i, nama_provinsi in enumerate(daftar_provinsi):
                with cols[i % 3]:
                    st.write(f"• {nama_provinsi}")

        # ------------------------------------------------------------------
        # REKOMENDASI KEBIJAKAN
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Rekomendasi Kebijakan")

        st.caption(
            "Berikut merupakan rekomendasi program yang dapat "
            "dipertimbangkan berdasarkan karakteristik kelompok wilayah."
        )

        for i, (program, penjelasan) in enumerate(
            rekomendasi['rekomendasi'],
            start=1
        ):
            with st.expander(
                f"{i}. {program}",
                expanded=(i == 1)
            ):
                st.write(penjelasan)
                    
        # ------------------------------------------------------------------
        # INFORMASI PCA
        # ------------------------------------------------------------------
        with st.expander("Informasi Koordinat PCA"):
            st.write(
                """
                Nilai PCA 1 dan PCA 2 merupakan hasil reduksi dimensi
                dari enam indikator kemiskinan. Koordinat ini digunakan
                untuk menggambarkan posisi data wilayah dalam ruang
                dua dimensi dan membantu visualisasi hasil pengelompokan.
                """
            )

        # ------------------------------------------------------------------
        # GENERATE PDF
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Laporan Hasil Simulasi")

        st.write(
            "Laporan PDF berikut berisi data input, hasil prediksi cluster, "
            "koordinat PCA, karakteristik kelompok, daftar provinsi, dan "
            "rekomendasi kebijakan."
        )

        try:
            pdf_laporan = buat_laporan_pdf(
                waktu_simulasi=waktu_simulasi,
                input_data=input_hasil,
                cluster_prediksi=cluster_prediksi,
                pca1=pca1,
                pca2=pca2,
                rekomendasi=rekomendasi
            )

            st.download_button(
                label="Download Laporan Hasil Simulasi (PDF)",
                data=pdf_laporan,
                file_name=(
                    f"laporan_simulasi_cluster_"
                    f"{cluster_prediksi}_"
                    f"{waktu_simulasi.replace(':', '-').replace(' ', '_')}.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )

            st.success(
                "Laporan PDF berhasil dibuat dan siap diunduh."
            )

        except ImportError:
            st.error(
                "Library ReportLab belum terpasang. "
                "Silakan jalankan: pip install reportlab"
            )

        except Exception as error_pdf:
            st.error(
                f"Laporan PDF gagal dibuat: {str(error_pdf)}"
            )

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.caption(
    "SIPROKEM — Dashboard Pengelompokan Profil Kemiskinan Provinsi di Indonesia "
    "Menggunakan K-Means"
)