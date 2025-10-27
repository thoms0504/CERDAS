"""
Script untuk menghasilkan data sample jabatan CASN dengan 5 tabel terpisah.
Jalankan: python generate_sample_data.py
Output: 5 file CSV terpisah
"""

import pandas as pd
import random
from datetime import datetime, timedelta

# ==================== TABEL 1: MASTER INSTANSI ====================
INSTANSI_DATA = [
    {
        "instansi_eselon_1": "BADAN PUSAT STATISTIK",
        "instansi_eselon_2": "BADAN PUSAT STATISTIK PROVINSI LAMPUNG",
        "instansi_eselon_3": "BADAN PUSAT STATISTIK KOTA BANDAR LAMPUNG",
        "lokasi": "KOTA BANDAR LAMPUNG, PROVINSI LAMPUNG"
    },
    {
        "instansi_eselon_1": "BADAN PUSAT STATISTIK",
        "instansi_eselon_2": "SEKRETARIAT UTAMA",
        "instansi_eselon_3": "BIRO HUMAS DAN HUKUM",
        "lokasi": "JAKARTA PUSAT, DKI JAKARTA"
    },
    {
        "instansi_eselon_1": "BADAN PUSAT STATISTIK",
        "instansi_eselon_2": "PUSAT PENDIDIKAN DAN PELATIHAN",
        "instansi_eselon_3": "-",
        "lokasi": "JAKARTA SELATAN, DKI JAKARTA"
    },
    {
        "instansi_eselon_1": "BADAN PUSAT STATISTIK",
        "instansi_eselon_2": "BADAN PUSAT STATISTIK PROVINSI LAMPUNG",
        "instansi_eselon_3": "-",
        "lokasi": "KOTA BANDAR LAMPUNG, PROVINSI LAMPUNG"
    },
    {
        "instansi_eselon_1": "BADAN PUSAT STATISTIK",
        "instansi_eselon_2": "BADAN PUSAT STATISTIK PROVINSI JAWA TENGAH",
        "instansi_eselon_3": "-",
        "lokasi": "KOTA SEMARANG, JAWA TENGAH"
    },
    {
        "instansi_eselon_1": "KEMENTERIAN KOMUNIKASI DAN INFORMATIKA",
        "instansi_eselon_2": "DIREKTORAT JENDERAL INFORMASI DAN KOMUNIKASI PUBLIK",
        "instansi_eselon_3": "DIREKTORAT KOMUNIKASI DAN MEDIA MASSA",
        "lokasi": "JAKARTA PUSAT, DKI JAKARTA"
    },
    {
        "instansi_eselon_1": "KEMENTERIAN KEUANGAN",
        "instansi_eselon_2": "DIREKTORAT JENDERAL PAJAK",
        "instansi_eselon_3": "KANTOR WILAYAH DJP JAKARTA SELATAN",
        "lokasi": "JAKARTA SELATAN, DKI JAKARTA"
    },
    {
        "instansi_eselon_1": "KEMENTERIAN KEUANGAN",
        "instansi_eselon_2": "DIREKTORAT JENDERAL BEA DAN CUKAI",
        "instansi_eselon_3": "KANTOR WILAYAH DJBC JAWA BARAT",
        "lokasi": "KOTA BANDUNG, JAWA BARAT"
    },
    {
        "instansi_eselon_1": "KEMENTERIAN PENDIDIKAN, KEBUDAYAAN, RISET, DAN TEKNOLOGI",
        "instansi_eselon_2": "DIREKTORAT JENDERAL PENDIDIKAN TINGGI",
        "instansi_eselon_3": "DIREKTORAT PEMBELAJARAN DAN KEMAHASISWAAN",
        "lokasi": "JAKARTA SELATAN, DKI JAKARTA"
    },
    {
        "instansi_eselon_1": "KEMENTERIAN KESEHATAN",
        "instansi_eselon_2": "DIREKTORAT JENDERAL PELAYANAN KESEHATAN",
        "instansi_eselon_3": "DIREKTORAT PELAYANAN KESEHATAN RUJUKAN",
        "lokasi": "JAKARTA SELATAN, DKI JAKARTA"
    },
    {
        "instansi_eselon_1": "BADAN KEPEGAWAIAN NEGARA",
        "instansi_eselon_2": "DEPUTI BIDANG MUTASI PEGAWAI",
        "instansi_eselon_3": "-",
        "lokasi": "JAKARTA PUSAT, DKI JAKARTA"
    },
    {
        "instansi_eselon_1": "KEMENTERIAN DALAM NEGERI",
        "instansi_eselon_2": "DIREKTORAT JENDERAL OTONOMI DAERAH",
        "instansi_eselon_3": "DIREKTORAT KEUANGAN DAERAH",
        "lokasi": "JAKARTA PUSAT, DKI JAKARTA"
    },
]

# ==================== TABEL 4: MASTER PROGRAM STUDI ====================
PROGRAM_STUDI_DATA = [
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "DESAIN KOMUNIKASI VISUAL",
        "deskripsi": "Jurusan Desain Komunikasi Visual atau DKV adalah bagian dari ilmu desain yang mempelajari tentang konsep komunikasi dan ungkapan kreatif, dengan memanfaatkan elemen visual untuk menyampaikan pesan dengan tujuan tertentu. Karena unsur pesan memiliki peran yang sangat penting, lulusan jurusan DKV diharapkan mampu mengelolah pesan tersebut menjadi sesuatu yang menarik, informatif, dan komunikatif, sehingga bisa disampaikan secara efektif."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "HUBUNGAN MASYARAKAT",
        "deskripsi": "Jurusan Hubungan Masyarakat (Humas) atau Public Relation (PR) mempelajari cara mencipatkan dan mengelola reputasi serta citra positif suatu organisasi. Mahasiswa akan belajar melakukan komunikasi yang baik, negosiasi, creative thinking, advertising, dan lain-lain. Hal tersebut berkaitan dengan profesi yang akan dijalani di bidang humas atau public relation nantinya. Profesi di bidang ini berkaitan dengan bagaimana cara mengelola hubungan komunikasi antara organisasi dengan pihak eksternal, seperti masyarakat, pemerintah, investor, serta dengan pihak internal perusahaan, yakni karyawan."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "HUKUM",
        "deskripsi": "Ilmu Hukum adalah studi yang mempelajari berbagai sistem hukum yang berkaitan dengan kehidupan kemasyarakatan. Di Prodi Ilmu Hukum, mahasiswa juga belajar mengenai perundang-undangan termasuk di dalamnya hukum dasar (konstitusi, hukum perdata, hukum dagang, hukum tata negara, hukum publik, hukum tata pidana) hingga hukum internasional dengan cakupan yang cukup luas. Pada akhir masa kuliah, biasanya mahasiswa diwajibkan mengikuti magang kerja atau praktik di berbagai firma hukum, lembaga pengadilan dan kantor kejaksaan."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "ILMU KOMUNIKASI",
        "deskripsi": "Jurusan Ilmu Komunikasi adalah studi yang mempelajari proses penyampaian pesan secara efektif dari komunikator (pemberi pesan) kepada komunikan (penerima pesan) melalui berbagai media. Di jurusan ini kamu akan mempelajari komunikasi dan interaksi antar individu, media, periklanan/publisitas, komunikasi interkultural, hingga komunikasi media sosial."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "JURNALISTIK",
        "deskripsi": "Jurusan Jurnalistik mengajak mahasiswa mendalami dunia jurnalistik beserta karakter dan dinamika yang terjadi di berbagai media massa (televisi, radio, cetak, dan online). Mahasiswa akan mempelajari bagaimana suatu informasi dikomunikasikan melalui media massa, dan mengetahui karakter masing-masing media beserta perbaharannya. Dalam perkuliahan, mahasiswa akan belajar berbagai hal termasuk mencari, menulis, serta menyunting berita. Pada akhir kuliah, Umumnya akan dibentuk jurnal atau koran kampus Jurusan Jurnalistik bila berdin sendiri. Kulminasi: Namun, di beberapa kampus Jurusan Jurnalistik bisa berdiri sendiri."
    },
    {
        "tingkat_pendidikan": "DIII",
        "nama_program_studi": "MANAJEMEN",
        "deskripsi": "Jurusan Manajemen adalah program studi yang mempelajari tentang kegiatan perusahaan atau korporasi, yang masih memiliki keterkaitan dengan ilmu ekonomi dan bisnis. Jika mendalami Ilmu Manajemen maka akan belajar menjadi seorang 'pengendali' layaknya tokoh Aang pada serial Avatar. Jika Aang mengendalikan elemen bumi, kamu akan mengendalikan roda perusahaan berupa kegiatan manajemen perusahaan, organisasi, strategi bisnis, ketenagakerjaan dan kepegawaian, manajemen produksi, pemasaran, administrasi, hingga organisasi nirlaba."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "MANAJEMEN KOMUNIKASI",
        "deskripsi": "Jurusan manajemen komunikasi adalah program studi yang fokus pada perencanaan, pelaksanaan, dan evaluasi strategi komunikasi secara efektif di berbagai konteks, seperti organisasi dan pemasaran. Lulusannya dibekali kemampuan untuk mengelola komunikasi internal dan eksternal, membangun citra perusahaan, serta merancang pesan yang persuasif untuk mencapai tujuan tertentu."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "MATEMATIKA",
        "deskripsi": "Jurusan Matematika fokus mempelajari tentang struktur, pola, dan hubungan matematika secara mendalam, seperti geometri, aljabar, teori matematika diskrit dan matematika terapan. Pasa jurusan ini akan belajar berpikir dengan beragam konsep dalam matematika seperti logika, analisis, dan pemodelan matematis."
    },
    {
        "tingkat_pendidikan": "DIII",
        "nama_program_studi": "SISTEM INFORMASI",
        "deskripsi": "Jurusan Sistem Informasi adalah bidang keilmuan yang menggabungkan ilmu komputer dengan bisnis dan manajemen. Di prodi ini kamu akan belajar gimana mengidentifikasi kebutuhan dan proses bisnis perusahaan berdasarkan data-data yang dimiliki perusahaan. Kemudian merancang sistem yang sesuai dengan kebutuhan perusahaan."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "STATISTIKA",
        "deskripsi": "Statistika merupakan perpanjangan dari belajar matematika. Bedanya, Statistika lebih fokus ke arah data dan angka dalam kuantitas besar. Sederhananya, jika kuliah di jurusan Statistika akan mempelajari cara mengumpulkan, menganalisis, dan menyajikan data ke dalam bahasa yang mudah dipahami sehingga bisa dijadikan sebuah informasi untuk banyak orang."
    },
    {
        "tingkat_pendidikan": "DIII",
        "nama_program_studi": "TEKNOLOGI INFORMASI",
        "deskripsi": "Teknik Informatika merupakan bidang ilmu yang mempelajari bagaimana menggunakan teknologi komputer secara optimal guna menangani masalah transformasi atau pengolahan data dengan proses logika. Di Jurusan Teknik Informatika kamu akan banyak belajar tentang pemrograman untuk pengembangan perangkat lunak, aplikasi, hingga sistem operasi perangkat lunak. Selama kuliah kamu akan banyak mengkaji pemrograman dan komputasi, dan dibekali pula dengan keterampilan merancang perangkat lunak."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "TEKNIK INFORMATIKA",
        "deskripsi": "Teknik Informatika merupakan bidang ilmu yang mempelajari bagaimana menggunakan teknologi komputer secara optimal guna menangani masalah transformasi atau pengolahan data dengan proses logika. Di Jurusan Teknik Informatika kamu akan banyak belajar tentang pemrograman untuk pengembangan perangkat lunak, aplikasi, hingga sistem operasi perangkat lunak. Selama kuliah kamu akan banyak mengkaji pemrograman dan komputasi, dan dibekali pula dengan keterampilan merancang perangkat lunak."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "ILMU KOMPUTER",
        "deskripsi": "Ilmu Komputer adalah studi tentang teori, eksperimen, dan rekayasa yang membentuk dasar untuk desain dan penggunaan komputer. Mahasiswa akan mempelajari algoritma, struktur data, kecerdasan buatan, sistem operasi, dan pengembangan software."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "ADMINISTRASI PUBLIK",
        "deskripsi": "Administrasi Publik mempelajari tentang kebijakan publik, manajemen pemerintahan, dan pelayanan publik. Mahasiswa akan dibekali kemampuan dalam merencanakan, mengorganisir, dan mengevaluasi program-program pemerintah untuk kesejahteraan masyarakat."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "AKUNTANSI",
        "deskripsi": "Akuntansi adalah studi tentang pencatatan, pengklasifikasian, dan pelaporan transaksi keuangan. Mahasiswa akan mempelajari audit, perpajakan, sistem informasi akuntansi, dan standar pelaporan keuangan."
    },
    {
        "tingkat_pendidikan": "S1",
        "nama_program_studi": "EKONOMI PEMBANGUNAN",
        "deskripsi": "Ekonomi Pembangunan fokus pada analisis dan perencanaan pembangunan ekonomi, terutama di negara berkembang. Mahasiswa mempelajari kebijakan ekonomi, pertumbuhan ekonomi, kemiskinan, dan distribusi pendapatan."
    },
]

# ==================== TABEL 5: MASTER JURUSAN SMA/SMK ====================
JURUSAN_SMA_DATA = [
    {
        "nama_jurusan": "IPA",
        "deskripsi": "Program Ilmu Pengetahuan Alam (IPA) mempersiapkan siswa melanjutkan pendidikan ke jenjang pendidikan tinggi yang berkaitan dengan Matematika dan Ilmu Pengetahuan Alam baik dalam bidang akademik maupun pendidikan profesional. Program ini juga mempersiapkan bekal kehidupan kepada siswa yang tidak langsung untuk bekerja di bidang yang sesuai dengan program ini."
    },
    {
        "nama_jurusan": "IPS",
        "deskripsi": "Ruang lingkup kajian IPS meliputi hal-hal yang berkaitan dengan keluarga, masyarakat setempat, uang, tabungan, pajak, ekonomi setempat, wilayah propinsi, wilayah kepulauan dan pengenalan kawasan dunia. Sedangkan ruang lingkup pengajaran sejarah meliputi: sejarah lokal, kerajaan di Indonesia, tokoh dan peristiwa, bangunan sejarah, Indonesia pada zaman penjajahan dan beberapa peristiwa penting masa kemerdekaan."
    },
    {
        "nama_jurusan": "BAHASA",
        "deskripsi": "Program Bahasa fokus pada pembelajaran bahasa dan sastra, baik bahasa Indonesia maupun bahasa asing seperti Inggris, Mandarin, Jepang, dll. Siswa akan mempelajari linguistik, sastra, dan budaya untuk menjadi ahli bahasa atau bekerja di bidang komunikasi dan media."
    },
    {
        "nama_jurusan": "TEKNIK KOMPUTER DAN JARINGAN",
        "deskripsi": "Jurusan TKJ mempelajari tentang cara instalasi PC, instalasi LAN, maintenance komputer dan troubleshooting. Lulusan TKJ diharapkan mampu mengelola jaringan komputer, instalasi perangkat keras dan lunak, serta melakukan perbaikan perangkat komputer."
    },
    {
        "nama_jurusan": "REKAYASA PERANGKAT LUNAK",
        "deskripsi": "Jurusan RPL fokus pada pengembangan software, mulai dari analisis kebutuhan, desain, coding, testing hingga maintenance. Siswa akan belajar berbagai bahasa pemrograman dan metodologi pengembangan perangkat lunak."
    },
    {
        "nama_jurusan": "AKUNTANSI",
        "deskripsi": "Jurusan Akuntansi SMK mempelajari proses pencatatan, pengelompokan, dan pelaporan transaksi keuangan. Siswa akan dibekali kemampuan membuat laporan keuangan, perpajakan dasar, dan penggunaan software akuntansi."
    },
    {
        "nama_jurusan": "ADMINISTRASI PERKANTORAN",
        "deskripsi": "Jurusan Administrasi Perkantoran mempelajari tata kelola perkantoran, kesekretariatan, korespondensi, kearsipan, dan manajemen administrasi. Lulusan diharapkan mampu mengelola administrasi kantor dengan baik."
    },
]

# ==================== TABEL 3: TUGAS POKOK DAN FUNGSI ====================
TUPOKSI_DATA = [
    {
        "nama_jabatan": "Pranata Komputer Terampil",
        "deskripsi_tugas_pokok": "Melaksanakan kegiatan teknologi informasi berbasis komputer yang meliputi tata kelola dan tata laksana teknologi informasi, infrastruktur teknologi informasi, serta sistem informasi dan multimedia",
        "rincian_kegiatan_fungsi": """melakukan penggandaan data;
melakukan deteksi dan/atau perbaikan terhadap permasalahan yang terjadi pada sistem jaringan lokal (local area network);
melakukan pencatatan infrastruktur teknologi informasi;
melakukan pemasangan kabel untuk infrastruktur teknologi informasi;
melakukan pemeliharaan perangkat teknologi informasi end user;
melakukan deteksi dan/atau perbaikan terhadap permasalahan perangkat teknologi informasi end user;
melakukan perekaman data dengan pemindaian;
melakukan perekaman data tanpa validasi;
melakukan validasi hasil perekaman data;
melakukan perekaman data dengan validasi;
membuat query sederhana;
melakukan konversi data;
melakukan kompilasi data pengolahan;
melakukan perekaman data spasial;
melakukan uji coba program multimedia interaktif;"""
    },
    {
        "nama_jabatan": "Analis Hukum Ahli Pertama",
        "deskripsi_tugas_pokok": "Melakukan analisis hukum, penelitian hukum, dan memberikan pendapat hukum serta saran terkait permasalahan hukum",
        "rincian_kegiatan_fungsi": """melakukan pengkajian terhadap peraturan perundang-undangan;
melakukan analisis terhadap permasalahan hukum;
menyusun konsep pendapat dan saran hukum;
melakukan penelitian hukum sederhana;
memberikan pertimbangan hukum;
menyusun laporan hasil analisis hukum;
melakukan koordinasi dengan unit terkait;"""
    },
    {
        "nama_jabatan": "Perencana Ahli Pertama",
        "deskripsi_tugas_pokok": "Melakukan penyusunan rencana program dan kegiatan, monitoring dan evaluasi pelaksanaan rencana",
        "rincian_kegiatan_fungsi": """mengumpulkan dan mengolah data perencanaan;
melakukan analisis data untuk penyusunan rencana;
menyusun konsep rencana program dan kegiatan;
melakukan monitoring pelaksanaan rencana;
menyusun laporan evaluasi pelaksanaan rencana;
melakukan koordinasi dengan unit terkait dalam penyusunan rencana;"""
    },
    {
        "nama_jabatan": "Pranata Hubungan Masyarakat Ahli Pertama",
        "deskripsi_tugas_pokok": "Melakukan kegiatan kehumasan meliputi publikasi, dokumentasi, dan pembinaan hubungan dengan media",
        "rincian_kegiatan_fungsi": """menyusun konsep materi publikasi;
melakukan dokumentasi kegiatan;
melakukan monitoring pemberitaan media;
menyusun press release;
melakukan koordinasi dengan media;
mengelola konten media sosial instansi;
menyusun laporan kegiatan kehumasan;"""
    },
    {
        "nama_jabatan": "Arsiparis Terampil",
        "deskripsi_tugas_pokok": "Melakukan kegiatan kearsipan yang meliputi penciptaan, penggunaan, pemeliharaan, penyusutan dan pemusnahan arsip",
        "rincian_kegiatan_fungsi": """melakukan penerimaan arsip;
melakukan pencatatan arsip;
melakukan penyimpanan arsip sesuai sistem;
melakukan pemeliharaan arsip;
melakukan pelayanan arsip;
melakukan penyusutan arsip;
menyusun daftar arsip;"""
    },
    {
        "nama_jabatan": "Analis Kebijakan Ahli Muda",
        "deskripsi_tugas_pokok": "Melakukan analisis kebijakan publik, evaluasi kebijakan, dan menyusun rekomendasi kebijakan",
        "rincian_kegiatan_fungsi": """melakukan kajian dan analisis kebijakan;
menyusun rekomendasi kebijakan;
melakukan evaluasi implementasi kebijakan;
menyusun laporan analisis kebijakan;
melakukan koordinasi dengan stakeholder;
mempresentasikan hasil analisis kebijakan;"""
    },
    {
        "nama_jabatan": "Auditor Ahli Muda",
        "deskripsi_tugas_pokok": "Melakukan audit internal terhadap pengelolaan keuangan dan kinerja organisasi",
        "rincian_kegiatan_fungsi": """menyusun program audit;
melakukan audit keuangan;
melakukan audit kinerja;
melakukan audit dengan tujuan tertentu;
menyusun laporan hasil audit;
melakukan monitoring tindak lanjut audit;
memberikan konsultasi terkait audit;"""
    },
]

# ==================== JABATAN TEMPLATES ====================
JABATAN_TEMPLATES = [
    {
        "nama": "Pranata Komputer Terampil",
        "tingkat": "DIII",
        "prodi": ["TEKNOLOGI INFORMASI", "SISTEM INFORMASI"],
        "gaji_min": 2918320,
        "gaji_max": 6994006
    },
    {
        "nama": "Analis Hukum Ahli Pertama",
        "tingkat": "S1",
        "prodi": ["HUKUM"],
        "gaji_min": 3398120,
        "gaji_max": 8600528
    },
    {
        "nama": "Perencana Ahli Pertama",
        "tingkat": "S1",
        "prodi": ["MATEMATIKA", "STATISTIK", "STATISTIKA"],
        "gaji_min": 3398120,
        "gaji_max": 8600528
    },
    {
        "nama": "Pranata Hubungan Masyarakat Ahli Pertama",
        "tingkat": "S1",
        "prodi": ["MANAJEMEN KOMUNIKASI", "DESAIN KOMUNIKASI VISUAL", "ILMU KOMUNIKASI", "JURNALISTIK", "HUBUNGAN MASYARAKAT"],
        "gaji_min": 3398120,
        "gaji_max": 8600528
    },
    {
        "nama": "Arsiparis Terampil",
        "tingkat": "DIII",
        "prodi": ["MANAJEMEN", "TEKNOLOGI INFORMASI", "SISTEM INFORMASI"],
        "gaji_min": 2908320,
        "gaji_max": 6984006
    },
    {
        "nama": "Analis Kebijakan Ahli Muda",
        "tingkat": "S1",
        "prodi": ["ADMINISTRASI PUBLIK", "ILMU PEMERINTAHAN"],
        "gaji_min": 3500000,
        "gaji_max": 9000000
    },
    {
        "nama": "Programmer Ahli Pertama",
        "tingkat": "S1",
        "prodi": ["TEKNIK INFORMATIKA", "ILMU KOMPUTER", "SISTEM INFORMASI"],
        "gaji_min": 3398120,
        "gaji_max": 8600528
    },
    {
        "nama": "Data Scientist Ahli Muda",
        "tingkat": "S1",
        "prodi": ["STATISTIKA", "MATEMATIKA", "ILMU KOMPUTER"],
        "gaji_min": 3500000,
        "gaji_max": 9500000
    },
    {
        "nama": "Auditor Ahli Muda",
        "tingkat": "S1",
        "prodi": ["AKUNTANSI"],
        "gaji_min": 3500000,
        "gaji_max": 9000000
    },
    {
        "nama": "Analis SDM Ahli Pertama",
        "tingkat": "S1",
        "prodi": ["PSIKOLOGI", "MANAJEMEN"],
        "gaji_min": 3398120,
        "gaji_max": 8600528
    },
]


def generate_tabel_instansi(n_samples=50):
    """Generate tabel 1: Master Instansi"""
    data = []
    
    for _ in range(n_samples):
        row = random.choice(INSTANSI_DATA).copy()
        data.append(row)
    
    return pd.DataFrame(data)


def generate_tabel_jabatan(n_samples=100):
    """Generate tabel 2: Data Jabatan dengan Formasi"""
    data = []
    periode_seleksi = "2024"
    
    for i in range(n_samples):
        jabatan = random.choice(JABATAN_TEMPLATES)
        instansi_row = random.choice(INSTANSI_DATA)
        
        # Pilih program studi dari list
        if len(jabatan["prodi"]) > 1:
            n_prodi = random.randint(1, min(3, len(jabatan["prodi"])))
            selected_prodi = random.sample(jabatan["prodi"], n_prodi)
            prodi_str = "\n".join(selected_prodi)
        else:
            prodi_str = jabatan["prodi"][0]
        
        row = {
            "periode_seleksi_casn": periode_seleksi,
            "nama_jabatan": jabatan["nama"],
            "kualifikasi_tingkat_pendidikan": jabatan["tingkat"],
            "kualifikasi_program_studi_jurusan": prodi_str,
            "alokasi_kebutuhan": random.randint(1, 5),
            "eselon_1_penempatan": instansi_row["instansi_eselon_1"],
            "eselon_2_penempatan": instansi_row["instansi_eselon_2"],
            "eselon_3_penempatan": instansi_row["instansi_eselon_3"],
            "lokasi": instansi_row["lokasi"],
            "rentang_penghasilan": f"{jabatan['gaji_min']}-{jabatan['gaji_max']}"
        }
        data.append(row)
    
    return pd.DataFrame(data)


def generate_tabel_tupoksi():
    """Generate tabel 3: Tugas Pokok dan Fungsi"""
    return pd.DataFrame(TUPOKSI_DATA)


def generate_tabel_program_studi():
    """Generate tabel 4: Master Program Studi"""
    return pd.DataFrame(PROGRAM_STUDI_DATA)


def generate_tabel_jurusan_sma():
    """Generate tabel 5: Master Jurusan SMA/SMK"""
    return pd.DataFrame(JURUSAN_SMA_DATA)


def main():
    print("🔄 Generating CASN sample data (5 tables)...\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate semua tabel
    print("📊 Table 1: Master Instansi")
    df_instansi = generate_tabel_instansi(n_samples=50)
    filename1 = f"tabel_1_instansi_{timestamp}.csv"
    df_instansi.to_csv(filename1, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved: {filename1} ({len(df_instansi)} rows)")
    
    print("\n📊 Table 2: Data Jabatan dan Formasi")
    df_jabatan = generate_tabel_jabatan(n_samples=200)
    filename2 = f"tabel_2_jabatan_formasi_{timestamp}.csv"
    df_jabatan.to_csv(filename2, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved: {filename2} ({len(df_jabatan)} rows)")
    
    print("\n📊 Table 3: Tugas Pokok dan Fungsi")
    df_tupoksi = generate_tabel_tupoksi()
    filename3 = f"tabel_3_tupoksi_{timestamp}.csv"
    df_tupoksi.to_csv(filename3, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved: {filename3} ({len(df_tupoksi)} rows)")
    
    print("\n📊 Table 4: Master Program Studi")
    df_prodi = generate_tabel_program_studi()
    filename4 = f"tabel_4_program_studi_{timestamp}.csv"
    df_prodi.to_csv(filename4, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved: {filename4} ({len(df_prodi)} rows)")
    
    print("\n📊 Table 5: Master Jurusan SMA/SMK")
    df_jurusan = generate_tabel_jurusan_sma()
    filename5 = f"tabel_5_jurusan_sma_{timestamp}.csv"
    df_jurusan.to_csv(filename5, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved: {filename5} ({len(df_jurusan)} rows)")
    
    # Summary
    print("\n" + "="*60)
    print("✨ SUMMARY")
    print("="*60)
    print(f"📁 Total files generated: 5")
    print(f"📊 Total records:")
    print(f"   - Instansi: {len(df_instansi)}")
    print(f"   - Jabatan & Formasi: {len(df_jabatan)}")
    print(f"   - Tugas Pokok & Fungsi: {len(df_tupoksi)}")
    print(f"   - Program Studi: {len(df_prodi)}")
    print(f"   - Jurusan SMA/SMK: {len(df_jurusan)}")
    
    print(f"\n📈 Statistics Tabel 2 (Jabatan):")
    print(f"   - Unique jabatan: {df_jabatan['nama_jabatan'].nunique()}")
    print(f"   - Total formasi: {df_jabatan['alokasi_kebutuhan'].sum()}")
    print(f"   - Jabatan by tingkat pendidikan:")
    print(df_jabatan['kualifikasi_tingkat_pendidikan'].value_counts().to_string())
    
    print("\n" + "="*60)
    print("🎉 Data generation completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()