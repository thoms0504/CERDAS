# generate_sample_data.py — Generate sample data untuk testing

"""
Script ini menghasilkan data sample jabatan PNS untuk keperluan testing.
Jalankan: python generate_sample_data.py
Output: sample_jabatan_pns.csv
"""

import pandas as pd
import random
from datetime import datetime

# Template data
INSTANSI = [
    "Kementerian Dalam Negeri",
    "Kementerian Keuangan",
    "Kementerian Komunikasi dan Informatika",
    "Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi",
    "Kementerian Kesehatan",
    "Kementerian Pertanian",
    "Kementerian BUMN",
    "Kementerian Hukum dan HAM",
    "Badan Pusat Statistik",
    "Badan Kepegawaian Negara"
]

UNIT_ORGANISASI = {
    "Kementerian Dalam Negeri": [
        "Direktorat Jenderal Otonomi Daerah",
        "Direktorat Jenderal Bina Keuangan Daerah",
        "Direktorat Jenderal Politik dan Pemerintahan Umum",
        "Inspektorat Jenderal"
    ],
    "Kementerian Keuangan": [
        "Direktorat Jenderal Pajak",
        "Direktorat Jenderal Bea dan Cukai",
        "Direktorat Jenderal Perbendaharaan",
        "Badan Kebijakan Fiskal"
    ],
    "Kementerian Komunikasi dan Informatika": [
        "Direktorat Jenderal Informasi dan Komunikasi Publik",
        "Direktorat Jenderal Aplikasi Informatika",
        "Direktorat e-Government",
        "Balai Sertifikasi Elektronik"
    ],
    "Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi": [
        "Direktorat Jenderal Pendidikan Tinggi",
        "Direktorat Jenderal Guru dan Tenaga Kependidikan",
        "Pusat Penelitian dan Pengembangan",
        "Balai Pengembangan Talenta Indonesia"
    ],
    "Kementerian Kesehatan": [
        "Direktorat Jenderal Pelayanan Kesehatan",
        "Direktorat Jenderal Pencegahan dan Pengendalian Penyakit",
        "Badan Penelitian dan Pengembangan Kesehatan",
        "Inspektorat Jenderal"
    ],
    "Kementerian Pertanian": [
        "Direktorat Jenderal Tanaman Pangan",
        "Direktorat Jenderal Perkebunan",
        "Badan Penyuluhan dan Pengembangan SDM Pertanian",
        "Badan Karantina Pertanian"
    ],
    "Kementerian BUMN": [
        "Deputi Bidang Usaha Pertambangan, Industri Strategis dan Media",
        "Deputi Bidang Usaha Infrastruktur dan Kawasan",
        "Biro Hukum dan Humas",
        "Inspektorat"
    ],
    "Kementerian Hukum dan HAM": [
        "Direktorat Jenderal Administrasi Hukum Umum",
        "Direktorat Jenderal Imigrasi",
        "Direktorat Jenderal Pemasyarakatan",
        "Badan Penelitian dan Pengembangan Hukum dan HAM"
    ],
    "Badan Pusat Statistik": [
        "Direktorat Statistik Kependudukan dan Ketenagakerjaan",
        "Direktorat Statistik Ekonomi",
        "Direktorat Diseminasi Statistik",
        "Pusat Data dan Informasi"
    ],
    "Badan Kepegawaian Negara": [
        "Deputi Bidang Mutasi Pegawai",
        "Deputi Bidang Pembinaan Pegawai",
        "Pusat Pengembangan Sistem Kepegawaian",
        "Pusat Penilaian Kompetensi ASN"
    ]
}

JABATAN_TEMPLATES = [
    {
        "judul": "Analis Kebijakan",
        "pendidikan": "S1/S2 Administrasi Publik/Ilmu Pemerintahan/Kebijakan Publik",
        "kompetensi": "Analisis Kebijakan, Riset, Komunikasi, Microsoft Office, Manajemen Proyek",
        "tupoksi": "Melakukan analisis dan evaluasi kebijakan publik, menyusun rekomendasi kebijakan, dan menyajikan hasil analisis kepada stakeholder terkait."
    },
    {
        "judul": "Programmer",
        "pendidikan": "S1 Teknik Informatika/Ilmu Komputer/Sistem Informasi",
        "kompetensi": "Python, Java, PHP, Database (MySQL/PostgreSQL), Git, Agile/Scrum",
        "tupoksi": "Mengembangkan, maintain, dan optimize sistem informasi pemerintah, melakukan troubleshooting, dan dokumentasi sistem."
    },
    {
        "judul": "Data Scientist",
        "pendidikan": "S1/S2 Statistika/Matematika/Ilmu Komputer/Data Science",
        "kompetensi": "Python, R, Machine Learning, Data Visualization, SQL, Big Data",
        "tupoksi": "Menganalisis data besar, membangun model prediktif, visualisasi data, dan memberikan insights untuk pengambilan keputusan."
    },
    {
        "judul": "Auditor",
        "pendidikan": "S1 Akuntansi/Manajemen Keuangan",
        "kompetensi": "Audit Internal, Risk Assessment, Financial Analysis, Compliance, Microsoft Excel",
        "tupoksi": "Melakukan audit internal, mengevaluasi sistem pengendalian internal, dan menyusun laporan audit serta rekomendasi perbaikan."
    },
    {
        "judul": "Perancang Peraturan Perundang-undangan",
        "pendidikan": "S1/S2 Ilmu Hukum",
        "kompetensi": "Legal Drafting, Penelitian Hukum, Analisis Regulasi, Komunikasi, Microsoft Office",
        "tupoksi": "Merancang naskah peraturan perundang-undangan, melakukan harmonisasi peraturan, dan memberikan legal advice."
    },
    {
        "judul": "Analis SDM",
        "pendidikan": "S1 Psikologi/Manajemen SDM",
        "kompetensi": "Rekrutmen & Seleksi, Assessment, Pengembangan SDM, HRIS, Komunikasi",
        "tupoksi": "Mengelola proses rekrutmen, mengembangkan sistem penilaian kinerja, dan merencanakan program pengembangan pegawai."
    },
    {
        "judul": "Public Relations Officer",
        "pendidikan": "S1 Ilmu Komunikasi/Public Relations",
        "kompetensi": "Media Relations, Press Release Writing, Social Media Management, Event Management, Komunikasi",
        "tupoksi": "Mengelola komunikasi publik instansi, membuat siaran pers, mengelola media sosial, dan koordinasi event."
    },
    {
        "judul": "Network Administrator",
        "pendidikan": "S1 Teknik Informatika/Teknik Komputer",
        "kompetensi": "Cisco, Network Security, Linux/Windows Server, Firewall, Troubleshooting",
        "tupoksi": "Mengelola infrastruktur jaringan, memastikan keamanan jaringan, troubleshooting, dan dokumentasi jaringan."
    },
    {
        "judul": "Graphic Designer",
        "pendidikan": "S1 Desain Komunikasi Visual/Desain Grafis",
        "kompetensi": "Adobe Photoshop, Illustrator, InDesign, Video Editing, Typography",
        "tupoksi": "Membuat desain visual untuk kampanye publik, infografis, publikasi, dan konten media sosial instansi."
    },
    {
        "judul": "Peneliti",
        "pendidikan": "S2/S3 sesuai bidang penelitian",
        "kompetensi": "Metodologi Penelitian, Analisis Data, Academic Writing, Publikasi Ilmiah, Presentasi",
        "tupoksi": "Melakukan penelitian di bidang tertentu, mempublikasikan hasil penelitian, dan memberikan rekomendasi kebijakan berbasis riset."
    },
    {
        "judul": "Analis Anggaran",
        "pendidikan": "S1 Akuntansi/Manajemen Keuangan/Ekonomi",
        "kompetensi": "Budget Planning, Financial Analysis, Microsoft Excel, Reporting, SAP/Oracle",
        "tupoksi": "Menyusun anggaran, menganalisis realisasi anggaran, melakukan forecast, dan menyajikan laporan keuangan."
    },
    {
        "judul": "Content Writer",
        "pendidikan": "S1 Jurnalistik/Sastra/Komunikasi",
        "kompetensi": "Writing, SEO, Content Strategy, Editing, Social Media",
        "tupoksi": "Membuat konten berkualitas untuk website, media sosial, dan publikasi instansi, serta melakukan editing dan proofreading."
    },
]

LOKASI = [
    "Jakarta", "Bandung", "Surabaya", "Medan", "Makassar", 
    "Semarang", "Palembang", "Denpasar", "Yogyakarta", "Manado"
]

def generate_sample_data(n_samples=100):
    """Generate n_samples data jabatan"""
    
    data = []
    
    for i in range(n_samples):
        # Random instansi
        instansi = random.choice(INSTANSI)
        unit = random.choice(UNIT_ORGANISASI.get(instansi, ["Unit Kerja"]))
        
        # Random jabatan template
        jabatan_template = random.choice(JABATAN_TEMPLATES)
        
        # Add variation
        judul_jabatan = f"{jabatan_template['judul']} {random.choice(['Ahli Pertama', 'Ahli Muda', 'Terampil'])}"
        
        # Random formasi & lokasi
        jumlah_formasi = random.randint(1, 10)
        lokasi = random.choice(LOKASI)
        
        # Generate kode formasi
        kode_formasi = f"FORM-2025-{i+1:04d}"
        
        # Construct row
        row = {
            "judul_jabatan": judul_jabatan,
            "instansi": instansi,
            "unit_organisasi": unit,
            "kualifikasi_pendidikan": jabatan_template["pendidikan"],
            "persyaratan_kompetensi": jabatan_template["kompetensi"],
            "tugas_pokok": jabatan_template["tupoksi"],
            "fungsi_jabatan": f"Fungsi {jabatan_template['judul']} di bidang {unit.split()[-1].lower()}",
            "jumlah_formasi": jumlah_formasi,
            "lokasi_penempatan": lokasi,
            "kode_formasi": kode_formasi,
            "tautan_detail": f"https://sscasn.bkn.go.id/formasi/{kode_formasi}",
            "keywords": ""  # Will be auto-generated by app
        }
        
        data.append(row)
    
    return pd.DataFrame(data)


def main():
    print("🔄 Generating sample data...")
    
    # Generate data
    df = generate_sample_data(n_samples=100)
    
    # Save to CSV
    filename = f"sample_jabatan_pns_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)
    
    print(f"✅ Sample data generated: {filename}")
    print(f"📊 Total rows: {len(df)}")
    print(f"📍 Unique instansi: {df['instansi'].nunique()}")
    print(f"🏢 Unique jabatan types: {df['judul_jabatan'].nunique()}")
    print(f"\n📋 Preview:")
    print(df.head())
    
    # Statistics
    print(f"\n📈 Statistics:")
    print(f"  - Avg formasi per jabatan: {df['jumlah_formasi'].mean():.1f}")
    print(f"  - Total formasi: {df['jumlah_formasi'].sum()}")
    print(f"  - Lokasi distribution:")
    print(df['lokasi_penempatan'].value_counts())


if __name__ == "__main__":
    main()
