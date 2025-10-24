# config.py — Configuration file untuk Smart PNS Recommender

"""
File ini berisi konfigurasi yang dapat disesuaikan tanpa mengubah kode utama.
Edit nilai-nilai di bawah sesuai kebutuhan Anda.
"""

# ===========================
# MODEL CONFIGURATION
# ===========================

# Model embedding yang digunakan
# Pilihan populer:
# - "paraphrase-multilingual-MiniLM-L12-v2" (default, balanced)
# - "paraphrase-multilingual-mpnet-base-v2" (lebih akurat, lebih lambat)
# - "distiluse-base-multilingual-cased-v2" (lebih cepat, kurang akurat)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Model LLM untuk reranking
LLM_MODEL = "gpt-4o-mini"  # atau "gpt-4o" untuk hasil lebih baik

# Batch size untuk embedding (sesuaikan dengan RAM)
EMBEDDING_BATCH_SIZE = 32  # Turunkan jika out of memory

# ===========================
# SEARCH CONFIGURATION
# ===========================

# Default weights untuk hybrid search
DEFAULT_WEIGHTS = {
    'bm25': 0.3,           # Keyword matching (BM25)
    'embedding': 0.3,      # Semantic similarity
    'education': 0.2,      # Pendidikan matching
    'skills': 0.1,         # Skills overlap
    'experience': 0.1      # Experience relevance
}

# Top-K default results
DEFAULT_TOP_K = 20

# Minimum score threshold (0-1)
# Kandidat dengan score di bawah ini tidak akan ditampilkan
MIN_SCORE_THRESHOLD = 0.0

# ===========================
# FUZZY MATCHING
# ===========================

# Threshold untuk fuzzy matching pendidikan/jurusan (0-100)
FUZZY_THRESHOLD = 80

# ===========================
# UI CONFIGURATION
# ===========================

# Jumlah hasil yang di-expand secara default
AUTO_EXPAND_RESULTS = 3

# Maksimal panjang text dalam preview (characters)
MAX_TEXT_PREVIEW = 200

# Warna tema (untuk visualisasi)
CHART_COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4',
    'tertiary': '#45B7D1',
    'success': '#51CF66',
    'warning': '#FFA07A'
}

# ===========================
# DATA CONFIGURATION
# ===========================

# Kolom CSV yang WAJIB ada
REQUIRED_COLUMNS = [
    "judul_jabatan",
    "instansi",
    "unit_organisasi"
]

# Kolom yang digunakan untuk ekstraksi keywords (jika kolom 'keywords' kosong)
KEYWORD_SOURCE_COLUMNS = [
    "judul_jabatan",
    "tugas_pokok",
    "persyaratan_kompetensi",
    "kualifikasi_pendidikan",
    "unit_organisasi"
]

# Maksimal keywords per row
MAX_KEYWORDS_PER_ROW = 50

# ===========================
# CACHE CONFIGURATION
# ===========================

# Directory untuk menyimpan cache
CACHE_DIR = "cache"

# Enable/disable caching
USE_CACHE = True

# Auto-clear cache jika lebih dari X hari
CACHE_MAX_AGE_DAYS = 30

# ===========================
# HISTORY & FEEDBACK
# ===========================

# Maksimal history yang disimpan
MAX_HISTORY_ENTRIES = 50

# File paths
FEEDBACK_FILE = "feedback.csv"
HISTORY_FILE = "search_history.json"
PROFILE_FILE = "saved_profiles.json"

# ===========================
# LLM CONFIGURATION
# ===========================

# Temperature untuk LLM (0-1, lower = lebih konsisten)
LLM_TEMPERATURE = 0.3

# Max tokens untuk LLM response
LLM_MAX_TOKENS = 500

# Timeout untuk LLM request (seconds)
LLM_TIMEOUT = 30

# ===========================
# SKILL SYNONYMS
# ===========================

# Dictionary untuk ekspansi skill dengan sinonim
# Format: "skill_utama": ["sinonim1", "sinonim2", ...]
SKILL_SYNONYMS = {
    # Programming Languages
    "python": ["python", "python3", "py", "python programming"],
    "java": ["java", "javase", "javaee", "java programming"],
    "javascript": ["javascript", "js", "node.js", "nodejs"],
    "php": ["php", "php programming"],
    "c++": ["cpp", "c++", "cplusplus"],
    "c#": ["csharp", "c#", "c sharp"],
    
    # Frameworks
    "django": ["django", "django framework", "python django"],
    "laravel": ["laravel", "php laravel"],
    "react": ["react", "reactjs", "react.js"],
    "vue": ["vue", "vuejs", "vue.js"],
    
    # Databases
    "mysql": ["mysql", "my sql", "mysql database"],
    "postgresql": ["postgresql", "postgres", "pgsql"],
    "mongodb": ["mongodb", "mongo", "mongo db"],
    "oracle": ["oracle", "oracle database", "oracle db"],
    
    # Office Skills
    "microsoft office": ["ms office", "microsoft office", "ms word", "ms excel", "ms powerpoint"],
    "excel": ["excel", "ms excel", "microsoft excel", "spreadsheet"],
    "word": ["word", "ms word", "microsoft word"],
    "powerpoint": ["powerpoint", "ms powerpoint", "ppt", "presentation"],
    
    # Soft Skills
    "komunikasi": ["komunikasi", "komunikatif", "public speaking", "presentasi"],
    "leadership": ["leadership", "kepemimpinan", "memimpin", "leader", "pemimpin"],
    "teamwork": ["teamwork", "kerja sama", "kolaborasi", "team player"],
    "problem solving": ["problem solving", "pemecahan masalah", "analisis masalah"],
    "analisis": ["analisis", "analisa", "analytic", "analytical", "analitik"],
    "manajemen": ["manajemen", "management", "mengelola", "pengelolaan"],
    "organisasi": ["organisasi", "organization", "pengorganisasian"],
    
    # Domain Specific
    "data analysis": ["data analysis", "analisis data", "data analytics"],
    "machine learning": ["machine learning", "ml", "pembelajaran mesin"],
    "artificial intelligence": ["artificial intelligence", "ai", "kecerdasan buatan"],
    "cyber security": ["cyber security", "cybersecurity", "keamanan siber", "security"],
    "network": ["network", "networking", "jaringan", "jaringan komputer"],
    "cloud computing": ["cloud computing", "cloud", "komputasi awan"],
    
    # Government Specific
    "kebijakan publik": ["kebijakan publik", "public policy", "kebijakan pemerintah"],
    "administrasi publik": ["administrasi publik", "public administration", "admin publik"],
    "pelayanan publik": ["pelayanan publik", "public service", "layanan publik"],
    "hukum": ["hukum", "law", "legal", "peraturan"],
    "akuntansi": ["akuntansi", "accounting", "akuntan"],
    "keuangan": ["keuangan", "finance", "finansial", "financial"],
}

# ===========================
# EDUCATION LEVELS
# ===========================

# Mapping level pendidikan ke numeric score
# Gunakan untuk matching hierarki pendidikan
EDUCATION_LEVELS = {
    "S3": 5, "DOKTOR": 5, "DOCTORAL": 5, "DR": 5,
    "S2": 4, "MAGISTER": 4, "MASTER": 4, "M.SC": 4, "M.A": 4, "M.M": 4,
    "S1": 3, "SARJANA": 3, "BACHELOR": 3, "B.SC": 3, "B.A": 3,
    "D4": 2.5, "D-IV": 2.5, "SARJANA TERAPAN": 2.5,
    "D3": 2, "D-III": 2, "DIPLOMA": 2,
    "D2": 1.5, "D-II": 1.5,
    "D1": 1, "D-I": 1,
    "SMA": 0.5, "SMK": 0.5, "SLTA": 0.5, "MA": 0.5,
}

# ===========================
# STOPWORDS INDONESIA
# ===========================

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada", "atas",
    "dalam", "oleh", "sebagai", "adalah", "bahwa", "ini", "itu", "atau",
    "tidak", "ada", "serta", "juga", "namun", "karena", "jika", "maka",
    "bisa", "dapat", "guna", "demi", "para", "kepada", "terhadap", "lebih",
    "tanpa", "sudah", "telah", "akan", "merupakan", "olehnya", "saja", "bagi",
    "agar", "hingga", "sampai", "dimana", "daripada", "ketika", "setiap", "per",
    "nya", "lah", "kami", "kita", "kalian", "mereka", "mu", "ku", "sang", "si",
    "pun", "kah", "tah", "kan", "lah", "toh", "dong", "kok", "sih", "deh"
}

# ===========================
# PROMPTS FOR LLM
# ===========================

# System prompt untuk LLM reranking
LLM_SYSTEM_PROMPT = """Anda adalah expert recruiter PNS yang memiliki pengalaman luas dalam:
- Menilai kecocokan kandidat dengan jabatan pemerintah
- Memahami kompetensi dan kualifikasi PNS
- Mengidentifikasi gap dan memberikan saran pengembangan karir
- Memberikan penilaian yang objektif dan terstruktur

Tugas Anda adalah menilai kecocokan kandidat dengan jabatan PNS secara komprehensif."""

# Template prompt untuk scoring kandidat
LLM_SCORING_TEMPLATE = """
Nilai kecocokan kandidat dengan jabatan berikut (skor 0-100):

**PROFIL KANDIDAT:**
- Pendidikan: {pendidikan} - {jurusan}
- IPK: {ipk}
- Hard Skills: {hard_skills}
- Soft Skills: {soft_skills}
- Sertifikasi: {sertifikasi}
- Pengalaman: {pengalaman}
- Minat: {minat}

**JABATAN:**
- Judul: {judul_jabatan}
- Instansi: {instansi} - {unit_organisasi}
- Lokasi: {lokasi}
- Kualifikasi: {kualifikasi}
- Kompetensi: {kompetensi}
- Tugas: {tugas}

Berikan output dalam format JSON:
{{
  "score": <0-100>,
  "reasoning": "<penjelasan singkat 2-3 kalimat kenapa cocok/tidak>",
  "strengths": ["<kekuatan1>", "<kekuatan2>"],
  "gaps": ["<gap1>", "<gap2>"],
  "recommendations": ["<rekomendasi1>", "<rekomendasi2>"]
}}

PENTING: 
- Score 80-100: Sangat cocok
- Score 60-79: Cocok dengan sedikit gap
- Score 40-59: Perlu pengembangan
- Score 0-39: Kurang cocok
"""

# ===========================
# EXPORT CONFIGURATION
# ===========================

# Kolom yang di-export ke Excel
EXPORT_COLUMNS = [
    'judul_jabatan',
    'instansi',
    'unit_organisasi',
    'match_score',
    'education_score',
    'skills_score',
    'experience_score',
    'kualifikasi_pendidikan',
    'persyaratan_kompetensi',
    'jumlah_formasi',
    'lokasi_penempatan',
    'kode_formasi',
    'tautan_detail'
]

# Format tanggal untuk export
EXPORT_DATE_FORMAT = "%Y-%m-%d %H:%M"

# ===========================
# VALIDATION RULES
# ===========================

# Minimum panjang input text fields
MIN_TEXT_LENGTH = {
    'jurusan': 3,
    'pengalaman_kerja': 20,
    'hard_skills': 3,  # per skill
    'soft_skills': 3,  # per skill
}

# Maximum panjang input
MAX_TEXT_LENGTH = {
    'jurusan': 100,
    'pengalaman_kerja': 2000,
}

# IPK range
IPK_MIN = 0.0
IPK_MAX = 4.0

# ===========================
# ERROR MESSAGES
# ===========================

ERROR_MESSAGES = {
    'no_data': '⚠️ Silakan upload data di tab "Data Management" terlebih dahulu',
    'no_profile': '⚠️ Silakan lengkapi profil di tab "Profil Kandidat" terlebih dahulu',
    'invalid_csv': '❌ Format CSV tidak valid. Pastikan kolom wajib tersedia.',
    'llm_error': '⚠️ Error dalam LLM processing. Menggunakan skor hybrid saja.',
    'embedding_error': '⚠️ Error dalam membuat embeddings. Fallback ke BM25 only.',
    'empty_results': 'Tidak ada hasil yang cocok. Coba adjust bobot pencarian atau lengkapi profil.',
}

# ===========================
# SUCCESS MESSAGES
# ===========================

SUCCESS_MESSAGES = {
    'data_loaded': '✅ Data berhasil dimuat: {count} jabatan',
    'profile_saved': '✅ Profil berhasil disimpan!',
    'search_complete': '✅ Ditemukan {count} rekomendasi jabatan!',
    'feedback_saved': 'Terima kasih atas feedback Anda!',
    'export_ready': '✅ File Excel siap diunduh!',
}

# ===========================
# FEATURE FLAGS
# ===========================

# Enable/disable fitur tertentu
FEATURES = {
    'enable_bm25': True,           # BM25 search
    'enable_embeddings': True,      # Semantic embeddings
    'enable_llm_rerank': True,      # LLM reranking
    'enable_visualizations': True,  # Charts & graphs
    'enable_export': True,          # Excel export
    'enable_history': True,         # Search history
    'enable_feedback': True,        # Feedback collection
    'enable_fuzzy_match': True,     # Fuzzy string matching
}

# ===========================
# ADVANCED SETTINGS
# ===========================

# Parallel processing
USE_MULTIPROCESSING = False  # Set True jika CPU banyak core
N_JOBS = -1  # -1 = use all cores

# Memory optimization
LOW_MEMORY_MODE = False  # Set True untuk dataset sangat besar

# Debug mode
DEBUG = False  # Set True untuk verbose logging
