import re
import json
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Optional imports dengan fallback
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="CERDAS - Smart CASN System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
def load_custom_css():
    """Load custom CSS untuk UI yang lebih menarik"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    div[data-testid="metric-container"] label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(90deg, #e0c3fc 0%, #8ec5fc 100%);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 8px;
        color: #764ba2;
        font-weight: 600;
        padding: 0 2rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f5f5f5;
        transform: scale(1.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Form Styling */
    .stForm {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    /* Input Fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus, .stNumberInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Download Button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.4);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        font-weight: 600;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Success/Error/Info Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
        padding: 1rem;
        font-weight: 500;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .stSpinner > div {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.3rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Info Box Custom */
    .custom-info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    /* Card Container */
    .card-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .card-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .badge-info {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #764ba2;
    }
    
    /* Pulse Animation for New Features */
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }
    }
    
    .glow {
        animation: glow 2s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("assets/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Provinsi Indonesia
PROVINSI_INDONESIA = [
    "ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "JAMBI",
    "SUMATERA SELATAN", "BENGKULU", "LAMPUNG", "KEP. BANGKA BELITUNG",
    "KEP. RIAU", "DKI JAKARTA", "JAWA BARAT", "JAWA TENGAH",
    "DI YOGYAKARTA", "JAWA TIMUR", "BANTEN", "BALI", "NUSA TENGGARA BARAT",
    "NUSA TENGGARA TIMUR", "KALIMANTAN BARAT", "KALIMANTAN TENGAH",
    "KALIMANTAN SELATAN", "KALIMANTAN TIMUR", "KALIMANTAN UTARA",
    "SULAWESI UTARA", "SULAWESI TENGAH", "SULAWESI SELATAN",
    "SULAWESI TENGGARA", "GORONTALO", "SULAWESI BARAT", "MALUKU",
    "MALUKU UTARA", "PAPUA BARAT", "PAPUA", "PAPUA TENGAH",
    "PAPUA PEGUNUNGAN", "PAPUA SELATAN", "PAPUA BARAT DAYA"
]

# Jenjang Pendidikan
JENJANG_PENDIDIKAN = ["SD", "SMP", "SMA/SMK", "D1", "D2", "D3", "D4", "S1", "S2", "S3"]

# ==================== UTILITAS ====================

def normalize_text(text: str) -> str:
    """Normalisasi teks"""
    if pd.isna(text):
        return ""
    return str(text).strip().lower()

def format_currency(amount: int) -> str:
    """Format mata uang IDR"""
    return f"Rp {amount:,.0f}".replace(",", ".")

def get_education_rank(edu: str) -> int:
    """Ranking pendidikan untuk sorting"""
    edu_map = {"SD": 1, "SMP": 2, "SMA/SMK": 3, "D1": 4, "D2": 5, 
               "D3": 6, "D4": 7, "S1": 7, "S2": 8, "S3": 9}
    for key in edu_map:
        if key in edu.upper():
            return edu_map[key]
    return 0

def extract_provinsi(lokasi: str) -> str:
    """Ekstrak nama provinsi dari lokasi"""
    lokasi = lokasi.upper()
    for prov in PROVINSI_INDONESIA:
        if prov in lokasi:
            return prov
    return "TIDAK DIKETAHUI"

# ==================== DATA MANAGER ====================

class CASNDataManager:
    """Mengelola 5 tabel data CASN dengan auto-load dari folder"""
    
    def __init__(self):
        self.df_instansi = None
        self.df_jabatan = None
        self.df_tupoksi = None
        self.df_prodi = None
        self.df_jurusan_sma = None
        self.df_merged = None
        self.data_loaded = False
    
    def auto_load_from_folder(self) -> bool:
        """Auto-load semua CSV dari folder assets/data - SILENT MODE"""
        try:
            csv_files = list(DATA_DIR.glob("*.csv"))
            
            if not csv_files:
                return False
            
            # Load berdasarkan pattern nama file (tanpa output)
            for csv_file in csv_files:
                filename = csv_file.name.lower()
                
                if 'instansi' in filename or 'tabel_1' in filename:
                    self.df_instansi = pd.read_csv(csv_file)
                
                elif 'jabatan' in filename or 'formasi' in filename or 'tabel_2' in filename:
                    self.df_jabatan = pd.read_csv(csv_file)
                
                elif 'tupoksi' in filename or 'tabel_3' in filename:
                    self.df_tupoksi = pd.read_csv(csv_file)
                
                elif 'prodi' in filename or 'program_studi' in filename or 'tabel_4' in filename:
                    self.df_prodi = pd.read_csv(csv_file)
                
                elif 'jurusan' in filename or 'sma' in filename or 'tabel_5' in filename:
                    self.df_jurusan_sma = pd.read_csv(csv_file)
            
            # Validasi: Minimal tabel jabatan harus ada
            if self.df_jabatan is None:
                return False
            
            # Merge tables
            self._merge_tables()
            self.data_loaded = True
            return True
            
        except Exception as e:
            return False
    
    def _merge_tables(self):
        """Gabungkan tabel untuk memudahkan search"""
        df = self.df_jabatan.copy()
        
        # Join dengan tupoksi
        if self.df_tupoksi is not None:
            df = df.merge(
                self.df_tupoksi[['nama_jabatan', 'deskripsi_tugas_pokok', 'rincian_kegiatan_fungsi']],
                left_on='nama_jabatan',
                right_on='nama_jabatan',
                how='left'
            )
        
        # Ekstraksi provinsi
        df['provinsi'] = df['lokasi'].apply(extract_provinsi)
        
        # Parse rentang gaji
        df['gaji_min'] = 0
        df['gaji_max'] = 0
        
        def parse_salary(s):
            if pd.isna(s):
                return 0, 0
            parts = str(s).split('-')
            try:
                min_sal = int(parts[0].strip())
                max_sal = int(parts[1].strip()) if len(parts) > 1 else min_sal
                return min_sal, max_sal
            except:
                return 0, 0
        
        df[['gaji_min', 'gaji_max']] = df['rentang_penghasilan'].apply(
            lambda x: pd.Series(parse_salary(x))
        )
        
        # Buat search text
        df['search_text'] = (
            df['nama_jabatan'].fillna('') + ' ' +
            df['kualifikasi_program_studi_jurusan'].fillna('') + ' ' +
            df['deskripsi_tugas_pokok'].fillna('') + ' ' +
            df['rincian_kegiatan_fungsi'].fillna('')
        ).apply(normalize_text)
        
        self.df_merged = df
    
    def get_statistics(self) -> Dict[str, Any]:
        """Statistik dataset"""
        if self.df_merged is None:
            return {}
        
        return {
            'total_jabatan': len(self.df_merged),
            'total_formasi': self.df_merged['alokasi_kebutuhan'].sum(),
            'total_instansi': self.df_merged['eselon_1_penempatan'].nunique(),
            'provinsi_count': self.df_merged['provinsi'].nunique(),
            'jenjang_pendidikan': self.df_merged['kualifikasi_tingkat_pendidikan'].value_counts().to_dict()
        }

# ==================== SEARCH ENGINE ====================

class HybridSearchEngine:
    """Search engine dengan semantic + keyword matching"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.embeddings = None
        self.model = None
    
    def build_embeddings(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """Build semantic embeddings - SILENT MODE"""
        if not HAS_SBERT:
            return
        
        try:
            self.model = SentenceTransformer(model_name)
            texts = self.df['search_text'].fillna('unknown').tolist()
            
            # Batch encoding tanpa progress bar
            batch_size = 32
            embeddings_list = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_emb = self.model.encode(batch, show_progress_bar=False)
                embeddings_list.extend(batch_emb)
            
            self.embeddings = np.array(embeddings_list)
        except:
            pass
    
    def search_for_candidate(
        self, 
        profile: Dict[str, Any], 
        top_k: int = 10
    ) -> pd.DataFrame:
        """Mode 1: Cari jabatan untuk kandidat"""
        results = self.df.copy()
        
        # Filter 1: Jurusan/Program Studi (PRIORITAS UTAMA)
        user_jurusan = profile.get('jurusan', '').strip()
        if user_jurusan:
            # Normalisasi input jurusan user
            user_jurusan_normalized = normalize_text(user_jurusan)
            
            # Filter jabatan yang memiliki jurusan yang cocok
            def check_jurusan_match(kualifikasi_prodi):
                if pd.isna(kualifikasi_prodi):
                    return False
                
                # Normalisasi kualifikasi prodi dari database
                kualifikasi_normalized = normalize_text(str(kualifikasi_prodi))
                
                # Split jika ada multiple prodi (dipisah dengan newline atau koma)
                prodi_list = []
                if '\n' in str(kualifikasi_prodi):
                    prodi_list = [normalize_text(p) for p in str(kualifikasi_prodi).split('\n') if p.strip()]
                elif ',' in str(kualifikasi_prodi):
                    prodi_list = [normalize_text(p) for p in str(kualifikasi_prodi).split(',') if p.strip()]
                else:
                    prodi_list = [kualifikasi_normalized]
                
                # Cek apakah jurusan user cocok dengan salah satu prodi
                for prodi in prodi_list:
                    if user_jurusan_normalized in prodi or prodi in user_jurusan_normalized:
                        return True
                    
                    # Cek kecocokan parsial (misal: "teknik informatika" cocok dengan "informatika")
                    user_words = set(user_jurusan_normalized.split())
                    prodi_words = set(prodi.split())
                    
                    # Jika ada minimal 2 kata yang sama atau 1 kata penting yang sama
                    common_words = user_words.intersection(prodi_words)
                    if len(common_words) >= 2:
                        return True
                    elif len(common_words) == 1:
                        # Kata penting (bukan kata umum seperti 'dan', 'atau', dll)
                        important_words = common_words - {'dan', 'atau', 'serta', 'di', 'dari', 'untuk'}
                        if important_words:
                            return True
                
                return False
            
            results['jurusan_match'] = results['kualifikasi_program_studi_jurusan'].apply(check_jurusan_match)
            results = results[results['jurusan_match'] == True]
            
            # Jika tidak ada hasil setelah filter jurusan, return empty dataframe
            if len(results) == 0:
                return pd.DataFrame()
        
        # Filter 2: Pendidikan
        user_edu = profile.get('pendidikan_terakhir', '')
        user_rank = get_education_rank(user_edu)
        
        results['edu_rank'] = results['kualifikasi_tingkat_pendidikan'].apply(get_education_rank)
        results = results[results['edu_rank'] <= user_rank]
        
        # Filter 3: Provinsi
        if profile.get('provinsi_penempatan') and profile['provinsi_penempatan'] != 'Semua':
            results = results[results['provinsi'] == profile['provinsi_penempatan'].upper()]
        
        # Filter 4: Salary
        if profile.get('gaji_minimum', 0) > 0:
            results = results[results['gaji_max'] >= profile['gaji_minimum']]
        
        if len(results) == 0:
            return pd.DataFrame()
        
        # Scoring: Semantic similarity
        if self.embeddings is not None and self.model is not None:
            query_text = self._build_candidate_query(profile)
            query_emb = self.model.encode([query_text])[0]
            
            filtered_indices = results.index.tolist()
            filtered_embeddings = self.embeddings[filtered_indices]
            
            # Cosine similarity (normalized to 0-1)
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([query_emb], filtered_embeddings)[0]
            results['match_score'] = similarities
        else:
            results['match_score'] = results['search_text'].apply(
                lambda x: self._keyword_match_score(x, profile)
            )
        
        # Ensure match_score is between 0 and 1
        results['match_score'] = results['match_score'].clip(0, 1)
        
        results = results.sort_values('match_score', ascending=False).head(top_k)
        return results.reset_index(drop=True)
    
    def search_for_job_requirement(
        self, 
        requirement: Dict[str, Any], 
        top_k: int = 10
    ) -> pd.DataFrame:
        """Mode 2: Cari jabatan yang sesuai dengan kebutuhan instansi - HANYA UNIQUE JABATAN"""
        results = self.df.copy()
        
        # Filter: Pendidikan
        req_edu = requirement.get('pendidikan_terakhir', '')
        if req_edu:
            results = results[
                results['kualifikasi_tingkat_pendidikan'].str.contains(req_edu, case=False, na=False)
            ]
        
        if len(results) == 0:
            return pd.DataFrame()
        
        # Scoring: Semantic similarity
        if self.embeddings is not None and self.model is not None:
            query_text = requirement.get('uraian_kebutuhan', '') + ' ' + \
                        requirement.get('uraian_pekerjaan', '')
            query_emb = self.model.encode([query_text])[0]
            
            filtered_indices = results.index.tolist()
            filtered_embeddings = self.embeddings[filtered_indices]
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([query_emb], filtered_embeddings)[0]
            results['match_score'] = similarities
        else:
            results['match_score'] = results['search_text'].apply(
                lambda x: self._keyword_match_score(x, requirement)
            )
        
        results['match_score'] = results['match_score'].clip(0, 1)
        
        # PERUBAHAN: Group by nama_jabatan dan ambil match_score tertinggi
        results_grouped = results.groupby('nama_jabatan').agg({
            'match_score': 'max',  # Ambil skor tertinggi
            'kualifikasi_tingkat_pendidikan': 'first',
            'kualifikasi_program_studi_jurusan': 'first',
            'deskripsi_tugas_pokok': 'first',
            'rincian_kegiatan_fungsi': 'first',
            'eselon_1_penempatan': lambda x: ', '.join(x.unique()[:3]),  # Gabung max 3 instansi
            'eselon_2_penempatan': 'first',
            'eselon_3_penempatan': 'first',
            'lokasi': lambda x: f"{x.nunique()} lokasi",  # Hitung jumlah lokasi
            'alokasi_kebutuhan': 'sum',  # Total formasi dari semua lokasi
            'gaji_min': 'min',
            'gaji_max': 'max',
            'rentang_penghasilan': 'first'
        }).reset_index()
        
        results_grouped = results_grouped.sort_values('match_score', ascending=False).head(top_k)
        return results_grouped.reset_index(drop=True)
    
    def _build_candidate_query(self, profile: Dict[str, Any]) -> str:
        """Build query text dari profil kandidat"""
        parts = [
            profile.get('pendidikan_terakhir', ''),
            profile.get('jurusan', ''),
            profile.get('pengalaman_kerja', ''),
            profile.get('pekerjaan_diharapkan', '')
        ]
        return ' '.join([p for p in parts if p])
    
    def _keyword_match_score(self, text: str, profile: Dict[str, Any]) -> float:
        """Simple keyword matching fallback"""
        keywords = self._build_candidate_query(profile).lower().split()
        text = text.lower()
        matches = sum(1 for kw in keywords if kw in text)
        return matches / max(len(keywords), 1)

# ==================== AI CHATBOT (FIXED) ====================

class GeminiChatbot:
    """Interactive chatbot dengan Gemini AI - OPTIMIZED untuk menghindari quota issues"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-lite"):
        if not HAS_GEMINI:
            raise ImportError("Install: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        # PERBAIKAN: Gunakan model yang lebih ringan (gemini-2.0-flash-lite)
        self.model = genai.GenerativeModel(model_name)
        self.chat_history = []
        self.context = {}
    
    def set_context(self, profile: Dict, results: pd.DataFrame, mode: str):
        """Set context untuk chatbot - OPTIMIZED"""
        # PERBAIKAN: Batasi hasil ke top 3 saja untuk menghemat token
        self.context = {
            'profile': profile,
            'results': results.head(3).to_dict('records'),  # Hanya ambil top 3
            'mode': mode
        }
    
    def chat(self, user_message: str) -> str:
        """Process chat message - OPTIMIZED"""
        # PERBAIKAN: Gunakan prompt yang lebih ringkas
        system_prompt = self._build_compact_system_prompt()
        
        # PERBAIKAN: Hanya ambil 2 percakapan terakhir (bukan 3)
        conversation = system_prompt + "\n\n"
        for msg in self.chat_history[-2:]:
            conversation += f"User: {msg['user'][:200]}\nBot: {msg['assistant'][:300]}\n\n"
        conversation += f"User: {user_message}\nBot:"
        
        try:
            response = self.model.generate_content(
                conversation,
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 500,  # PERBAIKAN: Kurangi dari 800 ke 500
                    'top_p': 0.9,
                    'top_k': 20,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                ]
            )
            
            if response and hasattr(response, 'text') and response.text:
                assistant_reply = response.text
                # PERBAIKAN: Simpan history dengan batasan panjang
                self.chat_history.append({
                    'user': user_message[:200],
                    'assistant': assistant_reply[:500]
                })
                # PERBAIKAN: Batasi history maksimal 5 percakapan
                if len(self.chat_history) > 5:
                    self.chat_history = self.chat_history[-5:]
                return assistant_reply
            
            elif response and hasattr(response, 'candidates'):
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                
                if finish_reason == 1:
                    return "Maaf, tidak ada response yang dihasilkan."
                elif finish_reason == 2:
                    return "Maaf, response terlalu panjang. Coba pertanyaan yang lebih spesifik."
                elif finish_reason == 3:
                    return self._retry_with_simpler_prompt(user_message)
                elif finish_reason == 4:
                    return "Maaf, tidak dapat memberikan response. Coba formulasikan pertanyaan dengan cara lain."
                else:
                    return f"Maaf, terjadi kesalahan (finish_reason: {finish_reason}). Silakan coba lagi."
            
            return "Maaf, tidak dapat memproses pertanyaan Anda. Silakan coba lagi."
                
        except Exception as e:
            error_msg = str(e).lower()
            
            if 'safety' in error_msg or 'blocked' in error_msg:
                return self._retry_with_simpler_prompt(user_message)
            elif 'quota' in error_msg or 'rate' in error_msg or 'resource' in error_msg:
                return "⚠️ API quota tercapai. Silakan tunggu beberapa saat (~1 menit) dan coba lagi dengan pertanyaan yang lebih ringkas."
            elif 'invalid' in error_msg:
                return "⚠️ Format pertanyaan tidak valid. Silakan coba dengan pertanyaan yang lebih sederhana."
            else:
                return f"⚠️ Terjadi kesalahan: {str(e)[:150]}"
    
    def _retry_with_simpler_prompt(self, user_message: str) -> str:
        """Retry dengan prompt yang lebih sederhana jika kena safety filter"""
        try:
            simple_prompt = f"""Asisten rekomendasi jabatan CASN.

Pertanyaan: {user_message[:150]}

Jawab singkat dan profesional dalam bahasa Indonesia."""
            
            response = self.model.generate_content(
                simple_prompt,
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 300,  # PERBAIKAN: Lebih pendek lagi
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                ]
            )
            
            if response and hasattr(response, 'text') and response.text:
                assistant_reply = response.text
                self.chat_history.append({
                    'user': user_message[:200],
                    'assistant': assistant_reply[:300]
                })
                if len(self.chat_history) > 5:
                    self.chat_history = self.chat_history[-5:]
                return assistant_reply
            
            return "Maaf, saya tidak dapat memproses pertanyaan tersebut. Silakan coba dengan kata-kata yang berbeda."
            
        except:
            return "Maaf, saya tidak dapat memproses pertanyaan tersebut saat ini. Silakan tunggu sebentar dan coba lagi."
    
    def _build_compact_system_prompt(self) -> str:
        """Build system prompt yang RINGKAS untuk menghemat token"""
        mode = self.context.get('mode', 'candidate')
        
        if mode == 'candidate':
            profile = self.context.get('profile', {})
            results = self.context.get('results', [])
            
            # PERBAIKAN: Hanya tampilkan info minimal
            results_text = "\n".join([
                f"{i+1}. {r.get('nama_jabatan', '-')[:60]} ({r.get('match_score', 0):.0%})"
                for i, r in enumerate(results)
            ])
            
            return f"""Asisten rekomendasi jabatan CASN.

PROFIL: {profile.get('pendidikan_terakhir', '-')}, {profile.get('jurusan', '-')}

TOP 3:
{results_text}

Jawab singkat, fokus, dan informatif."""
        
        else:  # job_requirement mode
            requirement = self.context.get('profile', {})
            results = self.context.get('results', [])
            
            # PERBAIKAN: Format lebih ringkas untuk mode 2
            results_text = "\n".join([
                f"{i+1}. {r.get('nama_jabatan', '-')[:60]} ({r.get('match_score', 0):.0%})"
                for i, r in enumerate(results)
            ])
            
            # PERBAIKAN: Potong deskripsi kebutuhan jika terlalu panjang
            kebutuhan = requirement.get('uraian_kebutuhan', '-')[:150]
            
            return f"""Asisten analisis jabatan CASN.

KEBUTUHAN: {requirement.get('pendidikan_terakhir', '-')}
{kebutuhan}

TOP 3:
{results_text}

Jawab singkat dan objektif."""

# ==================== VISUALISASI ====================

def plot_top_matches(df: pd.DataFrame, key_suffix: str = ""):
    """Plot top matches score dengan format label yang lebih informatif"""
    if not HAS_PLOTLY or len(df) == 0:
        return
    
    df_plot = df.head(10).copy()
    
    # Buat label yang menggabungkan jabatan + eselon 1, 2, 3
    def create_label(row):
        labels = []
        
        # Jabatan (potong jika terlalu panjang)
        jabatan = row['nama_jabatan']
        if len(jabatan) > 50:
            jabatan = jabatan[:47] + '...'
        labels.append(f"<b>{jabatan}</b>")
        
        # Eselon 1
        if pd.notna(row.get('eselon_1_penempatan')):
            eselon1 = str(row['eselon_1_penempatan'])
            if len(eselon1) > 50:
                eselon1 = eselon1[:47] + '...'
            labels.append(f"Eselon 1: {eselon1}")
        
        # Eselon 2
        if pd.notna(row.get('eselon_2_penempatan')):
            eselon2 = str(row['eselon_2_penempatan'])
            if len(eselon2) > 50:
                eselon2 = eselon2[:47] + '...'
            labels.append(f"Eselon 2: {eselon2}")
        
        # Eselon 3
        if pd.notna(row.get('eselon_3_penempatan')) and str(row.get('eselon_3_penempatan')) != '-':
            eselon3 = str(row['eselon_3_penempatan'])
            if len(eselon3) > 50:
                eselon3 = eselon3[:47] + '...'
            labels.append(f"Eselon 3: {eselon3}")
        
        return '<br>'.join(labels)
    
    df_plot['label_lengkap'] = df_plot.apply(create_label, axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_plot['label_lengkap'],
        y=df_plot['match_score'],
        text=[f"{score:.1%}" for score in df_plot['match_score']],
        textposition='outside',
        marker=dict(
            color=df_plot['match_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Skor Kecocokan")
        ),
        hovertemplate='<b>%{x}</b><br>Skor Kecocokan: %{y:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Top 10 Matches - Skor Kecocokan by Position',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        xaxis_title='Jabatan & Struktur Organisasi',
        yaxis_title='Skor Kecocokan',
        yaxis=dict(
            tickformat='.0%',
            range=[0, max(df_plot['match_score'].max() * 1.1, 0.1)]
        ),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        height=600,
        showlegend=False,
        hovermode='x unified',
        plot_bgcolor='rgba(240,240,240,0.5)',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"top_matches_{key_suffix}")

def show_statistics_dashboard(stats: Dict, df_merged: pd.DataFrame = None):
    """Dashboard statistik dengan visualisasi lengkap"""
    
    # ========== METRIC CARDS ==========
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jabatan", f"{stats.get('total_jabatan', 0):,}")
    with col2:
        st.metric("Total Formasi", f"{stats.get('total_formasi', 0):,}")
    with col3:
        st.metric("Total Instansi", stats.get('total_instansi', 0))
    with col4:
        st.metric("Provinsi", stats.get('provinsi_count', 0))
    
    if df_merged is None or not HAS_PLOTLY:
        return
    
    st.divider()
    
    # ========== ROW 1: Jenjang Pendidikan & Gaji ==========
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📚 Distribusi Jenjang Pendidikan")
        if 'jenjang_pendidikan' in stats:
            df_edu = pd.DataFrame(
                list(stats['jenjang_pendidikan'].items()),
                columns=['Jenjang', 'Jumlah']
            ).sort_values('Jumlah', ascending=False)
            
            fig_edu = go.Figure(data=[
                go.Bar(
                    x=df_edu['Jenjang'],
                    y=df_edu['Jumlah'],
                    text=df_edu['Jumlah'],
                    textposition='outside',
                    marker_color='lightblue',
                    hovertemplate='<b>%{x}</b><br>Jumlah: %{y}<extra></extra>'
                )
            ])
            fig_edu.update_layout(
                xaxis_title="Jenjang Pendidikan",
                yaxis_title="Jumlah Formasi",
                height=400,
                showlegend=False,
                hovermode='x'
            )
            st.plotly_chart(fig_edu, use_container_width=True, key="edu_bar")
    
    with col_right:
        st.subheader("💰 Distribusi Rentang Gaji")
        # Buat kategori gaji
        df_gaji = df_merged[df_merged['gaji_min'] > 0].copy()
        
        def kategorikan_gaji(gaji_min):
            if gaji_min < 3000000:
                return "< 3 Juta"
            elif gaji_min < 5000000:
                return "3-5 Juta"
            elif gaji_min < 7000000:
                return "5-7 Juta"
            elif gaji_min < 10000000:
                return "7-10 Juta"
            else:
                return "> 10 Juta"
        
        df_gaji['kategori_gaji'] = df_gaji['gaji_min'].apply(kategorikan_gaji)
        gaji_count = df_gaji['kategori_gaji'].value_counts().reindex(
            ["< 3 Juta", "3-5 Juta", "5-7 Juta", "7-10 Juta", "> 10 Juta"],
            fill_value=0
        )
        
        fig_gaji = go.Figure(data=[go.Pie(
            labels=gaji_count.index,
            values=gaji_count.values,
            hole=0.4,
            marker=dict(colors=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#5f27cd']),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<extra></extra>'
        )])
        fig_gaji.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="v", x=1.1, y=0.5)
        )
        st.plotly_chart(fig_gaji, use_container_width=True, key="gaji_pie")
    
    st.divider()
    
    # ========== ROW 2: Chart #1 - Provinsi vs Jumlah Eselon 1 ==========
    st.subheader("🗺️ Jumlah Instansi per Provinsi")
    
    # Hitung jumlah unique eselon 1 per provinsi
    provinsi_eselon = df_merged.groupby('provinsi')['eselon_1_penempatan'].nunique().reset_index()
    provinsi_eselon.columns = ['Provinsi', 'Jumlah_Eselon_1']
    provinsi_eselon = provinsi_eselon.sort_values('Jumlah_Eselon_1', ascending=False).head(15)
    
    fig_prov_eselon = go.Figure(data=[
        go.Bar(
            x=provinsi_eselon['Provinsi'],
            y=provinsi_eselon['Jumlah_Eselon_1'],
            text=provinsi_eselon['Jumlah_Eselon_1'],
            textposition='outside',
            marker=dict(
                color=provinsi_eselon['Jumlah_Eselon_1'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Jumlah")
            ),
            hovertemplate='<b>%{x}</b><br>Jumlah Instansi: %{y}<extra></extra>'
        )
    ])
    fig_prov_eselon.update_layout(
        xaxis_title="Provinsi",
        yaxis_title="Jumlah Instansi (Eselon 1)",
        height=500,
        showlegend=False,
        xaxis=dict(tickangle=-45),
        hovermode='x'
    )
    st.plotly_chart(fig_prov_eselon, use_container_width=True, key="prov_eselon")
    
    st.divider()
    
    # ========== ROW 3: Chart #2 - Filtered Chart per Provinsi ==========
    st.subheader("🏢 Jumlah Formasi per Instansi (Filter by Provinsi)")
    
    # Dropdown untuk memilih provinsi
    provinsi_list = sorted(df_merged['provinsi'].unique().tolist())
    selected_provinsi = st.selectbox(
        "Pilih Provinsi:",
        provinsi_list,
        key="select_provinsi_dash"
    )
    
    # Filter data berdasarkan provinsi
    df_filtered = df_merged[df_merged['provinsi'] == selected_provinsi]
    
    # Agregasi jumlah formasi per instansi (eselon 1)
    formasi_per_instansi = df_filtered.groupby('eselon_1_penempatan')['alokasi_kebutuhan'].sum().reset_index()
    formasi_per_instansi.columns = ['Instansi', 'Total_Formasi']
    formasi_per_instansi = formasi_per_instansi.sort_values('Total_Formasi', ascending=False).head(20)
    
    if len(formasi_per_instansi) > 0:
        # Potong nama instansi jika terlalu panjang
        formasi_per_instansi['Instansi_Short'] = formasi_per_instansi['Instansi'].apply(
            lambda x: x[:50] + '...' if len(x) > 50 else x
        )
        
        fig_formasi = go.Figure(data=[
            go.Bar(
                x=formasi_per_instansi['Instansi_Short'],
                y=formasi_per_instansi['Total_Formasi'],
                text=formasi_per_instansi['Total_Formasi'],
                textposition='outside',
                marker=dict(
                    color=formasi_per_instansi['Total_Formasi'],
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Formasi")
                ),
                hovertemplate='<b>%{x}</b><br>Total Formasi: %{y}<extra></extra>'
            )
        ])
        fig_formasi.update_layout(
            xaxis_title="Instansi (Eselon 1)",
            yaxis_title="Jumlah Formasi",
            height=500,
            showlegend=False,
            xaxis=dict(tickangle=-45),
            hovermode='x'
        )
        st.plotly_chart(fig_formasi, use_container_width=True, key="formasi_instansi")
        
        # Info tambahan
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Total Formasi di Provinsi Ini", f"{formasi_per_instansi['Total_Formasi'].sum():,}")
        with col_info2:
            st.metric("Jumlah Instansi", len(formasi_per_instansi))
        with col_info3:
            st.metric("Rata-rata Formasi/Instansi", f"{formasi_per_instansi['Total_Formasi'].mean():.0f}")
    else:
        st.info("Tidak ada data untuk provinsi ini.")
    
    st.divider()
    
    # ========== ROW 4: Chart Tambahan - Top 10 Jabatan Terpopuler ==========
    st.subheader("🎯 Top 10 Jabatan dengan Formasi Terbanyak")
    
    top_jabatan = df_merged.groupby('nama_jabatan')['alokasi_kebutuhan'].sum().reset_index()
    top_jabatan.columns = ['Jabatan', 'Total_Formasi']
    top_jabatan = top_jabatan.sort_values('Total_Formasi', ascending=False).head(10)
    
    # Potong nama jabatan
    top_jabatan['Jabatan_Short'] = top_jabatan['Jabatan'].apply(
        lambda x: x[:60] + '...' if len(x) > 60 else x
    )
    
    fig_top_jabatan = go.Figure(data=[
        go.Bar(
            y=top_jabatan['Jabatan_Short'][::-1],  # Reverse untuk top di atas
            x=top_jabatan['Total_Formasi'][::-1],
            text=top_jabatan['Total_Formasi'][::-1],
            textposition='outside',
            orientation='h',
            marker=dict(
                color=top_jabatan['Total_Formasi'][::-1],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Formasi")
            ),
            hovertemplate='<b>%{y}</b><br>Total Formasi: %{x}<extra></extra>'
        )
    ])
    fig_top_jabatan.update_layout(
        xaxis_title="Jumlah Formasi",
        yaxis_title="Nama Jabatan",
        height=600,
        showlegend=False,
        hovermode='y'
    )
    st.plotly_chart(fig_top_jabatan, use_container_width=True, key="top_jabatan")
    
    st.divider()
    
    
    
    # ========== ROW 6: Statistik Regional ==========
    st.subheader("📊 Statistik Regional")
    
    col_reg1, col_reg2 = st.columns(2)
    
    with col_reg1:
        st.markdown("**🏆 Top 5 Provinsi dengan Formasi Terbanyak**")
        top_prov_formasi = df_merged.groupby('provinsi')['alokasi_kebutuhan'].sum().nlargest(5).reset_index()
        top_prov_formasi.columns = ['Provinsi', 'Total Formasi']
        
        for idx, row in top_prov_formasi.iterrows():
            st.metric(
                f"{idx+1}. {row['Provinsi']}", 
                f"{row['Total Formasi']:,} formasi"
            )
    
    with col_reg2:
        st.markdown("**🎓 Top 5 Instansi dengan Formasi Terbanyak**")
        top_instansi = df_merged.groupby('eselon_1_penempatan')['alokasi_kebutuhan'].sum().nlargest(5).reset_index()
        top_instansi.columns = ['Instansi', 'Total Formasi']
        
        for idx, row in top_instansi.iterrows():
            instansi_name = row['Instansi'][:40] + '...' if len(row['Instansi']) > 40 else row['Instansi']
            st.metric(
                f"{idx+1}. {instansi_name}", 
                f"{row['Total Formasi']:,} formasi"
            )

# ==================== CHATBOT UI COMPONENT ====================

def render_chatbot_section(chatbot, mode: str, chat_key: str):
    """Render chatbot section dengan UI yang menarik"""
    
    st.markdown("---")
    st.markdown('<div class="section-header">💬 AI Assistant - Tanya Jawab Interaktif</div>', unsafe_allow_html=True)
    
    # Welcome message with styling
    st.markdown("""
    <div class="custom-info-box">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 2.5rem; margin-right: 1rem;">🤖</div>
            <div>
                <div style="font-weight: 600; font-size: 1.1rem; color: #667eea;">AI Assistant Siap Membantu!</div>
                <div style="font-size: 0.9rem; margin-top: 0.3rem;">
                    Tanyakan apa saja tentang rekomendasi jabatan di atas. Saya akan memberikan penjelasan detail dan insight yang berguna.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tips berdasarkan mode
    with st.expander("💡 Contoh Pertanyaan yang Bisa Ditanyakan", expanded=False):
        if mode == 'candidate':
            st.markdown("""
            <div style="line-height: 2;">
            ✨ <strong>Bisakah berikan uraian lebih detail tentang jabatan nomor 1?</strong><br>
            ✨ <strong>Kalau di provinsi lain apakah ada formasi lagi?</strong><br>
            ✨ <strong>Apakah benar tidak ada kemungkinan saya bekerja di lapangan jika memilih jabatan nomor 1?</strong><br>
            ✨ <strong>Bandingkan jabatan nomor 1 dan nomor 3, mana yang lebih cocok untuk saya?</strong><br>
            ✨ <strong>Apa saja skill yang harus saya tingkatkan untuk jabatan nomor 2?</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="line-height: 2;">
            ✨ <strong>Jabatan nomor 3 juga bisa bantu untuk mengelola keuangan?</strong><br>
            ✨ <strong>Apa perbedaan tugas antara jabatan nomor 1 dan nomor 2?</strong><br>
            ✨ <strong>Jabatan mana yang lebih fokus ke pekerjaan teknis?</strong><br>
            ✨ <strong>Apakah jabatan nomor 1 cocok untuk pekerjaan lapangan?</strong><br>
            ✨ <strong>Berapa total formasi yang tersedia untuk jabatan ini?</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Info box tentang batasan
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); 
                padding: 1rem; border-radius: 10px; margin: 1rem 0;
                border-left: 5px solid #e17055;">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.5rem; margin-right: 0.8rem;">⚡</div>
            <div style="font-size: 0.9rem;">
                <strong>Tips Penggunaan:</strong> Gunakan pertanyaan yang ringkas dan spesifik. 
                Jika terkena quota limit, tunggu sekitar 1 menit sebelum mencoba lagi.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container with custom styling
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    chat_container = st.container()
    
    # Display chat history with enhanced styling
    with chat_container:
        if len(chatbot.chat_history) == 0:
            st.markdown("""
            <div class="card-container" style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">💭</div>
                <div style="color: #666; font-size: 1.1rem;">
                    Belum ada percakapan. Mulai tanyakan sesuatu!
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, msg in enumerate(chatbot.chat_history):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"**Anda:** {msg['user']}")
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f"**AI Assistant:** {msg['assistant']}")
    
    # Chat input with custom placeholder
    placeholder_text = "Ketik pertanyaan Anda di sini... (misal: Jelaskan tugas jabatan nomor 1)"
    user_input = st.chat_input(placeholder_text, key=f"chat_input_{chat_key}")
    
    if user_input:
        # Get AI response with animated spinner
        with st.spinner("🤔 AI sedang berpikir dan menganalisis..."):
            response = chatbot.chat(user_input)
        
        # Display updated chat
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**Anda:** {user_input}")
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**AI Assistant:** {response}")
        
        # Rerun to update the display
        st.rerun()

# ==================== MAIN APP ====================

def main():
    # Load custom CSS first
    load_custom_css()
    
    # Animated Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 CERDAS</h1>
        <p><strong>CASN Expert Recommendation and Decision Assistance System</strong></p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Sistem Rekomendasi Cerdas untuk Kandidat & Formasi Jabatan CASN Indonesia</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar: Settings
    with st.sidebar:
        st.markdown("# 🎯 CERDAS")
        st.markdown("---")
        
        # Settings Section
        st.markdown("#### ⚙️ Pengaturan")
        top_k = st.slider(
            "Jumlah Rekomendasi",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Tentukan berapa banyak rekomendasi yang ingin ditampilkan"
        )
        
        st.markdown("---")
        
        # System Status Section
        st.markdown("#### 📊 Status Sistem")
        if 'data_initialized' in st.session_state and st.session_state.data_initialized:
            st.markdown("""
            <div class="custom-info-box">
                <div style="text-align: center;">
                    <div style="font-size: 3rem;">✅</div>
                    <div style="font-weight: 600; color: #667eea;">Data Siap</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if 'search_engine' in st.session_state and st.session_state.search_engine:
                if st.session_state.search_engine.embeddings is not None:
                    st.markdown("""
                    <div class="custom-info-box" style="margin-top: 1rem;">
                        <div style="text-align: center;">
                            <div style="font-size: 2rem;">🤖</div>
                            <div style="font-weight: 600; color: #667eea;">AI Search Aktif</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Keyword Search Aktif")
        else:
            st.warning("⚠️ Memuat data...")
        
        st.markdown("---")
        
        # Quick Stats (if data available)
        # if 'data_manager' in st.session_state and st.session_state.data_manager.df_merged is not None:
        #     st.markdown("#### 📈 Quick Stats")
        #     stats = st.session_state.data_manager.get_statistics()
            
        #     st.metric("Total Jabatan", f"{stats.get('total_jabatan', 0):,}", delta="Database")
        #     st.metric("Total Formasi", f"{stats.get('total_formasi', 0):,}", delta="Tersedia")
        #     st.metric("Provinsi", stats.get('provinsi_count', 0), delta="Wilayah")
        
        # st.markdown("---")
        
        # Info Section
        st.markdown("#### ℹ️ Tentang CERDAS")
        st.markdown("""
        <div style="font-size: 0.85rem; line-height: 1.6;">
        CERDAS menggunakan teknologi AI untuk memberikan rekomendasi jabatan CASN yang paling sesuai dengan profil dan kebutuhan Anda.
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize Session State
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = CASNDataManager()
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = None
    if 'results_mode1' not in st.session_state:
        st.session_state.results_mode1 = None
    if 'results_mode2' not in st.session_state:
        st.session_state.results_mode2 = None
    if 'data_initialized' not in st.session_state:
        st.session_state.data_initialized = False
    if 'chatbot_mode1' not in st.session_state:
        st.session_state.chatbot_mode1 = None
    if 'chatbot_mode2' not in st.session_state:
        st.session_state.chatbot_mode2 = None
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = None
    if 'current_profile_mode1' not in st.session_state:
        st.session_state.current_profile_mode1 = None
    if 'current_profile_mode2' not in st.session_state:
        st.session_state.current_profile_mode2 = None
    
    # Get Gemini API Key (dari environment atau secrets)
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", gemini_api_key)
    except:
        pass
    
    use_ai = bool(gemini_api_key and HAS_GEMINI)
    
    # Auto-load data pada startup (hanya sekali) - SILENT
    if not st.session_state.data_initialized:
        if st.session_state.data_manager.auto_load_from_folder():
            search_engine = HybridSearchEngine(st.session_state.data_manager.df_merged)
            
            if HAS_SBERT:
                search_engine.build_embeddings()
            
            st.session_state.search_engine = search_engine
            st.session_state.data_initialized = True
    
    # Tabs with Icons
    tabs = st.tabs([
        "📊 Dashboard Analytics",
        "🧑‍💼 Kandidat → Cari Jabatan",
        "🏢 Instansi → Cari Pegawai"
    ])
    
    # TAB 1: Dashboard
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Dashboard Analytics & Insights</div>', unsafe_allow_html=True)
        
        if st.session_state.data_manager.df_merged is not None:
            stats = st.session_state.data_manager.get_statistics()
            show_statistics_dashboard(stats, st.session_state.data_manager.df_merged)
            
            # Preview data with custom styling
            
        else:
            st.error("❌ Data tidak ditemukan di folder assets/data")
            st.info("💡 Pastikan file CSV ada di folder assets/data dengan format yang benar")
    
    # TAB 2: Mode 1 - Kandidat → Jabatan
    with tabs[1]:
        st.markdown('<div class="section-header">🧑‍💼 Mode Kandidat: Temukan Jabatan Impian Anda</div>', unsafe_allow_html=True)
        
        if st.session_state.search_engine is None:
            st.warning("⚠️ Data belum tersedia")
        else:
            # Info Banner
            st.markdown("""
            <div class="custom-info-box">
                <div style="display: flex; align-items: center;">
                    <div style="font-size: 2rem; margin-right: 1rem;">💡</div>
                    <div>
                        <strong>Tips:</strong> Lengkapi profil Anda dengan detail untuk mendapatkan rekomendasi jabatan yang paling sesuai dengan kemampuan dan preferensi Anda.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("candidate_form"):
                st.markdown("#### 📝 Profil Kandidat")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎓 Pendidikan & Keahlian**")
                    pendidikan = st.selectbox(
                        "Tingkat Pendidikan Terakhir*",
                        JENJANG_PENDIDIKAN,
                        index=7,
                        help="Pilih jenjang pendidikan tertinggi Anda"
                    )
                    jurusan = st.text_input(
                        "Jurusan*",
                        placeholder="Contoh: Teknik Informatika, Akuntansi",
                        help="Masukkan jurusan atau program studi Anda"
                    )
                    instansi = st.text_area(
                        "Preferensi Instansi (opsional)",
                        placeholder="Detail eselon 1-3 jika ada preferensi tertentu",
                        height=80,
                        help="Kosongkan jika tidak ada preferensi khusus"
                    )
                
                with col2:
                    st.markdown("**📍 Lokasi & Gaji**")
                    provinsi = st.selectbox(
                        "Preferensi Lokasi Penempatan",
                        ["Semua"] + PROVINSI_INDONESIA,
                        help="Pilih provinsi yang Anda inginkan"
                    )
                    gaji_min = st.number_input(
                        "Preferensi Pendapatan Bulanan (Min)",
                        min_value=0,
                        value=5000000,
                        step=500000,
                        format="%d",
                        help="Tentukan ekspektasi gaji minimum Anda"
                    )
      
                
                st.markdown("---")
                st.markdown("**💼 Pengalaman & Ekspektasi**")
                
                pengalaman = st.text_area(
                    "Pengalaman Bekerja*",
                    placeholder="Uraian pengalaman bekerja yang pernah dilakukan...\nContoh: Saya pernah bekerja sebagai admin call center yang banyak berkomunikasi dengan pengguna...",
                    height=100,
                    help="Ceritakan pengalaman kerja Anda secara singkat"
                )
                
                pekerjaan_diharapkan = st.text_area(
                    "Pekerjaan yang Diharapkan*",
                    placeholder="Uraian jenis pekerjaan seperti apa yang ingin dipenuhi...\nContoh: Saya mengharapkan pekerjaan yang tidak terlalu banyak teknis pakai komputer, lebih suka kerja yang di lapangan...",
                    height=100,
                    help="Jelaskan jenis pekerjaan yang Anda inginkan"
                )
                
                st.markdown("---")
                submit_candidate = st.form_submit_button(
                    "🔍 Cari Rekomendasi Jabatan Sekarang!",
                    use_container_width=True
                )
            
            if submit_candidate:
                if not all([jurusan, pengalaman, pekerjaan_diharapkan]):
                    st.error("❌ Mohon lengkapi field yang bertanda *")
                else:
                    profile = {
                        'pendidikan_terakhir': pendidikan,
                        'jurusan': jurusan,
                        'instansi': instansi,
                        'gaji_minimum': gaji_min,
                        'provinsi_penempatan': provinsi if provinsi != "Semua" else None,
                        'pengalaman_kerja': pengalaman,
                        'pekerjaan_diharapkan': pekerjaan_diharapkan
                    }
                    
                    with st.spinner("🔍 Mencari rekomendasi terbaik..."):
                        results = st.session_state.search_engine.search_for_candidate(profile, top_k)
                        st.session_state.results_mode1 = results
                        st.session_state.current_mode = 'candidate'
                        st.session_state.current_profile_mode1 = profile
                        
                        # Initialize chatbot
                        if use_ai:
                            try:
                                chatbot = GeminiChatbot(gemini_api_key, "gemini-2.0-flash-lite")
                                chatbot.set_context(profile, results, 'candidate')
                                st.session_state.chatbot_mode1 = chatbot
                            except Exception as e:
                                st.session_state.chatbot_mode1 = None
                    
                    if len(results) > 0:
                        st.success(f"✅ Ditemukan {len(results)} rekomendasi jabatan yang sesuai dengan jurusan **{profile.get('jurusan', '-')}**!")
                        st.info(f"📚 Filter aktif: Hanya menampilkan jabatan yang membutuhkan jurusan **{profile.get('jurusan', '-')}** atau sejenisnya")
                        
                        # Visualisasi
                        if HAS_PLOTLY:
                            plot_top_matches(results, "candidate")
                        
                        st.divider()
                        
                        # Display Results
                        st.subheader("📋 Daftar Rekomendasi Jabatan")
                        
                        for idx, row in results.iterrows():
                            with st.expander(
                                f"#{idx+1} — {row['nama_jabatan']} "
                                f"(Match: {row['match_score']:.2%}) — "
                                f"{row.get('eselon_1_penempatan', '-')}",
                                expanded=(idx < 3)
                            ):
                                col_a, col_b = st.columns([2, 1])
                                
                                with col_a:
                                    st.markdown(f"**🏢 Eselon 1:** {row.get('eselon_1_penempatan', '-')}")
                                    st.markdown(f"**📍 Eselon 2:** {row.get('eselon_2_penempatan', '-')}")
                                    if row.get('eselon_3_penempatan', '-') != '-':
                                        st.markdown(f"**📍 Eselon 3:** {row.get('eselon_3_penempatan', '-')}")
                                    st.markdown(f"**📍 Lokasi:** {row.get('lokasi', '-')}")
                                    st.markdown(f"**👥 Alokasi:** {row.get('alokasi_kebutuhan', 0)} orang")
                                    
                                    # Kualifikasi
                                    st.markdown("**🎓 Kualifikasi Pendidikan:**")
                                    kualifikasi = row.get('kualifikasi_program_studi_jurusan', '-')
                                    if '\n' in str(kualifikasi):
                                        kualifikasi = ', '.join([k.strip() for k in str(kualifikasi).split('\n') if k.strip()])
                                    st.info(kualifikasi)
                                    
                                    # Rentang Gaji
                                    if row.get('gaji_min', 0) > 0:
                                        st.markdown(
                                            f"**💰 Rentang Penghasilan:** "
                                            f"{format_currency(row['gaji_min'])} - {format_currency(row['gaji_max'])}"
                                        )
                                
                                with col_b:
                                    st.markdown("##### 📊 Skor Kecocokan")
                                    st.markdown(f"""
                                    <div class="card-container" style="text-align: center;">
                                        <div style="font-size: 3rem; font-weight: 700; color: #667eea;">
                                            {row['match_score']:.0%}
                                        </div>
                                        <div style="color: #666; margin-top: 0.5rem;">Overall Match</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    # Ketetatan Skor
                                    if pd.notna(row.get('passing_grade_persen')):
                                        tingkat_keketatan = row.get('tingkat_keketatan', '-').replace('_', ' ').title()
                                        rasio_keketatan = row.get('rasio_keketatan', '-')
                                        
                                    st.markdown("##### 🎯 Tingkat Keketatan")
                                    if pd.notna(row.get('passing_grade_persen')):
                                        st.markdown(f"""
                                        <div class="card-container" style="text-align: center;">
                                            <div style="font-size: 2rem; font-weight: 600; color: #ff6b6b;">
                                                {row.get('passing_grade_persen', 0):.2f}%
                                            </div>
                                            <div style="color: #666; margin-top: 0.5rem;">
                                                {tingkat_keketatan} (Rasio: {rasio_keketatan})
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="card-container" style="text-align: center;">
                                            <div style="font-size: 1.2rem; color: #666;">
                                                Data Keketatan Tidak Tersedia
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Tugas & Fungsi
                                if pd.notna(row.get('deskripsi_tugas_pokok')):
                                    st.markdown("---")
                                    with st.expander("📋 Lihat Tugas Pokok & Fungsi"):
                                        st.markdown("**Tugas Pokok:**")
                                        st.write(row.get('deskripsi_tugas_pokok', '-'))
                                        
                                        if pd.notna(row.get('rincian_kegiatan_fungsi')):
                                            st.markdown("**Rincian Kegiatan:**")
                                            kegiatan = row.get('rincian_kegiatan_fungsi', '').strip()
                                            for line in kegiatan.split(';'):
                                                if line.strip():
                                                    st.markdown(f"• {line.strip()}")
                        
                        # Export button with styling
                        st.markdown("---")
                        csv = results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Hasil Rekomendasi (CSV)",
                            csv,
                            f"rekomendasi_kandidat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key="export_candidate",
                            use_container_width=True
                        )
                        
                        # Chatbot Section
                        if use_ai and 'chatbot_mode1' in st.session_state and st.session_state.chatbot_mode1:
                            render_chatbot_section(st.session_state.chatbot_mode1, 'candidate', 'mode1')
                        elif not use_ai:
                            st.info("💡 Aktifkan AI Chatbot dengan mengatur GEMINI_API_KEY di environment atau .env file")
                    else:
                        st.error("❌ Tidak ditemukan jabatan yang sesuai dengan jurusan Anda")
                        
                        st.markdown("""
                        <div class="custom-info-box" style="background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);">
                            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
                                💡 Kemungkinan Penyebab:
                            </div>
                            <ul style="margin-left: 1.5rem; line-height: 1.8;">
                                <li>Jurusan <strong>{}</strong> tidak tersedia di database formasi CASN saat ini</li>
                                <li>Tidak ada jabatan yang membuka formasi untuk jurusan tersebut</li>
                                <li>Coba gunakan nama jurusan yang lebih umum (misal: "Informatika" bukan "Teknik Informatika")</li>
                            </ul>
                            <div style="margin-top: 1rem; padding: 1rem; background: white; border-radius: 8px;">
                                <strong>Saran:</strong> Coba ubah nama jurusan atau cek kembali ejaan jurusan Anda.
                            </div>
                        </div>
                        """.format(profile.get('jurusan', '-')), unsafe_allow_html=True)
                        
                        # Tampilkan beberapa jurusan yang tersedia sebagai referensi
                        if st.session_state.data_manager.df_merged is not None:
                            with st.expander("📋 Lihat Daftar Jurusan yang Tersedia di Database"):
                                all_prodi = set()
                                for _, row in st.session_state.data_manager.df_merged.iterrows():
                                    kualifikasi = row.get('kualifikasi_program_studi_jurusan', '')
                                    if pd.notna(kualifikasi):
                                        if '\n' in str(kualifikasi):
                                            prodi_list = [p.strip() for p in str(kualifikasi).split('\n') if p.strip()]
                                        elif ',' in str(kualifikasi):
                                            prodi_list = [p.strip() for p in str(kualifikasi).split(',') if p.strip()]
                                        else:
                                            prodi_list = [str(kualifikasi).strip()]
                                        all_prodi.update(prodi_list)
                                
                                all_prodi_sorted = sorted(list(all_prodi))
                                
                                # Tampilkan dalam kolom
                                col1, col2, col3 = st.columns(3)
                                third = len(all_prodi_sorted) // 3
                                
                                with col1:
                                    for prodi in all_prodi_sorted[:third]:
                                        st.markdown(f"• {prodi}")
                                with col2:
                                    for prodi in all_prodi_sorted[third:2*third]:
                                        st.markdown(f"• {prodi}")
                                with col3:
                                    for prodi in all_prodi_sorted[2*third:]:
                                        st.markdown(f"• {prodi}")
            
            # Display previous results (untuk chat interaction)
            if not submit_candidate and 'results_mode1' in st.session_state and st.session_state.results_mode1 is not None:
                results = st.session_state.results_mode1
                
                if len(results) > 0:
                    st.success(f"✅ Menampilkan {len(results)} rekomendasi jabatan sebelumnya")
                    
                    if HAS_PLOTLY:
                        plot_top_matches(results, "candidate_prev")
                    
                    st.markdown("---")
                    st.markdown('<div class="section-header">📋 Daftar Rekomendasi Jabatan</div>', unsafe_allow_html=True)
                    
                    for idx, row in results.iterrows():
                        match_pct = row['match_score'] * 100
                        emoji = "🌟" if match_pct >= 80 else "⭐" if match_pct >= 60 else "✨"
                        badge_color = "success" if match_pct >= 80 else "info"
                        
                        with st.expander(
                            f"{emoji} #{idx+1} — {row['nama_jabatan']} (Match: {row['match_score']:.0%})",
                            expanded=(idx < 2)
                        ):
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem;">
                                <span class="badge badge-{badge_color}">Match Score: {row['match_score']:.0%}</span>
                                <span class="badge badge-info">Formasi: {row.get('alokasi_kebutuhan', 0)} orang</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col_a, col_b = st.columns([2, 1])
                            
                            with col_a:
                                st.markdown(f"**🏢 Eselon 1:** {row.get('eselon_1_penempatan', '-')}")
                                st.markdown(f"**📍 Eselon 2:** {row.get('eselon_2_penempatan', '-')}")
                                if row.get('eselon_3_penempatan', '-') != '-':
                                    st.markdown(f"**📍 Eselon 3:** {row.get('eselon_3_penempatan', '-')}")
                                st.markdown(f"**📍 Lokasi:** {row.get('lokasi', '-')}")
                                
                                kualifikasi = row.get('kualifikasi_program_studi_jurusan', '-')
                                if '\n' in str(kualifikasi):
                                    kualifikasi = ', '.join([k.strip() for k in str(kualifikasi).split('\n') if k.strip()])
                                st.info(f"**🎓 Jurusan:** {kualifikasi}")
                                
                                if row.get('gaji_min', 0) > 0:
                                    st.markdown(f"**💰 Gaji:** {format_currency(row['gaji_min'])} - {format_currency(row['gaji_max'])}")
                            
                            with col_b:
                                st.metric("Match Score", f"{row['match_score']:.0%}")
                            
                            if pd.notna(row.get('deskripsi_tugas_pokok')):
                                with st.expander("📋 Tugas Pokok & Fungsi"):
                                    st.write(row.get('deskripsi_tugas_pokok', '-'))
                                    if pd.notna(row.get('rincian_kegiatan_fungsi')):
                                        kegiatan = row.get('rincian_kegiatan_fungsi', '').strip()
                                        for line in kegiatan.split(';'):
                                            if line.strip():
                                                st.markdown(f"• {line.strip()}")
                    
                    st.markdown("---")
                    csv = results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Hasil (CSV)",
                        csv,
                        f"rekomendasi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key="export_prev"
                    )
                    
                    if use_ai and 'chatbot_mode1' in st.session_state and st.session_state.chatbot_mode1:
                        render_chatbot_section(st.session_state.chatbot_mode1, 'candidate', 'mode1')
    
    # TAB 3: Mode 2 - Instansi → Pegawai
    with tabs[2]:
        st.markdown('<div class="section-header">🏢 Mode Instansi: Temukan Pegawai yang Tepat</div>', unsafe_allow_html=True)
        
        if st.session_state.search_engine is None:
            st.warning("⚠️ Data belum tersedia")
        else:
            # Info Banner
            st.markdown("""
            <div class="custom-info-box">
                <div style="display: flex; align-items: center;">
                    <div style="font-size: 2rem; margin-right: 1rem;">💡</div>
                    <div>
                        <strong>Tips:</strong> Jelaskan kebutuhan pegawai dengan detail untuk mendapatkan rekomendasi jabatan yang paling sesuai dengan kebutuhan instansi Anda.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("job_requirement_form"):
                st.markdown("#### 📝 Kebutuhan Pegawai Instansi")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎓 Kualifikasi Pendidikan**")
                    req_pendidikan = st.selectbox(
                        "Tingkat Pendidikan yang Dibutuhkan*",
                        JENJANG_PENDIDIKAN,
                        index=7,
                        key='req_edu',
                        help="Pilih minimal jenjang pendidikan yang dibutuhkan"
                    )
                
                with col2:
                    st.write("")
                
                st.markdown("---")
                st.markdown("**💼 Deskripsi Kebutuhan**")
                
                uraian_kebutuhan = st.text_area(
                    "Uraian Kebutuhan Pekerjaan yang Ingin Dipenuhi*",
                    placeholder="Contoh: Membutuhkan pegawai untuk mengelola sistem informasi, membuat aplikasi web, maintenance server, dan dokumentasi teknis...",
                    height=150,
                    key='req_desc',
                    help="Jelaskan tugas dan tanggung jawab yang akan dilakukan"
                )
                
                uraian_pekerjaan = st.text_area(
                    "Uraian Jenis Pekerjaan yang Diharapkan*",
                    placeholder="Contoh: Pekerjaan lebih banyak di lapangan atau di kantor, banyak interaksi dengan masyarakat atau lebih ke teknis...",
                    height=100,
                    key='req_job',
                    help="Jelaskan karakteristik pekerjaan yang diharapkan"
                )
                
                st.markdown("---")
                submit_requirement = st.form_submit_button(
                    "🔍 Cari Rekomendasi Jabatan Sekarang!",
                    use_container_width=True
                )
            
            if submit_requirement:
                if not all([uraian_kebutuhan, uraian_pekerjaan]):
                    st.error("❌ Mohon lengkapi field yang bertanda *")
                else:
                    requirement = {
                        'pendidikan_terakhir': req_pendidikan,
                        'uraian_kebutuhan': uraian_kebutuhan,
                        'uraian_pekerjaan': uraian_pekerjaan
                    }
                    
                    with st.spinner("🔍 Mencari jabatan yang sesuai..."):
                        results = st.session_state.search_engine.search_for_job_requirement(requirement, top_k)
                        st.session_state.results_mode2 = results
                        st.session_state.current_mode = 'job_requirement'
                        st.session_state.current_profile_mode2 = requirement
                        
                        # Initialize chatbot
                        if use_ai:
                            try:
                                chatbot = GeminiChatbot(gemini_api_key, "gemini-2.0-flash-lite")
                                chatbot.set_context(requirement, results, 'job_requirement')
                                st.session_state.chatbot_mode2 = chatbot
                            except Exception as e:
                                st.warning(f"⚠️ Chatbot tidak dapat diinisialisasi: {str(e)[:100]}")
                                st.session_state.chatbot_mode2 = None
                    
                    if len(results) > 0:
                        st.balloons()
                        st.success(f"✅ Ditemukan {len(results)} jabatan yang cocok dengan kebutuhan Anda!")
                        
                        if HAS_PLOTLY:
                            plot_top_matches(results, "requirement")
                        
                        st.markdown("---")
                        st.markdown('<div class="section-header">📋 Daftar Jabatan yang Sesuai</div>', unsafe_allow_html=True)
                        
                        for idx, row in results.iterrows():
                            match_pct = row['match_score'] * 100
                            emoji = "🌟" if match_pct >= 80 else "⭐" if match_pct >= 60 else "✨"
                            badge_color = "success" if match_pct >= 80 else "info"
                            
                            with st.expander(
                                f"{emoji} #{idx+1} — {row['nama_jabatan']} (Match: {row['match_score']:.0%})",
                                expanded=(idx < 2)
                            ):
                                st.markdown(f"""
                                <div style="margin-bottom: 1rem;">
                                    <span class="badge badge-{badge_color}">Match Score: {row['match_score']:.0%}</span>
                                    <span class="badge badge-info">Formasi: {row.get('alokasi_kebutuhan', 0)} orang</span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                col_a, col_b = st.columns([2, 1])
                                
                                with col_a:
                                    st.markdown("##### 🎓 Kualifikasi & Keahlian")
                                    st.markdown(f"**Pendidikan:** {row.get('kualifikasi_tingkat_pendidikan', '-')}")
                                    
                                    kualifikasi = row.get('kualifikasi_program_studi_jurusan', '-')
                                    if '\n' in str(kualifikasi):
                                        kualifikasi = ', '.join([k.strip() for k in str(kualifikasi).split('\n') if k.strip()])
                                    st.info(f"**📚 Jurusan:** {kualifikasi}")
                                    
                                    st.markdown("---")
                                    st.markdown("##### 📋 Tugas & Fungsi Jabatan")
                                    
                                    if pd.notna(row.get('deskripsi_tugas_pokok')):
                                        st.write(row.get('deskripsi_tugas_pokok', '-'))
                                    
                                    if pd.notna(row.get('rincian_kegiatan_fungsi')):
                                        with st.expander("📝 Lihat Rincian Kegiatan Lengkap"):
                                            kegiatan = row.get('rincian_kegiatan_fungsi', '').strip()
                                            for line in kegiatan.split(';'):
                                                if line.strip():
                                                    st.markdown(f"• {line.strip()}")
                                    
                                    st.markdown("---")
                                    st.markdown("##### 🏢 Informasi Formasi")
                                    col_x, col_y = st.columns(2)
                                    with col_x:
                                        st.markdown(f"**Instansi:** {row.get('eselon_1_penempatan', '-')}")
                                        st.markdown(f"**📍 Lokasi:** {row.get('lokasi', '-')}")
                                    with col_y:
                                        st.markdown(f"**👥 Kebutuhan:** {row.get('alokasi_kebutuhan', 0)} orang")
                                        if row.get('gaji_min', 0) > 0:
                                            st.markdown(f"**💰 Gaji:** {format_currency(row['gaji_min'])} - {format_currency(row['gaji_max'])}")
                                
                                
                        
                        st.markdown("---")
                        csv = results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Hasil Rekomendasi (CSV)",
                            csv,
                            f"rekomendasi_jabatan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key="export_requirement",
                            use_container_width=True
                        )
                        
                        # Chatbot Section
                        if use_ai and 'chatbot_mode2' in st.session_state and st.session_state.chatbot_mode2:
                            render_chatbot_section(st.session_state.chatbot_mode2, 'job_requirement', 'mode2')
                        elif not use_ai:
                            st.info("💡 Aktifkan AI Chatbot dengan mengatur GEMINI_API_KEY")
                    else:
                        st.warning("❌ Tidak ditemukan jabatan yang sesuai.")
            
            # Display previous results
            if not submit_requirement and 'results_mode2' in st.session_state and st.session_state.results_mode2 is not None:
                results = st.session_state.results_mode2
                
                if len(results) > 0:
                    st.success(f"✅ Menampilkan {len(results)} rekomendasi jabatan sebelumnya")
                    
                    if HAS_PLOTLY:
                        plot_top_matches(results, "requirement_prev")
                    
                    st.markdown("---")
                    st.markdown('<div class="section-header">📋 Daftar Jabatan yang Sesuai</div>', unsafe_allow_html=True)
                    
                    for idx, row in results.iterrows():
                        match_pct = row['match_score'] * 100
                        emoji = "🌟" if match_pct >= 80 else "⭐" if match_pct >= 60 else "✨"
                        badge_color = "success" if match_pct >= 80 else "info"
                        
                        with st.expander(
                            f"{emoji} #{idx+1} — {row['nama_jabatan']} (Match: {row['match_score']:.0%})",
                            expanded=(idx < 2)
                        ):
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem;">
                                <span class="badge badge-{badge_color}">Match: {row['match_score']:.0%}</span>
                                <span class="badge badge-info">Formasi: {row.get('alokasi_kebutuhan', 0)}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col_a, col_b = st.columns([2, 1])
                            
                            with col_a:
                                st.markdown(f"**🎓 Pendidikan:** {row.get('kualifikasi_tingkat_pendidikan', '-')}")
                                kualifikasi = row.get('kualifikasi_program_studi_jurusan', '-')
                                if '\n' in str(kualifikasi):
                                    kualifikasi = ', '.join([k.strip() for k in str(kualifikasi).split('\n') if k.strip()])
                                st.info(f"**📚 Jurusan:** {kualifikasi}")
                                
                                if pd.notna(row.get('deskripsi_tugas_pokok')):
                                    st.markdown("**📋 Tugas:**")
                                    st.write(row.get('deskripsi_tugas_pokok', '-'))
                                
                                st.markdown(f"**🏢 Instansi:** {row.get('eselon_1_penempatan', '-')}")
                                st.markdown(f"**📍 Lokasi:** {row.get('lokasi', '-')}")
                            
                            with col_b:
                                st.metric("Match", f"{row['match_score']:.0%}")
                    
                    st.markdown("---")
                    csv = results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Hasil (CSV)",
                        csv,
                        f"rekomendasi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key="export_req_prev"
                    )
                    
                    if use_ai and 'chatbot_mode2' in st.session_state and st.session_state.chatbot_mode2:
                        render_chatbot_section(st.session_state.chatbot_mode2, 'job_requirement', 'mode2')
    
    # TAB 3: Mode 2 - Instansi → Pegawai
    

if __name__ == "__main__":
    main()