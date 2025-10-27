"""
Smart CASN Bidirectional Recommender System
============================================
Fitur Utama:
1. Auto-load data dari folder assets/data
2. Rekomendasi Jabatan untuk Kandidat (Candidate → Job Matching)
3. Rekomendasi Pegawai untuk Instansi (Job → Candidate Matching)  
4. Integrated AI Chatbot di setiap mode

Run: streamlit run app_casn_v2.py
"""

import os
import re
import json
import glob
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
    page_title="CERDAS",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        
        # Filter 1: Pendidikan
        user_edu = profile.get('pendidikan_terakhir', '')
        user_rank = get_education_rank(user_edu)
        
        results['edu_rank'] = results['kualifikasi_tingkat_pendidikan'].apply(get_education_rank)
        results = results[results['edu_rank'] <= user_rank]
        
        # Filter 2: Provinsi
        if profile.get('provinsi_penempatan') and profile['provinsi_penempatan'] != 'Semua':
            results = results[results['provinsi'] == profile['provinsi_penempatan'].upper()]
        
        # Filter 3: Salary
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
        """Mode 2: Cari jabatan yang sesuai dengan kebutuhan instansi"""
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
            
            # Cosine similarity (normalized to 0-1)
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([query_emb], filtered_embeddings)[0]
            results['match_score'] = similarities
        else:
            results['match_score'] = results['search_text'].apply(
                lambda x: self._keyword_match_score(x, requirement)
            )
        
        # Ensure match_score is between 0 and 1
        results['match_score'] = results['match_score'].clip(0, 1)
        
        results = results.sort_values('match_score', ascending=False).head(top_k)
        return results.reset_index(drop=True)
    
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

# ==================== AI CHATBOT ====================

class GeminiChatbot:
    """Interactive chatbot dengan Gemini AI"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        if not HAS_GEMINI:
            raise ImportError("Install: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.chat_history = []
        self.context = {}
    
    def set_context(self, profile: Dict, results: pd.DataFrame, mode: str):
        """Set context untuk chatbot"""
        self.context = {
            'profile': profile,
            'results': results.to_dict('records'),
            'mode': mode
        }
    
    def chat(self, user_message: str) -> str:
        """Process chat message"""
        system_prompt = self._build_system_prompt()
        full_prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
        
        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 1000,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                ]
            )
            
            if response and response.text:
                assistant_reply = response.text
                self.chat_history.append({
                    'user': user_message,
                    'assistant': assistant_reply
                })
                return assistant_reply
            else:
                return "Maaf, saya tidak dapat memproses pertanyaan Anda saat ini."
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _build_system_prompt(self) -> str:
        """Build system prompt dengan context"""
        mode = self.context.get('mode', 'candidate')
        
        if mode == 'candidate':
            profile = self.context.get('profile', {})
            results = self.context.get('results', [])[:5]
            
            results_text = "\n".join([
                f"{i+1}. {r.get('nama_jabatan', '-')} - {r.get('eselon_1_penempatan', '-')} "
                f"(Score: {r.get('match_score', 0):.2f})"
                for i, r in enumerate(results)
            ])
            
            return f"""
Anda adalah asisten AI untuk rekomendasi jabatan CASN.

PROFIL KANDIDAT:
- Pendidikan: {profile.get('pendidikan_terakhir', '-')}
- Jurusan: {profile.get('jurusan', '-')}
- Provinsi: {profile.get('provinsi_penempatan', '-')}

TOP REKOMENDASI:
{results_text}

Tugas Anda:
- Jawab pertanyaan kandidat tentang jabatan yang direkomendasikan
- Berikan detail tugas & fungsi jabatan
- Bandingkan antar jabatan jika diminta
- Saran provinsi lain jika ada formasi serupa
- Berikan insight tentang peluang kerja lapangan/kantor

Jawab dalam bahasa Indonesia yang ramah dan profesional.
"""
        else:
            requirement = self.context.get('profile', {})
            results = self.context.get('results', [])[:5]
            
            results_text = "\n".join([
                f"{i+1}. {r.get('nama_jabatan', '-')}"
                for i, r in enumerate(results)
            ])
            
            return f"""
Anda adalah asisten AI untuk rekomendasi kebutuhan pegawai.

KEBUTUHAN INSTANSI:
- Pendidikan: {requirement.get('pendidikan_terakhir', '-')}
- Uraian: {requirement.get('uraian_kebutuhan', '-')}

JABATAN YANG COCOK:
{results_text}

Tugas Anda:
- Jelaskan mengapa jabatan tersebut cocok
- Jelaskan tugas & fungsi tiap jabatan
- Bandingkan kemampuan antar jabatan

Jawab dalam bahasa Indonesia yang profesional.
"""

# ==================== VISUALISASI ====================

def plot_top_matches(df: pd.DataFrame, key_suffix: str = ""):
    """Plot top matches score"""
    if not HAS_PLOTLY or len(df) == 0:
        return
    
    df_plot = df.head(10).copy()
    df_plot['jabatan_short'] = df_plot['nama_jabatan'].apply(
        lambda x: x[:40] + '...' if len(x) > 40 else x
    )
    
    fig = px.bar(
        df_plot,
        x='match_score',
        y='jabatan_short',
        orientation='h',
        title='Top 10 Matches',
        labels={'match_score': 'Match Score', 'jabatan_short': 'Jabatan'},
        color='match_score',
        color_continuous_scale='viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
    st.plotly_chart(fig, use_container_width=True, key=f"top_matches_{key_suffix}")

def show_statistics_dashboard(stats: Dict):
    """Dashboard statistik"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jabatan", stats.get('total_jabatan', 0))
    with col2:
        st.metric("Total Formasi", stats.get('total_formasi', 0))
    with col3:
        st.metric("Total Instansi", stats.get('total_instansi', 0))
    with col4:
        st.metric("Provinsi", stats.get('provinsi_count', 0))
    
    # Jenjang pendidikan chart
    if 'jenjang_pendidikan' in stats and HAS_PLOTLY:
        df_edu = pd.DataFrame(
            list(stats['jenjang_pendidikan'].items()),
            columns=['Jenjang', 'Jumlah']
        )
        fig = px.pie(df_edu, values='Jumlah', names='Jenjang', 
                     title='Distribusi Jenjang Pendidikan',
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True, key="edu_dist")

# ==================== CHATBOT UI COMPONENT ====================

def render_chatbot_section(chatbot, mode: str, chat_key: str):
    """Render chatbot section untuk mode tertentu"""
    
    st.divider()
    st.subheader("💬 AI Assistant - Tanya Jawab")
    
    # Tips berdasarkan mode
    with st.expander("💡 Tips Pertanyaan"):
        if mode == 'candidate':
            st.markdown("""
            - Bisakah berikan uraian lebih detail tentang jabatan nomor 1?
            - Kalau di provinsi lain apakah ada formasi lagi?
            - Apakah benar tidak ada kemungkinan saya bekerja di lapangan jika memilih jabatan nomor 1?
            - Bandingkan jabatan nomor 1 dan nomor 3, mana yang lebih cocok untuk saya?
            - Apa saja skill yang harus saya tingkatkan untuk jabatan nomor 2?
            """)
        else:
            st.markdown("""
            - Jabatan nomor 3 juga bisa bantu untuk mengelola keuangan?
            - Apa perbedaan tugas antara jabatan nomor 1 dan nomor 2?
            - Jabatan mana yang lebih fokus ke pekerjaan teknis?
            - Apakah jabatan nomor 1 cocok untuk pekerjaan lapangan?
            """)
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for i, msg in enumerate(chatbot.chat_history):
            with st.chat_message("user"):
                st.write(msg['user'])
            with st.chat_message("assistant"):
                st.write(msg['assistant'])
    
    # Chat input
    user_input = st.chat_input("Tanyakan sesuatu...", key=f"chat_input_{chat_key}")
    
    if user_input:
        # Add user message to history
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Berpikir..."):
                    response = chatbot.chat(user_input)
                    st.write(response)
        
        st.rerun()

# ==================== MAIN APP ====================

def main():
    st.title("CASN Expert Recommendation and Decision Assistance System")
    st.markdown("**Sistem Rekomendasi Jabatan dan Kebutuhan Formasi**")
    
    # Sidebar: Settings
    with st.sidebar:
        st.header("🎯 CERDAS")
        
        top_k = st.slider("Jumlah Rekomendasi", 5, 30, 10, 5)
        
        st.divider()
        
        # System status
        st.subheader("📊 Status Sistem")
        if 'data_initialized' in st.session_state and st.session_state.data_initialized:
            st.success("✅ Data siap")
            if 'search_engine' in st.session_state and st.session_state.search_engine:
                if st.session_state.search_engine.embeddings is not None:
                    st.success("✅ AI Search aktif")
                else:
                    st.info("ℹ️ Keyword search aktif")
        else:
            st.warning("⚠️ Memuat data...")
    
    # Initialize Session State
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = CASNDataManager()
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'data_initialized' not in st.session_state:
        st.session_state.data_initialized = False
    
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
    
    # Tabs
    tabs = st.tabs([
        "📊 Dashboard",
        "🧑‍💼 Kandidat → Jabatan",
        "🏢 Instansi → Pegawai"
    ])
    
    # TAB 1: Dashboard
    with tabs[0]:
        st.header("📊 Dashboard Overview")
        
        if st.session_state.data_manager.df_merged is not None:
            stats = st.session_state.data_manager.get_statistics()
            show_statistics_dashboard(stats)
            
            # Preview data
            with st.expander("👁️ Preview Data Jabatan"):
                st.dataframe(
                    st.session_state.data_manager.df_merged[
                        ['nama_jabatan', 'eselon_1_penempatan', 'kualifikasi_tingkat_pendidikan', 
                         'provinsi', 'alokasi_kebutuhan']
                    ].head(20),
                    use_container_width=True
                )
        else:
            st.error("❌ Data tidak ditemukan di folder assets/data")
            st.info("💡 Pastikan file CSV ada di folder assets/data dengan format yang benar")
    
    # TAB 2: Mode 1 - Kandidat → Jabatan
    with tabs[1]:
        st.header("🧑‍💼 Cari Jabatan untuk Kandidat")
        
        if st.session_state.search_engine is None:
            st.warning("⚠️ Data belum tersedia")
        else:
            with st.form("candidate_form"):
                st.subheader("📝 Formulir Kandidat")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    pendidikan = st.selectbox("Tingkat Pendidikan Terakhir*", JENJANG_PENDIDIKAN, index=7)
                    jurusan = st.text_input(
                        "Jurusan*",
                        placeholder="Contoh: Teknik Informatika, Akuntansi"
                    )
                    instansi = st.text_area(
                        "Instansi (opsional)",
                        placeholder="Detail eselon 1-3 jika ada preferensi tertentu",
                        height=80
                    )
                
                with col2:
                    gaji_min = st.number_input(
                        "Preferensi Pendapatan Bulanan (Min)",
                        min_value=0,
                        value=5000000,
                        step=500000,
                        format="%d"
                    )
                    provinsi = st.selectbox(
                        "Preferensi Lokasi Penempatan",
                        ["Semua"] + PROVINSI_INDONESIA
                    )
                
                pengalaman = st.text_area(
                    "Pengalaman Bekerja*",
                    placeholder="Uraian pengalaman bekerja yang pernah dilakukan...\nContoh: Saya pernah bekerja sebagai admin call center yang banyak berkomunikasi dengan pengguna...",
                    height=100
                )
                
                pekerjaan_diharapkan = st.text_area(
                    "Pekerjaan yang Diharapkan*",
                    placeholder="Uraian jenis pekerjaan seperti apa yang ingin dipenuhi...\nContoh: Saya mengharapkan pekerjaan yang tidak terlalu banyak teknis pakai komputer, lebih suka kerja yang di lapangan...",
                    height=100
                )
                
                submit_candidate = st.form_submit_button("🔍 Cari Rekomendasi Jabatan", use_container_width=True)
            
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
                        st.session_state.results = results
                        st.session_state.current_mode = 'candidate'
                        st.session_state.current_profile = profile
                        
                        # Initialize chatbot
                        if use_ai:
                            try:
                                chatbot = GeminiChatbot(gemini_api_key, "gemini-2.5-flash")
                                chatbot.set_context(profile, results, 'candidate')
                                st.session_state.chatbot_mode1 = chatbot
                            except Exception as e:
                                st.session_state.chatbot_mode1 = None
                    
                    if len(results) > 0:
                        st.success(f"✅ Ditemukan {len(results)} rekomendasi jabatan!")
                        
                        # Visualisasi
                        if HAS_PLOTLY:
                            plot_top_matches(results, "candidate")
                        
                        st.divider()
                        
                        # Display Results
                        st.subheader("📋 Daftar Rekomendasi Jabatan")
                        
                        for idx, row in results.iterrows():
                            with st.expander(
                                f"#{idx+1} — {row['nama_jabatan']} "
                                f"(Match: {row['match_score']:.2%})",
                                expanded=(idx < 3)
                            ):
                                col_a, col_b = st.columns([2, 1])
                                
                                with col_a:
                                    st.markdown(f"**🏢 Instansi:** {row.get('eselon_1_penempatan', '-')}")
                                    st.markdown(f"**📍 Unit:** {row.get('eselon_2_penempatan', '-')}")
                                    if row.get('eselon_3_penempatan', '-') != '-':
                                        st.markdown(f"**📍 Sub-unit:** {row.get('eselon_3_penempatan', '-')}")
                                    st.markdown(f"**📍 Lokasi:** {row.get('lokasi', '-')}")
                                    st.markdown(f"**👥 Alokasi:** {row.get('alokasi_kebutuhan', 0)} orang")
                                    
                                    # Kualifikasi
                                    st.markdown("**🎓 Kualifikasi Pendidikan:**")
                                    # Format kualifikasi dengan koma
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
                                    st.markdown("**📊 Match Score**")
                                    st.metric("Overall", f"{row['match_score']:.1%}")
                                
                                # Tugas & Fungsi
                                if pd.notna(row.get('deskripsi_tugas_pokok')):
                                    with st.expander("📋 Tugas Pokok & Fungsi"):
                                        st.markdown("**Tugas Pokok:**")
                                        st.write(row.get('deskripsi_tugas_pokok', '-'))
                                        
                                        if pd.notna(row.get('rincian_kegiatan_fungsi')):
                                            st.markdown("**Rincian Kegiatan:**")
                                            kegiatan = row.get('rincian_kegiatan_fungsi', '').strip()
                                            for line in kegiatan.split(';'):
                                                if line.strip():
                                                    st.markdown(f"- {line.strip()}")
                        
                        # Export button
                        if len(results) > 0:
                            st.divider()
                            csv = results.to_csv(index=False)
                            st.download_button(
                                "📥 Download Hasil (CSV)",
                                csv,
                                f"rekomendasi_kandidat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                key="export_candidate"
                            )
                        
                        # Chatbot Section
                        if use_ai and 'chatbot_mode1' in st.session_state and st.session_state.chatbot_mode1:
                            render_chatbot_section(st.session_state.chatbot_mode1, 'candidate', 'mode1')
                        elif not use_ai:
                            st.info("💡 Aktifkan AI Chatbot dengan mengatur GEMINI_API_KEY di environment atau .env file")
                    else:
                        st.warning("❌ Tidak ditemukan jabatan yang sesuai. Coba ubah filter pencarian.")
    
    # TAB 3: Mode 2 - Instansi → Pegawai
    with tabs[2]:
        st.header("🏢 Cari Jabatan untuk Kebutuhan Instansi")
        
        if st.session_state.search_engine is None:
            st.warning("⚠️ Data belum tersedia")
        else:
            with st.form("job_requirement_form"):
                st.subheader("📝 Formulir Kebutuhan Pegawai")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    req_pendidikan = st.selectbox(
                        "Tingkat Pendidikan*",
                        JENJANG_PENDIDIKAN,
                        index=7,
                        key='req_edu'
                    )
                
                with col2:
                    st.write("")  # Spacer
                
                uraian_kebutuhan = st.text_area(
                    "Uraian Kebutuhan Pekerjaan yang Ingin Dipenuhi*",
                    placeholder="Contoh: Membutuhkan pegawai untuk mengelola sistem informasi, "
                                "membuat aplikasi web, maintenance server, dan dokumentasi teknis...",
                    height=150,
                    key='req_desc'
                )
                
                uraian_pekerjaan = st.text_area(
                    "Uraian Jenis Pekerjaan yang Diharapkan*",
                    placeholder="Contoh: Pekerjaan lebih banyak di lapangan atau di kantor, "
                                "banyak interaksi dengan masyarakat atau lebih ke teknis...",
                    height=100,
                    key='req_job'
                )
                
                submit_requirement = st.form_submit_button(
                    "🔍 Cari Rekomendasi Jabatan",
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
                        results = st.session_state.search_engine.search_for_job_requirement(
                            requirement, top_k
                        )
                        st.session_state.results = results
                        st.session_state.current_mode = 'job_requirement'
                        st.session_state.current_profile = requirement
                        
                        # Initialize chatbot
                        if use_ai:
                            try:
                                chatbot = GeminiChatbot(gemini_api_key, "gemini-2.5-flash")
                                chatbot.set_context(requirement, results, 'job_requirement')
                                st.session_state.chatbot_mode2 = chatbot
                            except Exception as e:
                                st.session_state.chatbot_mode2 = None
                    
                    if len(results) > 0:
                        st.success(f"✅ Ditemukan {len(results)} jabatan yang cocok!")
                        
                        # Visualisasi
                        if HAS_PLOTLY:
                            plot_top_matches(results, "requirement")
                        
                        st.divider()
                        
                        # Display Results
                        st.subheader("📋 Daftar Jabatan yang Sesuai")
                        
                        for idx, row in results.iterrows():
                            with st.expander(
                                f"#{idx+1} — {row['nama_jabatan']} "
                                f"(Match: {row['match_score']:.2%})",
                                expanded=(idx < 3)
                            ):
                                st.markdown(f"**🎓 Kualifikasi:** {row.get('kualifikasi_tingkat_pendidikan', '-')}")
                                # Format kualifikasi dengan koma
                                kualifikasi = row.get('kualifikasi_program_studi_jurusan', '-')
                                if '\n' in str(kualifikasi):
                                    kualifikasi = ', '.join([k.strip() for k in str(kualifikasi).split('\n') if k.strip()])
                                st.markdown(f"**📚 Jurusan:** {kualifikasi}")
                                
                                # Rangkuman Tugas & Fungsi
                                st.markdown("**📋 Rangkuman Tugas & Fungsi:**")
                                
                                if pd.notna(row.get('deskripsi_tugas_pokok')):
                                    st.info(row.get('deskripsi_tugas_pokok', '-'))
                                
                                if pd.notna(row.get('rincian_kegiatan_fungsi')):
                                    with st.expander("Lihat Rincian Kegiatan"):
                                        kegiatan = row.get('rincian_kegiatan_fungsi', '').strip()
                                        for line in kegiatan.split(';'):
                                            if line.strip():
                                                st.markdown(f"- {line.strip()}")
                                
                                # Informasi Formasi
                                st.markdown("---")
                                col_x, col_y = st.columns(2)
                                with col_x:
                                    st.markdown(f"**🏢 Instansi:** {row.get('eselon_1_penempatan', '-')}")
                                    st.markdown(f"**📍 Lokasi:** {row.get('lokasi', '-')}")
                                with col_y:
                                    st.markdown(f"**👥 Formasi:** {row.get('alokasi_kebutuhan', 0)} orang")
                                    if row.get('gaji_min', 0) > 0:
                                        st.markdown(
                                            f"**💰 Gaji:** {format_currency(row['gaji_min'])} - "
                                            f"{format_currency(row['gaji_max'])}"
                                        )
                        
                        # Export
                        if len(results) > 0:
                            st.divider()
                            csv = results.to_csv(index=False)
                            st.download_button(
                                "📥 Download Hasil (CSV)",
                                csv,
                                f"rekomendasi_jabatan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                key="export_requirement"
                            )
                        
                        # Chatbot Section
                        if use_ai and 'chatbot_mode2' in st.session_state and st.session_state.chatbot_mode2:
                            render_chatbot_section(st.session_state.chatbot_mode2, 'job_requirement', 'mode2')
                        elif not use_ai:
                            st.info("💡 Aktifkan AI Chatbot dengan mengatur GEMINI_API_KEY di environment atau .env file")
                    else:
                        st.warning("❌ Tidak ditemukan jabatan yang sesuai.")


if __name__ == "__main__":
    main()