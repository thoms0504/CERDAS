# app_optimized.py — Rekomendasi Jabatan PNS (Enhanced & Optimized)
# =====================================================================
# Optimasi & Fitur Baru:
# - Smart caching (embedding, BM25, preprocessing)
# - Fuzzy matching untuk pendidikan/jurusan
# - Weighted multi-criteria scoring
# - Visual analytics & gap analysis
# - Export hasil (Excel, PDF)
# - Comparison mode
# - History tracking
# - Better error handling
# - Progress indicators
# - Optimized memory usage
# - FIXED: Gemini AI Integration dengan input manual API key
# =====================================================================

import os
import re
import io
import json
import time
import pickle
import hashlib
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Optional libraries dengan fallback
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    BM25Okapi = None
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    SentenceTransformer = None
    HAS_SBERT = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    genai = None
    HAS_GEMINI = False

try:
    from fuzzywuzzy import fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------------------------
# Konfigurasi
# ---------------------------
st.set_page_config(
    page_title="Smart PNS Recommender",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konstanta
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
FEEDBACK_FILE = "feedback.csv"
HISTORY_FILE = "search_history.json"
PROFILE_FILE = "saved_profiles.json"

# Model embedding default
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Stopwords Indonesia
STOPWORDS_ID = set("""
yang dan di ke dari dengan untuk pada atas dalam oleh sebagai adalah bahwa
ini itu atau tidak ada serta juga namun karena jika maka bisa dapat guna
demi para kepada terhadap lebih tanpa sudah telah akan merupakan olehnya
saja bagi agar hingga sampai dimana daripada ketika setiap per nya lah
kami kita kalian mereka mu ku sang si
""".split())

# Sinonim skill (untuk expanding query)
SKILL_SYNONYMS = {
    "python": ["python", "python3", "py"],
    "java": ["java", "javase", "javaee"],
    "komunikasi": ["komunikasi", "komunikatif", "public speaking", "presentasi"],
    "leadership": ["leadership", "kepemimpinan", "memimpin", "leader"],
    "analisis": ["analisis", "analisa", "analytic", "analytical"],
    "manajemen": ["manajemen", "management", "mengelola", "pengelolaan"],
    "microsoft office": ["ms office", "microsoft office", "word", "excel", "powerpoint"],
}

# Mapping pendidikan
EDUCATION_LEVELS = {
    "S3": 5, "DOKTOR": 5, "DOCTORAL": 5,
    "S2": 4, "MAGISTER": 4, "MASTER": 4,
    "S1": 3, "SARJANA": 3, "BACHELOR": 3,
    "D4": 2.5, "D-IV": 2.5, "SARJANA TERAPAN": 2.5,
    "D3": 2, "D-III": 2, "DIPLOMA": 2,
    "D2": 1.5, "D-II": 1.5,
    "D1": 1, "D-I": 1,
    "SMA": 0.5, "SMK": 0.5, "SLTA": 0.5,
}


# ---------------------------
# Utilitas
# ---------------------------

def normalize_text(s: Any) -> str:
    """Normalisasi teks dengan cleaning lebih baik"""
    if pd.isna(s) or not isinstance(s, str):
        return ""
    s = s.strip()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_id(s: str) -> List[str]:
    """Tokenisasi dengan stopword removal"""
    # Validasi input
    if pd.isna(s) or not isinstance(s, str):
        return []
    
    s = str(s)  # Pastikan string
    tokens = re.findall(r"[a-z0-9_\-]+", s.lower())
    return [t for t in tokens if t not in STOPWORDS_ID and len(t) > 1]


def expand_skills(skills: List[str]) -> List[str]:
    """Ekspansi skill dengan sinonim"""
    expanded = set(skills)
    for skill in skills:
        skill_lower = skill.lower()
        for key, synonyms in SKILL_SYNONYMS.items():
            if skill_lower in synonyms:
                expanded.update(synonyms)
    return list(expanded)


def fuzzy_match_education(user_edu: str, required_edu: str, threshold: int = 80) -> Tuple[bool, int]:
    """Fuzzy matching untuk pendidikan dengan scoring"""
    if not HAS_FUZZY:
        return user_edu.lower() in required_edu.lower(), 50
    
    score = fuzz.token_set_ratio(user_edu.lower(), required_edu.lower())
    return score >= threshold, score


def get_education_level(edu_str: str) -> float:
    """Extract education level dari string"""
    edu_str = normalize_text(edu_str).upper()
    for key, level in EDUCATION_LEVELS.items():
        if key in edu_str:
            return level
    return 0


def calculate_hash(data: Any) -> str:
    """Generate hash untuk caching"""
    return hashlib.md5(str(data).encode()).hexdigest()


def save_to_cache(key: str, data: Any):
    """Simpan data ke cache"""
    cache_path = CACHE_DIR / f"{key}.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)


def load_from_cache(key: str) -> Optional[Any]:
    """Load data dari cache"""
    cache_path = CACHE_DIR / f"{key}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


# ---------------------------
# Data Management
# ---------------------------

class DataManager:
    """Mengelola dataset jabatan dengan caching pintar"""
    
    def __init__(self):
        self.df = None
        self.data_hash = None
        
    def load_data(self, uploaded_file=None, csv_path=None) -> pd.DataFrame:
        """Load dan preprocess data"""
        try:
            if uploaded_file:
                self.df = pd.read_csv(uploaded_file)
            elif csv_path:
                self.df = pd.read_csv(csv_path)
            else:
                return None
            
            # Validasi kolom required
            required_cols = ["judul_jabatan", "instansi", "unit_organisasi"]
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                st.error(f"Kolom wajib tidak ditemukan: {missing_cols}")
                return None
            
            # Data cleaning
            self.df = self._clean_data(self.df)
            
            # Generate hash untuk cache invalidation
            self.data_hash = calculate_hash(self.df.to_dict())
            
            st.success(f"✅ Data berhasil dimuat: {len(self.df)} jabatan")
            return self.df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bersihkan dan standarisasi data"""
        df = df.copy()
        
        # Fill NaN
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            df[col] = df[col].fillna("")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
        
        # Normalisasi teks penting
        for col in ["judul_jabatan", "unit_organisasi", "instansi"]:
            if col in df.columns:
                df[f"{col}_normalized"] = df[col].apply(normalize_text)
        
        # Ekstraksi keywords jika belum ada
        if "keywords" not in df.columns or df["keywords"].isna().all():
            df["keywords"] = df.apply(self._extract_keywords, axis=1)
        
        # TAMBAHKAN INI: Pastikan keywords selalu string
        df["keywords"] = df["keywords"].fillna("").astype(str)
        
        return df
    
    def _extract_keywords(self, row) -> str:
        """Ekstraksi keywords dari row"""
        text_parts = []
        for col in ["judul_jabatan", "tugas_pokok", "persyaratan_kompetensi", 
                    "kualifikasi_pendidikan", "unit_organisasi"]:
            if col in row and pd.notna(row[col]):
                text_parts.append(str(row[col]))
        
        combined = " ".join(text_parts)
        tokens = tokenize_id(normalize_text(combined))
        return " ".join(tokens[:50])  # Top 50 keywords
    
    def get_statistics(self) -> Dict[str, Any]:
        """Dapatkan statistik dataset"""
        if self.df is None:
            return {}
        
        stats = {
            "total_jabatan": len(self.df),
            "total_instansi": self.df["instansi"].nunique() if "instansi" in self.df.columns else 0,
            "total_formasi": self.df["jumlah_formasi"].sum() if "jumlah_formasi" in self.df.columns else 0,
        }
        
        # Pendidikan distribution
        if "kualifikasi_pendidikan" in self.df.columns:
            edu_dist = self.df["kualifikasi_pendidikan"].value_counts().head(10)
            stats["top_education"] = edu_dist.to_dict()
        
        # Lokasi distribution
        if "lokasi_penempatan" in self.df.columns:
            loc_dist = self.df["lokasi_penempatan"].value_counts().head(10)
            stats["top_locations"] = loc_dist.to_dict()
        
        return stats


# ---------------------------
# Search Engine
# ---------------------------

class HybridSearchEngine:
    """Hybrid search dengan BM25 + Embedding + Weighted Scoring"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.bm25_index = None
        self.embeddings = None
        self.embedding_model = None
        self.corpus_tokens = None
        
    @st.cache_resource
    def build_bm25_index(_self, data_hash: str):
        """Build BM25 index dengan caching"""
        cache_key = f"bm25_{data_hash}"
        cached = load_from_cache(cache_key)
        if cached:
            return cached
        
        if not HAS_BM25:
            st.warning("⚠️ BM25 tidak tersedia. Install: pip install rank-bm25")
            return None, None
        
        # Pastikan keywords selalu string dan tidak kosong
        keywords_list = _self.df["keywords"].fillna("").astype(str)
        corpus_tokens = [tokenize_id(keywords) for keywords in keywords_list]
        
        # Filter corpus yang kosong
        corpus_tokens = [tokens if tokens else ["unknown"] for tokens in corpus_tokens]
        
        bm25_index = BM25Okapi(corpus_tokens)
        
        result = (bm25_index, corpus_tokens)
        save_to_cache(cache_key, result)
        return result
    
    @st.cache_resource
    def build_embeddings(_self, data_hash: str, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """Build embeddings dengan caching"""
        cache_key = f"embeddings_{data_hash}_{model_name}"
        cached = load_from_cache(cache_key)
        if cached:
            return cached
        
        if not HAS_SBERT:
            st.warning("⚠️ Sentence-Transformers tidak tersedia")
            return None, None
        
        try:
            model = SentenceTransformer(model_name)
            
            # PERBAIKAN: Pastikan texts selalu string dan tidak kosong
            texts = _self.df["keywords"].fillna("").astype(str).tolist()
            
            # Filter texts kosong, ganti dengan placeholder
            texts = [text if text.strip() else "unknown" for text in texts]
            
            # Batch encoding untuk efisiensi
            batch_size = 32
            embeddings = []
            
            progress_bar = st.progress(0)
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_emb = model.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_emb)
                progress_bar.progress(min((i + batch_size) / len(texts), 1.0))
            
            progress_bar.empty()  # Hapus progress bar setelah selesai
            
            embeddings = np.array(embeddings)
            result = (model, embeddings)
            save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            st.error(f"Error building embeddings: {e}")
            return None, None
    
    def search(self, profile: Dict[str, Any], top_k: int = 20, 
               weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Hybrid search dengan multi-criteria weighted scoring
        
        Weights: {
            'bm25': 0.3,
            'embedding': 0.3,
            'education': 0.2,
            'skills': 0.1,
            'experience': 0.1
        }
        """
        if weights is None:
            weights = {
                'bm25': 0.3,
                'embedding': 0.3,
                'education': 0.2,
                'skills': 0.1,
                'experience': 0.1
            }
        
        results = self.df.copy()
        
        # 1. BM25 Scoring
        bm25_scores = self._get_bm25_scores(profile)
        
        # 2. Embedding Scoring
        embedding_scores = self._get_embedding_scores(profile)
        
        # 3. Education Matching
        education_scores = self._get_education_scores(profile)
        
        # 4. Skills Matching
        skills_scores = self._get_skills_scores(profile)
        
        # 5. Experience Matching
        experience_scores = self._get_experience_scores(profile)
        
        # Combine scores
        final_scores = (
            weights.get('bm25', 0.3) * bm25_scores +
            weights.get('embedding', 0.3) * embedding_scores +
            weights.get('education', 0.2) * education_scores +
            weights.get('skills', 0.1) * skills_scores +
            weights.get('experience', 0.1) * experience_scores
        )
        
        results['match_score'] = final_scores
        results['bm25_score'] = bm25_scores
        results['embedding_score'] = embedding_scores
        results['education_score'] = education_scores
        results['skills_score'] = skills_scores
        results['experience_score'] = experience_scores
        
        # Sort and return top-k
        results = results.sort_values('match_score', ascending=False).head(top_k)
        return results.reset_index(drop=True)
    
    def _get_bm25_scores(self, profile: Dict[str, Any]) -> np.ndarray:
        """Calculate BM25 scores"""
        if not HAS_BM25 or self.bm25_index is None:
            return np.zeros(len(self.df))
        
        query_text = self._build_query_text(profile)
        query_tokens = tokenize_id(query_text)
        
        if not query_tokens:
            return np.zeros(len(self.df))
        
        scores = self.bm25_index.get_scores(query_tokens)
        return MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten()
    
    def _get_embedding_scores(self, profile: Dict[str, Any]) -> np.ndarray:
        """Calculate embedding similarity scores"""
        if not HAS_SBERT or self.embedding_model is None or self.embeddings is None:
            return np.zeros(len(self.df))
        
        query_text = self._build_query_text(profile)
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        return MinMaxScaler().fit_transform(similarities.reshape(-1, 1)).flatten()
    
    def _get_education_scores(self, profile: Dict[str, Any]) -> np.ndarray:
        """Calculate education matching scores"""
        user_edu = profile.get("pendidikan_terakhir", "")
        user_jurusan = profile.get("jurusan", "")
        user_level = get_education_level(user_edu)
        
        scores = []
        for _, row in self.df.iterrows():
            required_edu = str(row.get("kualifikasi_pendidikan", ""))
            required_level = get_education_level(required_edu)
            
            # Level matching
            if user_level >= required_level:
                level_score = 1.0
            elif user_level == required_level - 0.5:  # e.g., D3 vs S1
                level_score = 0.7
            else:
                level_score = 0.3
            
            # Jurusan matching (fuzzy)
            jurusan_score = 0.5
            if user_jurusan and HAS_FUZZY:
                fuzzy_score = fuzz.partial_ratio(
                    user_jurusan.lower(), 
                    required_edu.lower()
                ) / 100.0
                jurusan_score = fuzzy_score
            
            final_score = 0.6 * level_score + 0.4 * jurusan_score
            scores.append(final_score)
        
        return np.array(scores)
    
    def _get_skills_scores(self, profile: Dict[str, Any]) -> np.ndarray:
        """Calculate skills matching scores"""
        user_skills = profile.get("hard_skills", []) + profile.get("soft_skills", [])
        user_skills = [s.lower() for s in user_skills]
        user_skills_expanded = expand_skills(user_skills)
        
        scores = []
        for _, row in self.df.iterrows():
            required_skills = str(row.get("persyaratan_kompetensi", "")).lower()
            
            if not required_skills:
                scores.append(0.5)
                continue
            
            # Count matches
            matches = sum(1 for skill in user_skills_expanded if skill in required_skills)
            score = min(matches / max(len(user_skills), 1), 1.0)
            scores.append(score)
        
        return np.array(scores)
    
    def _get_experience_scores(self, profile: Dict[str, Any]) -> np.ndarray:
        """Calculate experience matching scores"""
        user_exp = profile.get("pengalaman_kerja", "")
        user_exp_lower = user_exp.lower()
        
        scores = []
        for _, row in self.df.iterrows():
            job_desc = str(row.get("tugas_pokok", "")).lower()
            
            if not job_desc or not user_exp:
                scores.append(0.5)
                continue
            
            # Simple keyword matching
            exp_tokens = set(tokenize_id(user_exp_lower))
            job_tokens = set(tokenize_id(job_desc))
            
            if exp_tokens and job_tokens:
                overlap = len(exp_tokens & job_tokens)
                score = min(overlap / len(exp_tokens), 1.0)
                scores.append(score)
            else:
                scores.append(0.5)
        
        return np.array(scores)
    
    def _build_query_text(self, profile: Dict[str, Any]) -> str:
        """Build comprehensive query text dari profile"""
        parts = []
        
        # Pendidikan & Jurusan
        if profile.get("pendidikan_terakhir"):
            parts.append(profile["pendidikan_terakhir"])
        if profile.get("jurusan"):
            parts.append(profile["jurusan"])
        
        # Skills
        if profile.get("hard_skills"):
            parts.extend(profile["hard_skills"])
        if profile.get("soft_skills"):
            parts.extend(profile["soft_skills"])
        
        # Minat & Preferensi
        if profile.get("minat_bidang"):
            parts.extend(profile["minat_bidang"])
        if profile.get("preferensi_kerja"):
            parts.extend(profile["preferensi_kerja"])
        
        # Pengalaman (keywords)
        if profile.get("pengalaman_kerja"):
            exp_keywords = tokenize_id(profile["pengalaman_kerja"])
            parts.extend(exp_keywords[:10])
        
        return " ".join(parts)


# ---------------------------
# LLM Reranker (FIXED)
# ---------------------------

class LLMReranker:
    """LLM-based reranking dengan Gemini - FIXED VERSION"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Args:
            api_key: Gemini API key
            model_name: Model Gemini yang digunakan
        """
        if not HAS_GEMINI:
            raise ImportError("Google Generative AI library tidak tersedia. Install: pip install google-generativeai")
        
        if not api_key or api_key.strip() == "":
            raise ValueError("API key tidak boleh kosong!")
        
        try:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name)
            self.model_name = model_name
            self.api_key = api_key
        except Exception as e:
            raise Exception(f"Gagal menginisialisasi Gemini: {str(e)}")
    
    def generate_overall_summary(self, profile: Dict[str, Any], top_results: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
        """Generate overall summary untuk top N recommendations"""
        
        # Ambil top N jabatan
        top_jobs = top_results.head(top_n)
        
        # Build ringkasan jabatan
        jobs_summary = []
        for idx, row in top_jobs.iterrows():
            jobs_summary.append({
                'judul': row.get('judul_jabatan', '-'),
                'instansi': row.get('instansi', '-'),
                'score': row.get('match_score', 0) * 100,
                'kualifikasi': row.get('kualifikasi_pendidikan', '-')
            })
        
        prompt = self._build_summary_prompt(profile, jobs_summary)
        
        try:
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            
            response = self.client.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.4,
                    'max_output_tokens': 800,
                    'top_p': 0.95,
                },
                safety_settings=safety_settings
            )
            
            # Cek response validity
            if not response or not response.candidates:
                return {
                    'summary': "Tidak dapat menghasilkan analisis AI saat ini (no response).",
                    'strengths': [],
                    'development_areas': [],
                    'recommendations': []
                }
            
            if response.candidates[0].finish_reason != 1:
                finish_reason_map = {
                    0: "Unspecified",
                    2: "Max tokens",
                    3: "Safety filter",
                    4: "Recitation",
                    5: "Other"
                }
                reason = finish_reason_map.get(response.candidates[0].finish_reason, "Unknown")
                return {
                    'summary': f"Response blocked: {reason}. Coba lagi atau gunakan model berbeda.",
                    'strengths': [],
                    'development_areas': [],
                    'recommendations': []
                }
            
            content = response.text
            return self._parse_summary_response(content)
            
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "401" in error_msg:
                return {
                    'summary': "API key tidak valid. Silakan periksa kembali API key Anda.",
                    'strengths': [],
                    'development_areas': [],
                    'recommendations': []
                }
            else:
                return {
                    'summary': f"Error: {error_msg}",
                    'strengths': [],
                    'development_areas': [],
                    'recommendations': []
                }

    def _build_summary_prompt(self, profile: Dict[str, Any], jobs_summary: List[Dict]) -> str:
        """Build prompt untuk overall summary"""
        
        # Format jobs list
        jobs_text = "\n".join([
            f"{i+1}. {job['judul']} - {job['instansi']} (Score: {job['score']:.1f}%)"
            for i, job in enumerate(jobs_summary)
        ])
        
        # Get skills dengan fallback
        hard_skills = profile.get('hard_skills', [])
        soft_skills = profile.get('soft_skills', [])
        minat = profile.get('minat_bidang', [])
        
        hard_skills_str = ', '.join(hard_skills[:8]) if hard_skills else 'Tidak disebutkan'
        soft_skills_str = ', '.join(soft_skills[:5]) if soft_skills else 'Tidak disebutkan'
        minat_str = ', '.join(minat[:3]) if minat else 'Tidak disebutkan'
        
        return f"""
Sebagai expert recruiter PNS, berikan analisis mendalam untuk kandidat berikut:

PROFIL KANDIDAT:
- Pendidikan: {profile.get('pendidikan_terakhir', '-')} - {profile.get('jurusan', '-')}
- IPK: {profile.get('ipk', '-')}
- Hard Skills: {hard_skills_str}
- Soft Skills: {soft_skills_str}
- Minat: {minat_str}

TOP REKOMENDASI JABATAN:
{jobs_text}

Berikan analisis dalam format JSON:
{{
"summary": "<ringkasan 2-3 kalimat tentang kecocokan kandidat dengan jabatan-jabatan tersebut>",
"strengths": ["<kekuatan 1>", "<kekuatan 2>", "<kekuatan 3>"],
"development_areas": ["<area pengembangan 1>", "<area pengembangan 2>"],
"recommendations": ["<saran 1>", "<saran 2>", "<saran 3>"]
}}
"""

    def _parse_summary_response(self, content: str) -> Dict[str, Any]:
        """Parse overall summary response"""
        try:
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'summary': data.get('summary', ''),
                    'strengths': data.get('strengths', []),
                    'development_areas': data.get('development_areas', []),
                    'recommendations': data.get('recommendations', [])
                }
        except Exception:
            pass
        
        # Fallback
        return {
            'summary': content[:300] if content else "Tidak dapat menghasilkan summary",
            'strengths': [],
            'development_areas': [],
            'recommendations': []
        }


# ---------------------------
# Visualization
# ---------------------------

def plot_match_scores(df: pd.DataFrame, top_n: int = 10, key_suffix: str = ""):
    """Plot match scores dengan breakdown"""
    if not HAS_PLOTLY:
        st.warning("Plotly tidak tersedia untuk visualisasi")
        return
    
    df_plot = df.head(top_n).copy()
    df_plot['jabatan_short'] = df_plot['judul_jabatan'].apply(
        lambda x: x[:40] + '...' if len(x) > 40 else x
    )
    
    # Stacked bar chart untuk score breakdown
    fig = go.Figure()
    
    score_components = ['bm25_score', 'embedding_score', 'education_score', 
                       'skills_score', 'experience_score']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for component, color in zip(score_components, colors):
        if component in df_plot.columns:
            fig.add_trace(go.Bar(
                name=component.replace('_', ' ').title(),
                x=df_plot['jabatan_short'],
                y=df_plot[component] * 100,
                marker_color=color
            ))
    
    fig.update_layout(
        title="Score Breakdown per Jabatan",
        xaxis_title="Jabatan",
        yaxis_title="Score (%)",
        barmode='stack',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"match_scores_{key_suffix}")


def plot_radar_chart(scores: Dict[str, float], key_suffix: str = ""):
    """Plot radar chart untuk score breakdown"""
    if not HAS_PLOTLY:
        return
    
    categories = list(scores.keys())
    values = [scores[k] * 100 for k in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"radar_{key_suffix}")


# ---------------------------
# Export Functions
# ---------------------------

def export_to_excel(results: pd.DataFrame, profile: Dict[str, Any]) -> bytes:
    """Export hasil ke Excel"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Profil
        df_profile = pd.DataFrame([{
            'Pendidikan': profile.get('pendidikan_terakhir', '-'),
            'Jurusan': profile.get('jurusan', '-'),
            'IPK': profile.get('ipk', '-'),
            'Hard Skills': ', '.join(profile.get('hard_skills', [])),
            'Soft Skills': ', '.join(profile.get('soft_skills', [])),
        }])
        df_profile.to_excel(writer, sheet_name='Profil', index=False)
        
        # Sheet 2: Results
        export_cols = ['judul_jabatan', 'instansi', 'unit_organisasi', 
                      'kualifikasi_pendidikan', 'match_score', 
                      'bm25_score', 'embedding_score', 'education_score',
                      'skills_score', 'experience_score']
        
        df_export = results[export_cols].copy()
        df_export['match_score'] = (df_export['match_score'] * 100).round(2)
        
        for col in ['bm25_score', 'embedding_score', 'education_score', 'skills_score', 'experience_score']:
            if col in df_export.columns:
                df_export[col] = (df_export[col] * 100).round(2)
        
        df_export.to_excel(writer, sheet_name='Rekomendasi', index=False)
    
    output.seek(0)
    return output.getvalue()


def save_search_history(profile: Dict[str, Any], results: pd.DataFrame):
    """Simpan riwayat pencarian"""
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'profile_summary': f"{profile.get('pendidikan_terakhir', '-')} - {profile.get('jurusan', '-')}",
        'results_count': len(results),
        'top_match': results.iloc[0]['judul_jabatan'] if len(results) > 0 else '-',
        'top_score': float(results.iloc[0]['match_score']) if len(results) > 0 else 0
    }
    
    # Load existing history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(history_entry)
    
    # Save
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history[-100:], f, indent=2)  # Keep last 100 entries


def save_feedback(jabatan: str, is_positive: bool, profile: Dict[str, Any]):
    """Simpan feedback untuk learning"""
    feedback_data = {
        'timestamp': datetime.now().isoformat(),
        'jabatan': jabatan,
        'is_positive': is_positive,
        'profile_snapshot': json.dumps(profile)
    }
    
    if os.path.exists(FEEDBACK_FILE):
        df_feedback = pd.read_csv(FEEDBACK_FILE)
        df_feedback = pd.concat([df_feedback, pd.DataFrame([feedback_data])], ignore_index=True)
    else:
        df_feedback = pd.DataFrame([feedback_data])
    
    df_feedback.to_csv(FEEDBACK_FILE, index=False)


# ---------------------------
# Main App (FIXED)
# ---------------------------


def main():
    # Header
    st.title("🧭 Smart PNS Job Recommender")
    st.markdown("**Sistem Rekomendasi Jabatan PNS dengan AI & Machine Learning**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        # ========================================
        # FIXED: Gemini API Key Configuration
        # ========================================
        gemini_api_key = None
        use_llm = False
        
        # 1. Coba ambil dari secrets.toml
        try:
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
            if gemini_api_key and gemini_api_key.strip():
                st.success("✅ API Key ditemukan di secrets.toml")
        except:
            pass
        
        # 2. Jika tidak ada di secrets, coba dari environment variable
        if not gemini_api_key:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key and gemini_api_key.strip():
                st.success("✅ API Key ditemukan di environment variable")
        
        # 3. Jika masih tidak ada, beri opsi input manual
        if not gemini_api_key:
            st.warning("⚠️ Gemini API Key tidak ditemukan")
            st.markdown("Dapatkan API key gratis di: [Google AI Studio](https://makersuite.google.com/app/apikey)")
            
            gemini_api_key = st.text_input(
                "Masukkan Gemini API Key",
                type="password",
                help="API key akan digunakan untuk fitur AI Analysis"
            )
        
        # 4. Validasi dan aktifkan fitur AI
        if gemini_api_key and gemini_api_key.strip() and HAS_GEMINI:
            use_llm = st.checkbox(
                "🤖 Aktifkan Gemini AI Analysis", 
                value=True,
                help="Menggunakan Gemini untuk analisis mendalam hasil rekomendasi"
            )
            
            if use_llm:
                # Pilihan model Gemini
                gemini_model = st.selectbox(
                    "Model Gemini",
                    ["gemini-2.0-flash-exp","gemini-2.5-flash"],
                    index=0,
                    help="Flash = cepat & murah, Pro = lebih akurat"
                )
                
                # Test connection
                with st.spinner("Testing API connection..."):
                    try:
                        genai.configure(api_key=gemini_api_key)
                        test_model = genai.GenerativeModel(gemini_model)
                        test_response = test_model.generate_content("Hello")
                        st.success(f"✅ Gemini AI siap! Model: {gemini_model}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("Periksa kembali API key Anda atau coba model lain")
                        use_llm = False
            else:
                gemini_model = "gemini-1.5-flash"
        else:
            use_llm = False
            gemini_model = "gemini-1.5-flash"
            if not HAS_GEMINI:
                st.info("📦 Install library: pip install google-generativeai")
        
        st.divider()
        
        # Search weights
        st.subheader("⚖️ Bobot Pencarian")
        weight_bm25 = st.slider("BM25 (keyword)", 0.0, 1.0, 0.3, 0.05)
        weight_embedding = st.slider("Embedding (semantic)", 0.0, 1.0, 0.3, 0.05)
        weight_education = st.slider("Pendidikan", 0.0, 1.0, 0.2, 0.05)
        weight_skills = st.slider("Skills", 0.0, 1.0, 0.1, 0.05)
        weight_experience = st.slider("Pengalaman", 0.0, 1.0, 0.1, 0.05)
        
        weights = {
            'bm25': weight_bm25,
            'embedding': weight_embedding,
            'education': weight_education,
            'skills': weight_skills,
            'experience': weight_experience
        }
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        st.info(f"Total bobot: {sum(weights.values()):.2f}")
        
        # Top-K
        top_k = st.number_input("Jumlah Rekomendasi", 5, 50, 20, 5)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "📁 Data Management", "👤 Profil Kandidat", 
        "🎯 Hasil Rekomendasi", "📈 Analytics"
    ])
    
    # Initialize session state
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = None
    if 'profile' not in st.session_state:
        st.session_state.profile = {}
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Tab 1: Dashboard
    with tab1:
        st.header("📊 Dashboard Overview")
        
        if st.session_state.data_manager.df is not None:
            stats = st.session_state.data_manager.get_statistics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Jabatan", stats.get('total_jabatan', 0))
            with col2:
                st.metric("Total Instansi", stats.get('total_instansi', 0))
            with col3:
                st.metric("Total Formasi", stats.get('total_formasi', 0))
            
            # Top Education
            if 'top_education' in stats:
                st.subheader("📚 Top 10 Kualifikasi Pendidikan")
                df_edu = pd.DataFrame(list(stats['top_education'].items()), 
                                     columns=['Pendidikan', 'Jumlah'])
                st.bar_chart(df_edu.set_index('Pendidikan'))
            
            # Top Locations
            if 'top_locations' in stats:
                st.subheader("📍 Top 10 Lokasi Penempatan")
                df_loc = pd.DataFrame(list(stats['top_locations'].items()), 
                                     columns=['Lokasi', 'Jumlah'])
                st.bar_chart(df_loc.set_index('Lokasi'))
        else:
            st.info("Silakan upload data di tab 'Data Management' terlebih dahulu")
    
    # Tab 2: Data Management
    with tab2:
        st.header("📁 Data Management")
        
        # Upload atau generate template
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload CSV Data Jabatan",
                type=['csv'],
                help="Format CSV dengan kolom: judul_jabatan, instansi, unit_organisasi, dst"
            )
            
            if uploaded_file:
                with st.spinner("Loading data..."):
                    df = st.session_state.data_manager.load_data(uploaded_file=uploaded_file)
                    
                    if df is not None:
                        # Build indexes
                        st.info("Building search indexes...")
                        search_engine = HybridSearchEngine(df)
                        
                        if HAS_BM25:
                            search_engine.bm25_index, search_engine.corpus_tokens = \
                                search_engine.build_bm25_index(st.session_state.data_manager.data_hash)
                        
                        if HAS_SBERT:
                            search_engine.embedding_model, search_engine.embeddings = \
                                search_engine.build_embeddings(st.session_state.data_manager.data_hash)
                        
                        st.session_state.search_engine = search_engine
                        st.success("✅ Data dan search engine siap!")
        
        with col2:
            st.subheader("Generate Template CSV")
            
            if st.button("📥 Download Template"):
                template_cols = [
                    "judul_jabatan", "instansi", "unit_organisasi",
                    "kualifikasi_pendidikan", "persyaratan_kompetensi",
                    "tugas_pokok", "fungsi_jabatan", "jumlah_formasi",
                    "lokasi_penempatan", "kode_formasi", "tautan_detail"
                ]
                df_template = pd.DataFrame(columns=template_cols)
                
                # Add sample row
                df_template.loc[0] = [
                    "Analis Kebijakan Publik",
                    "Kementerian Dalam Negeri",
                    "Direktorat Jenderal Otonomi Daerah",
                    "S1 Administrasi Publik/Ilmu Pemerintahan",
                    "Analisis Kebijakan, Komunikasi, MS Office",
                    "Menganalisis kebijakan otonomi daerah",
                    "Perencanaan dan evaluasi kebijakan",
                    5,
                    "Jakarta",
                    "FORM-2025-001",
                    "https://example.com/formasi/001"
                ]
                
                csv = df_template.to_csv(index=False)
                st.download_button(
                    "💾 Download Template CSV",
                    csv,
                    "template_jabatan_pns.csv",
                    "text/csv"
                )
        
        # Show data preview
        if st.session_state.data_manager.df is not None:
            st.subheader("Preview Data")
            st.dataframe(
                st.session_state.data_manager.df.head(100),
                use_container_width=True,
                height=400
            )
    
    # Tab 3: Profil Kandidat
    with tab3:
        st.header("👤 Profil Kandidat")
        
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Pendidikan")
                pendidikan = st.selectbox(
                    "Pendidikan Terakhir*",
                    ["", "SMA/SMK", "D3", "D4/S1", "S2", "S3"],
                    help="Jenjang pendidikan terakhir"
                )
                jurusan = st.text_input("Jurusan*", help="Contoh: Teknik Informatika")
                ipk = st.number_input("IPK", 0.0, 4.0, 0.0, 0.01, help="Opsional")
                
                st.subheader("Kompetensi")
                hard_skills = st.text_area(
                    "Hard Skills*",
                    help="Pisahkan dengan koma. Contoh: Python, Data Analysis, SQL"
                )
                soft_skills = st.text_area(
                    "Soft Skills*",
                    help="Pisahkan dengan koma. Contoh: Komunikasi, Leadership, Problem Solving"
                )
                sertifikasi = st.text_area(
                    "Sertifikasi",
                    help="Pisahkan dengan koma. Contoh: PMP, CISCO, AWS"
                )
            
            with col2:
                st.subheader("Preferensi")
                minat_bidang = st.text_area(
                    "Minat Bidang*",
                    help="Pisahkan dengan koma. Contoh: Teknologi Informasi, Kebijakan Publik"
                )
                preferensi_kerja = st.text_area(
                    "Preferensi Kerja*",
                    help="Pisahkan dengan koma. Contoh: Remote, Jakarta, Tim Kecil"
                )
                
                st.subheader("Pengalaman")
                pengalaman = st.text_area(
                    "Pengalaman Kerja*",
                    help="Deskripsikan pengalaman kerja relevan Anda (3-5 kalimat)",
                    height=150
                )
                
                st.subheader("Nilai Pribadi")
                nilai_pribadi = st.text_area(
                    "Nilai Pribadi",
                    help="Contoh: Integritas, Inovasi, Pelayanan Publik",
                    height=100
                )
            
            submitted = st.form_submit_button("💾 Simpan Profil", use_container_width=True)
            
            if submitted:
                # Validasi
                if not all([pendidikan, jurusan, hard_skills, soft_skills, 
                           minat_bidang, preferensi_kerja, pengalaman]):
                    st.error("❌ Mohon lengkapi semua field yang bertanda *")
                else:
                    profile = {
                        'pendidikan_terakhir': pendidikan,
                        'jurusan': jurusan,
                        'ipk': ipk if ipk > 0 else None,
                        'hard_skills': [s.strip() for s in hard_skills.split(',') if s.strip()],
                        'soft_skills': [s.strip() for s in soft_skills.split(',') if s.strip()],
                        'sertifikasi': [s.strip() for s in sertifikasi.split(',') if s.strip()],
                        'minat_bidang': [s.strip() for s in minat_bidang.split(',') if s.strip()],
                        'preferensi_kerja': [s.strip() for s in preferensi_kerja.split(',') if s.strip()],
                        'pengalaman_kerja': pengalaman,
                        'nilai_pribadi': nilai_pribadi
                    }
                    
                    st.session_state.profile = profile
                    st.success("✅ Profil berhasil disimpan!")
                    
                    # Show summary
                    with st.expander("📋 Ringkasan Profil"):
                        st.json(profile)
    
    # Tab 4: Hasil Rekomendasi
    with tab4:
        st.header("🎯 Hasil Rekomendasi")
        
        if not st.session_state.profile:
            st.warning("⚠️ Silakan lengkapi profil di tab 'Profil Kandidat' terlebih dahulu")
        elif st.session_state.search_engine is None:
            st.warning("⚠️ Silakan upload data di tab 'Data Management' terlebih dahulu")
        else:
            # Di tab "🎯 Hasil Rekomendasi", bagian tombol search
            if st.button("🔍 Cari Rekomendasi Jabatan", type="primary", use_container_width=True):
                with st.spinner("Memproses rekomendasi..."):
                    # Hybrid search
                    results = st.session_state.search_engine.search(
                        st.session_state.profile,
                        top_k=top_k,
                        weights=weights
                    )
                    
                    st.session_state.results = results
                    
                    # Save history
                    save_search_history(st.session_state.profile, results)
                    
                    st.success(f"✅ Ditemukan {len(results)} rekomendasi jabatan!")
                    
                    if use_llm:
                        st.info("💡 Scroll ke bawah untuk melihat AI Analysis Summary")   
                        
                        
            # Display results
            if st.session_state.results is not None and len(st.session_state.results) > 0:
                results = st.session_state.results
                
                # Score breakdown visualization
                if HAS_PLOTLY:
                    plot_match_scores(results, min(10, len(results)), key_suffix="main")
                
                # Export button
                col1, col2 = st.columns([3, 1])
                with col2:
                    excel_data = export_to_excel(results, st.session_state.profile)
                    st.download_button(
                        "📥 Export ke Excel",
                        excel_data,
                        f"rekomendasi_jabatan_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.divider()
                
                # ========================================
                # AI ANALYSIS SUMMARY (Hanya sekali di sini) - FIXED
                # ========================================
                if use_llm and gemini_api_key:
                    st.subheader("🤖 AI Analysis Summary")
                    
                    with st.spinner("Menghasilkan analisis mendalam dengan Gemini AI..."):
                        try:
                            llm_reranker = LLMReranker(
                                api_key=gemini_api_key,
                                model_name=gemini_model
                            )
                            
                            ai_summary = llm_reranker.generate_overall_summary(
                                st.session_state.profile,
                                results,
                                top_n=5
                            )
                            
                            # Display summary
                            if ai_summary.get('summary'):
                                col_sum1, col_sum2 = st.columns([2, 1])
                                
                                with col_sum1:
                                    # Overall summary
                                    st.markdown("**📊 Analisis Kecocokan:**")
                                    summary_text = ai_summary.get('summary', 'Tidak ada analisis tersedia')
                                    
                                    # Check for error messages
                                    if "Error" in summary_text or "blocked" in summary_text or "invalid" in summary_text.lower():
                                        st.error(summary_text)
                                    else:
                                        st.info(summary_text)
                                    
                                    # Strengths
                                    if ai_summary.get('strengths'):
                                        st.markdown("**✨ Kekuatan Kandidat:**")
                                        for strength in ai_summary['strengths']:
                                            st.success(f"✓ {strength}")
                                
                                with col_sum2:
                                    # Development areas
                                    if ai_summary.get('development_areas'):
                                        st.markdown("**📈 Area Pengembangan:**")
                                        for area in ai_summary['development_areas']:
                                            st.warning(f"• {area}")
                                
                                # Recommendations
                                if ai_summary.get('recommendations'):
                                    st.markdown("**💡 Rekomendasi Aksi:**")
                                    for i, rec in enumerate(ai_summary['recommendations'], 1):
                                        st.markdown(f"{i}. {rec}")
                            else:
                                st.warning("Tidak dapat menghasilkan AI analysis. Silakan coba lagi.")
                            
                        except ValueError as ve:
                            st.error(f"❌ {str(ve)}")
                            st.info("Pastikan API key Anda valid dan aktif.")
                        except Exception as e:
                            st.error(f"❌ Gagal menghasilkan AI analysis: {str(e)}")
                            st.info("Coba refresh halaman atau gunakan model Gemini yang berbeda.")
                    
                    st.divider()
                
                # ========================================
                # RESULTS DISPLAY (Tanpa AI per item)
                # ========================================
                st.subheader(f"Top {len(results)} Rekomendasi")
                
                for idx, row in results.iterrows():
                    with st.expander(
                        f"#{idx+1} — {row['judul_jabatan']} "
                        f"(Score: {row['match_score']*100:.1f}%)",
                        expanded=(idx < 3)
                    ):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Instansi:** {row.get('instansi', '-')}")
                            st.markdown(f"**Unit:** {row.get('unit_organisasi', '-')}")
                            st.markdown(f"**Lokasi:** {row.get('lokasi_penempatan', '-')}")
                            st.markdown(f"**Formasi:** {row.get('jumlah_formasi', '-')}")
                            
                            # Kualifikasi
                            st.markdown("**Kualifikasi:**")
                            st.info(row.get('kualifikasi_pendidikan', '-'))
                            
                            # Kompetensi
                            if pd.notna(row.get('persyaratan_kompetensi')):
                                st.markdown("**Kompetensi:**")
                                st.info(row.get('persyaratan_kompetensi', '-'))
                            
                            # Tupoksi
                            if pd.notna(row.get('tugas_pokok')):
                                with st.expander("📋 Tugas Pokok"):
                                    st.write(row.get('tugas_pokok', '-'))
                        
                        with col2:
                            # Score breakdown
                            st.markdown("**Score Breakdown:**")
                            
                            scores_to_show = {
                                'Match': row['match_score'],
                                'BM25': row.get('bm25_score', 0),
                                'Embedding': row.get('embedding_score', 0),
                                'Education': row.get('education_score', 0),
                                'Skills': row.get('skills_score', 0),
                                'Experience': row.get('experience_score', 0)
                            }
                            
                            if HAS_PLOTLY:
                                plot_radar_chart(scores_to_show, key_suffix=f"result_{idx}")
                            else:
                                for key, val in scores_to_show.items():
                                    st.metric(key, f"{val*100:.1f}%")
                        
                        # Feedback buttons
                        col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 4])
                        with col_fb1:
                            if st.button("👍 Cocok", key=f"pos_{idx}"):
                                save_feedback(row['judul_jabatan'], True, st.session_state.profile)
                                st.success("Terima kasih atas feedback Anda!")
                        with col_fb2:
                            if st.button("👎 Tidak Cocok", key=f"neg_{idx}"):
                                save_feedback(row['judul_jabatan'], False, st.session_state.profile)
                                st.info("Feedback tersimpan!")    
    
    # Tab 5: Analytics
    with tab5:
        st.header("📈 Analytics & Insights")
        
        # Search history
        if os.path.exists(HISTORY_FILE):
            st.subheader("📜 Riwayat Pencarian")
            
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            
            if history:
                df_history = pd.DataFrame(history)
                df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
                
                st.dataframe(df_history[['timestamp', 'results_count', 'top_match', 'top_score']], 
                        use_container_width=True)
                
                # Trend chart dengan key unik
                if HAS_PLOTLY and len(df_history) > 1:
                    fig = px.line(df_history, x='timestamp', y='top_score',
                                title='Trend Score Tertinggi')
                    st.plotly_chart(fig, use_container_width=True, key="analytics_trend")
        
        # Feedback analysis
        if os.path.exists(FEEDBACK_FILE):
            st.subheader("💬 Feedback Analysis")
            
            df_feedback = pd.read_csv(FEEDBACK_FILE)
            
            col1, col2 = st.columns(2)
            with col1:
                positive = df_feedback['is_positive'].sum()
                total = len(df_feedback)
                st.metric("Positive Feedback", f"{positive}/{total}", 
                        f"{positive/total*100:.1f}%" if total > 0 else "0%")
            
            with col2:
                st.metric("Total Feedback", total)
            
            # Top rated jabatan dengan key unik
            if total > 0:
                top_jabatan = df_feedback[df_feedback['is_positive']] \
                    .groupby('jabatan').size().sort_values(ascending=False).head(10)
                
                st.markdown("**Top 10 Jabatan (Most Positive Feedback):**")
                
                # Gunakan st.bar_chart atau plotly dengan key
                if HAS_PLOTLY:
                    fig = px.bar(x=top_jabatan.index, y=top_jabatan.values,
                                labels={'x': 'Jabatan', 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True, key="analytics_feedback")
                else:
                    st.bar_chart(top_jabatan)

# ---------------------------
# Run App
# ---------------------------

if __name__ == "__main__":
    main()