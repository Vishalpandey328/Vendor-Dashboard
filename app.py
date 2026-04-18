import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import os
import time
import json
import pickle
from datetime import datetime, timedelta
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import defaultdict
import torch
import chardet
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import io
import base64

# ------------------------
# PAGE CONFIG
st.set_page_config(
    page_title="AI Powered Center Matching System with Self-Learning",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .divider {
        margin: 1rem 0;
        border-top: 1px solid #e0e0e0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# REINFORCEMENT LEARNING MODEL
# --------------------------------------------------

class ReinforcementLearningMatcher:
    """Self-learning matcher using reinforcement learning principles"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.feature_weights = {
            'name_weight': 0.35,
            'address_weight': 0.25,
            'district_weight': 0.20,
            'state_weight': 0.10,
            'vector_weight': 0.10
        }
        self.successful_patterns = defaultdict(int)
        self.failure_patterns = defaultdict(int)
        self.match_history = []
        self.model_file = "rl_model.pkl"
        self.load_model()
    
    def get_state(self, input_text, candidate_text, similarity_scores):
        features = (
            round(similarity_scores.get('name', 0), 2),
            round(similarity_scores.get('address', 0), 2),
            round(similarity_scores.get('district', 0), 2),
            round(similarity_scores.get('state', 0), 2),
            round(similarity_scores.get('vector', 0), 2),
            len(input_text.split()),
            len(candidate_text.split())
        )
        return str(features)
    
    def get_action(self, state, available_actions=['accept', 'reject', 'adjust']):
        import random
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        if state in self.q_table:
            return max(self.q_table[state], key=self.q_table[state].get)
        return 'accept'
    
    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in ['accept', 'reject', 'adjust']}
        current_q = self.q_table[state].get(action, 0)
        max_future_q = 0
        if next_state in self.q_table:
            max_future_q = max(self.q_table[next_state].values())
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state][action] = new_q
    
    def update_weights(self, successful_match, features_used):
        for feature, value in features_used.items():
            if value > 0.8:
                self.feature_weights[f'{feature}_weight'] = min(0.5, self.feature_weights.get(f'{feature}_weight', 0.1) + self.learning_rate * 0.05)
            elif value < 0.3:
                self.feature_weights[f'{feature}_weight'] = max(0.05, self.feature_weights.get(f'{feature}_weight', 0.1) - self.learning_rate * 0.03)
        total = sum(self.feature_weights.values())
        if total > 0:
            for key in self.feature_weights:
                self.feature_weights[key] /= total
    
    def learn_from_match(self, input_text, matched_text, was_correct, confidence, similarity_scores, user_feedback=None):
        state = self.get_state(input_text, matched_text, similarity_scores)
        reward = 0
        if was_correct:
            reward = 1.0 + (confidence - 0.7)
            self.successful_patterns[state] += 1
            self.match_history.append({
                'timestamp': datetime.now(),
                'input': input_text[:100],
                'matched': matched_text[:100],
                'confidence': confidence,
                'success': True,
                'similarity_scores': similarity_scores
            })
            self.update_weights(True, similarity_scores)
        else:
            reward = -0.5 - (1 - confidence)
            self.failure_patterns[state] += 1
            self.match_history.append({
                'timestamp': datetime.now(),
                'input': input_text[:100],
                'matched': matched_text[:100],
                'confidence': confidence,
                'success': False,
                'similarity_scores': similarity_scores
            })
        
        if user_feedback:
            if user_feedback == 'thumbs_up':
                reward += 0.3
            elif user_feedback == 'thumbs_down':
                reward -= 0.5
            elif user_feedback == 'correct_match':
                reward += 0.5
        
        next_state = self.get_state(input_text, matched_text, similarity_scores)
        self.update_q_value(state, 'accept', reward, next_state)
        self.exploration_rate = max(0.05, self.exploration_rate * 0.995)
        
        if len(self.match_history) % 10 == 0:
            self.save_model()
    
    def get_adjusted_threshold(self, base_threshold=0.70):
        if len(self.match_history) < 10:
            return base_threshold
        recent_matches = self.match_history[-50:]
        if not recent_matches:
            return base_threshold
        success_rate = sum(1 for m in recent_matches if m['success']) / len(recent_matches)
        if success_rate > 0.9:
            return base_threshold + 0.05
        elif success_rate < 0.7:
            return base_threshold - 0.05
        return base_threshold
    
    def predict_match_quality(self, input_text, candidate_text, similarity_scores):
        state = self.get_state(input_text, candidate_text, similarity_scores)
        if state in self.q_table:
            q_value = self.q_table[state].get('accept', 0)
            probability = 1 / (1 + np.exp(-q_value))
            return probability
        return 0.5
    
    def save_model(self):
        model_data = {
            'q_table': self.q_table,
            'feature_weights': self.feature_weights,
            'successful_patterns': dict(self.successful_patterns),
            'failure_patterns': dict(self.failure_patterns),
            'match_history': self.match_history[-1000:],
            'exploration_rate': self.exploration_rate
        }
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.q_table = model_data.get('q_table', {})
                    self.feature_weights = model_data.get('feature_weights', self.feature_weights)
                    self.successful_patterns = defaultdict(int, model_data.get('successful_patterns', {}))
                    self.failure_patterns = defaultdict(int, model_data.get('failure_patterns', {}))
                    self.match_history = model_data.get('match_history', [])
                    self.exploration_rate = model_data.get('exploration_rate', 0.1)
            except:
                pass
    
    def get_learning_stats(self):
        total_matches = len(self.match_history)
        successful = sum(1 for m in self.match_history if m['success'])
        return {
            'total_learned_matches': total_matches,
            'success_rate': successful / total_matches if total_matches > 0 else 0,
            'unique_patterns': len(self.q_table),
            'exploration_rate': self.exploration_rate,
            'feature_weights': self.feature_weights,
            'recent_success_rate': self._get_recent_success_rate()
        }
    
    def _get_recent_success_rate(self, n=50):
        recent = self.match_history[-n:]
        if not recent:
            return 0
        return sum(1 for m in recent if m['success']) / len(recent)

# --------------------------------------------------
# ENHANCED TEXT CLEANING
# --------------------------------------------------

def enhanced_clean_text(text, synonyms_df):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s\-/]", " ", text)
    
    abbreviations = {
        r'\bvidhan sabha\b': 'vidhansabha',
        r'\bnagar nigam\b': 'nagarnigam',
        r'\bnagar palika\b': 'nagarparishad',
        r'\bgram panchayat\b': 'grampanchayat',
        r'\bst\b': 'saint',
        r'\bmt\b': 'mount',
        r'\bnr\b': 'near',
        r'\b\&\b': 'and',
        r'\bph\b': 'public high',
        r'\bghs\b': 'government high school',
        r'\bgps\b': 'government primary school',
        r'\bggic\b': 'government girls inter college',
        r'\bgic\b': 'government inter college',
        r'\bno\.\b': 'number',
        r'\bdist\.\b': 'district',
        r'\bdist\b': 'district',
        r'\bblk\b': 'block',
        r'\bpo\b': 'post office',
        r'\bps\b': 'primary school',
        r'\bhs\b': 'high school',
        r'\binter\b': 'intermediate',
    }
    
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
    
    if synonyms_df is not None and len(synonyms_df) > 0:
        for _, row in synonyms_df.iterrows():
            word = str(row["word"]).lower()
            replacement = str(row["replacement"]).lower()
            text = re.sub(rf'\b{word}\b', replacement, text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --------------------------------------------------
# FILE LOADING WITH ENCODING HANDLING
# --------------------------------------------------

def load_csv_with_encoding(file):
    try:
        raw_data = file.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
        file.seek(0)
        try:
            df = pd.read_csv(file, encoding=encoding)
        except:
            file.seek(0)
            for enc in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']:
                try:
                    df = pd.read_csv(file, encoding=enc)
                    break
                except:
                    continue
            else:
                raise Exception("Could not read CSV file with any common encoding")
        return df
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")

def load_excel_with_fallback(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        try:
            file.seek(0)
            df = pd.read_excel(file, engine='openpyxl')
            return df
        except:
            try:
                file.seek(0)
                df = pd.read_excel(file, engine='xlrd')
                return df
            except Exception as e2:
                raise Exception(f"Could not load Excel file. Error: {str(e2)}")

# --------------------------------------------------
# ENHANCED MATCHING WITH MULTIPLE STRATEGIES
# --------------------------------------------------

def advanced_matching(input_row, master_df, model, index, synonyms_df, rl_model, strategies):
    """Enhanced matching with multiple strategies"""
    clean_name = enhanced_clean_text(input_row['center_name'], synonyms_df)
    clean_district = enhanced_clean_text(input_row['district'], synonyms_df)
    clean_state = enhanced_clean_text(input_row['state'], synonyms_df)
    clean_address = enhanced_clean_text(input_row['address'], synonyms_df)
    
    query_text = f"{clean_name} {clean_district} {clean_state} {clean_address}"
    query_embedding = model.encode([query_text])
    query_embedding = normalize(query_embedding.astype(np.float32))
    
    # Vector search with dynamic k
    k = strategies.get('top_k', 15)
    distances, indices = index.search(query_embedding, k)
    
    scored_candidates = []
    
    for master_idx, distance in zip(indices[0], distances[0]):
        if master_idx < len(master_df):
            master_row = master_df.iloc[master_idx]
            
            # Calculate multiple similarity scores
            name_score = fuzz.token_set_ratio(clean_name, master_row['clean_name']) / 100
            address_score = fuzz.token_set_ratio(clean_address, master_row['clean_address']) / 100
            district_score = fuzz.ratio(clean_district, master_row['clean_district']) / 100
            state_score = fuzz.ratio(clean_state, master_row['clean_state']) / 100
            vector_score = 1 / (1 + distance)
            
            # Additional fuzzy matching strategies
            partial_ratio = fuzz.partial_ratio(clean_name, master_row['clean_name']) / 100
            token_sort_ratio = fuzz.token_sort_ratio(clean_name, master_row['clean_name']) / 100
            
            similarity_scores = {
                'name': name_score,
                'address': address_score,
                'district': district_score,
                'state': state_score,
                'vector': vector_score,
                'partial': partial_ratio,
                'token_sort': token_sort_ratio
            }
            
            # Dynamic weight adjustment based on strategies
            if strategies.get('boost_address', False):
                rl_model.feature_weights['address_weight'] = min(0.4, rl_model.feature_weights['address_weight'] + 0.05)
            if strategies.get('boost_name', False):
                rl_model.feature_weights['name_weight'] = min(0.5, rl_model.feature_weights['name_weight'] + 0.05)
            
            final_score = (
                rl_model.feature_weights.get('name_weight', 0.35) * name_score +
                rl_model.feature_weights.get('address_weight', 0.25) * address_score +
                rl_model.feature_weights.get('district_weight', 0.20) * district_score +
                rl_model.feature_weights.get('state_weight', 0.10) * state_score +
                rl_model.feature_weights.get('vector_weight', 0.10) * vector_score
            )
            
            # Apply strategy modifiers
            if strategies.get('strict_matching', False) and final_score < 0.8:
                final_score *= 0.9
            if strategies.get('fuzzy_enhanced', False):
                final_score = max(final_score, partial_ratio * 0.8 + token_sort_ratio * 0.2)
            
            predicted_quality = rl_model.predict_match_quality(query_text, master_row['clean_name'], similarity_scores)
            
            scored_candidates.append({
                'master_id': master_row['center_id'] if 'center_id' in master_row else master_idx,
                'master_name': master_row['center_name'],
                'master_address': master_row['address'],
                'master_district': master_row['district'],
                'master_state': master_row['state'],
                'score': final_score,
                'predicted_quality': predicted_quality,
                'similarity_scores': similarity_scores,
                'master_row': master_row
            })
    
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    return scored_candidates

def match_with_rl(input_data, master_df, model, index, synonyms_df, rl_model, confidence_threshold=0.70, strategies=None):
    if strategies is None:
        strategies = {}
    
    results = []
    match_details = []
    learning_opportunities = []
    
    # Add center_id if not exists
    if 'center_id' not in master_df.columns:
        master_df['center_id'] = range(len(master_df))
    
    master_df['clean_name'] = master_df['center_name'].apply(lambda x: enhanced_clean_text(x, synonyms_df))
    master_df['clean_address'] = master_df['address'].apply(lambda x: enhanced_clean_text(x, synonyms_df))
    master_df['clean_district'] = master_df['district'].apply(lambda x: enhanced_clean_text(x, synonyms_df))
    master_df['clean_state'] = master_df['state'].apply(lambda x: enhanced_clean_text(x, synonyms_df))
    
    adjusted_threshold = rl_model.get_adjusted_threshold(confidence_threshold)
    
    total = len(input_data)
    progress_bar = st.progress(0)
    
    for idx, row in input_data.iterrows():
        progress_bar.progress((idx + 1) / total)
        
        scored_candidates = advanced_matching(row, master_df, model, index, synonyms_df, rl_model, strategies)
        
        if scored_candidates and scored_candidates[0]['score'] >= adjusted_threshold:
            best = scored_candidates[0]
            results.append(best['master_name'])
            match_details.append({
                'master_id': best['master_id'],
                'master_name': best['master_name'],
                'master_address': best['master_address'],
                'master_district': best['master_district'],
                'master_state': best['master_state'],
                'confidence': best['score'],
                'predicted_quality': best['predicted_quality'],
                'similarity_scores': best['similarity_scores'],
                'matched_text': best['master_name']
            })
            
            learning_opportunities.append({
                'input_text': f"{row['center_name']} {row['district']} {row['state']}",
                'matched_text': best['master_name'],
                'confidence': best['score'],
                'similarity_scores': best['similarity_scores'],
                'row_index': idx
            })
        else:
            results.append("⚡ No Match")
            match_details.append({
                'master_id': 'NULL',
                'master_name': None,
                'master_address': None,
                'master_district': None,
                'master_state': None,
                'confidence': scored_candidates[0]['score'] if scored_candidates else 0,
                'predicted_quality': 0,
                'similarity_scores': {},
                'matched_text': None
            })
    
    progress_bar.empty()
    return results, match_details, learning_opportunities

# --------------------------------------------------
# SYNOPSIS MANAGEMENT
# --------------------------------------------------

def load_synonyms():
    synonym_file = "synonyms.csv"
    if os.path.exists(synonym_file):
        try:
            df = pd.read_csv(synonym_file)
            return df
        except:
            return pd.DataFrame(columns=["word", "replacement"])
    else:
        return pd.DataFrame(columns=["word", "replacement"])

def save_synonyms(synonyms_df):
    synonym_file = "synonyms.csv"
    synonyms_df.to_csv(synonym_file, index=False)

def add_synonym(word, replacement, synonyms_df):
    new_row = pd.DataFrame({"word": [word], "replacement": [replacement]})
    synonyms_df = pd.concat([synonyms_df, new_row], ignore_index=True)
    return synonyms_df

def delete_synonym(index, synonyms_df):
    synonyms_df = synonyms_df.drop(index).reset_index(drop=True)
    return synonyms_df

# --------------------------------------------------
# VISUALIZATION FUNCTIONS
# --------------------------------------------------

def create_confidence_distribution(match_details):
    confidences = [d['confidence'] for d in match_details if d['confidence'] > 0]
    if confidences:
        fig = px.histogram(confidences, nbins=20, title="Confidence Score Distribution",
                          labels={'value': 'Confidence Score', 'count': 'Number of Matches'})
        fig.update_layout(showlegend=False)
        return fig
    return None

def create_match_quality_chart(match_details):
    qualities = [d['predicted_quality'] for d in match_details if d['predicted_quality'] > 0]
    if qualities:
        fig = px.box(qualities, title="Match Quality Distribution",
                    labels={'value': 'Predicted Quality'})
        return fig
    return None

def create_feature_importance_chart(rl_model):
    weights = rl_model.feature_weights
    fig = px.bar(x=list(weights.keys()), y=list(weights.values()),
                 title="Feature Importance (Learned Weights)",
                 labels={'x': 'Features', 'y': 'Weight'})
    fig.update_layout(showlegend=False)
    return fig

# --------------------------------------------------
# DOWNLOAD FUNCTIONS
# --------------------------------------------------

def create_detailed_download(input_df, match_details):
    """Create detailed download file with master address information"""
    download_df = input_df.copy()
    
    # Add master information columns
    download_df['Master ID'] = [d['master_id'] for d in match_details]
    download_df['Matched Center'] = [d.get('master_name', 'No Match') if d.get('master_name') else '⚡ No Match' for d in match_details]
    download_df['Master Address'] = [d.get('master_address', 'No Match') for d in match_details]
    download_df['Master District'] = [d.get('master_district', 'No Match') for d in match_details]
    download_df['Master State'] = [d.get('master_state', 'No Match') for d in match_details]
    download_df['Confidence Score'] = [d['confidence'] for d in match_details]
    download_df['Predicted Quality'] = [d['predicted_quality'] for d in match_details]
    
    # Add similarity scores breakdown
    download_df['Name Similarity'] = [d.get('similarity_scores', {}).get('name', 0) for d in match_details]
    download_df['Address Similarity'] = [d.get('similarity_scores', {}).get('address', 0) for d in match_details]
    download_df['District Similarity'] = [d.get('similarity_scores', {}).get('district', 0) for d in match_details]
    download_df['State Similarity'] = [d.get('similarity_scores', {}).get('state', 0) for d in match_details]
    download_df['Vector Similarity'] = [d.get('similarity_scores', {}).get('vector', 0) for d in match_details]
    
    return download_df

def get_table_download_link(df, filename):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

# --------------------------------------------------
# USER FEEDBACK COMPONENT
# --------------------------------------------------

def feedback_component(row_idx, match_info):
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        if st.button("👍", key=f"thumb_up_{row_idx}"):
            return "thumbs_up"
    with col2:
        if st.button("👎", key=f"thumb_down_{row_idx}"):
            return "thumbs_down"
    with col3:
        if st.button("✓ Correct", key=f"correct_{row_idx}"):
            return "correct_match"
    with col4:
        if st.button("✗ Wrong", key=f"wrong_{row_idx}"):
            return "wrong_match"
    return None

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

# Initialize RL model
@st.cache_resource
def init_rl_model():
    return ReinforcementLearningMatcher(learning_rate=0.1, exploration_rate=0.1)

rl_model = init_rl_model()

# Load synonyms
synonyms_df = load_synonyms()

# Main navigation
selected = option_menu(
    menu_title=None,
    options=["Matching", "Synonym Management", "Analytics", "Settings"],
    icons=["search", "book", "graph-up", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# Matching Tab
if selected == "Matching":
    st.markdown("<h1 class='main-header'>🎯 AI Powered Center Matching System</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        master_file = st.file_uploader("📁 Upload Master Database", type=["xlsx", "csv"], key="master")
    
    with col2:
        input_file = st.file_uploader("📁 Upload Input Stream", type=["xlsx", "csv"], key="input")
    
    # Matching strategies in sidebar
    with st.sidebar:
        st.markdown("<div class='section-header'>🎯 Matching Strategies</div>", unsafe_allow_html=True)
        
        strategies = {
            'boost_name': st.checkbox("Boost Name Matching", value=False, help="Give more weight to name similarity"),
            'boost_address': st.checkbox("Boost Address Matching", value=False, help="Give more weight to address similarity"),
            'strict_matching': st.checkbox("Strict Matching Mode", value=False, help="Require higher confidence scores"),
            'fuzzy_enhanced': st.checkbox("Enhanced Fuzzy Matching", value=True, help="Use additional fuzzy matching algorithms"),
            'top_k': st.slider("Number of Candidates to Consider", min_value=5, max_value=30, value=15, help="How many potential matches to evaluate")
        }
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>⚙️ Model Settings</div>", unsafe_allow_html=True)
        
        model_option = st.selectbox(
            "Choose Embedding Model",
            ["multi-qa-mpnet-base-dot-v1", "all-mpnet-base-v2", "BAAI/bge-large-en-v1.5", "all-MiniLM-L6-v2"],
            index=0
        )
        
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.50, max_value=0.95, value=0.70, step=0.05)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📊 Learning Stats</div>", unsafe_allow_html=True)
        
        learning_stats = rl_model.get_learning_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Matches Learned", learning_stats['total_learned_matches'])
            st.metric("Success Rate", f"{learning_stats['success_rate']*100:.1f}%")
        with col2:
            st.metric("Unique Patterns", learning_stats['unique_patterns'])
            st.metric("Exploration Rate", f"{learning_stats['exploration_rate']*100:.1f}%")
        
        if st.button("🔄 Reset Model", use_container_width=True):
            rl_model = ReinforcementLearningMatcher()
            st.success("Model reset!")
            st.rerun()
    
    if master_file and input_file:
        try:
            # Load files
            with st.spinner("Loading files..."):
                if master_file.name.endswith(".csv"):
                    master_df = load_csv_with_encoding(master_file)
                else:
                    master_df = load_excel_with_fallback(master_file)
                
                if input_file.name.endswith(".csv"):
                    input_df = load_csv_with_encoding(input_file)
                else:
                    input_df = load_excel_with_fallback(input_file)
            
            st.success(f"✅ Loaded {len(master_df)} master records and {len(input_df)} input records")
            
            # Validate required columns
            required_cols = ['center_name', 'district', 'state', 'address']
            for col in required_cols:
                if col not in master_df.columns:
                    st.error(f"Master file missing required column: {col}")
                    st.stop()
                if col not in input_df.columns:
                    st.error(f"Input file missing required column: {col}")
                    st.stop()
            
            # Load model
            @st.cache_resource
            def load_embedding_model(model_name):
                return SentenceTransformer(model_name)
            
            with st.spinner(f"Loading {model_option}..."):
                model = load_embedding_model(model_option)
            
            # Process master data
            with st.spinner("Generating embeddings for master data..."):
                master_df['clean_text'] = master_df.apply(
                    lambda x: enhanced_clean_text(f"{x['center_name']} {x['district']} {x['state']} {x['address']}", synonyms_df), 
                    axis=1
                )
                embeddings = model.encode(master_df['clean_text'].tolist(), show_progress_bar=True)
                embeddings = normalize(embeddings.astype(np.float32))
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings)
            
            # Perform matching
            with st.spinner("Matching with AI..."):
                results, match_details, learning_opportunities = match_with_rl(
                    input_df, master_df, model, index, synonyms_df, rl_model, confidence_threshold, strategies
                )
            
            # Create detailed dataframe
            detailed_df = create_detailed_download(input_df, match_details)
            
            # Display metrics
            st.markdown("<div class='section-header'>📊 Matching Results Summary</div>", unsafe_allow_html=True)
            
            matches_found = detailed_df[detailed_df['Matched Center'] != "⚡ No Match"]
            match_rate = (len(matches_found) / len(detailed_df)) * 100
            avg_confidence = matches_found['Confidence Score'].mean() * 100 if len(matches_found) > 0 else 0
            high_quality_matches = len(matches_found[matches_found['Predicted Quality'] > 0.8])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Match Rate", f"{match_rate:.1f}%", delta=f"{len(matches_found)}/{len(detailed_df)}")
            with col2:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            with col3:
                st.metric("High Quality Matches", high_quality_matches)
            with col4:
                st.metric("Total Processed", len(detailed_df))
            
            # Tabular display
            st.markdown("<div class='section-header'>📋 Matching Results Table</div>", unsafe_allow_html=True)
            
            # Display dataframe with custom formatting
            display_df = detailed_df[['center_name', 'district', 'state', 'Matched Center', 'Master Address', 
                                     'Confidence Score', 'Predicted Quality', 'Name Similarity', 'Address Similarity']].copy()
            display_df['Confidence Score'] = display_df['Confidence Score'].apply(lambda x: f"{x*100:.1f}%")
            display_df['Predicted Quality'] = display_df['Predicted Quality'].apply(lambda x: f"{x*100:.1f}%")
            display_df['Name Similarity'] = display_df['Name Similarity'].apply(lambda x: f"{x*100:.1f}%")
            display_df['Address Similarity'] = display_df['Address Similarity'].apply(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Detailed results with feedback
            st.markdown("<div class='section-header'>🔍 Detailed Results with Feedback</div>", unsafe_allow_html=True)
            
            for idx, row in detailed_df.iterrows():
                with st.expander(f"Record {idx+1}: {row['center_name']} → {row['Matched Center']} (Confidence: {row['Confidence Score']*100:.1f}%)"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Input Data:**")
                        st.write(f"• Name: {row['center_name']}")
                        st.write(f"• District: {row['district']}")
                        st.write(f"• State: {row['state']}")
                        st.write(f"• Address: {row['address']}")
                        
                        st.write("**Matched Master Data:**")
                        st.write(f"• Name: {row['Matched Center']}")
                        st.write(f"• Address: {row['Master Address']}")
                        st.write(f"• District: {row['Master District']}")
                        st.write(f"• State: {row['Master State']}")
                        
                        st.write("**Similarity Breakdown:**")
                        st.write(f"• Name: {row['Name Similarity']*100:.1f}%")
                        st.write(f"• Address: {row['Address Similarity']*100:.1f}%")
                        st.write(f"• District: {row['District Similarity']*100:.1f}%")
                        st.write(f"• State: {row['State Similarity']*100:.1f}%")
                    
                    with col2:
                        if row['Matched Center'] != "⚡ No Match":
                            feedback = feedback_component(idx, row['Matched Center'])
                            if feedback:
                                match_detail = match_details[idx]
                                rl_model.learn_from_match(
                                    input_text=f"{row['center_name']} {row['district']} {row['state']}",
                                    matched_text=row['Matched Center'],
                                    was_correct=True,
                                    confidence=row['Confidence Score'],
                                    similarity_scores=match_detail.get('similarity_scores', {}),
                                    user_feedback=feedback
                                )
                                st.success("✅ Thank you! AI learned from your feedback")
                                st.rerun()
            
            # Download options
            st.markdown("<div class='section-header'>💾 Download Results</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                # Full detailed download
                csv_full = detailed_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (with Master Address)",
                    data=csv_full,
                    file_name=f"matching_results_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Summary download
                summary_df = detailed_df[['center_name', 'district', 'state', 'Matched Center', 'Confidence Score']]
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary Only",
                    data=csv_summary,
                    file_name=f"matching_results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

# Synonym Management Tab
elif selected == "Synonym Management":
    st.markdown("<h1 class='main-header'>📚 Synonym Management</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='section-header'>Current Synonyms</div>", unsafe_allow_html=True)
        
        if len(synonyms_df) > 0:
            edited_df = st.data_editor(synonyms_df, use_container_width=True, num_rows="dynamic")
            if st.button("💾 Save Changes", use_container_width=True):
                save_synonyms(edited_df)
                synonyms_df = edited_df
                st.success("Synonyms saved successfully!")
                st.rerun()
        else:
            st.info("No synonyms added yet. Add some below.")
    
    with col2:
        st.markdown("<div class='section-header'>Add New Synonym</div>", unsafe_allow_html=True)
        
        with st.form("add_synonym_form"):
            word = st.text_input("Original Word/Abbreviation")
            replacement = st.text_input("Replacement Word")
            submitted = st.form_submit_button("➕ Add Synonym", use_container_width=True)
            
            if submitted and word and replacement:
                synonyms_df = add_synonym(word.lower(), replacement.lower(), synonyms_df)
                save_synonyms(synonyms_df)
                st.success(f"Added synonym: {word} → {replacement}")
                st.rerun()
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Import/Export</div>", unsafe_allow_html=True)
        
        # Export synonyms
        if len(synonyms_df) > 0:
            csv = synonyms_df.to_csv(index=False)
            st.download_button("📤 Export Synonyms", csv, "synonyms.csv", "text/csv", use_container_width=True)
        
        # Import synonyms
        uploaded_file = st.file_uploader("📥 Import Synonyms CSV", type=["csv"], key="synonym_import")
        if uploaded_file:
            try:
                imported_df = pd.read_csv(uploaded_file)
                synonyms_df = imported_df
                save_synonyms(synonyms_df)
                st.success("Synonyms imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error importing synonyms: {str(e)}")

# Analytics Tab
elif selected == "Analytics":
    st.markdown("<h1 class='main-header'>📈 Learning Analytics</h1>", unsafe_allow_html=True)
    
    learning_stats = rl_model.get_learning_stats()
    
    # Learning progress metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Learning Iterations", learning_stats['total_learned_matches'])
    with col2:
        st.metric("Overall Success Rate", f"{learning_stats['success_rate']*100:.1f}%")
    with col3:
        st.metric("Recent Success Rate", f"{learning_stats['recent_success_rate']*100:.1f}%")
    with col4:
        st.metric("Patterns Learned", learning_stats['unique_patterns'])
    
    # Feature importance visualization
    st.markdown("<div class='section-header'>Feature Importance (Learned Weights)</div>", unsafe_allow_html=True)
    fig_importance = create_feature_importance_chart(rl_model)
    if fig_importance:
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Learning history
    st.markdown("<div class='section-header'>Learning History</div>", unsafe_allow_html=True)
    
    if len(rl_model.match_history) > 0:
        history_df = pd.DataFrame(rl_model.match_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df['success'] = history_df['success'].map({True: 'Success', False: 'Failure'})
        
        # Success rate over time
        history_df['cumulative_success'] = history_df.groupby('success').cumcount() / (history_df.index + 1)
        
        fig = px.line(history_df, x='timestamp', y='confidence', color='success',
                     title='Match Confidence Over Time',
                     labels={'timestamp': 'Time', 'confidence': 'Confidence Score'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent matches table
        st.markdown("### Recent Matches")
        recent_df = history_df.tail(20)[['timestamp', 'input', 'matched', 'confidence', 'success']]
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No learning history yet. Start matching to see analytics!")

# Settings Tab
elif selected == "Settings":
    st.markdown("<h1 class='main-header'>⚙️ System Settings</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>Model Configuration</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                                       help="Controls how quickly the model learns from new data")
        discount_factor = st.number_input("Discount Factor", min_value=0.5, max_value=0.99, value=0.95, step=0.01,
                                         help="How much importance to give to future rewards")
    
    with col2:
        exploration_rate = st.number_input("Exploration Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                                          help="How often the model tries new strategies")
        save_frequency = st.number_input("Auto-save Frequency", min_value=5, max_value=100, value=10, step=5,
                                        help="Save model after every N learning iterations")
    
    if st.button("💾 Apply Settings", use_container_width=True):
        # Update RL model settings
        rl_model.learning_rate = learning_rate
        rl_model.discount_factor = discount_factor
        rl_model.exploration_rate = exploration_rate
        rl_model.save_model()
        st.success("Settings applied successfully!")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Data Management</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Learning History", use_container_width=True):
            rl_model.match_history = []
            rl_model.successful_patterns = defaultdict(int)
            rl_model.failure_patterns = defaultdict(int)
            rl_model.save_model()
            st.success("Learning history cleared!")
            st.rerun()
    
    with col2:
        if st.button("📤 Export Model", use_container_width=True):
            if os.path.exists("rl_model.pkl"):
                with open("rl_model.pkl", "rb") as f:
                    st.download_button("Download Model", f, "rl_model.pkl", "application/octet-stream")
            else:
                st.warning("No saved model found")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>About</div>", unsafe_allow_html=True)
    
    st.info("""
    **AI Powered Center Matching System with Self-Learning**
    
    This system uses advanced AI and reinforcement learning to match center records:
    
    - **Self-Learning**: Improves over time based on user feedback
    - **Multiple Strategies**: Combines vector search, fuzzy matching, and semantic understanding
    - **Dynamic Thresholds**: Automatically adjusts matching criteria based on success rate
    - **Feature Learning**: Continuously optimizes feature weights for better matches
    
    **How to use:**
    1. Upload Master Database and Input Stream files
    2. Configure matching strategies in sidebar
    3. Review results and provide feedback to help AI learn
    4. Download detailed results with master address information
    """)