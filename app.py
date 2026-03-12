import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import os
import time
from datetime import datetime
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Center Matching System",
    page_icon="🎯",
    layout="wide"
)

# --------------------------------------------------
# FUTURISTIC CYBERPUNK UI STYLE
# --------------------------------------------------

st.markdown("""
<style>
    /* Import futuristic fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');
    
    /* Global styles */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0a0f1f 0%, #030514 100%);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Main title with neon effect */
    .neon-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #00ffff, #ff00ff, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        letter-spacing: 3px;
        margin: 20px 0 5px 0;
        animation: glowPulse 3s infinite;
    }
    
    @keyframes glowPulse {
        0%, 100% { filter: brightness(1); text-shadow: 0 0 30px rgba(0,255,255,0.5); }
        50% { filter: brightness(1.3); text-shadow: 0 0 50px rgba(255,0,255,0.8); }
    }
    
    /* Subtitle with cyber effect */
    .cyber-subtitle {
        font-family: 'Orbitron', sans-serif;
        font-size: 18px;
        text-align: center;
        color: #00ffff;
        text-shadow: 0 0 15px #00ffff;
        letter-spacing: 4px;
        margin-bottom: 30px;
        position: relative;
    }
    
    .cyber-subtitle::before,
    .cyber-subtitle::after {
        content: "⚡";
        color: #ff00ff;
        margin: 0 15px;
        text-shadow: 0 0 15px #ff00ff;
    }
    
    /* Company name */
    .company-name {
        font-family: 'Share Tech Mono', monospace;
        font-size: 16px;
        text-align: center;
        color: #ff00ff;
        text-shadow: 0 0 10px #ff00ff;
        margin-bottom: 30px;
        letter-spacing: 2px;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(0, 255, 255, 0.05);
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #ff00ff;
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.3);
    }
    
    .metric-label {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
        font-size: 14px;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 24px;
        font-weight: 700;
        color: #ff00ff;
        text-shadow: 0 0 15px #ff00ff;
    }
    
    /* Button styling */
    .stButton > button {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        color: #000;
        font-weight: 700;
        border: none;
        padding: 12px 30px;
        border-radius: 30px;
        letter-spacing: 2px;
        text-transform: uppercase;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.7);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
        background-size: 200% 100%;
        animation: gradientMove 2s linear infinite;
        height: 20px;
        border-radius: 10px;
    }
    
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    /* FUTURISTIC POPUP STYLES */
    .popup-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        z-index: 9998;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .holographic-popup {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 600px;
        background: rgba(10, 20, 40, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 2px solid;
        border-image: linear-gradient(45deg, #00ffff, #ff00ff, #00ffff) 1;
        border-radius: 30px;
        padding: 30px;
        z-index: 9999;
        box-shadow: 
            0 0 50px rgba(0, 255, 255, 0.5),
            0 0 100px rgba(255, 0, 255, 0.3),
            inset 0 0 50px rgba(0, 255, 255, 0.2);
        animation: popupPulse 2s infinite, slideIn 0.5s ease;
    }
    
    @keyframes popupPulse {
        0%, 100% { border-color: #00ffff; box-shadow: 0 0 50px rgba(0,255,255,0.5); }
        50% { border-color: #ff00ff; box-shadow: 0 0 80px rgba(255,0,255,0.7); }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translate(-50%, -40%);
        }
        to {
            opacity: 1;
            transform: translate(-50%, -50%);
        }
    }
    
    .popup-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 25px;
        border-bottom: 1px solid rgba(0, 255, 255, 0.3);
        padding-bottom: 15px;
    }
    
    .popup-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    .progress-container {
        margin: 30px 0;
    }
    
    .progress-label {
        font-family: 'Share Tech Mono', monospace;
        color: #00ffff;
        font-size: 16px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .progress-bar-container {
        width: 100%;
        height: 20px;
        background: rgba(0, 0, 0, 0.5);
        border: 1px solid #00ffff;
        border-radius: 10px;
        overflow: hidden;
        position: relative;
        margin: 10px 0;
    }
    
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        width: 0%;
        transition: width 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.3), 
            transparent
        );
        animation: scan 2s linear infinite;
    }
    
    @keyframes scan {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .progress-percentage {
        font-family: 'Orbitron', sans-serif;
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 20px 0;
        text-shadow: 0 0 30px rgba(255, 0, 255, 0.5);
    }
    
    .status-messages {
        background: rgba(0, 0, 0, 0.5);
        border: 1px solid #00ffff;
        border-radius: 15px;
        padding: 20px;
        height: 200px;
        overflow-y: auto;
        font-family: 'Share Tech Mono', monospace;
        margin: 20px 0;
    }
    
    .status-message {
        padding: 8px;
        margin: 5px 0;
        border-left: 3px solid #ff00ff;
        color: #00ffff;
        animation: messageSlide 0.3s ease;
    }
    
    @keyframes messageSlide {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .message-time {
        color: #ff00ff;
        font-size: 12px;
        margin-right: 10px;
    }
    
    .message-icon {
        display: inline-block;
        width: 20px;
        text-align: center;
        margin-right: 10px;
    }
    
    .glowing-text {
        animation: textGlow 2s infinite;
        font-family: 'Share Tech Mono', monospace;
        color: #00ffff;
        font-size: 16px;
    }
    
    @keyframes textGlow {
        0%, 100% { text-shadow: 0 0 10px #00ffff; }
        50% { text-shadow: 0 0 20px #ff00ff; }
    }
    
    /* Time display */
    .time-display {
        position: fixed;
        top: 10px;
        right: 20px;
        font-family: 'Share Tech Mono', monospace;
        color: #00ffff;
        background: rgba(0, 0, 0, 0.5);
        padding: 5px 15px;
        border-radius: 20px;
        border: 1px solid #00ffff;
        z-index: 1000;
        font-size: 12px;
        backdrop-filter: blur(5px);
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00ff00;
        border-radius: 50%;
        box-shadow: 0 0 10px #00ff00;
        margin-right: 5px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(0, 255, 255, 0.05);
        border: 2px dashed #00ffff;
        border-radius: 10px;
        padding: 20px;
    }
    
    .stFileUploader:hover {
        border-color: #ff00ff;
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.3);
    }
    
    /* Divider */
    .cyber-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ffff, #ff00ff, #00ffff, transparent);
        margin: 30px 0;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 24px;
        color: #00ffff;
        text-shadow: 0 0 10px #00ffff;
        margin: 20px 0;
        border-left: 4px solid #ff00ff;
        padding-left: 15px;
    }
    
    /* Download button */
    .stDownloadButton > button {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(45deg, #ff00ff, #00ffff);
        color: #000;
        font-weight: 700;
        border: none;
        padding: 12px 30px;
        border-radius: 30px;
        letter-spacing: 2px;
        text-transform: uppercase;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.7);
    }
</style>

<div class="time-display">
    <span class="status-indicator"></span>
    SYSTEM ONLINE // <span id="time"></span>
</div>

<script>
    function updateTime() {
        const now = new Date();
        const timeString = now.getUTCFullYear() + '-' + 
            String(now.getUTCMonth() + 1).padStart(2, '0') + '-' +
            String(now.getUTCDate()).padStart(2, '0') + ' ' +
            String(now.getUTCHours()).padStart(2, '0') + ':' +
            String(now.getUTCMinutes()).padStart(2, '0') + ':' +
            String(now.getUTCSeconds()).padStart(2, '0');
        document.getElementById('time').textContent = timeString;
    }
    setInterval(updateTime, 1000);
    updateTime();
</script>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER WITH CYBER STYLING
# --------------------------------------------------

st.markdown("<h1 class='neon-title'>AI POWERED CENTER MATCHING SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<div class='cyber-subtitle'>NEURAL MATCHING ENGINE</div>", unsafe_allow_html=True)
st.markdown("<div class='company-name'>INNOVATIVIEW INDIA LIMITED</div>", unsafe_allow_html=True)

# --------------------------------------------------
# REALTIME DASHBOARD WITH METRIC CARDS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>⚡ AI ENGINE</div>
        <div class='metric-value'>ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>🌀 EMBEDDING MODEL</div>
        <div class='metric-value'>BGE-BASE</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>🔍 VECTOR SEARCH</div>
        <div class='metric-value'>FAISS HNSW</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>📡 SYSTEM STATUS</div>
        <div class='metric-value'>ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='cyber-divider'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# DOWNLOAD TEMPLATE SECTION
# --------------------------------------------------

st.markdown("<h2 class='section-header'>📥 DOWNLOAD TEMPLATES</h2>", unsafe_allow_html=True)

colA, colB = st.columns(2)

master_template = pd.DataFrame({
    "center_id": ["1001"],
    "center_name": ["ABC Public School"],
    "district": ["Lucknow"],
    "state": ["Uttar Pradesh"],
    "address": ["Near City Mall"]
})

input_template = pd.DataFrame({
    "center_name": ["ABC Public School"],
    "district": ["Lucknow"],
    "state": ["Uttar Pradesh"],
    "address": ["Near City Mall"]
})

with colA:
    st.download_button(
        "⬇️ MASTER TEMPLATE",
        master_template.to_csv(index=False),
        "master_format.csv"
    )

with colB:
    st.download_button(
        "⬇️ INPUT TEMPLATE",
        input_template.to_csv(index=False),
        "input_format.csv"
    )

st.markdown("<div class='cyber-divider'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# FILE UPLOAD SECTION
# --------------------------------------------------

st.markdown("<h2 class='section-header'>📡 DATA UPLOAD INTERFACE</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

master_file = col1.file_uploader("MASTER DATABASE", type=["xlsx", "csv"], key="master")
input_file = col2.file_uploader("INPUT STREAM", type=["xlsx", "csv"], key="input")

# --------------------------------------------------
# LOAD MODEL (GPU SUPPORT)
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = SentenceTransformer("BAAI/bge-base-en")
    try:
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda")
    except:
        pass
    return model

model = load_model()

# --------------------------------------------------
# TEXT CLEAN FUNCTION
# --------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --------------------------------------------------
# MEMORY LEARNING SYSTEM
# --------------------------------------------------

memory_file = "learning_memory.csv"

if os.path.exists(memory_file):
    memory = pd.read_csv(memory_file)
else:
    memory = pd.DataFrame(columns=["input_text", "match_center", "master_id"])

# --------------------------------------------------
# PROCESSING SECTION WITH SIMPLE POPUP
# --------------------------------------------------

if master_file and input_file:

    # Load files
    if master_file.name.endswith(".csv"):
        master = pd.read_csv(master_file)
    else:
        master = pd.read_excel(master_file)

    if input_file.name.endswith(".csv"):
        input_data = pd.read_csv(input_file)
    else:
        input_data = pd.read_excel(input_file)

    st.success("✅ FILES LOADED SUCCESSFULLY")
    
    # Create containers for progress display
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # --------------------------------------------------
    # STEP 1: CLEAN MASTER DATA
    # --------------------------------------------------
    status_text.markdown("""
    <div style='text-align: center; font-family: Orbitron; color: #00ffff; font-size: 20px; margin: 20px;'>
        ⚡ CLEANING MASTER DATA...
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(10)
    time.sleep(0.5)
    
    master["clean_name"] = master["center_name"].apply(clean_text)
    master["clean_address"] = master["address"].apply(clean_text)
    
    master["combined"] = (
        master["center_name"].astype(str) + " " +
        master["district"].astype(str) + " " +
        master["state"].astype(str) + " " +
        master["address"].astype(str)
    ).apply(clean_text)
    
    progress_bar.progress(20)
    time.sleep(0.5)
    
    # --------------------------------------------------
    # STEP 2: GENERATE EMBEDDINGS
    # --------------------------------------------------
    status_text.markdown("""
    <div style='text-align: center; font-family: Orbitron; color: #ff00ff; font-size: 20px; margin: 20px;'>
        🌀 GENERATING QUANTUM EMBEDDINGS...
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(30)
    
    embeddings = model.encode(master["combined"].tolist())
    embeddings = np.array(embeddings).astype("float32")
    
    progress_bar.progress(45)
    time.sleep(0.5)
    
    # --------------------------------------------------
    # STEP 3: BUILD FAISS INDEX
    # --------------------------------------------------
    status_text.markdown("""
    <div style='text-align: center; font-family: Orbitron; color: #00ffff; font-size: 20px; margin: 20px;'>
        🔍 BUILDING FAISS INDEX...
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(50)
    
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    
    progress_bar.progress(60)
    time.sleep(0.5)
    
    # --------------------------------------------------
    # STEP 4: EXECUTE MATCHING
    # --------------------------------------------------
    status_text.markdown("""
    <div style='text-align: center; font-family: Orbitron; color: #ff00ff; font-size: 20px; margin: 20px;'>
        🎯 EXECUTING NEURAL MATCHING ALGORITHM...
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(65)
    
    results = []
    ids = []
    scores = []
    explanation = []
    
    total = len(input_data)
    
    # Create a container for record progress
    record_text = st.empty()
    
    for i, row in input_data.iterrows():
        # Update progress
        progress = 65 + int((i / total) * 30)
        progress_bar.progress(progress)
        
        # Show current record
        record_text.markdown(f"""
        <div style='text-align: center; font-family: Share Tech Mono; color: #00ffff; font-size: 16px; margin: 10px;'>
            PROCESSING RECORD {i+1}/{total} • {int((i+1)/total*100)}%
        </div>
        """, unsafe_allow_html=True)
        
        name = clean_text(row["center_name"])
        address = clean_text(row["address"])
        
        combined = f"{name} {row['district']} {row['state']} {address}"
        
        # Check memory first
        mem_check = memory[memory["input_text"] == combined]
        if not mem_check.empty:
            results.append(mem_check.iloc[0]["match_center"])
            ids.append(mem_check.iloc[0]["master_id"])
            scores.append(1.0)
            explanation.append("MEMORY RECALL")
            continue
        
        # Vector search
        emb = model.encode([combined])
        emb = np.array(emb).astype("float32")
        
        k = 5
        D, I = index.search(emb, k)
        candidates = master.iloc[I[0]]
        
        best_score = 0
        best = None
        best_reason = ""
        
        for _, m in candidates.iterrows():
            name_score = fuzz.token_set_ratio(name, m["clean_name"]) / 100
            addr_score = fuzz.token_set_ratio(address, m["clean_address"]) / 100
            score = (0.6 * name_score) + (0.4 * addr_score)
            
            if score > best_score:
                best_score = score
                best = m
                best_reason = f"N:{name_score:.2f}|A:{addr_score:.2f}"
        
        if best_score >= 0.90:
            results.append(best["center_name"])
            ids.append(best["center_id"])
            scores.append(best_score)
            explanation.append(best_reason)
        else:
            results.append("⚡ NO MATCH")
            ids.append("NULL")
            scores.append(best_score)
            explanation.append(f"LOW CONF:{best_score:.2f}")
    
    progress_bar.progress(95)
    record_text.empty()
    
    # Add results to dataframe
    input_data["Matched Center"] = results
    input_data["Master ID"] = ids
    input_data["Score"] = scores
    input_data["Explanation"] = explanation
    
    # --------------------------------------------------
    # STEP 5: FINALIZE
    # --------------------------------------------------
    status_text.markdown("""
    <div style='text-align: center; font-family: Orbitron; color: #00ff00; font-size: 20px; margin: 20px;'>
        ✅ PROCESSING COMPLETE!
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(100)
    time.sleep(1)
    
    # Clear status
    status_text.empty()
    
    # --------------------------------------------------
    # REAL TIME STATISTICS
    # --------------------------------------------------
    
    st.markdown("<h2 class='section-header'>📊 MATCHING STATISTICS</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    match_rate = len([s for s in scores if s > 0.9]) / len(scores) * 100
    avg_score = np.mean(scores) * 100
    high_conf = len([s for s in scores if s > 0.95])
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>🎯 MATCH RATE</div>
            <div class='metric-value'>{match_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>📈 AVG CONFIDENCE</div>
            <div class='metric-value'>{avg_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>📊 RECORDS</div>
            <div class='metric-value'>{len(scores)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>⭐ HIGH CONFIDENCE</div>
            <div class='metric-value'>{high_conf}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # --------------------------------------------------
    # HEATMAP STYLING
    # --------------------------------------------------
    
    def highlight_score(val):
        if val > 0.95:
            return 'background: rgba(0, 255, 0, 0.3); color: #00ff00; font-weight: bold'
        elif val > 0.90:
            return 'background: rgba(255, 255, 0, 0.3); color: #ffff00; font-weight: bold'
        else:
            return 'background: rgba(255, 0, 0, 0.3); color: #ff6666'
    
    styled_df = input_data.style.map(highlight_score, subset=["Score"])
    st.dataframe(styled_df, use_container_width=True)
    
    # --------------------------------------------------
    # SAVE CORRECTION MEMORY
    # --------------------------------------------------
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("💾 SAVE TO NEURAL MEMORY BANK", use_container_width=True):
            for _, r in input_data.iterrows():
                if r["Matched Center"] != "⚡ NO MATCH":
                    txt = clean_text(
                        f"{r['center_name']} {r['district']} {r['state']} {r['address']}"
                    )
                    memory.loc[len(memory)] = [
                        txt,
                        r["Matched Center"],
                        r["Master ID"]
                    ]
            
            memory.drop_duplicates(inplace=True)
            memory.to_csv(memory_file, index=False)
            st.success("✅ MEMORY BANK UPDATED SUCCESSFULLY")
            st.balloons()
    
    # --------------------------------------------------
    # DOWNLOAD RESULT
    # --------------------------------------------------
    
    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        st.download_button(
            "⬇️ DOWNLOAD MATCHING RESULTS",
            input_data.to_csv(index=False),
            f"matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            use_container_width=True
        )