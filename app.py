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
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROFESSIONAL CYBERPUNK UI STYLE
# --------------------------------------------------

st.markdown("""
<style>
    /* Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0a0f1f 0%, #1a1f2f 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main title */
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 32px;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 10px 0 5px 0;
        letter-spacing: -0.5px;
    }
    
    /* Subtitle */
    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        text-align: center;
        color: #a0a0a0;
        margin-bottom: 15px;
        font-weight: 400;
    }
    
    /* Company name */
    .company-name {
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        text-align: center;
        color: #667eea;
        margin-bottom: 20px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Metric cards - Compact */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        color: #a0a0a0;
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #667eea;
        margin-top: 4px;
    }
    
    /* Button styling */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 13px;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader - Compact */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed #667eea;
        border-radius: 6px;
        padding: 10px;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Upload labels */
    .upload-label {
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        font-weight: 500;
        color: #667eea;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* File info */
    .file-info {
        font-family: 'Roboto Mono', monospace;
        font-size: 11px;
        color: #a0a0a0;
        background: rgba(0, 0, 0, 0.2);
        padding: 4px 8px;
        border-radius: 4px;
        margin-top: 5px;
    }
    
    /* Section headers - Minimal */
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
        font-weight: 500;
        color: white;
        margin: 15px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 15px 0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 6px;
        border-radius: 3px;
    }
    
    /* POPUP STYLES */
    .popup-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(5px);
        z-index: 9998;
    }
    
    .popup-container {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 450px;
        background: #1a1f2f;
        border: 1px solid #667eea;
        border-radius: 12px;
        padding: 25px;
        z-index: 9999;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translate(-50%, -45%);
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
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .popup-title {
        font-family: 'Poppins', sans-serif;
        font-size: 20px;
        font-weight: 600;
        color: #667eea;
    }
    
    .popup-close {
        font-family: 'Inter', sans-serif;
        color: #a0a0a0;
        font-size: 12px;
        cursor: pointer;
        padding: 4px 12px;
        border: 1px solid #a0a0a0;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .popup-close:hover {
        border-color: #667eea;
        color: #667eea;
    }
    
    .popup-progress-container {
        margin: 20px 0;
    }
    
    .popup-progress-label {
        font-family: 'Inter', sans-serif;
        color: #a0a0a0;
        font-size: 12px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .popup-progress-bar {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .popup-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.3s ease;
    }
    
    .popup-percentage {
        font-family: 'Poppins', sans-serif;
        font-size: 32px;
        font-weight: 600;
        color: #667eea;
        text-align: center;
        margin: 15px 0;
    }
    
    .popup-status {
        font-family: 'Inter', sans-serif;
        color: white;
        font-size: 14px;
        margin: 15px 0;
        padding: 10px;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 6px;
        text-align: center;
    }
    
    .popup-messages {
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 6px;
        padding: 12px;
        height: 150px;
        overflow-y: auto;
        font-family: 'Roboto Mono', monospace;
        font-size: 11px;
        margin-top: 15px;
    }
    
    .popup-message {
        padding: 4px 0;
        color: #a0a0a0;
        border-left: 2px solid #667eea;
        padding-left: 8px;
        margin: 5px 0;
    }
    
    .popup-message-time {
        color: #667eea;
        margin-right: 8px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #0a0f1f;
    }
    
    .sidebar-header {
        font-family: 'Poppins', sans-serif;
        font-size: 16px;
        font-weight: 600;
        color: white;
        margin: 20px 0 15px 0;
    }
    
    .template-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 6px;
        padding: 12px;
        margin: 10px 0;
    }
    
    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER - COMPACT
# --------------------------------------------------

st.markdown("<h1 class='main-title'>AI Center Matching System</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Neural Matching Engine • Innovatiview India Limited</div>", unsafe_allow_html=True)

# --------------------------------------------------
# COMPACT METRICS ROW
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>AI Engine</div>
        <div class='metric-value'>ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Model</div>
        <div class='metric-value'>BGE-BASE</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Vector Search</div>
        <div class='metric-value'>FAISS</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Status</div>
        <div class='metric-value'><span class='status-dot'></span>ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# COMPACT FILE UPLOAD SECTION
# --------------------------------------------------

st.markdown("<h2 class='section-header'>Data Upload</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='upload-label'>Master Database</div>", unsafe_allow_html=True)
    master_file = st.file_uploader("", type=["xlsx", "csv"], key="master", label_visibility="collapsed")
    if master_file:
        file_size = len(master_file.getvalue()) / 1024
        st.markdown(f"<div class='file-info'>📁 {master_file.name} • {file_size:.1f} KB</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='upload-label'>Input Stream</div>", unsafe_allow_html=True)
    input_file = st.file_uploader("", type=["xlsx", "csv"], key="input", label_visibility="collapsed")
    if input_file:
        file_size = len(input_file.getvalue()) / 1024
        st.markdown(f"<div class='file-info'>📁 {input_file.name} • {file_size:.1f} KB</div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR WITH TEMPLATES
# --------------------------------------------------

with st.sidebar:
    st.markdown("<div class='sidebar-header'>📥 Templates</div>", unsafe_allow_html=True)
    
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
    
    st.markdown("<div class='template-card'>", unsafe_allow_html=True)
    st.download_button(
        "📋 Master Template",
        master_template.to_csv(index=False),
        "master_format.csv",
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='template-card'>", unsafe_allow_html=True)
    st.download_button(
        "📋 Input Template",
        input_template.to_csv(index=False),
        "input_format.csv",
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-header'>⚡ System Info</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: Inter; font-size: 12px; color: #a0a0a0;'>
        • GPU: Available<br>
        • Memory: 16GB<br>
        • Version: 2.0.0
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
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
# POPUP FUNCTION
# --------------------------------------------------

def show_progress_popup(progress, message, status, step_name, record_info=""):
    """Display a professional popup with progress"""
    
    popup_html = f"""
    <div class='popup-overlay'></div>
    <div class='popup-container'>
        <div class='popup-header'>
            <div class='popup-title'>⚡ Neural Processing</div>
            <div class='popup-close' onclick="this.closest('.popup-container').style.display='none'; document.querySelector('.popup-overlay').style.display='none';">✕</div>
        </div>
        
        <div class='popup-progress-container'>
            <div class='popup-progress-label'>Current Operation</div>
            <div style='font-family: Poppins; color: white; font-size: 16px; margin-bottom: 15px;'>{step_name}</div>
            
            <div class='popup-progress-label'>Progress</div>
            <div class='popup-progress-bar'>
                <div class='popup-progress-fill' style='width: {progress}%;'></div>
            </div>
            <div class='popup-percentage'>{progress}%</div>
            
            <div class='popup-status'>
                {message}
            </div>
            
            {f"<div style='font-family: Roboto Mono; color: #667eea; text-align: center; margin: 10px 0; font-size: 12px;'>{record_info}</div>" if record_info else ""}
            
            <div class='popup-progress-label'>System Log</div>
            <div class='popup-messages' id='popup-messages'>
                {''.join([f"<div class='popup-message'><span class='popup-message-time'>[{datetime.now().strftime('%H:%M:%S')}]</span> {msg}</div>" for msg in status[-8:]])}
            </div>
        </div>
    </div>
    """
    
    return popup_html

# --------------------------------------------------
# PROCESSING SECTION WITH POPUP
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

    st.success("✅ Files loaded successfully")
    
    # Create placeholder for popup
    popup_placeholder = st.empty()
    
    # Initialize logs
    logs = []
    
    # --------------------------------------------------
    # STEP 1: CLEAN MASTER DATA
    # --------------------------------------------------
    logs.append("Initializing data cleaning...")
    popup_placeholder.markdown(show_progress_popup(
        10, "Cleaning master data", logs, "Data Preprocessing"
    ), unsafe_allow_html=True)
    time.sleep(0.5)
    
    master["clean_name"] = master["center_name"].apply(clean_text)
    master["clean_address"] = master["address"].apply(clean_text)
    
    master["combined"] = (
        master["center_name"].astype(str) + " " +
        master["district"].astype(str) + " " +
        master["state"].astype(str) + " " +
        master["address"].astype(str)
    ).apply(clean_text)
    
    logs.append(f"✓ Cleaned {len(master)} master records")
    popup_placeholder.markdown(show_progress_popup(
        20, "Data cleaning complete", logs, "Data Preprocessing"
    ), unsafe_allow_html=True)
    time.sleep(0.5)
    
    # --------------------------------------------------
    # STEP 2: GENERATE EMBEDDINGS
    # --------------------------------------------------
    logs.append("Generating embeddings...")
    popup_placeholder.markdown(show_progress_popup(
        30, "Creating vector representations", logs, "Embedding Generation"
    ), unsafe_allow_html=True)
    
    embeddings = model.encode(master["combined"].tolist())
    embeddings = np.array(embeddings).astype("float32")
    
    logs.append(f"✓ Generated {len(embeddings)} embeddings")
    logs.append(f"✓ Dimension: {embeddings.shape[1]}")
    popup_placeholder.markdown(show_progress_popup(
        45, "Embeddings ready", logs, "Embedding Generation"
    ), unsafe_allow_html=True)
    time.sleep(0.5)
    
    # --------------------------------------------------
    # STEP 3: BUILD FAISS INDEX
    # --------------------------------------------------
    logs.append("Building FAISS index...")
    popup_placeholder.markdown(show_progress_popup(
        50, "Creating search index", logs, "Index Construction"
    ), unsafe_allow_html=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    
    logs.append("✓ FAISS index built successfully")
    popup_placeholder.markdown(show_progress_popup(
        60, "Index ready", logs, "Index Construction"
    ), unsafe_allow_html=True)
    time.sleep(0.5)
    
    # --------------------------------------------------
    # STEP 4: EXECUTE MATCHING
    # --------------------------------------------------
    logs.append("Starting neural matching...")
    popup_placeholder.markdown(show_progress_popup(
        65, "Matching in progress", logs, "Pattern Recognition"
    ), unsafe_allow_html=True)
    
    results = []
    ids = []
    scores = []
    explanation = []
    
    total = len(input_data)
    
    for i, row in input_data.iterrows():
        # Update progress (65% to 95%)
        progress = 65 + int((i / total) * 30)
        
        if i % max(1, total // 10) == 0:
            logs.append(f"⚡ Processing batch {i+1}-{min(i+10, total)}/{total}")
            popup_placeholder.markdown(show_progress_popup(
                progress, f"Matching records...", logs, "Pattern Recognition",
                f"Record {i+1} of {total} • {int((i+1)/total*100)}%"
            ), unsafe_allow_html=True)
        
        name = clean_text(row["center_name"])
        address = clean_text(row["address"])
        
        combined = f"{name} {row['district']} {row['state']} {address}"
        
        # Check memory
        mem_check = memory[memory["input_text"] == combined]
        if not mem_check.empty:
            results.append(mem_check.iloc[0]["match_center"])
            ids.append(mem_check.iloc[0]["master_id"])
            scores.append(1.0)
            explanation.append("Memory recall")
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
            results.append("⚡ No Match")
            ids.append("NULL")
            scores.append(best_score)
            explanation.append(f"Low conf:{best_score:.2f}")
    
    # Add results to dataframe
    input_data["Matched Center"] = results
    input_data["Master ID"] = ids
    input_data["Score"] = scores
    input_data["Explanation"] = explanation
    
    # --------------------------------------------------
    # STEP 5: FINALIZE
    # --------------------------------------------------
    match_rate = len([s for s in scores if s > 0.9]) / len(scores) * 100
    logs.append(f"✓ Match rate: {match_rate:.1f}%")
    logs.append("✓ Processing complete")
    
    popup_placeholder.markdown(show_progress_popup(
        100, "Processing complete!", logs, "Finalizing",
        f"Processed {total} records • {match_rate:.1f}% match rate"
    ), unsafe_allow_html=True)
    
    time.sleep(2)
    popup_placeholder.empty()
    
    # --------------------------------------------------
    # RESULTS SECTION
    # --------------------------------------------------
    
    st.markdown("<h2 class='section-header'>Results</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Match Rate</div>
            <div class='metric-value'>{match_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_score = np.mean(scores) * 100
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Avg Confidence</div>
            <div class='metric-value'>{avg_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Records</div>
            <div class='metric-value'>{len(scores)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_conf = len([s for s in scores if s > 0.95])
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>High Confidence</div>
            <div class='metric-value'>{high_conf}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataframe with styling
    def highlight_score(val):
        if val > 0.95:
            return 'background: rgba(16, 185, 129, 0.2); color: #10b981;'
        elif val > 0.90:
            return 'background: rgba(245, 158, 11, 0.2); color: #f59e0b;'
        else:
            return 'background: rgba(239, 68, 68, 0.2); color: #ef4444;'
    
    styled_df = input_data.style.map(highlight_score, subset=["Score"])
    st.dataframe(styled_df, use_container_width=True, height=300)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save to Memory", use_container_width=True):
            for _, r in input_data.iterrows():
                if r["Matched Center"] != "⚡ No Match":
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
            st.success("✅ Memory updated")
    
    with col2:
        st.download_button(
            "📥 Download CSV",
            input_data.to_csv(index=False),
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            use_container_width=True
        )
    
    with col3:
        if st.button("🔄 New Match", use_container_width=True):
            st.rerun()