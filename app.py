import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import os
from datetime import datetime
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Center Matching System", layout="wide")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-base-en")

model = load_model()

# --------------------------------------------------
# SYNONYMS
# --------------------------------------------------
synonym_file = "synonyms.csv"

if os.path.exists(synonym_file):
    synonyms = pd.read_csv(synonym_file)
else:
    synonyms = pd.DataFrame({
        "word": ["govt", "rajkiya", "mahila"],
        "replacement": ["government", "government", "girls"]
    })

# --------------------------------------------------
# CLEAN TEXT
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)

    for _, row in synonyms.iterrows():
        text = re.sub(rf"\b{row['word']}\b", row["replacement"], text)

    return re.sub(r"\s+", " ", text).strip()

# --------------------------------------------------
# MEMORY SYSTEM (FIXED)
# --------------------------------------------------
memory_file = "learning_memory.csv"

def load_memory():
    if os.path.exists(memory_file):
        df = pd.read_csv(memory_file)
        df.columns = df.columns.str.strip().str.lower()
    else:
        df = pd.DataFrame()

    required = ["input_text", "match_center", "master_id"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    return df[required]

# auto-create file (important for Streamlit Cloud)
if not os.path.exists(memory_file):
    pd.DataFrame(columns=["input_text","match_center","master_id"]).to_csv(memory_file, index=False)

memory = load_memory()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🎯 AI Center Matching System")

col1, col2 = st.columns(2)

with col1:
    master_file = st.file_uploader("Upload Master File", type=["csv","xlsx"])

with col2:
    input_file = st.file_uploader("Upload Input File", type=["csv","xlsx"])

# --------------------------------------------------
# PROCESS
# --------------------------------------------------
if master_file and input_file:

    master = pd.read_csv(master_file) if master_file.name.endswith(".csv") else pd.read_excel(master_file)
    input_data = pd.read_csv(input_file) if input_file.name.endswith(".csv") else pd.read_excel(input_file)

    st.success("Files loaded")

    # CLEAN MASTER
    master["clean_name"] = master["center_name"].apply(clean_text)
    master["clean_address"] = master["address"].apply(clean_text)

    master["combined"] = (
        master["center_name"].astype(str) + " " +
        master["district"].astype(str) + " " +
        master["state"].astype(str) + " " +
        master["address"].astype(str)
    ).apply(clean_text)

    # EMBEDDINGS
    embeddings = model.encode(master["combined"].tolist())
    embeddings = np.array(embeddings).astype("float32")

    # FAISS
    index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
    index.add(embeddings)

    results, ids, scores = [], [], []

    for _, row in input_data.iterrows():

        name = clean_text(row["center_name"])
        address = clean_text(row["address"])

        combined = clean_text(
            f"{row['center_name']} {row['district']} {row['state']} {row['address']}"
        )

        # MEMORY CHECK (FIXED)
        if not memory.empty and "input_text" in memory.columns:
            mem = memory[memory["input_text"] == combined]
        else:
            mem = pd.DataFrame()

        if not mem.empty:
            results.append(mem.iloc[0]["match_center"])
            ids.append(mem.iloc[0]["master_id"])
            scores.append(1.0)
            continue

        # VECTOR SEARCH
        emb = model.encode([combined]).astype("float32")
        D, I = index.search(emb, 5)
        candidates = master.iloc[I[0]]

        best_score = 0
        best = None

        for _, m in candidates.iterrows():
            name_score = fuzz.token_set_ratio(name, m["clean_name"]) / 100
            addr_score = fuzz.token_set_ratio(address, m["clean_address"]) / 100
            score = (0.6 * name_score) + (0.4 * addr_score)

            if score > best_score:
                best_score = score
                best = m

        if best_score >= 0.9:
            results.append(best["center_name"])
            ids.append(best["center_id"])
            scores.append(best_score)
        else:
            results.append("No Match")
            ids.append("NULL")
            scores.append(best_score)

    # OUTPUT
    input_data["Matched Center"] = results
    input_data["Master ID"] = ids
    input_data["Score"] = scores

    st.dataframe(input_data)

    # SAVE MEMORY
    if st.button("Save Memory"):
        new_rows = []

        for _, r in input_data.iterrows():
            if r["Matched Center"] != "No Match":
                txt = clean_text(
                    f"{r['center_name']} {r['district']} {r['state']} {r['address']}"
                )

                new_rows.append({
                    "input_text": txt,
                    "match_center": r["Matched Center"],
                    "master_id": r["Master ID"]
                })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            memory_updated = pd.concat([memory, new_df])
            memory_updated.drop_duplicates(subset=["input_text"], inplace=True)
            memory_updated.to_csv(memory_file, index=False)

            st.success("Memory saved ✅")

    st.download_button(
        "Download Results",
        input_data.to_csv(index=False),
        f"results_{datetime.now().strftime('%H%M%S')}.csv"
    )