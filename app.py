import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import os
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

st.title("🎯 AI Powered Center Matching System")

# --------------------------------------------------
# DOWNLOAD TEMPLATES
# --------------------------------------------------

master_template = pd.DataFrame({
    "center_id":["1001"],
    "center_name":["ABC Public School"],
    "district":["Lucknow"],
    "state":["Uttar Pradesh"],
    "address":["Near City Mall"]
})

input_template = pd.DataFrame({
    "center_name":["ABC Public School"],
    "district":["Lucknow"],
    "state":["Uttar Pradesh"],
    "address":["Near City Mall"]
})

c1,c2=st.columns(2)

with c1:
    st.download_button(
        "Download Master Template",
        master_template.to_csv(index=False),
        "master_template.csv"
    )

with c2:
    st.download_button(
        "Download Input Template",
        input_template.to_csv(index=False),
        "input_template.csv"
    )

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-base-en")

model=load_model()

# --------------------------------------------------
# SYNONYM SYSTEM
# --------------------------------------------------

synonym_file="synonyms.xlsx"

if os.path.exists(synonym_file):
    synonym_df=pd.read_excel(synonym_file)
else:
    synonym_df=pd.DataFrame(columns=["main_word","synonym"])

def build_synonym_dict(df):

    syn_dict={}

    for _,row in df.iterrows():

        main=row["main_word"].lower()
        syn=row["synonym"].lower()

        if main not in syn_dict:
            syn_dict[main]=[]

        syn_dict[main].append(syn)

    return syn_dict

synonyms=build_synonym_dict(synonym_df)

# --------------------------------------------------
# SYNONYM EDITOR
# --------------------------------------------------

st.sidebar.header("Synonym Manager")

edited=st.sidebar.data_editor(
    synonym_df,
    num_rows="dynamic"
)

if st.sidebar.button("Save Synonyms"):
    edited.to_excel(synonym_file,index=False)
    st.sidebar.success("Synonyms saved")
    synonyms=build_synonym_dict(edited)

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def normalize_synonyms(text):

    text=str(text).lower()

    for main_word, variations in synonyms.items():

        for v in variations:

            pattern=r"\b"+re.escape(v)+r"\b"

            text=re.sub(pattern,main_word,text)

    return text


def clean_text(text):

    text=str(text).lower()

    text=normalize_synonyms(text)

    text=re.sub(r"[^\w\s]","",text)

    text=re.sub(r"\s+"," ",text)

    return text.strip()

# --------------------------------------------------
# STATE STANDARDIZATION
# --------------------------------------------------

state_map={
"jammu & kashmir":"jammu and kashmir",
"jk":"jammu and kashmir"
}

def standardize_state(s):

    s=str(s).lower()

    return state_map.get(s,s)

# --------------------------------------------------
# MEMORY SYSTEM
# --------------------------------------------------

memory_file="learning_memory.csv"

if os.path.exists(memory_file):
    memory=pd.read_csv(memory_file)
else:
    memory=pd.DataFrame(columns=["input_text","match_center","master_id"])

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

col1,col2=st.columns(2)

with col1:
    master_file=st.file_uploader("Upload Master File",type=["csv","xlsx"])

with col2:
    input_file=st.file_uploader("Upload Input File",type=["csv","xlsx"])

# --------------------------------------------------
# PROCESS FILES
# --------------------------------------------------

if master_file and input_file:

    if master_file.name.endswith(".csv"):
        master=pd.read_csv(master_file)
    else:
        master=pd.read_excel(master_file)

    if input_file.name.endswith(".csv"):
        input_data=pd.read_csv(input_file)
    else:
        input_data=pd.read_excel(input_file)

    st.success("Files Loaded")

# --------------------------------------------------
# CLEAN MASTER
# --------------------------------------------------

    master["state_clean"]=master["state"].apply(standardize_state)
    master["district_clean"]=master["district"].apply(clean_text)
    master["clean_name"]=master["center_name"].apply(clean_text)
    master["clean_address"]=master["address"].apply(clean_text)

    master["combined"]=(
        master["center_name"].astype(str)+" "+
        master["district"].astype(str)+" "+
        master["state"].astype(str)+" "+
        master["address"].astype(str)
    ).apply(clean_text)

# --------------------------------------------------
# RESULTS STORAGE
# --------------------------------------------------

    results=[]
    ids=[]
    scores=[]
    explanation=[]

    matched_address=[]
    matched_district=[]
    matched_state=[]

# --------------------------------------------------
# MATCHING ENGINE
# --------------------------------------------------

    progress=st.progress(0)

    total=len(input_data)

    for i,row in input_data.iterrows():

        progress.progress((i+1)/total)

        name=clean_text(row["center_name"])
        address=clean_text(row["address"])
        district=clean_text(row["district"])
        state=standardize_state(row["state"])

        combined=f"{name} {district} {state} {address}"

# --------------------------------------------------
# MEMORY CHECK
# --------------------------------------------------

        mem=memory[memory["input_text"]==combined]

        if not mem.empty:

            m=master[master["center_id"]==mem.iloc[0]["master_id"]].iloc[0]

            results.append(mem.iloc[0]["match_center"])
            ids.append(mem.iloc[0]["master_id"])
            scores.append(1.0)
            explanation.append("Memory Recall")

            matched_address.append(m["address"])
            matched_district.append(m["district"])
            matched_state.append(m["state"])

            continue

# --------------------------------------------------
# FILTER MASTER
# --------------------------------------------------

        filtered_master=master[
            (master["state_clean"]==state)&
            (master["district_clean"]==district)
        ]

        if filtered_master.empty:

            filtered_master=master[
                master["state_clean"]==state
            ]

# --------------------------------------------------
# FAISS SEARCH
# --------------------------------------------------

        texts=filtered_master["combined"].tolist()

        emb=model.encode(texts)

        emb=np.array(emb).astype("float32")

        dim=emb.shape[1]

        index=faiss.IndexFlatL2(dim)

        index.add(emb)

        query_emb=model.encode([combined])
        query_emb=np.array(query_emb).astype("float32")

        k=5

        D,I=index.search(query_emb,k)

        candidates=filtered_master.iloc[I[0]]

# --------------------------------------------------
# FUZZY RANKING
# --------------------------------------------------

        best_score=0
        best=None
        reason=""

        for _,m in candidates.iterrows():

            name_score=fuzz.token_set_ratio(name,m["clean_name"])/100
            addr_score=fuzz.token_set_ratio(address,m["clean_address"])/100

            score=(0.6*name_score)+(0.4*addr_score)

            if score>best_score:

                best_score=score
                best=m
                reason=f"N:{name_score:.2f}|A:{addr_score:.2f}"

# --------------------------------------------------
# FINAL DECISION
# --------------------------------------------------

        if best_score>=0.90:

            results.append(best["center_name"])
            ids.append(best["center_id"])
            scores.append(best_score)
            explanation.append(reason)

            matched_address.append(best["address"])
            matched_district.append(best["district"])
            matched_state.append(best["state"])

# save learning

            memory.loc[len(memory)]=[
                combined,
                best["center_name"],
                best["center_id"]
            ]

        else:

            results.append("No Match")
            ids.append("NULL")
            scores.append(best_score)
            explanation.append(f"Low Confidence {best_score:.2f}")

            matched_address.append("")
            matched_district.append("")
            matched_state.append("")

# --------------------------------------------------
# SAVE MEMORY
# --------------------------------------------------

    memory.drop_duplicates(inplace=True)

    memory.to_csv(memory_file,index=False)

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

    input_data["Matched Center"]=results
    input_data["Master ID"]=ids
    input_data["Matched Address"]=matched_address
    input_data["Matched District"]=matched_district
    input_data["Matched State"]=matched_state
    input_data["Score"]=scores
    input_data["Explanation"]=explanation

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

    st.subheader("Results")

    st.dataframe(input_data,use_container_width=True)

# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

    st.download_button(
        "Download Result CSV",
        input_data.to_csv(index=False),
        f"matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )