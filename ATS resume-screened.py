# ==============================
# ATS Resume Analyzer
# Designed & Developed by Kajal Dadas
# Email: kajaldadas149@gmail.com
# Phone: 7972244559
# ==============================

# ------------------------------
# CONFIG
# ------------------------------

import streamlit as st
import pdfplumber
import docx
import re
import os
import shutil
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ------------------------------
# CONFIG
# ------------------------------

st.set_page_config(page_title="ATS Resume Analyzer", layout="wide")
st.title("ATS Resume Analyzer")
st.markdown("<p style='text-align:center; font-size:14px;'>Designed & Developed by Kajal Dadas</p>", unsafe_allow_html=True)

# ------------------------------
# UTILITY FUNCTIONS
# ------------------------------

def extract_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def extract_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def extract_keywords(text, top_n=30):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    vectorizer.fit([text])
    return vectorizer.get_feature_names_out()

def keyword_score(resume, jd_keywords):
    matched, missing = [], []
    for word in jd_keywords:
        if word in resume:
            matched.append(word)
        else:
            missing.append(word)
    score = (len(matched)/len(jd_keywords))*100 if len(jd_keywords)>0 else 0
    return score, matched, missing

def semantic_score(resume, jd):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode([resume, jd])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]*100

def calculate_final_score(keyword_score_val, semantic_score_val):
    return 0.4*keyword_score_val + 0.6*semantic_score_val

def move_to_screened(resume_path):
    target_folder = "screened_resumes"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    shutil.move(resume_path, os.path.join(target_folder, os.path.basename(resume_path)))

# ------------------------------
# RESUME SELECTION
# ------------------------------

resume_folder = "resumes"
resume_files = [f for f in os.listdir(resume_folder) if f.endswith((".pdf", ".docx"))]

selected_resume = st.selectbox("Select Resume", ["-- Select --"] + resume_files)
jd_text = st.text_area("Paste Job Description Here")

if st.button("Analyze"):

    if selected_resume == "-- Select --" or not jd_text.strip():
        st.warning("Please select a resume and paste job description")
    else:
        resume_path = os.path.join(resume_folder, selected_resume)
        if resume_path.endswith(".pdf"):
            resume_text = extract_pdf(resume_path)
        else:
            resume_text = extract_docx(resume_path)

        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(jd_text)

        jd_keywords = extract_keywords(jd_clean)
        kw_score, matched, missing = keyword_score(resume_clean, jd_keywords)
        sem_score = semantic_score(resume_clean, jd_clean)
        final_score = calculate_final_score(kw_score, sem_score)

        # Classification
        status = "Highly ATS Friendly" if final_score>=85 else "Moderately ATS Friendly" if final_score>=65 else "Needs Optimization"

        # ------------------------------
        # COMPACT DASHBOARD
        # ------------------------------
        st.subheader("Compact Dashboard")
        col1, col2, col3 = st.columns([1,1,2])

        with col1:
            st.metric("ATS Score (%)", f"{round(final_score,2)}")
            st.progress(int(final_score))

        with col2:
            st.metric("Keyword Match (%)", f"{round(kw_score,2)}")
            st.progress(int(kw_score))

        with col3:
            st.metric("Semantic Similarity (%)", f"{round(sem_score,2)}")
            st.progress(int(sem_score))

        with st.expander("Matched Keywords"):
            st.write(", ".join(matched) if matched else "None")

        with st.expander("Missing Keywords"):
            st.write(", ".join(missing) if missing else "None")

        st.markdown(f"**Resume Status:** {status}")

        # ------------------------------
        # Proceed Decision
        # ------------------------------
        st.subheader("Do you want to proceed with this resume?")
        proceed = st.radio("", ["Yes", "No"], horizontal=True)
        if proceed:
            st.success(f"You selected: {proceed}")

        # ------------------------------
        # Screen Resume Button
        # ------------------------------
        if st.button("Screen Resume"):
            move_to_screened(resume_path)
            st.success(f"Resume '{selected_resume}' has been moved to 'screened_resumes' folder.")
