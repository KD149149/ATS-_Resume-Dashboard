
# ATS Resume Intelligence Dashboard

## Problem Statement

Recruiters often use Applicant Tracking Systems (ATS) to filter resumes before human review. Many candidates are unaware if their resume is ATS-friendly, resulting in qualified applicants being rejected automatically. 

This project provides a solution to analyze resumes and assess their ATS compatibility against a given Job Description (JD), providing actionable insights to improve the resume.

---

## Overview

The ATS Resume Intelligence Dashboard allows users to:

- Upload a resume (PDF or DOCX)
- Paste or upload a Job Description
- Analyze the resume for ATS compatibility
- Display ATS Score visually
- Highlight matched and missing keywords
- Provide a breakdown of keyword match and semantic similarity
- Classify resume as highly ATS-friendly, moderately ATS-friendly, or needs optimization

The system combines keyword extraction using TF-IDF and semantic similarity analysis using transformer-based embeddings to give an accurate ATS compatibility score.

---

## How It Works

1. **Resume Parsing**: Extract text from PDF or DOCX resumes.
2. **Job Description Parsing**: Extract text and identify top keywords from the Job Description.
3. **Keyword Matching**: Compare resume against JD keywords to calculate keyword match score.
4. **Semantic Analysis**: Use sentence embeddings to calculate semantic similarity between resume and JD.
5. **Final ATS Score**: Weighted combination of keyword match score (40%) and semantic similarity (60%).
6. **Visualization**: Display results using a gauge chart and radar chart. Show matched and missing keywords, along with recommendations.

---

## Tech Stack

- Python
- Streamlit
- PDFPlumber (for PDF text extraction)
- python-docx (for DOCX text extraction)
- scikit-learn (TF-IDF vectorizer)
- sentence-transformers (semantic similarity)
- matplotlib (charts and dashboard)

---

## Project Structure

```

ats-resume-intelligence/
│
├── ats_resume_ai.py       # Main application file
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

````

---

## Installation and Setup

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/ats-resume-intelligence.git
cd ats-resume-intelligence
````

2. **Create a Virtual Environment (Recommended)**

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run ats_resume_ai.py
```

* The app will open in your default browser.
* Upload a resume and paste the Job Description.
* Click "Analyze Resume" to see the ATS score, keyword match, semantic similarity, and missing keywords.

---

## Future Enhancements

* Auto-suggestion for missing keywords
* Multi-job comparison
* Resume formatting analysis for ATS optimization
* Deployable SaaS version with API support
* Recruiter simulation mode

---

## Author

Kajal Dadas | kajaldadas149@gmail.com | 7972244559

---

# 📦 requirements.txt

```txt
streamlit==1.30.0
pdfplumber==0.10.3
python-docx==1.1.0
scikit-learn==1.3.2
sentence-transformers==2.2.2
matplotlib==3.8.2
numpy==1.26.3
````

---
