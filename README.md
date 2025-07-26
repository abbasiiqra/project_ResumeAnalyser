
<p align="center">
	<em><code>Automated AI Powered Resume Ranking & Analysis Tool</code></em>
</p>
<p align="center">
	<!-- local repository, no metadata badges. -->
</p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=default&logo=Streamlit&logoColor=white" alt="Streamlit">
	<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=default&logo=Jupyter&logoColor=white" alt="Jupyter">
	<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=default&logo=scikit-learn&logoColor=white" alt="scikitlearn">
	<img src="https://img.shields.io/badge/spaCy-09A3D5.svg?style=default&logo=spaCy&logoColor=white" alt="spaCy">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="pandas">
</p>
<br>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
  - [Project Index](#project-index)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)
- [Project Roadmap](#project-roadmap)

---

## Overview

**ResumeAnalyser** is an automated resume ranking and analysis application designed for recruiters and hiring managers. This tool accepts a ZIP file containing multiple PDF resumes and an optional job description text file. It then processes the resumes using advanced NLP techniques and scoring algorithms to rank candidates based on:

- **Technical & Managerial Skills:** Evaluated using years of experience, education level, and keyword-based skills extraction.
- **Resume Quality:** Assessed via spell-check ratios, section identification, and overall brevity.
- **Job Matching:** Uses TF-IDF similarity measures to compare resumes against the provided job description.

The final output is a ranked list of resumes with scores and downloadable results for further review.

---

## Features

- **Interactive UI with Streamlit:** Upload resumes and job descriptions, view processing status, and download results.
- **PDF Text Extraction:** Converts resumes in PDF format into text using libraries like PyPDF2 and pymupdf.
- **NLP-Powered Analysis:** 
  - **Preprocessing:** Tokenization and lemmatization using spaCy.
  - **Skill Extraction:** Identifies both general and technical skills.
  - **Experience & Education:** Extracts years of experience and detects education level.
  - **Resume Quality Metrics:** Spell-check, section identification, and brevity evaluation.
- **Scoring and Ranking:** Combines technical, managerial, and overall quality scores with job match scores (via TF-IDF similarity) to generate final rankings.
- **Downloadable Results:** 
  - CSV file containing all calculated scores.
  - Top 3 ranked resumes available as downloadable PDFs with an embedded viewer.
- **Default Job Skills Management:** Automatically loads or creates a default job skills file (`job_skills.json`) for reference during analysis.

---

## Project Structure

Below is a sample layout of the project's folder structure:

    └── ResumeAnalyser/
        ├── README.md
        ├── model.py
        ├── app.py
        ├── output
        |      └── .gitkeep
        ├── .streamlit
        |      └── config.toml
        └── requirements.txt

### Project Index

- **requirements.txt:** Lists all project dependencies.
- **model.py:** Main logic for resume processing, scoring, and ranking.
- **app.py:** Streamlit application for interactive resume analysis.
- **.streamlit:** Configuration for streamlit environment.
- **output:** Placeholder for output stream
---

## Getting Started

### Prerequisites

- **Programming Language:** Python 3.7 or higher
- **Package Manager:** Pip
- **Required Libraries:** Streamlit, pandas, scikit-learn, spaCy, pymupdf, PyPDF2, language_tool_python, textblob, and others listed in [requirements.txt](requirements.txt).

### Installation

1. **Clone the Repository:**

       git clone https://github.com/yourusername/ResumeAnalyser.git

2. **Navigate to the Project Directory:**

       cd ResumeAnalyser

3. **Install Dependencies:**

       pip install -r requirements.txt
       python -m spacy download en_core_web_sm

### Usage

To launch the Resume Ranking Application, use the following command:

    streamlit run app.py

This will start a local Streamlit server where you can:
- Upload a ZIP file containing PDF resumes.
- Optionally upload a job description (TXT file).
- View the analysis results, download the final ranked CSV, and retrieve the top ranked resume PDFs.

---

## Project Roadmap

- [X] **Task 1:** Implement resume extraction and text conversion from PDF files.
- [X] **Task 2:** Enhance scoring algorithms with additional metrics and dynamic weighting.
- [X] **Task 3:** Integrate more robust job description parsing and candidate matching features.
- [X] **Task 4:** Improve UI/UX and add more visualization options in the Streamlit app.


---

## Future Plans

- Improve the regex and pattern matching for extraction
- Enhance the NLP pipeline
- Host on AWS/Google Cloud
- Suggest improvements to resumes
- Generate cover letters
- Incorporate weightage to personal recommendations on the basis of content of recommendation letters

---



