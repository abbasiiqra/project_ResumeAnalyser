#!/usr/bin/env python

import pymupdf
import zipfile
import os
import re
import pandas as pd
import spacy
import language_tool_python
import datetime
import json
import subprocess
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import tempfile
from PyPDF2 import PdfReader
from io import BytesIO


def extract_text_from_pdf(pdf_path):
    with pymupdf.open(pdf_path) as pdf:
        text = ""
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text += page.get_text()
        return text


def extract_text_from_pdf_zip(zip_file, output_folder=None):
    """
    Extracts text from PDF files within a zip archive and saves them as text files.

    Args:
        zip_file (str or bytes or file-like object): Path to the zip file containing PDF files,
            or a bytes object, or a file-like object.
        output_folder (str, optional): Path to the folder where the extracted text files will be saved.
            If None, a temporary directory will be created.

    Returns:
        str: Path to the output folder containing the extracted text files.
    """
    if isinstance(zip_file, str):
        zip_file_path = zip_file
    elif isinstance(zip_file, bytes):
        temp_zip = tempfile.NamedTemporaryFile(delete=False)
        temp_zip.write(zip_file)
        temp_zip.close()
        zip_file_path = temp_zip.name
    elif hasattr(zip_file, "read"):
        temp_zip = tempfile.NamedTemporaryFile(delete=False)
        temp_zip.write(zip_file.read())
        temp_zip.close()
        zip_file_path = temp_zip.name
    else:
        raise ValueError(
            "zip_file must be a file path, bytes object, or file-like object."
        )

    if output_folder is None:
        output_folder = tempfile.mkdtemp()
    else:
        os.makedirs(output_folder, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        pdf_files = [
            f for f in zip_ref.namelist()
            if f.lower().endswith(".pdf") and not f.startswith("__MACOSX/")
        ]
        if not pdf_files:
            raise ValueError("No PDF files found in the zip archive.")

        print("PDF files recognized in the zip archive:")
        for pdf_file in pdf_files:
            print(pdf_file)

        for pdf_file_name in pdf_files:
            with zip_ref.open(pdf_file_name) as pdf_file:
                pdf_stream = BytesIO(pdf_file.read())
                pdf_reader = PdfReader(pdf_stream)

                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

                base_name = os.path.splitext(os.path.basename(pdf_file_name))[0]
                match = re.match(r"candidate_(\d+)", base_name, re.IGNORECASE)
                if match:
                    candidate_id = match.group(1)
                    text_file_name = f"Resume_of_ID_{candidate_id}.txt"
                else:
                    text_file_name = f"{base_name}.txt"

                text_file_path = os.path.join(output_folder, text_file_name)
                with open(text_file_path, "w", encoding="utf-8") as text_file:
                    text_file.write(text)

    if not isinstance(zip_file, str):
        os.remove(zip_file_path)

    print("\nText files saved in the output folder:")
    for file_name in os.listdir(output_folder):
        print(file_name)

    return output_folder


def load_and_clean_data(directory):
    text_files_folder = extract_text_from_pdf_zip(directory)
    print("Now we have a folder of text files")

    data = []
    for filename in os.listdir(text_files_folder):
        if filename.startswith("Resume_of_ID_") and filename.endswith(".txt"):
            file_path = os.path.join(text_files_folder, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            id_number = int(filename.split("_")[3].split(".")[0])
            clean_text = " ".join(text.split())
            data.append({"ID": id_number, "Text": clean_text})
    return pd.DataFrame(data)


def preprocess_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return tokens


def extract_years_of_experience(text):
    years = re.findall(r"\b(19[7-9]\d|20[0-2]\d)\b", text)
    if len(years) >= 2:
        earliest_year = min(int(year) for year in years)
        latest_year = max(int(year) for year in years)
        current_year = datetime.datetime.now().year
        if latest_year > current_year:
            latest_year = current_year
        return latest_year - earliest_year
    return 0


def detect_education_level(text):
    education_patterns = {
        "PhD": r"\bPh\.?D\.?\b|\bDoctor(ate)?\b",
        "Postgraduate": r"\bM\.?S\.?\b|\bM\.?A\.?\b|\bM\.?Tech\b|\bM\.?Sc\b|\bMaster(s)?\b|\bPost\s?Graduation\b|\bPostgraduate\b",
        "Bachelor": r"\bB\.?S\.?\b|\bB\.?A\.?\b|\bB\.?Tech\b|\bB\.?Sc\b|\bBachelor(s)?\b",
    }
    for level, pattern in education_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return level
    return "Other"


import time

def calculate_spell_check_ratio(text):
    tool = language_tool_python.LanguageToolPublicAPI("en-US")
    time.sleep(4)  # sleep to avoid hitting rate limit
    matches = tool.check(text)

def identify_resume_sections(text):
    important_sections = ["education", "experience", "skills", "projects", "achievements"]
    optional_sections = ["summary", "objective", "interests", "activities"]
    unnecessary_sections = ["references"]

    section_score = 0
    for section in important_sections:
        if re.search(r"\b" + section + r"\b", text, re.IGNORECASE):
            section_score += 1
    for section in optional_sections:
        if re.search(r"\b" + section + r"\b", text, re.IGNORECASE):
            section_score += 0.5
    for section in unnecessary_sections:
        if re.search(r"\b" + section + r"\b", text, re.IGNORECASE):
            section_score -= 0.5
    return min(section_score / len(important_sections), 1)


def quantify_brevity(text):
    word_count = len(text.split())
    if word_count < 200:
        return 0.5
    elif word_count > 1000:
        return 0.5
    else:
        return 1 - (abs(600 - word_count) / 400)


def calculate_word_sentence_counts(text):
    sentences = re.split(r"[.!?]+", text)
    word_count = len(text.split())
    sentence_count = len([s for s in sentences if s.strip()])
    return word_count, sentence_count


def calculate_skill_match_score(resume_skills, job_skills):
    if not job_skills:
        return 0
    matched_skills = set(resume_skills) & set(job_skills)
    return len(matched_skills) / len(job_skills)


def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


def quantify_achievement_impact(text):
    impact_score = 0
    achievements = re.findall(
        r"\b(increased|decreased|improved|reduced|saved|generated|expanded).*?(\d+(?:\.\d+)?%?)",
        text,
        re.IGNORECASE,
    )
    for _, value in achievements:
        if "%" in value:
            impact_score += float(value.strip("%")) / 100
        else:
            impact_score += float(value) / 1000
    return min(impact_score, 1)


def calculate_technical_score(row):
    skill_count = min(len(row["extracted_skills"]), 10)
    experience_score = min(row["years_of_experience"] / 10, 1)
    education_score = {
        "PhD": 1,
        "Master": 0.8,
        "Bachelor": 0.6,
        "Associate": 0.4,
        "Other": 0.2,
    }.get(row["education_level"], 0.2)
    return skill_count / 10 * 0.4 + experience_score * 0.3 + education_score * 0.3


def calculate_managerial_score(row):
    soft_skills_score = analyze_sentiment(row["text"])
    achievement_impact = quantify_achievement_impact(row["text"])
    leadership_score = min(row["years_of_experience"] / 15, 1)
    return soft_skills_score * 0.3 + achievement_impact * 0.4 + leadership_score * 0.3


def calculate_overall_score(row):
    technical_score = row["technical_score"]
    managerial_score = row["managerial_score"]
    resume_quality_score = (
        row["spell_check_ratio"] + row["section_score"] + row["brevity_score"]
    ) / 3
    return technical_score * 0.4 + managerial_score * 0.3 + resume_quality_score * 0.3


def job_description_matching(resume_text: str, job_description: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def match_resume_to_job_description(resume_text, job_description):
    match_score = job_description_matching(resume_text, job_description)
    return {"job_match_score": match_score}


def load_job_skills(file_path: str) -> List[str]:
    default_skills = [
        "communication", "teamwork", "leadership", "problem-solving",
        "time management", "analytical skills", "creativity", "adaptability",
        "programming", "data analysis", "project management", "software development",
        "database management", "web development", "Python", "Java",
        "Machine Learning", "Deep Learning", "NLP", "SQL", "C++", "JavaScript",
        "Data Science", "TensorFlow", "PyTorch", "Linux", "Docker",
        "Kubernetes", "Git", "REST API", "Flask", "Django", "BERT",
        "Transformers", "Siamese", "Neural Networks",
    ]
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                skills_data = json.load(file)
            if isinstance(skills_data, list):
                return skills_data
            elif isinstance(skills_data, dict) and "skills" in skills_data:
                return skills_data["skills"]
            else:
                print(f"Unexpected format in '{file_path}'. Using default skills list.")
                return default_skills
        else:
            print(f"'{file_path}' not found. Using default skills list.")
            return default_skills
    except json.JSONDecodeError:
        print(f"Error reading '{file_path}'. Using default skills list.")
        return default_skills

# Load spaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"]) 
    nlp = spacy.load("en_core_web_sm")


def extract_skills(text: str) -> List[str]:
    if not text:
        return []
    doc = nlp(text)
    keyword_skills = set()
    ner_skills = set()
    for skill in general_skills:
        if skill.lower() in text.lower():
            keyword_skills.add(skill)
    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "WORK_OF_ART"}:
            ner_skills.add(ent.text)
    return sorted(keyword_skills) + sorted(ner_skills)


def process_resume(row):
    text = row["Text"]
    return {
        "years_of_experience": extract_years_of_experience(text),
        "education_level": detect_education_level(text),
        "spell_check_ratio":0.9,
        "section_score": identify_resume_sections(text),
        "brevity_score": quantify_brevity(text),
        "extracted_skills": extract_skills(text),
    }


def resumemain(resume_directory: str, job_description_path: str=None):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print("1. Starting resume analysis process...")

    print(resume_directory)
    print("2. Loading general skills...")
    global general_skills
    general_skills = load_job_skills("job_skills.json")

    print("3. Loading and preprocessing resumes...")
    df = load_and_clean_data(resume_directory)

    processed_df = df.apply(process_resume, axis=1, result_type="expand")
    print("Processed columns:", processed_df.columns.tolist())
    df = pd.concat([df, processed_df], axis=1)

    # Use list comprehension for column normalization (avoid .str)
    df.columns = [
        col.strip().replace(" ", "_").lower()
        for col in df.columns
    ]

    if "extracted_skills" not in df.columns:
        df["extracted_skills"] = [[] for _ in range(len(df))]
    assert "extracted_skills" in df.columns, (
        f"'extracted_skills' not found! Have: {df.columns}"
    )

    print("4. Calculating scores...")
    # compute technical_score safely
    temp_tech = df.apply(calculate_technical_score, axis=1)
    if isinstance(temp_tech, pd.DataFrame):
        temp_tech = temp_tech.iloc[:, 0]
    df["technical_score"] = temp_tech

    # compute managerial_score safely
    temp_mgr = df.apply(calculate_managerial_score, axis=1)
    if isinstance(temp_mgr, pd.DataFrame):
        temp_mgr = temp_mgr.iloc[:, 0]
    df["managerial_score"] = temp_mgr

    # compute overall_featured_score safely
    temp_ovr = df.apply(calculate_overall_score, axis=1)
    if isinstance(temp_ovr, pd.DataFrame):
        temp_ovr = temp_ovr.iloc[:, 0]
    df["overall_featured_score"] = temp_ovr

    if job_description_path:
        with open(job_description_path, "r", encoding="utf-8") as f:
            job_description = f.read()
    else:
        job_description = None

    df["tf_idf_score"] = df.apply(
        lambda row: (
            match_resume_to_job_description(row["text"], job_description).get(
                "job_match_score", 1.0
            )
            if job_description
            else 1.0
        ),
        axis=1,
    )

    df["final_score"] = df["overall_featured_score"] * 0.7 + df["tf_idf_score"] * 0.3
    # compute skill_count after scores
    df["skill_count"] = df["extracted_skills"].apply(len)

    print("5. Ranking resumes...")
    ranked_df = df.sort_values("final_score", ascending=False).reset_index(drop=True)

    print("6. Saving results...")
    # define expected output columns
    final_columns = [
        "id",
        "final_score",
        "overall_featured_score",
        "tf_idf_score",
        "education_level",
        "technical_score",
        "managerial_score",
        "spell_check_ratio",
        "section_score",
        "brevity_score",
        "years_of_experience",
        "skill_count",
        "extracted_skills",
    ]
    # ensure all expected columns exist in ranked_df
    for col in final_columns:
        if col not in ranked_df.columns:
            ranked_df[col] = None
    # restrict and reorder columns
    ranked_df = ranked_df[final_columns]
    ranked_df.to_csv("final_ranked_resumes.csv", index=False)

    print("7. Resume analysis complete. Results saved to 'final_ranked_resumes.csv' !")
    return ranked_df


def main():
    resume_directory = os.path.join(os.getcwd(), "extracted_text_files")
    resumemain(resume_directory)

if __name__ == "__main__":
    main()
