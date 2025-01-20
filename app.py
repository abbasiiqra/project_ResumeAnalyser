from flask import Flask, request, render_template, redirect, url_for
import os
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
from docx import Document
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Load pre-trained model and vectorizer
MODEL_PATH = r'C:\Users\Asus\Desktop\project_ResumeAnalyser\trained_model.pkl'
VECTORIZER_PATH = r'C:\Users\Asus\Desktop\project_ResumeAnalyser\vectorizer.pkl'

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Example ATS keywords
ATS_KEYWORDS = ['Python', 'SQL', 'machine learning', 'data analysis', 'TensorFlow', 'AI', 'NLP']

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_text(text):
    """Clean and preprocess resume text."""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def extract_text_from_file(file_path):
    """Extract text from uploaded resume file."""
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            return file.read()
    elif file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        return ' '.join(page.extract_text() for page in reader.pages)
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        return ' '.join(paragraph.text for paragraph in doc.paragraphs)
    else:
        return ""

def calculate_ats_score(text):
    """Calculate ATS score based on keyword matches."""
    text = clean_text(text)
    keywords = set(word.lower() for word in ATS_KEYWORDS)
    words = set(text.split())
    matching_keywords = keywords.intersection(words)
    return len(matching_keywords)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/get_predict', methods=['POST'])
def get_predict():
    """Handle file uploads, calculate ATS score, and display results."""
    # Check if a file is part of the request
    if 'file' not in request.files:
        return render_template('error.html', message='No file part in the request')

    file = request.files['file']

    # Check if the file is valid
    if file.filename == '':
        return render_template('error.html', message='No file selected')
    if file and allowed_file(file.filename):
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Extract text from the uploaded file
        resume_text = extract_text_from_file(file_path)
        if not resume_text:
            os.remove(file_path)
            return render_template('error.html', message='Unable to extract text from the file')

        # Preprocess resume and calculate ATS score
        ats_score = calculate_ats_score(resume_text)
        # Load LabelEncoder
        with open('label_encoder.pkl', 'rb') as le_file:
            label_encoder = pickle.load(le_file)


        # Transform the text and predict category
        processed_text = clean_text(resume_text)
        input_vector = vectorizer.transform([processed_text])  # Transform text to vector
        prediction = model.predict(input_vector)  # Predict using the model
        predicted_category = label_encoder.inverse_transform(prediction)[0]

        os.remove(file_path)  # Clean up the uploaded file

        # Render the results page
        return render_template('result.html', predicted_category= predicted_category, ats_score=ats_score)

    return render_template('error.html', message='Invalid file format')

if __name__ == '__main__':
    # Ensure uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
