from flask import Flask, request, render_template, redirect, url_for
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

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

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_resume(file_path):
    """Dummy function to preprocess the uploaded file."""
    # Add code to extract text from PDF/Word files if needed
    # For now, this is a placeholder returning sample text
    return "Sample processed text from the uploaded resume."

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/get_predict', methods=['POST'])
def get_predict():
    """Handle file uploads and display results aesthetically."""
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

        # Preprocess the uploaded file
        processed_text = preprocess_resume(file_path)
        input_vector = vectorizer.transform([processed_text])  # Transform text to vector
        prediction = model.predict(input_vector)  # Predict using the model

        os.remove(file_path)  # Clean up the uploaded file

        # Render the results page
        return render_template('result.html', prediction=prediction[0])

    return render_template('error.html', message='Invalid file format')

if __name__ == '__main__':
    # Ensure uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
