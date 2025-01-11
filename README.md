Resume Screening Application
This project is a Flask-based web application for screening resumes using a machine learning model. 
Users can upload their resumes in various formats (PDF, DOCX, TXT), and the application predicts the category of resume.

Features
Upload resumes in PDF, DOCX, or TXT formats.
Preprocesses and extracts text from uploaded resumes.
Uses a pre-trained machine learning model for predictions.
Displays the category of resume.

Setup and Installation
1. Clone the Repository
bash
Copy code
git clone https://github.com/your-username/resume-screening-app.git
cd resume-screening-app
2. Create a Virtual Environment
bash
Copy code
python -m venv venv
Activate the virtual environment:

Windows: venv\Scripts\activate
Linux/Mac: source venv/bin/activate
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Add Pre-trained Model and Vectorizer
Place your pre-trained model file (trained_model.pkl) and vectorizer file (vectorizer.pkl) in the project directory.
Update the paths in the Python script if necessary.
5. Create Necessary Directories
Create an uploads directory for storing temporary uploaded files:
mkdir uploads
How to Run
Start the Flask server:
bash
Copy code
python app.py
Open your browser and navigate to:
arduino
Copy code
http://127.0.0.1:5000/

File Structure:

resume-screening-app/
├── app.py                 # Main Flask application
├── requirements.txt       # List of Python dependencies
├── templates/
│   ├── index.html         # Upload page
│   ├── result.html        # Results display page
│   ├── error.html         # Error display page
├── uploads/               # Temporary directory for uploaded files
├── trained_model.pkl      # Pre-trained ML model
├── vectorizer.pkl         # Pre-trained vectorizer
└── README.md              # Project documentation

Dependencies:
Ensure the following libraries are installed (see requirements.txt):
Flask
scikit-learn
pandas
pickle5 (for loading the model/vectorizer)
python-docx (optional, for DOCX processing)
PyPDF2 (optional, for PDF processing)

Install dependencies using:
pip install -r requirements.txt
Usage
Upload a Resume: Select a file in PDF, DOCX, or TXT format on the homepage.
View Predictions: After submission, the application will process the resume and display the model's predictions in a styled result page.
Handle Errors: If invalid files are uploaded, clear error messages will be displayed.
