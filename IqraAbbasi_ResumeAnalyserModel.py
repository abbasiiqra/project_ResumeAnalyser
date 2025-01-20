# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from collections import Counter
import pickle

# %matplotlib inline
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv('resdata.csv')

# Display basic info
print("Data Info:")
df.info()
print("\nData Shape:", df.shape)

# Plot categories
plt.figure(figsize=(20, 5))
plt.xticks(rotation=90)
sns.countplot(x='Category', data=df)
plt.grid()
plt.show()

# Check and remove duplicates
print("Duplicates:", df.duplicated(subset=['Resume']).value_counts())
df.drop_duplicates(subset=['Resume'], keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)

# Clean resume text
def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)
    resume_text = re.sub('RT|cc', ' ', resume_text)
    resume_text = re.sub('#\S+', '', resume_text)
    resume_text = re.sub('@\S+', ' ', resume_text)
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)
    resume_text = re.sub('\s+', ' ', resume_text)
    return resume_text

df['Resume'] = df['Resume'].apply(clean_resume)

# Generate word frequency distribution
stop_words = set(stopwords.words('english') + ['``', "''"])
all_words = []
for resume in df['Resume']:
    words = nltk.word_tokenize(resume)
    all_words.extend([word for word in words if word not in stop_words and word not in string.punctuation])

word_freq_dist = nltk.FreqDist(all_words)
most_common_words = word_freq_dist.most_common(50)
print("\nMost Common Words:", most_common_words)

# Prepare data for machine learning
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
tfidf_vectorizer.fit(df['Resume'])
features = tfidf_vectorizer.transform(df['Resume'])
target = df['Category']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

# Train the model
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)

# Save the model
with open('/content/trained_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

# Evaluate the model
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)
print(f"\nTraining Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Classification reports
print("\nClassification Report on Training Data:")
print(metrics.classification_report(y_train, clf.predict(X_train)))
print("\nClassification Report on Test Data:")
print(metrics.classification_report(y_test, clf.predict(X_test)))

# Extract ATS keywords
frequency_threshold = 10
potential_keywords = [word for word, count in word_freq_dist.items() if count >= frequency_threshold]
potential_keywords = [word for word in potential_keywords if word not in ['work', 'experience', 'skills']]
potential_keywords.extend(['Python', 'SQL', 'machine learning', 'data analysis'])
ats_keywords = list(set(potential_keywords))
print("\nPotential ATS Keywords:", ats_keywords)

# Function to predict ATS score and category
def predict_ats_score_and_category(resume_text):
    cleaned_resume = clean_resume(resume_text)
    ats_score = sum(1 for keyword in ats_keywords if keyword.lower() in cleaned_resume.lower())

    try:
        resume_features = tfidf_vectorizer.transform([resume_text])
        predicted_category_index = clf.predict(resume_features)[0]
        predicted_category = le.inverse_transform([predicted_category_index])[0]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0, "Unknown"

    return ats_score, predicted_category

# Test prediction with a sample resume
new_resume_data = df.iloc[19]['Resume']
ats_score, predicted_category = predict_ats_score_and_category(new_resume_data)
print("\nPredicted ATS Score:", ats_score)
print("Predicted Category:", predicted_category)
