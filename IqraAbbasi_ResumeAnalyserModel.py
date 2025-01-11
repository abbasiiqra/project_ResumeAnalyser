# -*- coding: utf-8 -*-
"""IqraAbbasi_resumeAnalyser
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

df = pd.read_csv('/content/UpdatedResumeDataSet.csv')

df.head()

df.info()

df.shape

"""# Exploring Categories"""

df['Category'].value_counts()

plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
sns.countplot(x='Category',data=df)
plt.grid()

"""# Cleaning Data"""

df.duplicated(subset=['Resume']).value_counts()

df[df.duplicated()==True]

df.iloc[19]['Resume']

df.drop_duplicates(subset=['Resume'], keep='first',inplace = True)
df.reset_index(inplace=True,drop=True)
df.head()

df.info()

import re
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\S+', '', resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

df['Resume'] = df.Resume.apply(lambda x: cleanResume(x))

df.head()

import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt_tab')

oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
totalWords =[]
Sentences = df['Resume'].values
cleanedSentences = ""

def clean_resume(resume_text):

    words = nltk.word_tokenize(resume_text)
    cleaned_words = [word for word in words if word not in oneSetOfStopWords and word not in string.punctuation]
    cleaned_text = " ".join(cleaned_words)
    return cleaned_text


for records in Sentences:
    cleanedText = clean_resume(records)
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)


wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)

"""# Words into Categorical Values"""

from sklearn.preprocessing import LabelEncoder

var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

df.head()

"""# vectorization and train test split"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

requiredText = df['Resume'].values
requiredTarget = df['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english')
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

print ("Feature completed .....")

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=42, test_size=0.2,
                                                 shuffle=True, stratify=requiredTarget)
print(X_train.shape)
print(X_test.shape)

from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format
(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set:     {:.2f}'.format
(clf.score(X_test, y_test)))

"""# Classification reports"""

print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_train, clf.predict(X_train))))

print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))

"""# Prediction System"""

new_resume_data = df.iloc[19]['Resume']

import pickle

filename = 'trained_model.pkl'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))


new_resume_features = word_vectorizer.transform([new_resume_data])

predicted_category = loaded_model.predict(new_resume_features)[0]
print("Predicted Category:", predicted_category)

original_category = le.inverse_transform([predicted_category])

print("Original Category:", original_category[0])

pickle.dump(word_vectorizer, open('vectorizer.pkl', 'wb'))