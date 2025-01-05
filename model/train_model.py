import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv("data/sentiment_dataset.csv") 

X = data['text']
y = data['label']

vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)
y = LabelEncoder().fit_transform(y)

model = LogisticRegression()
model.fit(X_vect, y)

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')