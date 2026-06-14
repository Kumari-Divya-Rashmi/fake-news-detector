import pandas as pd
import joblib
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1, random_state=42)

df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["content"] = df["content"].apply(clean_text)

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, pred)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model trained successfully")
print("Accuracy:", accuracy)
