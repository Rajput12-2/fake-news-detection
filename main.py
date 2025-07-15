# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# Load dataset
df = pd.read_csv("news_dataset.csv")
print("Data loaded successfully!\n")

# Show basic info
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# Visualize data
sns.countplot(x='label', data=df)
plt.title("Distribution of Fake and Real News")
plt.savefig("visuals/data_distribution.png")
plt.show()

# Preprocessing
df = df.dropna()
df['text'] = df['text'].astype(str)

# Train-test split
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
with open("model/logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully in 'model/' folder.")
