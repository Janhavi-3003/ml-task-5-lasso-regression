import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.title("Spam Analysis with L1 Regularization")

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

total_features = X.shape[1]
st.write("Total features:", total_features)

# Train model with L1 penalty
model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
model.fit(X_train, y_train)

coef = model.coef_[0]

non_zero = (coef != 0).sum()
zero = (coef == 0).sum()

st.write("Selected features:", non_zero)
st.write("Eliminated features:", zero)

# Alpha comparison (C inverse of alpha)
st.subheader("Regularization Comparison")

for C in [0.01, 0.1, 1]:
    model = LogisticRegression(penalty='l1', solver='liblinear', C=C)
    model.fit(X_train, y_train)

    coef = model.coef_[0]
    selected = (coef != 0).sum()

    st.write("C:", C, "| Selected:", selected)

# Reduction %
reduction = ((total_features - non_zero) / total_features) * 100
st.write("Feature reduction: {:.2f}%".format(reduction))
