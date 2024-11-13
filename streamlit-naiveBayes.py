import pandas as pd
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

dtsName = "uci-news-aggregator.csv"
data = pd.read_csv(dtsName)

data = data.dropna(subset=["TITLE", "CATEGORY"])


def remove_special_chars(text):
    if isinstance(text, str):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


data["TITLE"] = data["TITLE"].apply(remove_special_chars)
data["PUBLISHER"] = data["PUBLISHER"].apply(remove_special_chars)
data["CATEGORY"] = data["CATEGORY"].map(
    {
        "b": "business",
        "t": "science and technology",
        "e": "entertainment",
        "m": "health",
    }
)

st.title("News Classifier using Naive Bayes")
st.write("Application of news categories from the title using the Naive Bayes model.")

search_query = st.text_input("Search for news")

col1, col2, col3 = st.columns(3)

with col1:
    selected_category = st.selectbox(
        "Choose a News Category", ["--"] + list(data["CATEGORY"].unique())
    )

with col2:
    selected_publisher = st.selectbox(
        "Choose a News Publisher", ["--"] + list(data["PUBLISHER"].unique())
    )

with col3:
    selected_datetime = st.selectbox(
        "Choose a Date (Timestamp)", ["--"] + list(data["TIMESTAMP"].unique())
    )

if search_query:
    filtered_data = data[
        data["TITLE"].str.contains(search_query, case=False, na=False)
        | data["PUBLISHER"].str.contains(search_query, case=False, na=False)
        | data["TIMESTAMP"].astype(str).str.contains(search_query, case=False, na=False)
    ]
else:
    if (
        selected_category == "--"
        and selected_publisher == "--"
        and selected_datetime == "--"
    ):
        filtered_data = data.head(5)
    else:
        filtered_data = data[
            (
                data["CATEGORY"] == selected_category
                if selected_category != "--"
                else True
            )
            & (
                data["PUBLISHER"] == selected_publisher
                if selected_publisher != "--"
                else True
            )
            & (
                data["TIMESTAMP"] == selected_datetime
                if selected_datetime != "--"
                else True
            )
        ]

num_headlines_to_show = 5
filtered_data_limited = filtered_data.head(num_headlines_to_show)

st.write("Headlines for News with filter:")

for index, row in filtered_data_limited.iterrows():
    st.write(f"- {row['TITLE']}")

X = data["TITLE"]
y = data["CATEGORY"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Classification of news")
sample_titles = st.text_area(
    "Enter the titles of news articles (one per line):",
    value="",
)

if sample_titles:
    sample_titles = sample_titles.split("\n")
    sample_titles_tfidf = vectorizer.transform(sample_titles)
    predictions = model.predict(sample_titles_tfidf)

    for title, prediction in zip(sample_titles, predictions):
        st.write(f"News Title: {title}")
        st.write(f"News Category: {prediction}")
