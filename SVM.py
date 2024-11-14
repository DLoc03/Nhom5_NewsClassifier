import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime
from scipy.sparse import hstack
import nltk
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
from nltk.stem import PorterStemmer

df = pd.read_csv("uci-news-aggregator.csv")
ps = PorterStemmer()
df.info()

df = df[["TITLE", "HOSTNAME", "CATEGORY", "PUBLISHER", "TIMESTAMP"]]

print(df["CATEGORY"].value_counts())

df.dropna(inplace=True)

print(df.duplicated().sum())

category_mapping = {
    "b": "Business",
    "t": "Science and technology",
    "e": "Entertainment",
    "m": "Health",
}

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def preprocess_title(title):
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"[^a-zA-Z\s]", "", title)
    title = title.lower()
    tokens = title.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]  # Stemming
    return " ".join(tokens)


def convert_date(timestamp):
    date_obj = datetime.fromtimestamp(timestamp / 1000)
    return date_obj.strftime("%d-%m-%Y")


def process_host(host):
    host = re.sub(r"^www\.", "", host)
    host = re.sub(r"\.[a-z]+$", "", host)
    return host


df["TIMESTAMP"] = df["TIMESTAMP"].apply(convert_date)
df["TITLE"] = df["TITLE"].apply(preprocess_title)
df["PUBLISHER"] = df["PUBLISHER"].apply(preprocess_title)
df["HOSTNAME"] = df["HOSTNAME"].apply(process_host)
df["CATEGORY"] = df["CATEGORY"].apply(lambda x: category_mapping.get(x, "unknown"))

label_encoder = LabelEncoder()
df["CATEGORY"] = label_encoder.fit_transform(df["CATEGORY"])

X = df[["TITLE", "HOSTNAME", "PUBLISHER", "TIMESTAMP"]]
y = df["CATEGORY"]

tfidf_title = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), max_df=0.85, min_df=5
)
X_title = tfidf_title.fit_transform(X["TITLE"])

tfidf_host = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), max_df=0.85, min_df=5
)
X_host = tfidf_host.fit_transform(X["HOSTNAME"])

tfidf_publisher = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), max_df=0.85, min_df=5
)
X_publisher = tfidf_publisher.fit_transform(X["PUBLISHER"])

X_combined = hstack([X_title, X_host, X_publisher])

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
print("y shape: ", y.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)


# Function to simulate progress during training
def train_with_progress(model, X_train, y_train, batch_size=10000):
    num_batches = X_train.shape[0] // batch_size
    with tqdm(total=num_batches, desc="Training LinearSVC Model") as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            model.fit(X_train[start_idx:end_idx], y_train[start_idx:end_idx])
            pbar.update(1)  # Update progress bar after each batch


# Initialize and train the model
svm_model = LinearSVC(max_iter=1000)
train_with_progress(svm_model, X_train, y_train)  # Train with progress updates

# Predicting and evaluating the model
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
