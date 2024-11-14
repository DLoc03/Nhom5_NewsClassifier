import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime
from scipy.sparse import hstack
import nltk
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from tqdm import tqdm
from nltk.stem import PorterStemmer

# Đọc dữ liệu
df = pd.read_csv("./archive/uci-news-aggregator.csv")
ps = PorterStemmer()
df.info()

df = df[["TITLE", "HOSTNAME", "CATEGORY", "PUBLISHER", "TIMESTAMP"]]

# Xử lý dữ liệu
df.dropna(inplace=True)
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
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
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

# Chuyển đổi dữ liệu thành dạng vector sử dụng Tfidf
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

# Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


# Function to simulate progress during training
def train_with_progress(model, X_train, y_train, batch_size=10000):
    num_batches = X_train.shape[0] // batch_size
    with tqdm(total=num_batches, desc="Training LinearSVC Model") as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            model.fit(X_train[start_idx:end_idx], y_train[start_idx:end_idx])
            pbar.update(1)  # Update progress bar after each batch


# Đo thời gian huấn luyện
start_time = time.time()

# Huấn luyện mô hình với tiến trình
svm_model = LinearSVC(max_iter=1000)
train_with_progress(svm_model, X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = svm_model.predict(X_test)

end_time = time.time()
execution_time = end_time - start_time
print(f"Total Training Time: {execution_time:.4f} seconds")

# Accuracy và báo cáo phân loại
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Lưu mô hình
joblib.dump(svm_model, "save_model/svm_model.pkl")

# Lưu vectorizers
joblib.dump(tfidf_title, "save_model/svm_tfidf_title.pkl")
joblib.dump(tfidf_host, "save_model/svm_tfidf_host.pkl")
joblib.dump(tfidf_publisher, "save_model/svm_tfidf_publisher.pkl")

# Lưu metrics và thời gian huấn luyện
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "classification_report": classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    ),
    "training_time": execution_time,
}

with open("save_model/svm_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("True Category")
plt.savefig("save_model/svm_confusion_matrix.png")
plt.close()

# Category Distribution Plot
plt.figure(figsize=(8, 6))
sns.countplot(x="CATEGORY", data=df)
plt.title("Distribution of News Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.savefig("save_model/svm_category_distribution.png")
plt.close()

# Performance Metrics Bar Plot
report = classification_report(
    y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
)
report_df = pd.DataFrame(report).transpose()
metrics = report_df[["precision", "recall", "f1-score"]]

plt.figure(figsize=(12, 8))
metrics.plot(kind="bar", figsize=(10, 6))
plt.title("Performance Metrics per Category")
plt.xlabel("Category")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(title="Metrics")
plt.savefig("save_model/svm_performance_metrics.png")
plt.close()

print("Training Success!")
