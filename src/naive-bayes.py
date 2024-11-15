import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import time
import joblib
import json

dtsName = "./data/uci-news-aggregator.csv"
data = pd.read_csv(dtsName)

print(data.head())

data = data.dropna(subset=["TITLE", "CATEGORY"])


def remove_special_chars(text):
    if isinstance(text, str):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower().strip())
    return text


data["TITLE"] = data["TITLE"].apply(remove_special_chars)
data["CATEGORY"] = data["CATEGORY"].map(
    {
        "b": "Business",
        "t": "Science and technology",
        "e": "Entertainment",
        "m": "Health",
    }
)

# Tách dữ liệu thành dữ liệu huấn luyện và kiểm tra
X = data["TITLE"]  # Dữ liệu đầu vào (tiêu đề)
y = data["CATEGORY"]  # Nhãn mục tiêu (thể loại)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

start_time = time.time()

# Vector hóa dữ liệu văn bản sử dụng TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Huấn luyện mô hình Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test_tfidf)

# Đánh giá độ chính xác và in ra báo cáo phân loại
accuracy = accuracy_score(y_test, y_pred)

# Các tiêu chí đánh giá khác
precision = precision_score(
    y_test, y_pred, average="weighted"
)  # Precision theo phương pháp "weighted"
recall = recall_score(
    y_test, y_pred, average="weighted"
)  # Recall theo phương pháp "weighted"
f1 = f1_score(
    y_test, y_pred, average="weighted"
)  # F1-Score theo phương pháp "weighted"

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1-Score (Weighted): {f1:.4f}")

# Dự đoán thử với một vài bài viết mới
sample_titles = [
    "\nFDA Approves New Drug to Combat Rare Genetic Disorder",
    "Global Health Crisis: Rising Cases of Diabetes Among Young Adults",
    "Snack stole 2 millions dollar from VietNamA Bank",
]
sample_titles_tfidf = vectorizer.transform(sample_titles)
predictions = model.predict(sample_titles_tfidf)

for title, prediction in zip(sample_titles, predictions):
    print(f"Title: {title}\nPredicted Category: {prediction}\n")

# Đo thời gian kết thúc
end_time = time.time()

# Tính toán thời gian chạy tổng cộng
execution_time = end_time - start_time
print(f"\nTotal Execution Time: {execution_time:.4f} seconds")

joblib.dump(model, "models/NaiveBayes/naive_bayes_model.joblib")
joblib.dump(vectorizer, "models/NaiveBayes/nvb_vectorizer.joblib")

metrics_df = pd.DataFrame(
    {
        "Accuracy": [accuracy],
        "Precision (Weighted)": [precision],
        "Recall (Weighted)": [recall],
        "F1-Score (Weighted)": [f1],
        "Training Time (seconds)": [execution_time],
    }
)
metrics_df.to_csv("models/NaiveBayes/nvb_metrics.csv", index=False)

with open("models/NaiveBayes/nvb_classi_report.json", "w") as f:
    json.dump(classification_report(y_test, y_pred, output_dict=True), f)


plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("True Category")
plt.savefig("plots/NaiveBayes/confusion_matrix.png")

plt.figure(figsize=(8, 6))
sns.countplot(x="CATEGORY", data=data)
plt.title("Distribution of News Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.savefig("plots/NaiveBayes/category_distribution.png")

metrics_df_filtered = metrics_df.drop(columns=["Training Time (seconds)"])

plt.figure(figsize=(12, 8))
metrics_df_filtered.plot(kind="bar", figsize=(10, 6))
plt.title("Performance Metrics per Category")
plt.xlabel("Category")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(title="Metrics")
plt.savefig("plots/NaiveBayes/performance_metrics.png")
