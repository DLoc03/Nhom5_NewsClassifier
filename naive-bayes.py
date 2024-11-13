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

dtsName = "uci-news-aggregator.csv"
# Đọc dữ liệu từ CSV
data = pd.read_csv(dtsName)

# Xem qua thông tin dữ liệu
print(data.head())

# Tiền xử lý dữ liệu
# Loại bỏ các giá trị thiếu
data = data.dropna(subset=["TITLE", "CATEGORY"])


# Làm sạch văn bản: loại bỏ ký tự đặc biệt trong tiêu đề và câu chuyện
def remove_special_chars(text):
    if isinstance(text, str):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower().strip())
    return text


data["TITLE"] = data["TITLE"].apply(remove_special_chars)
data["CATEGORY"] = data["CATEGORY"].map(
    {
        "b": "business",
        "t": "science and technology",
        "e": "entertainment",
        "m": "health",
    }
)

# Tách dữ liệu thành dữ liệu huấn luyện và kiểm tra
X = data["TITLE"]  # Dữ liệu đầu vào (tiêu đề)
y = data["CATEGORY"]  # Nhãn mục tiêu (thể loại)

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Đo thời gian bắt đầu
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
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

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

print(f"\nPrecision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1-Score (Weighted): {f1:.4f}")

# Dự đoán thử với một vài bài viết mới
sample_titles = [
    "FDA Approves New Drug to Combat Rare Genetic Disorder",
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

# Vẽ confusion matrix bằng heatmap (dùng seaborn)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("True Category")
plt.show()

# Biểu đồ phân phối các thể loại tin tức trong bộ dữ liệu
plt.figure(figsize=(8, 6))
sns.countplot(x="CATEGORY", data=data)
plt.title("Distribution of News Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()
