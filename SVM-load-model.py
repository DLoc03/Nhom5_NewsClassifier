import joblib
import pickle
import time
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)  # Cần import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack  # Cần import hstack từ scipy.sparse

# Đọc lại dữ liệu
df = pd.read_csv("./archive/uci-news-aggregator.csv")

# Load mô hình và các vectorizer
svm_model = joblib.load("save_model/svm_model.pkl")
tfidf_title = joblib.load("save_model/svm_tfidf_title.pkl")
tfidf_host = joblib.load("save_model/svm_tfidf_host.pkl")
tfidf_publisher = joblib.load("save_model/svm_tfidf_publisher.pkl")

# Load metrics và thời gian huấn luyện từ pickle
with open("save_model/svm_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

# In thời gian huấn luyện và accuracy
print(f"Total Training Time: {metrics['training_time']:.4f} seconds")
print(f"Accuracy: {metrics['accuracy']}")
print("Classification Report:")
print(metrics["classification_report"])

# Tạo X_test và y_test
X = df[["TITLE", "HOSTNAME", "PUBLISHER", "TIMESTAMP"]]
y = df["CATEGORY"]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Tiền xử lý dữ liệu tương tự như khi huấn luyện
X_title = tfidf_title.transform(X["TITLE"])
X_host = tfidf_host.transform(X["HOSTNAME"])
X_publisher = tfidf_publisher.transform(X["PUBLISHER"])

X_combined = hstack([X_title, X_host, X_publisher])

# Dự đoán với mô hình đã lưu
y_pred = svm_model.predict(X_combined)

# In kết quả dự đoán và đánh giá lại
print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred, target_names=label_encoder.classes_))

# Vẽ confusion matrix
cm = confusion_matrix(y, y_pred)
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
plt.show()
