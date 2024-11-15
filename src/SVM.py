import os
import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re
import time
from scipy.sparse import hstack

# ==================== Data Preprocessing ====================
# Read data
df = pd.read_csv("./data/uci-news-aggregator.csv")
ps = PorterStemmer()

# Displaying initial dataset structure
print("Dataset info:")
df.info()

df = df[["TITLE", "HOSTNAME", "CATEGORY", "PUBLISHER", "TIMESTAMP"]]
df.dropna(inplace=True)

# Map categories
category_mapping = {
    "b": "Business",
    "t": "Science and technology",
    "e": "Entertainment",
    "m": "Health",
}
df["CATEGORY"] = df["CATEGORY"].map(category_mapping)

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


# Text preprocessing
def preprocess_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


df["TITLE"] = df["TITLE"].apply(preprocess_text)
df["PUBLISHER"] = df["PUBLISHER"].apply(preprocess_text)

# Label Encoding
label_encoder = LabelEncoder()
df["CATEGORY"] = label_encoder.fit_transform(df["CATEGORY"])

# Splitting the dataset
X = df[["TITLE", "HOSTNAME", "PUBLISHER"]]
y = df["CATEGORY"]

# Vectorization
tfidf_title = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), max_df=0.85, min_df=5
)
X_title = tfidf_title.fit_transform(X["TITLE"])

X_combined = hstack([X_title])
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

# ==================== Model Training ====================
# Setting model parameters for LinearSVC
C = 1.0
svm_model = LinearSVC(C=C, max_iter=1000)

# Tracking training time
start_time = time.time()
svm_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Predictions and Metrics
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_micro = f1_score(y_test, y_pred, average="micro")
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")

# Display metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score (Macro): {f1_macro}")
print(f"F1 Score (Micro): {f1_micro}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Training time: {training_time} seconds")

# Classification report as dictionary
class_report_dict = classification_report(
    y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
)

print("\nClassification Report:\n", json.dumps(class_report_dict, indent=4))

# ==================== Saving Files ====================
# Tạo thư mục nếu chưa tồn tại
os.makedirs("models/SVM", exist_ok=True)


# Save metrics and classification report as JSON
def save_metrics_and_report_json(
    metrics, report_dict, metrics_filename, report_filename
):
    metrics_filepath = os.path.join("models/SVM", metrics_filename)
    report_filepath = os.path.join("models/SVM", report_filename)

    with open(metrics_filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    with open(report_filepath, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Metrics và báo cáo đã được lưu vào: {metrics_filepath}, {report_filepath}")


# Save metrics as CSV
def save_metrics_as_csv(metrics, filename="metrics.csv"):
    metrics_filepath = os.path.join("models/SVM", filename)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_filepath, index=False)
    print(f"Metrics đã được lưu vào {metrics_filepath}.")


# Metrics dictionary
metrics = {
    "Accuracy": accuracy,
    "F1 Score (Macro)": f1_macro,
    "F1 Score (Micro)": f1_micro,
    "Precision": precision,
    "Recall": recall,
    "Training time": training_time,
}

# Save files
save_metrics_and_report_json(
    metrics, class_report_dict, "metrics.json", "classification_report.json"
)
save_metrics_as_csv(metrics, "metrics.csv")


# Save Model and Vectorizer
def save_model(
    model,
    vectorizer,
    model_filename="svm_model.joblib",
    vectorizer_filename="tfidf_title.joblib",
):
    model_filepath = os.path.join("models/SVM", model_filename)
    vectorizer_filepath = os.path.join("models/SVM", vectorizer_filename)

    joblib.dump(model, model_filepath)
    joblib.dump(vectorizer, vectorizer_filepath)
    print(f"Mô hình đã được lưu vào {model_filepath}.")
    print(f"Vectorizer đã được lưu vào {vectorizer_filepath}.")


save_model(svm_model, tfidf_title)

# Save Confusion Matrix as Image
conf_matrix_filepath = os.path.join("models/SVM", "confusion_matrix.png")
plt.figure(figsize=(10, 7))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(conf_matrix_filepath)
plt.show()

print(f"Confusion Matrix đã được lưu vào {conf_matrix_filepath}.")
