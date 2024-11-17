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

df = pd.read_csv("./data/uci-news-aggregator.csv")
ps = PorterStemmer()
df.info()
df = df[["TITLE", "HOSTNAME", "CATEGORY", "PUBLISHER", "TIMESTAMP"]]
df.dropna(inplace=True)
category_mapping = {
    "b": "Business",
    "t": "Science and technology",
    "e": "Entertainment",
    "m": "Health",
}
df["CATEGORY"] = df["CATEGORY"].map(category_mapping)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


df["TITLE"] = df["TITLE"].apply(preprocess_text)
df["PUBLISHER"] = df["PUBLISHER"].apply(preprocess_text)
label_encoder = LabelEncoder()
df["CATEGORY"] = label_encoder.fit_transform(df["CATEGORY"])
X = df[["TITLE", "HOSTNAME", "PUBLISHER"]]
y = df["CATEGORY"]

tfidf_title = TfidfVectorizer(
    max_features=10000, ngram_range=(1, 3), max_df=0.85, min_df=5
)

X_title = tfidf_title.fit_transform(X["TITLE"])
X_combined = hstack([X_title])
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)
C = 1.0

svm_model = LinearSVC(C=C, max_iter=1000)

start_time = time.time()
svm_model.fit(X_train, y_train)
training_time = time.time() - start_time
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_micro = f1_score(y_test, y_pred, average="micro")
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
print(f"Accuracy: {accuracy}")
print(f"F1 Score (Macro): {f1_macro}")
print(f"F1 Score (Micro): {f1_micro}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Training time: {training_time} seconds")
class_report_dict = classification_report(
    y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
)
print("\nClassification Report:\n", json.dumps(class_report_dict, indent=4))
os.makedirs("models/SVM", exist_ok=True)


def save_metrics_and_report_json(
    metrics, report_dict, metrics_filename, report_filename
):
    metrics_filepath = os.path.join("models/SVM", metrics_filename)
    report_filepath = os.path.join("models/SVM", report_filename)
    with open(metrics_filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    with open(report_filepath, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Metrics and report saved to: {metrics_filepath}, {report_filepath}")


def save_metrics_as_csv(metrics, filename="metrics.csv"):
    metrics_filepath = os.path.join("models/SVM", filename)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_filepath, index=False)
    print(f"Metrics saved to {metrics_filepath}.")


metrics =  {
    "Accuracy": [accuracy],
    "F1 Score (Macro)": f1_macro,
    "F1 Score (Micro)": f1_micro,
    "Precision": precision,
    "Recall": recall,
    "Training time": training_time,
}
save_metrics_and_report_json(
    metrics, class_report_dict, "metrics.json", "classification_report.json"
)
save_metrics_as_csv(metrics, "metrics.csv")


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
    print(f"Model saved to {model_filepath}.")
    print(f"Vectorizer saved to {vectorizer_filepath}.")


save_model(svm_model, tfidf_title)
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
plt.savefig("plots/SVM/confusion_matrix.png")
plt.show()

metrics_names = ["Accuracy", "F1 Macro", "F1 Micro", "Precision", "Recall"]
metrics_values = [accuracy, f1_macro, f1_micro, precision, recall]
plt.figure(figsize=(10, 6))
sns.barplot(x=metrics_names, y=metrics_values, palette="muted")
plt.title("Performance Metrics Comparison")
plt.ylabel("Score")
plt.xlabel("Metrics")
plt.savefig("plots/SVM/metrics_comparison.png")
plt.show()
