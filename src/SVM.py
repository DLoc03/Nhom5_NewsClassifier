import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import re
from scipy.sparse import hstack
import nltk
from nltk.stem import PorterStemmer
import seaborn as sns
import matplotlib.pyplot as plt
import time

df = pd.read_csv("uci-news-aggregator.csv")
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
tfidf_title = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), max_df=0.85, min_df=5)
X_title = tfidf_title.fit_transform(X["TITLE"])

X_combined = hstack([X_title])
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

#------------------------------------------------------------------Pre-Processing End------------------------------------------------------------------------------->

# Setting model parameters for LinearSVC
C = 1.0  # You can experiment with different C values for optimal results
svm_model = LinearSVC(C=C, max_iter=1000)

# Tracking training time
start_time = time.time()
svm_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Predictions and Metrics
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# Display metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score (Macro): {f1_macro}")
print(f"F1 Score (Micro): {f1_micro}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Training time: {training_time} seconds")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Sample Input/Output
print("\nSample inputs and outputs:")
print("Sample input titles:", X["TITLE"].iloc[:5].values)
print("Sample predictions:", label_encoder.inverse_transform(y_pred[:5]))

# Accuracy Comparison Chart
plt.figure(figsize=(10, 5))
metrics = ["Accuracy", "F1 Macro", "F1 Micro", "Precision", "Recall"]
values = [accuracy, f1_macro, f1_micro, precision, recall]
sns.barplot(x=metrics, y=values, palette="viridis")
plt.title("Performance Metrics Comparison")
plt.ylabel("Score")
plt.show()
import joblib


def save_model(model, tfidf_title, model_filename="svm_model.joblib", vectorizer_filename="tfidf_title.joblib"):
    joblib.dump(model, model_filename)
    joblib.dump(tfidf_title, vectorizer_filename)
    print("Mô hình và vectorizer đã được lưu thành công.")

def load_model(model_filename="svm_model.joblib", vectorizer_filename="tfidf_title.joblib"):
    model = joblib.load(model_filename)
    tfidf_title = joblib.load(vectorizer_filename)
    print("Mô hình và vectorizer đã được tải thành công.")
    return model, tfidf_title


save_model(svm_model, tfidf_title)


loaded_model, loaded_vectorizer = load_model()
sample_texts = ["This is a test news title", "New advancements in AI technology"]
sample_features = loaded_vectorizer.transform([preprocess_text(text) for text in sample_texts])
sample_predictions = loaded_model.predict(sample_features)
sample_predictions_labels = label_encoder.inverse_transform(sample_predictions)

print("Sample texts:", sample_texts)
print("Predicted categories:", sample_predictions_labels)