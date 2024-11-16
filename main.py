import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# ==================== Tải mô hình Naive Bayes ====================
# Tải mô hình Naive Bayes và vectorizer
try:
    nb_model = joblib.load("models/NaiveBayes/naive_bayes_model.joblib")
    nb_vectorizer = joblib.load("models/NaiveBayes/nvb_vectorizer.joblib")
    print("Naive Bayes model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading Naive Bayes model/vectorizer: {e}")

# Tải các chỉ số mô hình Naive Bayes
try:
    metrics_df = pd.read_csv("models/NaiveBayes/nvb_metrics.csv")
    print("Naive Bayes Model Metrics:\n", metrics_df)
except FileNotFoundError:
    print("Naive Bayes metrics file not found.")

# Tải báo cáo phân loại của Naive Bayes
try:
    with open("models/NaiveBayes/nvb_classi_report.json", "r") as f:
        nb_report = json.load(f)
    nb_report_df = pd.DataFrame(nb_report).transpose()
    print("Naive Bayes Classification Report:\n", nb_report_df)
except FileNotFoundError:
    print("Naive Bayes classification report file not found.")


# Hiển thị ma trận nhầm lẫn Naive Bayes và các biểu đồ khác
def display_image(file_path, title):
    try:
        img = mpimg.imread(file_path)
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
        plt.show()
    except FileNotFoundError:
        print(f"{title} image not found at {file_path}.")


display_image("plots/NaiveBayes/confusion_matrix.png", "Naive Bayes Confusion Matrix")
display_image(
    "plots/NaiveBayes/category_distribution.png", "Naive Bayes Category Distribution"
)
display_image(
    "plots/NaiveBayes/metrics_comparison.png", "Naive Bayes Performance Metrics"
)

# ==================== Tải mô hình SVM ====================
# Tải mô hình SVM và vectorizer
try:
    svm_model = joblib.load("models/SVM/svm_model.joblib")
    svm_vectorizer = joblib.load("models/SVM/tfidf_title.joblib")
    print("SVM model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading SVM model/vectorizer: {e}")

# Ví dụ: sử dụng mô hình SVM đã tải để dự đoán
sample_texts = ["This is a test news title", "New advancements in AI technology"]
try:
    sample_features = svm_vectorizer.transform(sample_texts)
    sample_predictions = svm_model.predict(sample_features)
    print("Sample texts:", sample_texts)
    print("Predicted categories (SVM):", sample_predictions)
except Exception as e:
    print(f"Error during SVM prediction: {e}")

# Tải các chỉ số mô hình SVM
try:
    metrics_svm = pd.read_csv("models/SVM/metrics.csv")
    print("SVM Model Metrics:\n", metrics_svm)
except FileNotFoundError:
    print("Naive Bayes metrics file not found.")

# Tải báo cáo phân loại của SVM
try:
    with open("models/SVM/classification_report.json", "r") as f:
        svm_report = json.load(f)
    svm_report_df = pd.DataFrame(svm_report).transpose()
    print("SVM Classification Report:\n", svm_report_df)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading SVM classification report: {e}")

# Hiển thị ma trận nhầm lẫn của SVM và các biểu đồ khác
display_image("plots/SVM/confusion_Matrix.png", "SVM Confusion Matrix")
display_image("plots/SVM/metrics_comparison.png", "SVM Performance Metrics")

# So sánh hiệu suất của Naive Bayes và SVM
comparison_data = {
    "Metrics": [
        "Accuracy",
        "F1-Score (Macro)",
        "F1-Score (Micro)",
        "Precision (Macro)",
        "Recall (Macro)",
        "Training Time (seconds)",
    ],
    "Naive Bayes": [
        metrics_df["Accuracy"][0],
        metrics_df["F1-Score (Macro)"][0],
        metrics_df["F1-Score (Micro)"][0],
        metrics_df["Precision (Macro)"][0],
        metrics_df["Recall (Macro)"][0],
        metrics_df["Training Time (seconds)"][0],
    ],
    "SVM": [
        metrics_svm["Accuracy"][0],
        metrics_svm["F1 Score (Macro)"][0],
        metrics_svm["F1 Score (Micro)"][0],
        metrics_svm["Precision"][0],
        metrics_svm["Recall"][0],
        metrics_svm["Training time"][0],
    ],
}

df_comparison = pd.DataFrame(comparison_data)

# In ra giá trị của các metrics
print("Metrics Comparison between Naive Bayes and SVM:")
print(df_comparison)

display_image("plots/comparison_metrics.png", "Compare Models")
