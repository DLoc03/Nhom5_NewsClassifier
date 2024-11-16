import json
import matplotlib.pyplot as plt
import pandas as pd

# Đường dẫn tới các file metrics của hai mô hình
naive_bayes_metrics_path = "models/NaiveBayes/nvb_metrics.csv"
svm_metrics_path = "models/SVM/metrics.csv"

# Load metrics từ các file CSV
naive_bayes_metrics = pd.read_csv(naive_bayes_metrics_path)
svm_metrics = pd.read_csv(svm_metrics_path)

# Tạo một DataFrame để chứa dữ liệu so sánh các metrics
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
        naive_bayes_metrics["Accuracy"][0],
        naive_bayes_metrics["F1-Score (Macro)"][0],
        naive_bayes_metrics["F1-Score (Micro)"][0],
        naive_bayes_metrics["Precision (Macro)"][0],
        naive_bayes_metrics["Recall (Macro)"][0],
        naive_bayes_metrics["Training Time (seconds)"][0],
    ],
    "SVM": [
        svm_metrics["Accuracy"][0],
        svm_metrics["F1 Score (Macro)"][0],
        svm_metrics["F1 Score (Micro)"][0],
        svm_metrics["Precision"][0],
        svm_metrics["Recall"][0],
        svm_metrics["Training time"][0],
    ],
}

df_comparison = pd.DataFrame(comparison_data)

# In ra giá trị của các metrics
print("Metrics Comparison between Naive Bayes and SVM:")
print(df_comparison)

# Vẽ biểu đồ so sánh
plt.figure(figsize=(12, 8))
df_comparison.set_index("Metrics").plot(
    kind="bar", figsize=(12, 8), color=["#1f77b4", "#ff7f0e"]
)
plt.title("Performance Metrics Comparison between Naive Bayes and SVM")
plt.ylabel("Score")
plt.xlabel("Metrics")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/comparison_metrics.png")
plt.show()
