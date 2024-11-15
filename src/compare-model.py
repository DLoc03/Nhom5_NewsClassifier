import json
import matplotlib.pyplot as plt
import pandas as pd

# Đường dẫn tới các file metrics của hai mô hình
naive_bayes_metrics_path = "models/NaiveBayes/nvb_metrics.csv"
svm_metrics_path = "models/SVM/metrics.csv"

# Load metrics từ các file CSV
naive_bayes_metrics = pd.read_csv(naive_bayes_metrics_path)
svm_metrics = pd.read_csv(svm_metrics_path)

# Chỉ lấy các giá trị của Accuracy và Training Time
metrics_names = [
    "Accuracy",
    "Training Time (seconds)",
]

# Tạo một DataFrame để chứa dữ liệu so sánh
comparison_data = {
    "Metrics": metrics_names,
    "Naive Bayes": [
        naive_bayes_metrics["Accuracy"][0],
        naive_bayes_metrics["Training Time (seconds)"][0],
    ],
    "SVM": [
        svm_metrics["Accuracy"][0],
        svm_metrics["Training time"][0],
    ],
}

df_comparison = pd.DataFrame(comparison_data)

# In ra giá trị của các metrics
print("Metrics Comparison between Naive Bayes and SVM:")
print(df_comparison)

# Vẽ biểu đồ so sánh
plt.figure(figsize=(10, 6))
df_comparison.set_index("Metrics").plot(
    kind="bar", figsize=(10, 6), color=["#1f77b4", "#ff7f0e"]
)
plt.title("Performance Metrics Comparison between Naive Bayes and SVM")
plt.ylabel("Score")
plt.xlabel("Metrics")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/comparison_metrics.png")
plt.show()
