import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==================== Load Naive Bayes Model ====================

# Load Naive Bayes model and vectorizer
nb_model = joblib.load("models/NaiveBayes/naive_bayes_model.joblib")
nb_vectorizer = joblib.load("models/NaiveBayes/nvb_vectorizer.joblib")

# Load Naive Bayes metrics and reports
metrics_df = pd.read_csv("models/NaiveBayes/nvb_metrics.csv")
print("Naive Bayes Model Metrics:\n", metrics_df)

# Load the classification report for Naive Bayes
with open("models/NaiveBayes/nvb_classi_report.json", "r") as f:
    nb_report = json.load(f)

nb_report_df = pd.DataFrame(nb_report).transpose()
print("Naive Bayes Classification Report:\n", nb_report_df)

# Load the training time for Naive Bayes
with open("models/NaiveBayes/nvb_train_time.txt", "r") as f:
    nb_training_time = f.read()

print("Naive Bayes Training Time:\n", nb_training_time)

# Display Naive Bayes confusion matrix and other plots
# Naive Bayes Confusion Matrix Plot
nb_confusion_matrix_img = mpimg.imread("plots/NaiveBayes/confusion_matrix.png")
plt.imshow(nb_confusion_matrix_img)
plt.axis("off")
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# Naive Bayes Category Distribution Plot
nb_category_distribution_img = mpimg.imread(
    "plots/NaiveBayes/category_distribution.png"
)
plt.imshow(nb_category_distribution_img)
plt.axis("off")
plt.title("Naive Bayes Category Distribution")
plt.show()

# Naive Bayes Performance Metrics Plot
nb_performance_metrics_img = mpimg.imread("plots/NaiveBayes/performance_metrics.png")
plt.imshow(nb_performance_metrics_img)
plt.axis("off")
plt.title("Naive Bayes Performance Metrics")
plt.show()

# ==================== Load SVM Model ====================

# Load SVM model and vectorizer
svm_model = joblib.load("models/SVM/svm_model.joblib")
svm_vectorizer = joblib.load("models/SVM/tfidf_title.joblib")

print("SVM model and vectorizer have been loaded successfully.")

# Example: using the loaded SVM model for predictions
sample_texts = ["This is a test news title", "New advancements in AI technology"]
sample_features = svm_vectorizer.transform(sample_texts)
sample_predictions = svm_model.predict(sample_features)

# Load SVM metrics if available
svm_metrics_df = pd.read_csv("models/SVM/svm_metrics.csv")
print("SVM Model Metrics:\n", svm_metrics_df)

# Load the classification report for SVM
with open("models/SVM/svm_classi_report.json", "r") as f:
    svm_report = json.load(f)

svm_report_df = pd.DataFrame(svm_report).transpose()
print("SVM Classification Report:\n", svm_report_df)

# Load the training time for SVM
with open("models/SVM/svm_train_time.txt", "r") as f:
    svm_training_time = f.read()

print("SVM Training Time:\n", svm_training_time)

# Display SVM confusion matrix and other plots
# SVM Confusion Matrix Plot
svm_confusion_matrix_img = mpimg.imread("plots/SVM/Confusion_Matrix.png")
plt.imshow(svm_confusion_matrix_img)
plt.axis("off")
plt.title("SVM Confusion Matrix")
plt.show()

# SVM Performance Metrics Plot
svm_performance_metrics_img = mpimg.imread("plots/SVM/Performance_Metrics.png")
plt.imshow(svm_performance_metrics_img)
plt.axis("off")
plt.title("SVM Performance Metrics")
plt.show()

# ==================== Display SVM Predictions ====================

# Print sample predictions for SVM
print("Sample texts:", sample_texts)
print("Predicted categories (SVM):", sample_predictions)
