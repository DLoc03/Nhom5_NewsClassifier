import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==================== Load Naive Bayes Model ====================
# Load Naive Bayes model and vectorizer
try:
    nb_model = joblib.load("models/NaiveBayes/naive_bayes_model.joblib")
    nb_vectorizer = joblib.load("models/NaiveBayes/nvb_vectorizer.joblib")
    print("Naive Bayes model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading Naive Bayes model/vectorizer: {e}")

# Load Naive Bayes metrics
try:
    metrics_df = pd.read_csv("models/NaiveBayes/nvb_metrics.csv")
    print("Naive Bayes Model Metrics:\n", metrics_df)
except FileNotFoundError:
    print("Naive Bayes metrics file not found.")

# Load the classification report for Naive Bayes
try:
    with open("models/NaiveBayes/nvb_classi_report.json", "r") as f:
        nb_report = json.load(f)
    nb_report_df = pd.DataFrame(nb_report).transpose()
    print("Naive Bayes Classification Report:\n", nb_report_df)
except FileNotFoundError:
    print("Naive Bayes classification report file not found.")

# Load the training time for Naive Bayes
try:
    with open("models/NaiveBayes/nvb_train_time.txt", "r") as f:
        nb_training_time = f.read()
    print("Naive Bayes Training Time:\n", nb_training_time)
except FileNotFoundError:
    print("Naive Bayes training time file not found.")


# Display Naive Bayes confusion matrix and other plots
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
    "plots/NaiveBayes/performance_metrics.png", "Naive Bayes Performance Metrics"
)

# ==================== Load SVM Model ====================
# Load SVM model and vectorizer
try:
    svm_model = joblib.load("models/SVM/svm_model.joblib")
    svm_vectorizer = joblib.load("models/SVM/tfidf_title.joblib")
    print("SVM model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading SVM model/vectorizer: {e}")

# Example: using the loaded SVM model for predictions
sample_texts = ["This is a test news title", "New advancements in AI technology"]
try:
    sample_features = svm_vectorizer.transform(sample_texts)
    sample_predictions = svm_model.predict(sample_features)
    print("Sample texts:", sample_texts)
    print("Predicted categories (SVM):", sample_predictions)
except Exception as e:
    print(f"Error during SVM prediction: {e}")

# Load SVM metrics
try:
    with open("models/SVM/metrics.txt", "r") as f:
        svm_metrics = f.read()
    print("SVM Model Metrics:\n", svm_metrics)
except FileNotFoundError:
    print("SVM metrics file not found.")

# Load the classification report for SVM
try:
    with open("models/SVM/classification_report.txt", "r") as f:
        svm_report = json.load(f)
    svm_report_df = pd.DataFrame(svm_report).transpose()
    print("SVM Classification Report:\n", svm_report_df)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading SVM classification report: {e}")

# Display SVM confusion matrix and other plots
display_image("plots/SVM/Confusion_Matrix.png", "SVM Confusion Matrix")
display_image("plots/SVM/Performance_Metrics.png", "SVM Performance Metrics")
