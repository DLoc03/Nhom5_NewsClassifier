import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

# Tải lại mô hình Naive Bayes và Vectorizer
nb_model_path = "models/NaiveBayes/naive_bayes_model.joblib"
vectorizer_path = "models/NaiveBayes/nvb_vectorizer.joblib"
nb_model = joblib.load(nb_model_path)
vectorizer = joblib.load(vectorizer_path)

# Tải lại dữ liệu và thực hiện các bước tiền xử lý cần thiết
import pandas as pd
import re

# Đọc dữ liệu
dtsName = "./data/uci-news-aggregator.csv"
data = pd.read_csv(dtsName)
data = data.dropna(subset=["TITLE", "CATEGORY"])


# Tiền xử lý dữ liệu
def remove_special_chars(text):
    if isinstance(text, str):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower().strip())
    return text


data["TITLE"] = data["TITLE"].apply(remove_special_chars)
data["CATEGORY"] = data["CATEGORY"].map(
    {
        "b": "Business",
        "t": "Science and technology",
        "e": "Entertainment",
        "m": "Health",
    }
)

# Tách dữ liệu thành X, y
X = data["TITLE"]
y = data["CATEGORY"]

# Vector hóa dữ liệu
X_tfidf = vectorizer.transform(X)

# Chia dữ liệu thành train và test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Tính toán ROC cho Naive Bayes
y_test_binarized = label_binarize(y_test, classes=list(set(y_test)))
nb_pred_proba = nb_model.predict_proba(X_test)
fpr_nb, tpr_nb, _ = roc_curve(y_test_binarized.ravel(), nb_pred_proba.ravel())
auc_nb = roc_auc_score(y_test_binarized, nb_pred_proba, multi_class="ovr")

# Tải lại mô hình SVM
svm_model_path = "models/SVM/svm_model.joblib"
svm_model = joblib.load(svm_model_path)

# Tính toán ROC cho SVM
if hasattr(svm_model, "decision_function"):
    svm_pred_proba = svm_model.decision_function(X_test)
    fpr_svm, tpr_svm, _ = roc_curve(y_test_binarized.ravel(), svm_pred_proba.ravel())
    auc_svm = roc_auc_score(y_test_binarized, svm_pred_proba, multi_class="ovr")
else:
    print("SVM model does not support probability estimation for ROC.")

# Vẽ biểu đồ ROC
plt.figure(figsize=(10, 6))

# ROC cho Naive Bayes
plt.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC = {auc_nb:.2f})")

# ROC cho SVM
if hasattr(svm_model, "decision_function"):
    plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.2f})")

# Đường chéo (Random Classifier)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")

# Định dạng biểu đồ
plt.title("ROC Curve for Naive Bayes and SVM")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid(alpha=0.5)

# Lưu biểu đồ
plt.savefig("plots/ROC_comparison.png")
plt.show()
