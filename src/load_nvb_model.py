import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = joblib.load("save_model/naive_bayes_model.joblib")
vectorizer = joblib.load("save_model/nvb_vectorizer.joblib")

metrics_df = pd.read_csv("save_model/nvb_metrics.csv")
print("Model Metrics:\n", metrics_df)

with open("save_model/nvb_classi_report.json", "r") as f:
    report = json.load(f)

report_df = pd.DataFrame(report).transpose()

print("Classification Report:\n", report_df)

with open("save_model/nvb_train_time.txt", "r") as f:
    training_time = f.read()

print("Training Time:\n", training_time)

confusion_matrix_img = mpimg.imread("chart/confusion_matrix.png")
plt.imshow(confusion_matrix_img)
plt.axis("off")
plt.title("Confusion Matrix")
plt.show()

category_distribution_img = mpimg.imread("chart/category_distribution.png")
plt.imshow(category_distribution_img)
plt.axis("off")
plt.title("Category Distribution")
plt.show()

performance_metrics_img = mpimg.imread("chart/performance_metrics.png")
plt.imshow(performance_metrics_img)
plt.axis("off")
plt.title("Performance Metrics")
plt.show()
