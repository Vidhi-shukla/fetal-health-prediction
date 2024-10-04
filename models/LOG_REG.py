# Importing Necessary Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, learning_curve
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from joblib import dump

# Load preprocessed data
X_train = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_train.pkl")
X_test = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_test.pkl")
y_train = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_train.pkl")
y_test = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_test.pkl")
X = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X.pkl")
y = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y.pkl")    

# Adjusting the class labels to start from 0
y_train_adjusted = y_train - 1
y_test_adjusted = y_test - 1

# Initialize Logistic Regression pipeline
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))

# Train the model
pipe.fit(X_train, y_train_adjusted)

# Predictions 
y_pred_train_logistic = pipe.predict(X_train)
y_pred_test_logistic = pipe.predict(X_test)

# Metrics calculation
logistic_accuracy_test = accuracy_score(y_test_adjusted, y_pred_test_logistic) * 100
logistic_accuracy_train = accuracy_score(y_train_adjusted, y_pred_train_logistic) * 100
logistic_f1 = f1_score(y_test_adjusted, y_pred_test_logistic, average='weighted') * 100
logistic_accuracy_overall = (logistic_accuracy_test + logistic_accuracy_train) / 2

# Create a dictionary to store different metrics
metrics_dict_logistic = {
    "Model": ["Logistic Regression"],
    "Train Accuracy": [f"{logistic_accuracy_train:.3f}%"],
    "Test Accuracy": [f"{logistic_accuracy_test:.3f}%"],
    "Overall Accuracy": [f"{logistic_accuracy_overall:.3f}%"],
    "F1-Score": [f"{logistic_f1:.3f}%"],
}

# Convert the dictionary to a DataFrame
metrics_df_logistic = pd.DataFrame(metrics_dict_logistic)

# Define the class names
class_names = ['Normal', 'Suspect', 'Pathologic']

# Save the DataFrame to a CSV file
metrics_df_logistic.to_csv("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/LOG_REG/log_metrics.csv", index=False)

# Classification report
report = classification_report(y_test_adjusted, y_pred_test_logistic, output_dict=True, target_names=class_names)
df_report = pd.DataFrame(report).transpose()
print("\nClassification Report:")
print(tabulate(df_report, headers='keys', tablefmt='fancy_grid'))
print("\n")

# Confusion matrix
cm = confusion_matrix(y_test_adjusted, y_pred_test_logistic)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Learning Curve
train_sizes, train_scores, valid_scores = learning_curve(pipe, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color="r", label="Training score")
plt.plot(train_sizes, valid_scores.mean(axis=1), 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.title("Learning Curves (Logistic Regression)")
plt.legend(loc="best")
plt.grid()
plt.show()

# Save the trained model to a file
import pickle

model_filename = 'C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/best_log_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(pipe, model_file)

print(f"Model trained and saved to file: {model_filename}")


