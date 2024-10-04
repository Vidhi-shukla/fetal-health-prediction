# Importing Necessary Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tabulate import tabulate

# Load preprocessed data
X_train = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_train.pkl")
X_test = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_test.pkl")
y_train = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_train.pkl")
y_test = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_test.pkl")
X = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X.pkl")
y = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y.pkl")    

# Create a LightGBM classifier
lgbm = LGBMClassifier(random_state=42)

# Train the classifier
lgbm.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred_train_lgbm = lgbm.predict(X_train)
y_pred_test_lgbm = lgbm.predict(X_test)

# Evaluation metrics calculations
# Calculate accuracy, f1-score
lgbm_accuracy_test = accuracy_score(y_test, y_pred_test_lgbm) * 100
lgbm_accuracy_train = accuracy_score(y_train, y_pred_train_lgbm) * 100
lgbm_f1 = f1_score(y_test, y_pred_test_lgbm, average='weighted') * 100
lgbm_accuracy_overall = (lgbm_accuracy_test + lgbm_accuracy_train) / 2

# Create a dictionary to store different metrics
metrics_dict_lgbm = {
    "Model": ["LightGBM"],
    "Train Accuracy": [f"{lgbm_accuracy_train:.3f}%"],
    "Test Accuracy": [f"{lgbm_accuracy_test:.3f}%"],
    "Overall Accuracy": [f"{lgbm_accuracy_overall:.3f}%"],
    "F1-Score": [f"{lgbm_f1:.3f}%"]
}

# Convert the dictionary to a DataFrame
metrics_df_lgbm = pd.DataFrame(metrics_dict_lgbm)
print("\nLGB Metrics:")
print(tabulate(metrics_df_lgbm, headers='keys', tablefmt='fancy_grid', showindex=False))

# Save the DataFrame to a CSV file
metrics_df_lgbm.to_csv("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/LGB/lgb_metrics.csv", index=False)

# Classification report
class_names = ['Normal', 'Suspect', 'Pathologic']
report = classification_report(y_test, y_pred_test_lgbm, output_dict=True, target_names=class_names)
df_report = pd.DataFrame(report).transpose()
print("\nClassification Report:")
print(tabulate(df_report, headers='keys', tablefmt='fancy_grid'))
print("\n")

# Confusion Matrix
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred_test_lgbm)

# Convert the confusion matrix to a DataFrame
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
print("\n")

# Learning Curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot the learning curve
title = "Learning Curves (LightGBM)"
cv = 5  # Number of folds for cross-validation
plot_learning_curve(lgbm, title, X_train, y_train, cv=cv, n_jobs=-1)
plt.show()

# Save the trained model to a file
import pickle

with open('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/best_lgb_model.pkl', 'wb') as model_file:
    pickle.dump(lgbm, model_file)

print("Model trained and saved to file.")


