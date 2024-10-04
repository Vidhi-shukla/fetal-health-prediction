# Importing Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tabulate import tabulate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from joblib import dump
from joblib import load


# Load preprocessed data
X_train = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_train.pkl")
X_test = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_test.pkl")
y_train = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_train.pkl")
y_test = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_test.pkl")
X = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X.pkl")
y = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y.pkl")    

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    'kernel': ['rbf', 'linear', 'poly']  # Type of SVM
}

# Initialize SVC model
svc = SVC(class_weight='balanced', random_state=42, probability=True)

# GridSearchCV setup
grid = GridSearchCV(svc, param_grid, refit=True, verbose=2, scoring='f1_weighted', cv=5)
grid.fit(X_train, y_train)

# Displaying the best parameters from GridSearchCV
print("Best parameters found: ", grid.best_params_)

# Using the best model from GridSearchCV to make predictions
best_svc = grid.best_estimator_
y_pred = best_svc.predict(X_test)

# Saving the model
dump(best_svc, 'C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/best_svc_model.pkl')

# Using saved model
# Load the model from the file
loaded_model = load('/Users/pranavkhot/Documents/Fetal Health Project/trained_models/best_svc_model.pkl')

# Use the loaded model to make predictions
y_pred_loaded = loaded_model.predict(X_test)

# Calculating accuracy and f1-score
svc_accuracy_test = accuracy_score(y_test, y_pred) * 100
svc_accuracy_train = accuracy_score(y_train, best_svc.predict(X_train)) * 100
svc_f1 = f1_score(y_test, y_pred, average='weighted') * 100
svc_accuracy_overall = (svc_accuracy_test + svc_accuracy_train) / 2

# SVC Model metrics
# Create a dictionary to store different metrics
metrics_dict_svc = {
    "Model": ["Support Vector Classifier (Optimized)"],
    "Train Accuracy": [f"{svc_accuracy_train:.3f}%"],
    "Test Accuracy": [f"{svc_accuracy_test:.3f}%"],
    "Overall Accuracy": [f"{svc_accuracy_overall:.3f}%"],
    "F1-Score": [f"{svc_f1:.3f}%"],
}

# Convert the dictionary to a DataFrame for display
metrics_df_svc = pd.DataFrame(metrics_dict_svc)
print("\nSupport Vector Classifier Metrics (After Optimization):")
print(tabulate(metrics_df_svc, headers='keys', tablefmt='fancy_grid', showindex=False))

# Save the DataFrame to a CSV file
metrics_df_svc.to_csv("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/SVC/svc_metrics.csv", index=False)


# Classification report
class_names = ['Normal', 'Suspect', 'Pathologic']
report = classification_report(y_test, y_pred, output_dict=True, target_names=class_names)
df_report = pd.DataFrame(report).transpose()
print("\nClassification Report:")
print(tabulate(df_report, headers='keys', tablefmt='fancy_grid'))

# Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Learning Curve 
train_sizes, train_scores, test_scores = learning_curve(best_svc, X, y, cv=5, scoring='f1_weighted')

# Compute the mean and standard deviation for the training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Compute the mean and standard deviation for the test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
plt.title('Learning Curve for Optimized SVC')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score (Weighted)')
plt.legend(loc='best')
plt.grid()
plt.show()

unique_labels = np.unique(y_test)
print("Unique labels in y_test:", unique_labels)



