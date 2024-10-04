# Importing Necessary Libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from joblib import dump
warnings.filterwarnings('ignore')

# Load preprocessed data
X_train = pd.read_pickle("C:/UsersVidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_train.pkl")
X_test = pd.read_pickle("C:/UsersVidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_test.pkl")
y_train = pd.read_pickle("C:/UsersVidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_train.pkl")
y_test = pd.read_pickle("C:/UsersVidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_test.pkl")
X = pd.read_pickle("C:/UsersVidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X.pkl")
y = pd.read_pickle("C:/UsersVidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y.pkl")    

# Define the parameter range for 
param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40, 50]
}

# Custom scorer for grid search
f1_scorer = make_scorer(f1_score, average='weighted')

# Instantiate the grid search model with StratifiedKFold``
strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=strat_k_fold, scoring=f1_scorer, verbose=2, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

#  Train the classifier with the best parameters
knn = KNeighborsClassifier(**best_params)
knn.fit(X_train, y_train)

# Make predictions
y_pred_train_knn = knn.predict(X_train)
y_pred_test_knn = knn.predict(X_test)

# Evaluation metrics calculations
# Calculate accuracy, f1-score
knn_accuracy_test = accuracy_score(y_test, y_pred_test_knn) * 100
knn_accuracy_train = accuracy_score(y_train, y_pred_train_knn) * 100
knn_f1 = f1_score(y_test, y_pred_test_knn, average='weighted') * 100
knn_accuracy_overall = (knn_accuracy_test + knn_accuracy_train) / 2

# Create a dictionary to store different metrics
metrics_dict_knn = {
    "Model": ["K-Nearest Neighbors"],
    "Train Accuracy": [f"{knn_accuracy_train:.3f}%"],
    "Test Accuracy": [f"{knn_accuracy_test:.3f}%"],
    "Overall Accuracy": [f"{knn_accuracy_overall:.3f}%"],
    "F1-Score": [f"{knn_f1:.3f}%"],
}
# Convert the dictionary to a DataFrame
metrics_df_knn = pd.DataFrame(metrics_dict_knn)

# Print the results
print("\nK-Nearest Neighbors Metrics:")
print(tabulate(metrics_df_knn, headers='keys', tablefmt='fancy_grid', showindex=False))


# Save the DataFrame to a CSV file
metrics_df_knn.to_csv("C:/UsersVidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/KNN/knn_metrics.csv", index=False)

# Classification report
# Print the classification report
class_names = ['Normal', 'Suspect', 'Pathologic']
report = classification_report(y_test, y_pred_test_knn, output_dict=True, target_names=class_names)
df_report = pd.DataFrame(report).transpose()
print("\nClassification Report:")
print(tabulate(df_report, headers='keys', tablefmt='fancy_grid'))
print("\n")

# Confusion Matrix
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred_test_knn)
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
# Plot the learning curve
train_sizes, train_scores, valid_scores = learning_curve(knn, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=strat_k_fold)
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color="r", label="Training score")
plt.plot(train_sizes, valid_scores.mean(axis=1), 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.title("Learning Curves (K-Nearest Neighbors)")
plt.legend(loc="best")
plt.grid()
plt.show()

# Save the model to a file
import pickle

# Save the trained model to a file
with open('C:/UsersVidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/best_knn_model.pkl', 'wb') as model_file:
    pickle.dump(knn, model_file)

print("Model trained and saved to file.")


