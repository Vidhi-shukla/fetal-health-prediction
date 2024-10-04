# Importing Necessary Libraries
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import learning_curve
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

# Initialize CatBoost model with predefined hyperparameters for tuning
model = CatBoostClassifier(random_state=42, verbose=0)  # verbose=0 for silent training

#  Define the hyperparameters and their possible values
param_grid = {
    'depth': [3, 6, 10],
    'learning_rate': [0.03, 0.1],
    'iterations': [500, 1000]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

#  Fit the model to the training data
grid_search.fit(X_train, y_train_adjusted)

#  Best model
best_model = grid_search.best_estimator_

#  Predictions
y_pred_train_catboost = best_model.predict(X_train)
y_pred_test_catboost = best_model.predict(X_test)
dump(best_model, 'C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/best_cat_model.pkl')

#  Calculate accuracy and f1-score
catboost_accuracy_test = accuracy_score(y_test_adjusted, y_pred_test_catboost) * 100
catboost_accuracy_train = accuracy_score(y_train_adjusted, y_pred_train_catboost) * 100
catboost_f1 = f1_score(y_test_adjusted, y_pred_test_catboost, average='weighted') * 100
catboost_accuracy_overall = (catboost_accuracy_test + catboost_accuracy_train) / 2

# Create a dictionary to store different metrics
metrics_dict_catboost = {
    "Model": ["CatBoost Classifier"],
    "Train Accuracy": [f"{catboost_accuracy_train:.3f}%"],
    "Test Accuracy": [f"{catboost_accuracy_test:.3f}%"],
    "Overall Accuracy": [f"{catboost_accuracy_overall:.3f}%"],
    "F1-Score": [f"{catboost_f1:.3f}%"],
}

# Convert the dictionary to a DataFrame
metrics_df_catboost = pd.DataFrame(metrics_dict_catboost)

# Print metrics
print("\nCatBoost Classifier Metrics:")
print(tabulate(metrics_df_catboost, headers='keys', tablefmt='fancy_grid', showindex=False))

# Save the DataFrame to a CSV file
metrics_df_catboost.to_csv("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/CAT/cat_metrics.csv", index=False)

#  Classification report
class_names = ['Normal', 'Suspect', 'Pathologic']
report = classification_report(y_test_adjusted, y_pred_test_catboost, output_dict=True, target_names=class_names)
df_report = pd.DataFrame(report).transpose()
print("\nClassification Report:")
print(tabulate(df_report, headers='keys', tablefmt='fancy_grid'))

#  Confusion matrix
cm = confusion_matrix(y_test_adjusted, y_pred_test_catboost)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

#  Plot the learning curve
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

title = "Learning Curves (CatBoost)"
cv = 5  # Number of folds for cross-validation
plot_learning_curve(best_model, title, X_train, y_train_adjusted, cv=cv, n_jobs=-1)
plt.show()

import pandas as pd
from joblib import load

# Load the trained CatBoost model
model = load('/Users/pranavkhot/Documents/Fetal Health Project/trained_models/best_cat_model.pkl')

# Feature names
features = [
    "baseline value", "accelerations", "fetal_movement", 
    "uterine_contractions", "light_decelerations", "severe_decelerations", 
    "prolongued_decelerations", "abnormal_short_term_variability", 
    "mean_value_of_short_term_variability", "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability", "histogram_width", 
    "histogram_min", "histogram_max", "histogram_number_of_peaks", 
    "histogram_number_of_zeroes", "histogram_mode", "histogram_mean", 
    "histogram_median", "histogram_variance", "histogram_tendency"
]

# Take user inputs from the terminal
print("Please enter the following fetal health parameters:")

user_input = []
for feature in features:
    value = input(f"{feature}: ")
    user_input.append(float(value))

# Convert the user input into a DataFrame
input_data = pd.DataFrame([user_input], columns=features)

# Make the prediction
prediction = model.predict(input_data)

# Mapping of the predicted class label to the fetal health
mapping = {0: 'Normal', 1: 'Suspect', 2: 'Pathologic'}

# Print the predicted class
print(f"\nPredicted Fetal Health: {mapping[int(prediction[0])]}")

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

# Load the dataset
data = pd.read_csv('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/processed_data.csv')

# Selecting features and target
X = data.drop('fetal_health', axis=1)  # Assuming 'fetal_health' is the target variable
y = data['fetal_health']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balancing the training dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# CatBoost model definition
catboost_model = CatBoostClassifier(random_state=42)

# Hyperparameter Grid
param_grid = {
    'iterations': [500, 1000],
    'depth': [4, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    # You can add more parameters here
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(catboost_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Training the model
grid_search.fit(X_train_balanced, y_train_balanced, verbose=100)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Predictions and evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))



