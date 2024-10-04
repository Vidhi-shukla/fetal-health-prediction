# Importing Necessary Libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from sklearn.model_selection import train_test_split

# Importing Necessary Libraries
data = pd.read_csv("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/processed_data.csv")

# Feature Selection
# Prepare the data (Feature Selection)
target_data = data.drop(['fetal_health'], axis=1)
selector = SelectKBest(f_classif, k=10)
X_important = selector.fit_transform(target_data, data['fetal_health'])

# Get the mask of selected features
mask = selector.get_support()

# Get the names of the important features
important_feature_names = target_data.columns[mask]

# Create a DataFrame with feature names and their scores
important_features_scores = pd.DataFrame({'Important Features': important_feature_names, 'Scores': selector.scores_[mask]})

# Sort the DataFrame by scores in descending order
important_features_scores = important_features_scores.sort_values(by='Scores', ascending=False)

# Reset index of the DataFrame
important_features_scores.reset_index(drop=True, inplace=True)

# Display the DataFrame
print(tabulate(important_features_scores, headers='keys', tablefmt='fancy_grid', showindex=False))

# Standardizing the features
# Select the top 10 features for scaling
A = data[important_feature_names]
y = data["fetal_health"]
col_names = list(A.columns)

# Standard Scaler
std_scaler = StandardScaler()
X = std_scaler.fit_transform(A)
X = pd.DataFrame(X, columns=col_names)

# Get descriptive statistics of the scaled features
desc_std = X.describe().T

# Set the index name of the DataFrame
desc_std.index.name = 'Feature'

# Display the DataFrame
print("\nStandard Scaler:")
print(tabulate(desc_std, headers='keys', tablefmt='fancy_grid'))

import pickle

with open('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y.pkl', 'wb') as f:
    pickle.dump(y, f)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=45)

# Print the shapes of the training and test sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
import pickle

with open('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
    
with open('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
    
with open('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)



