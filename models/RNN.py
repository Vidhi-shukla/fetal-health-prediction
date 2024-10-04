# Importing Necessary Libraries
from tabulate import tabulate
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load Preprocessed Data
X_train = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_train.pkl")
X_test = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/X_test.pkl")
y_train = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_train.pkl")
y_test = pd.read_pickle("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/y_test.pkl")

# Reshaping data for RNN
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Building the RNN Model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Model Evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.round(y_pred).astype(int).reshape(-1)

accuracy = accuracy_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix and Classification Report
cm = confusion_matrix(y_test, y_pred_classes)
class_names = ['Normal', 'Suspect', 'Pathologic']
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Generate the classification report
report = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)

# Convert the report to a DataFrame
df_report = pd.DataFrame(report).transpose()

# Print the classification report using tabulate
print("\nClassification Report:")
print(tabulate(df_report, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))

# Calculate accuracy, f1-score for RNN
rnn_accuracy_test = accuracy_score(y_test, y_pred_classes) * 100
rnn_accuracy_train = accuracy_score(y_train, np.round(model.predict(X_train)).astype(int).reshape(-1)) * 100
rnn_f1 = f1_score(y_test, y_pred_classes, average='weighted') * 100
rnn_accuracy_overall = (rnn_accuracy_test + rnn_accuracy_train) / 2

# Create a dictionary to store different metrics for RNN
metrics_dict_rnn = {
    "Model": ["Recurrent Neural Network"],
    "Train Accuracy": [f"{rnn_accuracy_train:.3f}%"],
    "Test Accuracy": [f"{rnn_accuracy_test:.3f}%"],
    "Overall Accuracy": [f"{rnn_accuracy_overall:.3f}%"],
    "F1-Score": [f"{rnn_f1:.3f}%"],
}

# Convert the dictionary to a DataFrame for RNN
metrics_df_rnn = pd.DataFrame(metrics_dict_rnn)

# Print the results for RNN
print("\nRecurrent Neural Network Metrics:")
print(tabulate(metrics_df_rnn, headers='keys', tablefmt='fancy_grid', showindex=False))

metrics_df_rnn.to_csv("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/RNN/rnn_metrics.csv", index=False)

print("Metrics saved to CSV file.")


# Plotting Training History
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Save the Model to a File
import pickle

# Save the trained model to a file
with open('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/RNN.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved to file.")


