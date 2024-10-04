import pandas as pd
from joblib import load
import tkinter as tk
from tkinter import messagebox, StringVar
import numpy as np

# Models 
models = {
    "CatBoost": "C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/best_cat_model.pkl",
    "KNN": "C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/best_knn_model.pkl",
    "LightGBM": "C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/best_lgb_model.pkl",
    "Logistic Regression": "C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/best_log_model.pkl",
    "SVC": "C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/best_svc_model.pkl",
    "CNN": "C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/CNN.pkl",
    "LSTM": "C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/LSTM.pkl",
    "RNN": "C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/trained_models/RNN.pkl"
}

# Selected features 
features = [
    "Baseline Value", "Accelerations", "Prolongued Decelerations",
    "Abnormal Short Term Variability", "Mean Value of Short Term Variability",
    "Percentage of Time with Abnormal Long Term Variability",
    "Histogram Mode", "Histogram Mean", "Histogram Median", "Histogram Variance"
]

# Original feature names as used in the model
original_feature_names = [
    "baseline value", "accelerations", "prolongued_decelerations",
    "abnormal_short_term_variability", "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance"
]

def predict_fetal_health():
    model_path = models[model_var.get()]
    model = load(model_path)

    mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}
    if model_var.get() == "LVQ":
        mapping = {0: 'Normal', 1: 'Suspect', 2: 'Pathologic'}

    try:
        user_input = [float(entries[feature].get()) for feature in original_feature_names]
    except ValueError as e:
        messagebox.showerror("Error", "Invalid input. Please enter valid numerical values for all features.")
        return

    try:
        if model_var.get() in ["CNN", "LSTM", "RNN"]:
            # Adjust reshape based on model type
            if model_var.get() in ["LSTM", "RNN"]:
                # For LSTM and RNN, expecting (1, 1, 10) based on previous error context
                input_data = np.array(user_input).reshape((1, 1, len(user_input)))
            elif model_var.get() == "CNN":
                # Assuming CNN expects (1, 10, 1) based on a typical 1D CNN setup for sequences
                input_data = np.array(user_input).reshape((1, len(user_input), 1))
        else:
            input_data = pd.DataFrame([user_input], columns=original_feature_names)
        
        prediction = model.predict(input_data)
        
        if model_var.get() in ["CNN", "LSTM", "RNN"]:
            # Assuming the output needs to be processed for class label extraction
            prediction = np.argmax(prediction, axis=-1)
        
        result.set(f"Predicted Fetal Health: {mapping[int(prediction[0])]}")
    except Exception as e:
        messagebox.showerror("Error", "An error occurred during prediction: " + str(e))

# Create the main window
root = tk.Tk()
root.title("Fetal Health Predictor")

# Create a drop-down menu for model selection
model_var = StringVar()
model_var.set(list(models.keys())[0])  # set the default value

model_menu = tk.OptionMenu(root, model_var, *models.keys())
model_menu.pack(padx=5, pady=5)

# Create a Frame for input fields
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Dictionary to hold the input fields
entries = {}

# Create input fields for each feature
for feature, original_feature in zip(features, original_feature_names):
    label = tk.Label(input_frame, text=f"{feature}:")
    label.grid(row=features.index(feature), column=0, padx=5, pady=5)
    entry = tk.Entry(input_frame)
    entry.grid(row=features.index(feature), column=1, padx=5, pady=5)
    entries[original_feature] = entry

# Create a Frame for the predict button
button_frame = tk.Frame(root)
button_frame.pack()

# Predict button
predict_button = tk.Button(button_frame, text="Predict", command=predict_fetal_health)
predict_button.pack(padx=5, pady=5)

# Result label
mapping = {0: 'Normal', 1: 'Suspect', 2: 'Pathologic'}
result = tk.StringVar()
result.set("Predicted Fetal Health: ")

result_label = tk.Label(root, textvariable=result, font=("Arial", 12))
result_label.pack(padx=5, pady=10)

# Run the main loop
root.mainloop()


