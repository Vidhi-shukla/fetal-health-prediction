import pandas as pd
from tabulate import tabulate

# Read metrics from CSV files
SVC = pd.read_csv('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/SVC/svc_metrics.csv')
CAT = pd.read_csv('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/CAT/cat_metrics.csv')
LOG = pd.read_csv('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/LOG_REG/log_metrics.csv')
LGB = pd.read_csv('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/LGB/lgb_metrics.csv')
KNN = pd.read_csv('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/KNN/knn_metrics.csv')
LSTM = pd.read_csv('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/LSTM/lstm_metrics.csv')
CNN = pd.read_csv('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/CNN/cnn_metrics.csv')
RNN = pd.read_csv('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/results/RNN/rnn_metrics.csv')

# Concatenate the DataFrames along the row axis
all_metrics = pd.concat([SVC, XGB, CAT, LOG, LVQ, LGB, RF, KNN, EEC, LSTM, ADABOOST, RNN], axis=0)

# Print the combined DataFrame
print("\nAll Model Metrics:")
print(tabulate(all_metrics, headers='keys', tablefmt='fancy_grid', showindex=False))



