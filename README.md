# Fetal Health Assessment Using Machine Learning

## Project Overview

**Fetal Health Prediction** is a machine learning-based tool aimed at improving the accuracy of fetal health assessments using **Cardiotocography (CTG)** data. CTG measures fetal heart rate and uterine contractions during pregnancy, and while it provides crucial insights into fetal well-being, its manual interpretation can be subjective and error-prone.

This project automates the analysis of CTG data using machine learning models, which classify fetal health into three categories:
- **Normal**: The fetus is healthy with no signs of distress.
- **Suspect**: Potential signs of mild distress, requiring observation.
- **Pathological**: Clear signs of fetal distress, requiring immediate medical intervention.

## Models Included

The project implements and compares the performance of various machine learning models, including:
- Logistic Regression (`LOG_REG.py`)
- Support Vector Classifier (SVC) (`SVC.py`)
- K-Nearest Neighbors (KNN) (`KNN.py`)
- LightGBM (`LGB.py`)
- Convolutional Neural Network (CNN) (`CNN.py`)
- Recurrent Neural Network (RNN) (`RNN.py`)
- Long Short-Term Memory (LSTM) (`LSTM.py`)
- CatBoost Classifier (`CAT.py`)

Each model is evaluated on multiple metrics such as accuracy, precision, recall, and F1-score. The performance results help determine which model best predicts fetal health status, assisting healthcare professionals in making more informed and accurate decisions.

## Motivation

Traditional fetal health assessment using CTG can be prone to variability between different clinicians. By applying machine learning techniques, this project provides an objective, data-driven tool for assessing fetal health. The goal is to:
- Improve diagnostic accuracy
- Enable faster, automated analysis of CTG data
- Facilitate early detection of potential fetal health complications

## How It Works

1. **Data Collection**: CTG data is collected, recording vital signs such as fetal heart rate and uterine contractions.
2. **Preprocessing**: The data is cleaned, normalized, and relevant features are extracted to ensure optimal model performance.
3. **Modeling**: The models are trained to predict the fetal health category (Normal, Suspect, Pathological) based on the input data.
4. **Prediction**: Once trained, the models can predict fetal health status from new CTG data, providing timely and actionable insights for clinicians.

## Results

By comparing the models, the project identifies the best-performing algorithms based on accuracy and other key metrics. The `RESULT.py` script provides a comprehensive analysis of the model results, highlighting the most effective approaches for fetal health classification.


