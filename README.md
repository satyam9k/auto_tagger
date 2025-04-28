# Stack Overflow Question Auto-Tagger

This project implements a system to automatically assign relevant tags (multi-label classification) to Stack Overflow question posts based on their title and body content.

## Overview

The system processes question text using basic NLP techniques and extracts features using TF-IDF. It then trains and compares multi-label classification models to predict appropriate tags.

## Features

*   **Text Preprocessing:** Cleans question titles and bodies (lowercase, removes punctuation/HTML/stopwords).
*   **Feature Extraction:** Uses TF-IDF (Term Frequency-Inverse Document Frequency) with uni-grams and bi-grams.
*   **Tag Encoding:** Handles multiple tags per question using `MultiLabelBinarizer`.
*   **Model Comparison:** Trains and evaluates:
    *   Logistic Regression (with `OneVsRestClassifier`)
    *   Linear Support Vector Classifier (LinearSVC) (with `OneVsRestClassifier`)
*   **Evaluation:** Measures performance using metrics like F1-score (Micro, Weighted), Precision, Recall, Subset Accuracy, and Hamming Loss.

## Results Summary

Based on evaluation with generated data, LinearSVC provided better results:

*   **Logistic Regression Micro F1:** 0.7462 (Train: ~2.5s, Predict: ~0.1s)
*   **Linear SVC Micro F1:** **0.8003** (Train: ~2.3s, Predict: ~0.1s)

## Example Prediction (using LinearSVC)

*   **Input Question (Processed Text):** `'reading specific columns csv using pandas load column column large csv file pandas dataframe save memory'`
*   **Predicted Tags:** `['dataframe', 'pandas', 'python']`

