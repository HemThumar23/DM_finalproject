import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the dataset
data = pd.read_csv(r"C:\Users\DELL\Downloads\archive\heart.csv")

# Separate features (X) and target (y)
X = data.iloc[:, :-1].values  # Features (all columns except the last one)
y = data.iloc[:, -1].values  # Target (last column)


# Function to calculate metrics
def calculate_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tp, fn, fp, tn = cm.ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tss = recall - fpr
    hss = (2 * (tp * tn - fp * fn)) / (
            (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    ) if (tp + fn + fp + tn) > 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "TSS": tss,
        "HSS": hss,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "FPR": fpr,
        "FNR": fnr
    }


# Results dictionary
results = {
    'Model': [],
    'Fold': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'TP': [],
    'TN': [],
    'FP': [],
    'FN': [],
    'FPR': [],
    'FNR': [],
    'TSS': [],
    'HSS': []
}

# 10-Fold Cross-Validation Setup for Random Forest
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # Changed to 10 splits
fold = 1

# Random Forest Classifier (Cross-validation)
rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10)  # Hyperparameters changed
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)
    for metric, value in metrics.items():
        results[metric].append(value)
    results['Model'].append('Random Forest')
    results['Fold'].append(fold)
    fold += 1

# Standardize data for LSTM and Naïve Bayes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split for LSTM and Naïve Bayes
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# LSTM Model (modified architecture)
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

lstm_model = Sequential([
    LSTM(128, activation='relu', input_shape=(X_train_lstm.shape[1], 1)),  # Changed number of units and activation
    Dropout(0.3),  # Increased dropout rate
    Dense(64, activation='relu'),
    Dropout(0.3),  # Increased dropout rate
    Dense(1, activation='sigmoid')
])

# Add EarlyStopping callback to stop training early if no improvement
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0,
               callbacks=[early_stopping])

y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int)
metrics = calculate_metrics(y_test, y_pred_lstm)
for metric, value in metrics.items():
    results[metric].append(value)
results['Model'].append('LSTM')
results['Fold'].append('N/A')  # LSTM doesn't use cross-validation

# Naïve Bayes Model (GaussianNB)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
metrics = calculate_metrics(y_test, y_pred_nb)
for metric, value in metrics.items():
    results[metric].append(value)
results['Model'].append('Naïve Bayes')
results['Fold'].append('N/A')  # Naïve Bayes doesn't use cross-validation

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Define output file paths in Documents folder
output_file = "C:\\Users\\DELL\\Downloads\\heart_disease_model_comparison_results.csv"
comparison_output_file = "C:\\Users\\DELL\\Downloads\\heart_disease_model_comparison_summary.csv"

# Save results to CSV with unique filename if needed
if os.path.exists(output_file):
    base, ext = os.path.splitext(output_file)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    output_file = f"{base}_{counter}{ext}"

# Save to CSV with UTF-8 encoding
results_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Results saved to {os.path.abspath(output_file)}")

# Print the results
print(results_df)

# Calculate average metrics for each model and compare
average_metrics = results_df.groupby('Model').mean(numeric_only=True)

# Compare models based on average metrics
comparison = {
    "Model": [],
    "Average Accuracy": [],
    "Average F1-Score": [],
    "Average TSS": [],
    "Average HSS": []
}

for model in average_metrics.index:
    comparison["Model"].append(model)
    comparison["Average Accuracy"].append(average_metrics.loc[model, "Accuracy"])
    comparison["Average F1-Score"].append(average_metrics.loc[model, "F1-Score"])
    comparison["Average TSS"].append(average_metrics.loc[model, "TSS"])
    comparison["Average HSS"].append(average_metrics.loc[model, "HSS"])

comparison_df = pd.DataFrame(comparison)

# Print model comparison
print("\nModel Comparison:")
print(comparison_df)

# Save model comparison to CSV with UTF-8 encoding
comparison_df.to_csv(comparison_output_file, index=False, encoding='utf-8')
print(f"Comparison summary saved to {os.path.abspath(comparison_output_file)}")

# Identify the best model based on Average Accuracy (or any other metric you prefer)
best_model_row = comparison_df.loc[comparison_df['Average Accuracy'].idxmax()]
best_model_name = best_model_row['Model']
best_model_accuracy = best_model_row['Average Accuracy']

# Add a line to print which model is best
best_model_summary = f"\nThe best model for this dataset based on Average Accuracy is: {best_model_name} with an Accuracy of {best_model_accuracy:.4f}."
print(best_model_summary)

# Save models
joblib.dump(rf_model, "C:\\Users\\DELL\\Downloads\\rf_model_heart.pkl")
lstm_model.save("C:\\Users\\DELL\\Downloads\\lstm_model_heart.keras")
joblib.dump(nb_model, "C:\\Users\\DELL\\Downloads\\nb_model_heart.pkl")
