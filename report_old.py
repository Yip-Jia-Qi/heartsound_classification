import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Extract targets and predictions
    targets = data['Targets']
    predictions = data['Predictions']

    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')

    # Generate classification report
    report = classification_report(targets, predictions)

    # Save metrics as a text file
    txt_file = csv_file[:-4] + '.txt'
    with open(txt_file, 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 score: {f1}\n')
        f.write('\nClassification Report:\n')
        f.write(report)

    # Compute and save classification matrix as a JPG image
    matrix = pd.crosstab(targets, predictions, rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize=(10, 7))
    plt.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(matrix.columns)), matrix.columns)
    plt.yticks(np.arange(len(matrix.index)), matrix.index)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(csv_file[:-4] + '.jpg')

    print(f"Metrics and classification matrix saved successfully.")
