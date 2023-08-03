import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


def calculate_metrics(csv_file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the target and prediction columns from the DataFrame
    targets = df['Targets']
    predictions = df['Predictions']

    # Calculate performance metrics
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='weighted')
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')

    # Generate classification report
    report = classification_report(targets, predictions)

    # Create a confusion matrix
    cm = confusion_matrix(targets, predictions)

    # Save metrics as a text file
    output_file_path = csv_file_path.replace('.csv', '.txt')
    with open(output_file_path, 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write('\nClassification Report:\n')
        f.write(report)

    # Save the confusion matrix as a JPEG image
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Add count values to each cell of the confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', verticalalignment='center')

    # Save the image as a JPEG file
    image_file_path = csv_file_path.replace('.csv', '.jpg')
    plt.savefig(image_file_path, format='jpeg')
    plt.close()

    print('Metrics and confusion matrix saved successfully.')


# # Example usage
# calculate_metrics('path/to/your/file.csv')
