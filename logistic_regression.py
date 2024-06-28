import time

# Record the start time
start_time = time.time()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import seaborn as sns
import joblib

# Load the dataset
df = pd.read_csv('datasets/model_data/truncated_train_data.csv')

X = df['full_text']
y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Apply SMOTE on the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_vec, y_train)

# Define and train the logistic regression model with specified parameters
model = LogisticRegression()
model.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_vec)[:, 1])

# Save the vectorizer
joblib.dump(vectorizer, 'app_artifacts/vectorizer.joblib')

# Save the model
joblib.dump(model, 'app_artifacts/model.joblib')

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Start an MLflow run
with mlflow.start_run(run_name='Logistic Regression Truncated data'):
    # Log parameters
    mlflow.log_param('model', 'LogisticRegression')
    mlflow.log_param('C', 7)
    mlflow.log_param('penalty', 'l2')
    mlflow.log_param('solver', 'lbfgs')
    
    # Log detailed classification report metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)
    
    # Log model
    mlflow.sklearn.log_model(model, 'model')

    # Log the classification report as a text file
    with open('classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact('classification_report.txt')
    
    # Log the confusion matrix as an artifact
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    
    # Plot and log the ROC curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_vec)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    mlflow.log_artifact('roc_curve.png')

    # Log classification report as a text file
    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact("classification_report.txt")

print('MLflow run logged as Logistic Regression with specified parameters.')

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
