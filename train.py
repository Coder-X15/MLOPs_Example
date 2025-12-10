#! /usr/bin/env python3

# train.py
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train_model():
    # setup MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run() as run:

        # Load dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize model
        model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_pred)

        # log model
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(sk_model = model, 
                                              name="mlp_iris_model", 
                                              signature=signature,
                                              registered_model_name="mlp_iris_model")

        # Log metrics with MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", recall)

        # End MLflow run
        mlflow.end_run()

        # Log accuracy, precision, f1, recall into a file
        with open("metrics.txt", "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"Recall: {recall}\n")

        # Log confusion matrix plot
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=iris.target_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Write model info into a .env file
        with open(".env", "w") as f:
            f.write(f"MODEL_URI={model_info.model_uri}\n")
        mlflow.log_artifact(".env")
        

if __name__ == "__main__":
    train_model()