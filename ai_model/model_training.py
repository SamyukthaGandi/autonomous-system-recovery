import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


class AdvancedModelTrainer:
    def __init__(self, model_dir="ai_model"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # DATA GENERATION
    # ------------------------------------------------------------------
    def generate_realistic_training_data(self, samples=2000):
        print("[INFO] Generating synthetic system metrics...")

        np.random.seed(42)
        X, y = [], []

        # Normal data (70%)
        for _ in range(int(samples * 0.7)):
            X.append([
                np.clip(np.random.normal(35, 10), 0, 100),
                np.clip(np.random.normal(45, 12), 0, 100),
                np.clip(np.random.normal(60, 8), 0, 100),
                np.clip(np.random.normal(15, 5), 0, 100),
                np.clip(np.random.normal(30, 10), 0, 100),
                np.clip(np.random.normal(80, 15), 0, 500),
                np.clip(np.random.normal(20, 8), 0, 100),
                np.clip(np.random.normal(50, 15), 0, 100),
                np.clip(np.random.normal(200, 30), 0, 1000),
                np.clip(np.random.normal(100, 20), 0, 500),
            ])
            y.append(0)

        # Anomalies (30%)
        for _ in range(int(samples * 0.3)):
            X.append([
                np.random.uniform(75, 100),
                np.random.uniform(70, 95),
                np.random.uniform(70, 100),
                np.random.uniform(50, 90),
                np.random.uniform(50, 90),
                np.random.uniform(150, 400),
                np.random.uniform(40, 90),
                np.random.uniform(70, 100),
                np.random.uniform(400, 900),
                np.random.uniform(250, 500),
            ])
            y.append(1)

        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # TRAIN MODELS
    # ------------------------------------------------------------------
    def train_models(self, X, y):
        print("[INFO] Training models...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        results = {}

        # Isolation Forest
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(X_train)
        preds = (iso.predict(X_test) == -1).astype(int)

        results["isolation_forest"] = {
            "model": iso,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }

        # LOF
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        lof.fit(X_train)
        preds = (lof.predict(X_test) == -1).astype(int)

        results["lof"] = {
            "model": lof,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }

        # One-Class SVM
        svm = OneClassSVM(gamma="auto", nu=0.1)
        svm.fit(X_train)
        preds = (svm.predict(X_test) == -1).astype(int)

        results["svm"] = {
            "model": svm,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }

        # Optional LSTM
        results["lstm"] = self.try_lstm(X_train)

        return results, scaler

    # ------------------------------------------------------------------
    # OPTIONAL LSTM (SAFE)
    # ------------------------------------------------------------------
    def try_lstm(self, X_train):
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models

            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

            model = models.Sequential([
                layers.Input(shape=(1, X_train.shape[2])),
                layers.LSTM(32, return_sequences=False),
                layers.RepeatVector(1),
                layers.LSTM(32, return_sequences=True),
                layers.TimeDistributed(layers.Dense(X_train.shape[2]))
            ])

            model.compile(optimizer="adam", loss="mse")
            model.fit(X_train, X_train, epochs=10, batch_size=32, verbose=0)

            print("[OK] LSTM trained")
            return {"model": model}

        except Exception as e:
            print("[SKIP] TensorFlow not available:", e)
            return {"model": None}

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    def save(self, results, scaler):
        print("[INFO] Saving models...")

        joblib.dump(results["isolation_forest"]["model"], f"{self.model_dir}/iso.pkl")
        joblib.dump(results["lof"]["model"], f"{self.model_dir}/lof.pkl")
        joblib.dump(results["svm"]["model"], f"{self.model_dir}/svm.pkl")
        joblib.dump(scaler, f"{self.model_dir}/scaler.pkl")

        if results["lstm"]["model"] is not None:
            results["lstm"]["model"].save(f"{self.model_dir}/lstm.h5")

        with open(f"{self.model_dir}/metadata.json", "w") as f:
            json.dump({"timestamp": datetime.now().isoformat()}, f, indent=2)

        print("[SUCCESS] Models saved")

    # ------------------------------------------------------------------
    # PIPELINE
    # ------------------------------------------------------------------
    def run(self):
        print("=" * 55)
        print(" AUTONOMOUS SYSTEM RECOVERY - TRAINING ")
        print("=" * 55)

        X, y = self.generate_realistic_training_data()
        results, scaler = self.train_models(X, y)
        self.save(results, scaler)

        print("[DONE] Training pipeline finished")


if __name__ == "__main__":
    trainer = AdvancedModelTrainer()
    trainer.run()
