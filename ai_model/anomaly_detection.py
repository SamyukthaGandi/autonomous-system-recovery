import os
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json

class AnomalyDetector:
    def __init__(self, model_dir='ai_model'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        self.svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        self.scaler = StandardScaler()
        self.lstm_model = None
        
        self.metrics_history = []
        self.predictions_history = []
        self.model_performance = {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'auc': 0
        }
        
    def build_lstm_autoencoder(self, input_shape=(30, 10)):
        """Build LSTM autoencoder for time-series anomaly detection"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
            tf.keras.layers.LSTM(32, activation='relu', return_sequences=False),
            tf.keras.layers.RepeatVector(input_shape[0]),
            tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
            tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_shape[1]))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_ensemble(self, training_data):
        """Train all ensemble models"""
        print("[INFO] Training Ensemble Models...")
        
        # Normalize data
        scaled_data = self.scaler.fit_transform(training_data)
        
        # Train Isolation Forest
        print("  → Training Isolation Forest...")
        self.isolation_forest.fit(scaled_data)
        
        # Train LOF
        print("  → Training Local Outlier Factor...")
        self.lof.fit(scaled_data)
        
        # Train One-Class SVM
        print("  → Training One-Class SVM...")
        self.svm.fit(scaled_data)
        
        # Train LSTM Autoencoder
        print("  → Training LSTM Autoencoder...")
        self.lstm_model = self.build_lstm_autoencoder(input_shape=(30, training_data.shape[1]))
        
        # Prepare time-series data (30 timesteps)
        if len(training_data) > 30:
            ts_data = np.array([training_data[i:i+30] for i in range(len(training_data)-30)])
            self.lstm_model.fit(ts_data, ts_data, epochs=50, batch_size=16, verbose=0)
        
        self._save_models()
        print("[SUCCESS] Ensemble models trained and saved!")
    
    def detect_anomalies(self, data, return_details=False):
        """
        Multi-algorithm ensemble anomaly detection
        Returns: anomaly_score (0-1), is_anomaly (bool), details (dict)
        """
        # Normalize
        scaled_data = self.scaler.transform(np.array([data]))[0]
        
        scores = {}
        
        # Isolation Forest score
        if_score = -self.isolation_forest.score_samples(np.array([scaled_data]))[0]
        scores['isolation_forest'] = (if_score + 10) / 20  # Normalize to 0-1
        
        # LOF score
        lof_score = self.lof.score_samples(np.array([scaled_data]))[0]
        scores['lof'] = 1 / (1 + np.exp(lof_score))  # Sigmoid normalization
        
        # SVM score
        svm_score = self.svm.score_samples(np.array([scaled_data]))[0]
        scores['svm'] = 1 if svm_score < 0 else 0
        
        # LSTM reconstruction error
        lstm_score = 0
        if self.lstm_model and len(self.metrics_history) >= 30:
            recent_data = np.array(self.metrics_history[-30:])
            try:
                reconstruction = self.lstm_model.predict(np.array([recent_data]), verbose=0)
                mse = np.mean((recent_data - reconstruction[0]) ** 2)
                lstm_score = min(mse / 100, 1.0)  # Normalize
            except:
                lstm_score = 0
        
        scores['lstm'] = lstm_score
        
        # Ensemble vote (weighted average)
        ensemble_score = np.mean([
            scores['isolation_forest'] * 0.3,
            scores['lof'] * 0.3,
            scores['svm'] * 0.2,
            scores['lstm'] * 0.2
        ])
        
        is_anomaly = ensemble_score > 0.5
        
        # Store history
        self.metrics_history.append(data)
        self.predictions_history.append({
            'timestamp': datetime.now().isoformat(),
            'score': float(ensemble_score),
            'is_anomaly': bool(is_anomaly),
            'scores': {k: float(v) for k, v in scores.items()}
        })
        
        if return_details:
            return ensemble_score, is_anomaly, scores
        return ensemble_score, is_anomaly
    
    def _save_models(self):
        """Save trained models"""
        joblib.dump(self.isolation_forest, f'{self.model_dir}/isolation_forest.pkl')
        joblib.dump(self.lof, f'{self.model_dir}/lof.pkl')
        joblib.dump(self.svm, f'{self.model_dir}/svm.pkl')
        joblib.dump(self.scaler, f'{self.model_dir}/scaler.pkl')
        if self.lstm_model:
            self.lstm_model.save(f'{self.model_dir}/lstm_model.h5')
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.isolation_forest = joblib.load(f'{self.model_dir}/isolation_forest.pkl')
            self.lof = joblib.load(f'{self.model_dir}/lof.pkl')
            self.svm = joblib.load(f'{self.model_dir}/svm.pkl')
            self.scaler = joblib.load(f'{self.model_dir}/scaler.pkl')
            self.lstm_model = tf.keras.models.load_model(f'{self.model_dir}/lstm_model.h5')
            print("[INFO] Models loaded successfully!")
            return True
        except Exception as e:
            print(f"[WARNING] Could not load models: {e}")
            return False
    
    def get_performance_metrics(self, y_true, y_pred):
        """Calculate model performance metrics"""
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc = roc_auc_score(y_true, y_pred)
            
            self.model_performance = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc)
            }
            return self.model_performance
        except Exception as e:
            print(f"[ERROR] Could not calculate metrics: {e}")
            return self.model_performance
    
    def export_predictions(self, filepath='predictions.json'):
        """Export prediction history"""
        with open(filepath, 'w') as f:
            json.dump(self.predictions_history[-1000:], f, indent=2)


def generate_training_data(samples=1000):
    """Generate synthetic training data"""
    print("[INFO] Generating training data...")
    
    # Normal data: CPU/Memory between 10-60%
    normal_cpu = np.random.uniform(10, 60, (samples, 1))
    normal_mem = np.random.uniform(10, 60, (samples, 1))
    normal_disk = np.random.uniform(20, 70, (samples, 1))
    normal_io = np.random.uniform(5, 30, (samples, 1))
    normal_network = np.random.uniform(10, 50, (samples, 1))
    
    normal = np.hstack([normal_cpu, normal_mem, normal_disk, normal_io, normal_network,
                       np.random.rand(samples, 5)])  # 5 additional features
    
    # Anomaly data: High CPU/Memory
    anom_cpu = np.random.uniform(80, 100, (samples // 10, 1))
    anom_mem = np.random.uniform(80, 100, (samples // 10, 1))
    anom_disk = np.random.uniform(85, 99, (samples // 10, 1))
    anom_io = np.random.uniform(70, 95, (samples // 10, 1))
    anom_network = np.random.uniform(80, 99, (samples // 10, 1))
    
    anomaly = np.hstack([anom_cpu, anom_mem, anom_disk, anom_io, anom_network,
                        np.random.rand(samples // 10, 5)])
    
    data = np.vstack([normal, anomaly])
    return data


if __name__ == "__main__":
    # Initialize detector
    detector = AnomalyDetector()
    
    # Generate and train
    training_data = generate_training_data(samples=1000)
    detector.train_ensemble(training_data)
    
    # Test detection
    normal_test = np.array([30, 40, 50, 15, 25, 0.1, 0.2, 0.3, 0.4, 0.5])
    anomaly_test = np.array([95, 92, 88, 80, 90, 0.9, 0.8, 0.85, 0.9, 0.95])
    
    score1, is_anom1, details1 = detector.detect_anomalies(normal_test, return_details=True)
    score2, is_anom2, details2 = detector.detect_anomalies(anomaly_test, return_details=True)
    
    print(f"\n[TEST] Normal System:")
    print(f"  Score: {score1:.4f}, Anomaly: {is_anom1}")
    print(f"  Details: {details1}\n")
    
    print(f"[TEST] Anomalous System:")
    print(f"  Score: {score2:.4f}, Anomaly: {is_anom2}")
    print(f"  Details: {details2}")