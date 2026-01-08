# anomaly_logs.py

from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

# Load model & tokenizer once
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def log_to_vector(log_text):
    inputs = tokenizer(log_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def detect_log_anomaly(log_text, threshold=1.5):
    vector = log_to_vector(log_text)
    magnitude = np.linalg.norm(vector)
    print(f"Log vector magnitude: {magnitude:.4f}")
    return magnitude > threshold
