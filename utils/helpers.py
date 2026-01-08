"""Helper functions"""
import json
from datetime import datetime

def serialize_metric(metric):
    """Serialize metric for JSON"""
    if isinstance(metric, datetime):
        return metric.isoformat()
    raise TypeError(f"Type {type(metric)} not serializable")

def format_bytes(bytes_val):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def get_timestamp():
    """Get current ISO timestamp"""
    return datetime.now().isoformat()

def calculate_percentage_change(old, new):
    """Calculate percentage change"""
    if old == 0:
        return 0
    return ((new - old) / old) * 100
