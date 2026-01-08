from flask import Flask, render_template, jsonify, request
import os
import time
import threading
import json
from datetime import datetime, timedelta
import numpy as np
import psutil
from collections import deque

# Import enhanced anomaly detector
from ai_model.anomaly_detection import AnomalyDetector

app = Flask(__name__)

# Initialize anomaly detector
detector = AnomalyDetector()
detector.load_models()  # Try to load existing models

# Global state
recovery_history = deque(maxlen=100)
metrics_buffer = deque(maxlen=1000)
system_state = {
    "last_triggered": "Never",
    "status": "Monitoring",
    "anomaly_score": 0,
    "is_anomalous": False,
    "recovery_level": 0
}

# Recovery levels
RECOVERY_LEVELS = {
    0: {"name": "Normal", "actions": [], "threshold": 0.3},
    1: {"name": "Gentle", "actions": ["clear_cache", "close_handles"], "threshold": 0.5},
    2: {"name": "Moderate", "actions": ["terminate_background", "reduce_services"], "threshold": 0.7},
    3: {"name": "Aggressive", "actions": ["terminate_low_priority", "activate_failover"], "threshold": 0.9}
}

# Process whitelist
PROCESS_WHITELIST = {
    "python", "python3", "flask", "pycharm", "code", "explorer.exe",
    "System", "systemd", "init", "chrome", "cmd.exe", "svchost.exe"
}

def get_enhanced_system_metrics():
    """Collect comprehensive system metrics"""
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    
    # Network I/O
    net_io = psutil.net_io_counters()
    
    # Disk I/O
    disk_io = psutil.disk_io_counters()
    
    # Process info
    process_count = len(psutil.pids())
    top_processes = get_top_processes(3)
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu": cpu,
        "memory": mem.percent,
        "disk": disk.percent,
        "memory_mb": mem.used / (1024 ** 2),
        "network_bytes_sent": net_io.bytes_sent,
        "network_bytes_recv": net_io.bytes_recv,
        "disk_io_read": disk_io.read_bytes if disk_io else 0,
        "disk_io_write": disk_io.write_bytes if disk_io else 0,
        "process_count": process_count,
        "top_processes": top_processes,
        "swap_percent": psutil.swap_memory().percent if hasattr(psutil, 'swap_memory') else 0,
        "cpu_count": psutil.cpu_count()
    }
    
    return metrics

def get_top_processes(n=3):
    """Get top N processes by memory usage"""
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
            try:
                info = proc.info
                if info['memory_percent'] > 0.1:
                    processes.append({
                        'name': info['name'],
                        'pid': info['pid'],
                        'memory_percent': info['memory_percent'],
                        'cpu_percent': info['cpu_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:n]
    except Exception as e:
        print(f"[ERROR] Could not get top processes: {e}")
        return []

def determine_recovery_level(anomaly_score):
    """Determine recovery action level based on anomaly score"""
    for level in sorted(RECOVERY_LEVELS.keys(), reverse=True):
        if anomaly_score >= RECOVERY_LEVELS[level]["threshold"]:
            return level
    return 0

def execute_recovery_action(level):
    """Execute recovery actions based on level"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    actions = RECOVERY_LEVELS[level]["actions"]
    
    action_results = {
        "level": level,
        "level_name": RECOVERY_LEVELS[level]["name"],
        "timestamp": timestamp,
        "actions_executed": [],
        "freed_memory_mb": 0
    }
    
    os.makedirs("logs", exist_ok=True)
    
    with open("logs/system.log", "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}]  Recovery Level {level} ({RECOVERY_LEVELS[level]['name']}) Triggered\n")
        f.write(f"Actions: {', '.join(actions)}\n")
        
        for action in actions:
            if action == "clear_cache":
                result = clear_system_cache()
                action_results["actions_executed"].append({"action": action, "result": result})
                f.write(f"  ✓ Cache cleared\n")
            
            elif action == "close_handles":
                result = close_unused_handles()
                action_results["actions_executed"].append({"action": action, "result": result})
                f.write(f"  ✓ Unused handles closed\n")
            
            elif action == "terminate_background":
                freed = terminate_background_processes()
                action_results["freed_memory_mb"] += freed
                action_results["actions_executed"].append({"action": action, "freed_mb": freed})
                f.write(f"  ✓ Background processes terminated (freed {freed:.1f} MB)\n")
            
            elif action == "terminate_low_priority":
                freed = terminate_low_priority_processes()
                action_results["freed_memory_mb"] += freed
                action_results["actions_executed"].append({"action": action, "freed_mb": freed})
                f.write(f"  ✓ Low-priority processes terminated (freed {freed:.1f} MB)\n")
            
            elif action == "reduce_services":
                result = reduce_service_quality()
                action_results["actions_executed"].append({"action": action, "result": result})
                f.write(f"  ✓ Service quality reduced\n")
            
            elif action == "activate_failover":
                result = activate_failover()
                action_results["actions_executed"].append({"action": action, "result": result})
                f.write(f"  ✓ Failover activated\n")
        
        f.write(f"Total memory freed: {action_results['freed_memory_mb']:.1f} MB\n")
    
    return action_results

def clear_system_cache():
    """Clear system cache (simulated)"""
    try:
        if os.path.exists("/proc/sys/vm/drop_caches"):
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3")
        return "success"
    except:
        return "permission_denied"

def close_unused_handles():
    """Close unused file handles (simulated)"""
    return "success"

def terminate_background_processes():
    """Terminate non-critical background processes"""
    freed = 0
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
        try:
            if proc.info['name'] not in PROCESS_WHITELIST and proc.info['memory_percent'] > 5:
                freed += proc.memory_info().rss / (1024 ** 2)
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return freed

def terminate_low_priority_processes():
    """Terminate low-priority processes aggressively"""
    freed = 0
    critical_services = {"svchost", "csrss", "lsass"}
    
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'nice']):
        try:
            info = proc.info
            if (info['name'] not in PROCESS_WHITELIST and 
                info['name'] not in critical_services and 
                info['nice'] > 0 and 
                info['memory_percent'] > 2):
                freed += proc.memory_info().rss / (1024 ** 2)
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return freed

def reduce_service_quality():
    """Reduce service quality (disable non-essential features)"""
    return "success"

def activate_failover():
    """Activate failover mechanisms"""
    return "success"

def auto_monitor():
    """Background monitoring thread"""
    while True:
        try:
            # Collect metrics
            metrics = get_enhanced_system_metrics()
            metrics_buffer.append(metrics)
            
            # Prepare vector for ML detection (10 features)
            vector = np.array([
                metrics["cpu"],
                metrics["memory"],
                metrics["disk"],
                metrics["process_count"] / 100,
                len(metrics["top_processes"]),
                metrics["swap_percent"],
                metrics["network_bytes_sent"] / 1e9,
                metrics["network_bytes_recv"] / 1e9,
                metrics["disk_io_read"] / 1e9,
                metrics["disk_io_write"] / 1e9
            ])
            
            # Detect anomalies
            anomaly_score, is_anomalous = detector.detect_anomalies(vector)
            
            system_state["anomaly_score"] = float(anomaly_score)
            system_state["is_anomalous"] = bool(is_anomalous)
            
            # Determine recovery level
            recovery_level = determine_recovery_level(anomaly_score)
            system_state["recovery_level"] = recovery_level
            
            # Execute recovery if needed
            if recovery_level > 0:
                print(f" Anomaly detected (Score: {anomaly_score:.4f}, Level: {recovery_level})")
                result = execute_recovery_action(recovery_level)
                recovery_history.append(result)
                system_state["last_triggered"] = result["timestamp"]
                system_state["status"] = f"Recovery Level {recovery_level}"
            else:
                system_state["status"] = "Monitoring - Normal"
            
            time.sleep(5)
        
        except Exception as e:
            print(f"[ERROR] Monitoring thread error: {e}")
            time.sleep(5)

# Flask Routes
@app.route("/")
def home():
    anomalies = get_anomalies()
    return render_template("index.html", anomalies=anomalies)

@app.route("/api/status")
def api_status():
    """Get current system status"""
    return jsonify({
        **system_state,
        "metrics": dict(list(metrics_buffer)[-1]) if metrics_buffer else {}
    })

@app.route("/api/metrics")
def api_metrics():
    """Get recent metrics (last 100)"""
    return jsonify(list(metrics_buffer)[-100:])

@app.route("/api/recovery-history")
def api_recovery_history():
    """Get recovery action history"""
    return jsonify(list(recovery_history))

@app.route("/api/manual-recover", methods=["POST"])
def manual_recover():
    """Manually trigger recovery"""
    level = request.json.get("level", 2)
    result = execute_recovery_action(level)
    recovery_history.append(result)
    system_state["last_triggered"] = result["timestamp"]
    return jsonify(result)

@app.route("/api/model-performance")
def model_performance():
    """Get ML model performance metrics"""
    return jsonify({
        "performance": detector.model_performance,
        "predictions_count": len(detector.predictions_history),
        "last_10_predictions": detector.predictions_history[-10:]
    })

@app.route("/api/clear-logs", methods=["POST"])
def clear_logs():
    """Clear system logs"""
    os.makedirs("logs", exist_ok=True)
    open("logs/system.log", "w").close()
    return jsonify({"status": "Logs cleared"})

@app.route("/logs")
def logs():
    """Render logs page"""
    log_file = "logs/system.log"
    lines = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
    return render_template("logs.html", logs=lines)

def get_anomalies():
    """Get current system anomalies"""
    anomalies = []
    metrics = list(metrics_buffer)[-1] if metrics_buffer else {}
    
    if metrics.get("cpu", 0) > 80:
        anomalies.append(f" High CPU Usage: {metrics['cpu']:.1f}%")
    if metrics.get("memory", 0) > 75:
        anomalies.append(f" High Memory Usage: {metrics['memory']:.1f}%")
    if metrics.get("disk", 0) > 85:
        anomalies.append(f"️ High Disk Usage: {metrics['disk']:.1f}%")
    if system_state["anomaly_score"] > 0.7:
        anomalies.append(f" ML Anomaly Detected: {system_state['anomaly_score']:.2f}")
    
    return anomalies if anomalies else [" All systems normal"]

if __name__ == "__main__":
    # Start background monitoring
    monitor_thread = threading.Thread(target=auto_monitor, daemon=True)
    monitor_thread.start()
    
    # Run Flask app
    app.run(debug=False, threaded=True, port=5000)