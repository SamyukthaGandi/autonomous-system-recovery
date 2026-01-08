import time
from ai_model.anomaly_detection import detect_anomalies
from os_interaction.monitor_system import monitor_system
from os_interaction.recovery_actions import recover_system

if __name__ == "__main__":
    while True:
        # Collect system metrics
        metrics = monitor_system()
        print(f"System Metrics: CPU: {metrics[0]}%, RAM: {metrics[1]}%, Disk: {metrics[2]}%")

        # Detect anomalies
        status = detect_anomalies(metrics)
        print(f"Anomaly Detection Status: {status}")

        # Execute recovery actions if anomaly detected
        if status == "Anomalous":
            if metrics[0] > 85:
                recover_system("High CPU")
            elif metrics[1] > 90:
                recover_system("High Memory")
            elif metrics[2] > 95:
                recover_system("High Disk")

        time.sleep(5)  # Monitor every 5 seconds
