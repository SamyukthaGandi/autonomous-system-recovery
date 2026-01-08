import psutil

def monitor_system():
    """
    Collects system metrics (CPU, Memory, Disk usage).
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent

    return [cpu_usage, memory_usage, disk_usage]
