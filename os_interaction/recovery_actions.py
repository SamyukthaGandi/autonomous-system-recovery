import os
import subprocess

def recover_system(issue_type):
    """
    Executes system recovery actions based on the detected issue.
    """
    if issue_type == "High CPU":
        print("Restarting high CPU-consuming processes...")
        os.system("pkill -f heavy_process")  # Example command

    elif issue_type == "High Memory":
        print("Clearing memory cache...")
        os.system("sync; echo 3 > /proc/sys/vm/drop_caches")

    elif issue_type == "High Disk":
        print("Cleaning temporary files...")
        subprocess.run(["rm", "-rf", "/tmp/*"])

    else:
        print("No critical issues detected.")
