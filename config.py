"""Main configuration file"""
import os
from utils.config import get_config, DevelopmentConfig, ProductionConfig

# Get environment
ENV = os.getenv('ENV', 'development')

# Select config
if ENV == 'production':
    Config = ProductionConfig
else:
    Config = DevelopmentConfig

# Export config
DEBUG = Config.DEBUG
HOST = Config.HOST
PORT = Config.PORT
MODEL_DIR = Config.MODEL_DIR
ANOMALY_THRESHOLD = Config.ANOMALY_THRESHOLD
MONITORING_INTERVAL = Config.MONITORING_INTERVAL
CPU_THRESHOLD = Config.CPU_THRESHOLD
MEMORY_THRESHOLD = Config.MEMORY_THRESHOLD
DISK_THRESHOLD = Config.DISK_THRESHOLD
PROCESS_WHITELIST = Config.PROCESS_WHITELIST
LOG_DIR = Config.LOG_DIR