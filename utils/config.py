"""Configuration management"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    HOST = os.getenv('HOST', '127.0.0.1')
    PORT = int(os.getenv('PORT', 5000))
    
    # Model Configuration
    MODEL_DIR = 'ai_model'
    ANOMALY_THRESHOLD = 0.5
    
    # Monitoring
    MONITORING_INTERVAL = 5
    METRICS_BUFFER_SIZE = 1000
    
    # Recovery
    CPU_THRESHOLD = 80
    MEMORY_THRESHOLD = 85
    DISK_THRESHOLD = 90
    
    # Process Management
    PROCESS_WHITELIST = {
        'python', 'python3', 'flask', 'pycharm', 'code',
        'explorer.exe', 'System', 'systemd', 'init', 'chrome'
    }
    
    # Logging
    LOG_DIR = 'logs'
    LOG_LEVEL = 'INFO'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

def get_config():
    """Get appropriate config"""
    env = os.getenv('ENV', 'development')
    if env == 'production':
        return ProductionConfig
    return DevelopmentConfig
