import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Server settings
    SERVER_HOST: str = os.getenv('SERVER_HOST', '0.0.0.0')
    SERVER_PORT: int = int(os.getenv('SERVER_PORT', 8000))
    
    # GPU settings
    DEVICE: str = os.getenv('DEVICE', 'cpu')  # 'cuda' or 'cpu'
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', 1))
    
    # Model settings
    YOLO_MODEL: str = os.getenv('YOLO_MODEL', 'yolov8n.pt')
    CLIP_MODEL: str = os.getenv('CLIP_MODEL', 'ViT-B/32')
    
    # Inference settings
    CONF_THRESHOLD: float = float(os.getenv('CONF_THRESHOLD', 0.5))
    IOU_THRESHOLD: float = float(os.getenv('IOU_THRESHOLD', 0.45))
    
    # File storage
    UPLOAD_FOLDER: str = os.getenv('UPLOAD_FOLDER', './uploads')
    MODEL_CACHE_FOLDER: str = os.getenv('MODEL_CACHE_FOLDER', './models')

settings = Settings()
