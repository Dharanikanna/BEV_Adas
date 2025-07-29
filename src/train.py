import sys
import os
from ultralytics import YOLO

# Set working directory
os.chdir('/content/drive/MyDrive/Desseration_proj/BEV_Adas')

# Define log file path
log_path = 'training.log'

# Redirect stdout and stderr to log file
log_file = open(log_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

# Load custom architecture
model = YOLO('Model_Architecture/Yolov8n.yaml')

# Train and save inside existing 'model' folder
model.train(
    data='Proj_dataset/Combined_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    project='model',
    name='BEV_yolov8n',
    device='0'
)

# Restore stdout and close log file
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
