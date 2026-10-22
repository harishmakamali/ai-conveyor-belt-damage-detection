
from ultralytics import YOLO

# Load the trained conveyor belt damage detection model
model = YOLO("better.pt")

# Start webcam-based detection
model.predict(
    source=0,
    show=True,
    conf=0.50
)
