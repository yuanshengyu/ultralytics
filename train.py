from ultralytics import YOLO

# Load a model
model = YOLO('/root/autodl-tmp/yolov8/yolov8n.yaml').load('/root/autodl-tmp/weight/yolov8n.pt')
# Train the model
results = model.train(data='/root/autodl-tmp/yolov8/luosi.yaml', batch=32, epochs=150, imgsz=640, degrees=180)