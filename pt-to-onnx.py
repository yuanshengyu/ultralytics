from ultralytics import YOLO

model = YOLO('material/best.pt')
success = model.export(format="onnx", half = True, imgsz=320, optimize=True)
print(success)