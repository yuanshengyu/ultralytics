from ultralytics import YOLO

model = YOLO('material/best.pt')
success = model.export(format="onnx", task='detect', opset = 13, simplify=True)
print(success)