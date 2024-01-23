import os
os.getcwd()

# arrange an instance segmentation model for test
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
import time

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    # YOLOv8模型的路径
    model_path='material/bestn.pt',
    # YOLOv8模型的路径
    confidence_threshold=0.2,
    # 设备类型。
    # 如果您的计算机配备 NVIDIA GPU，则可以通过将 'device' 标志更改为'cuda:0'来启用 CUDA 加速；否则，将其保留为'cpu'
    device="cpu", # or 'cuda:0'
)

img_path = 'material/test2.jpg'
start = time.time()

result = get_sliced_prediction(
    img_path,
    detection_model,
    slice_height = 300,
    slice_width = 300,
    overlap_height_ratio = 0.1,
    overlap_width_ratio = 0.1
)

# result = get_prediction(img_path, detection_model)

msec = (time.time() - start) * 1000
print(f'耗时 {msec} ms')
result.export_visuals(export_dir="material/")

