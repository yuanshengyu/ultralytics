import cv2
from ultralytics import YOLO
import glob
import os
import time
import shutil

def save_label(lb_path, lines):
    with open(lb_path, 'w') as f:
        for line in lines:
            f.write(line+'\n')

# 加载模型
model = YOLO('material/bestn2.pt')
img_dir = '/Users/gaosheng/Work/Document/Hitachi/test'
output_dir = '/Users/gaosheng/Work/Document/Hitachi/test2'
no_dir = '/Users/gaosheng/Work/Document/Hitachi/test3'
img_paths = glob.glob(os.path.join(img_dir, '**.jpg'))
for img_path in img_paths:
    start = time.time()
    img_name, _ = os.path.splitext(os.path.basename(img_path))
    results = model(img_path)
    boxes = results[0].boxes
    img_size = (boxes.orig_shape[1], boxes.orig_shape[0])
    lines = []
    for box in boxes.xywhn:
        lines.append("0 " + " ".join([str(a.item()) for a in box]))
    if len(lines) == 0:
        new_img_path = os.path.join(no_dir, os.path.basename(img_path))
        shutil.copy(img_path, new_img_path)
        continue
    new_img_path = os.path.join(output_dir, os.path.basename(img_path))
    shutil.copy(img_path, new_img_path)
    label_path = os.path.join(output_dir, img_name+'.txt')
    save_label(label_path, lines)
    print(f'耗时 {int((time.time() - start) * 1000)} ms')
        
