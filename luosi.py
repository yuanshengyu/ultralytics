import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO('material/bestn.pt')
rotate = False
# 打开视频文件
# video_path = "material/VID_20231120_140850.mp4"
# video_name = "qyj0001_1700460294295"
# video_name = "qyj0001_1700461636531"
video_name = "qyj260100001_1700460790508"
# video_name = "qyj260100001_1700101544566"
video_path = f"material/{video_name}.mp4"
save_video_path = f'material/{video_name}_detect.mp4'
cap = cv2.VideoCapture(video_path)
# 获取视频帧的维度
if rotate:
    frame_width = int(cap.get(4))
    frame_height = int(cap.get(3))
else:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
#创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_video_path, fourcc, 20.0, (frame_width, frame_height))
#循环视频帧
while cap.isOpened():
    # 读取某一帧
    success, frame = cap.read()
    if success:
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # 使用yolov8进行预测
        results = model(frame)
        #可视化结果
        annotated_frame = results[0].plot()
        #将带注释的帧写入视频文件
        out.write(annotated_frame)
    else:
        # 最后结尾中断视频帧循环
        break
#释放读取和写入对象
cap.release()
out.release()
