from ultralytics import YOLO

# --- 实验 1: 使用 Nano 模型 (yolo11n.pt) ---
# Slide 2 显示使用的是 yolo11n，如果报错找不到，可以使用 yolov8n.pt
print("开始训练 Nano 模型...")
model_n = YOLO('yolo11n.pt') 
model_n.train(data='distribution_box.yaml', epochs=50, imgsz=640, name='yolo_n_default')

# --- 实验 2: 使用 Small 模型 (yolo11s.pt) ---
print("开始训练 Small 模型...")
model_s = YOLO('yolo11s.pt')
model_s.train(data='distribution_box.yaml', epochs=50, imgsz=640, name='yolo_s_default')
# --- 实验 3: 修改超参数的模型 (以 Nano 为例) ---
print("开始训练修改超参数的模型...")
model_custom = YOLO('yolo11n.pt')

# 在 train 函数中传入参数即可覆盖默认值
model_custom.train(
    data='distribution_box.yaml',
    epochs=50,         # 修改点 1: 轮次
    batch=8,           # 修改点 2: 批次大小
    lr0=0.001,         # 修改点 3: 初始学习率
    optimizer='AdamW', # 修改点 4: 优化器
    imgsz=640,
    name='yolo_n_custom_params'
)