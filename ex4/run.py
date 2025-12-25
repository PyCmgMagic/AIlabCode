from ultralytics import YOLO

# 1. 定义配置文件路径
yaml_file = 'distribution_box.yaml'

# =======================================================
# 任务一：训练两个不同尺寸的模型 (使用默认参数)
# =======================================================

# --- 模型 A: Nano (最小尺寸 n) ---
print("\n>>> [1/3] 开始训练 Nano 模型 (默认参数)...")
model_n = YOLO('yolo11n.pt') 
model_n.train(data=yaml_file, epochs=50, imgsz=640, device=0, name='yolo_n_default')

# --- 模型 B: Small (稍大尺寸 s) ---
print("\n>>> [2/3] 开始训练 Small 模型 (默认参数)...")
model_s = YOLO('yolo11s.pt')
model_s.train(data=yaml_file, epochs=50, imgsz=640, device=0, name='yolo_s_default')

# =======================================================
# 任务二：修改至少4个超参数进行对比 (实验要求)
# =======================================================

# --- 模型 C: Nano (修改超参数) ---
print("\n>>> [3/3] 开始训练 Nano 模型 (修改超参数)...")
model_custom = YOLO('yolo11n.pt')

# 修改点：
# 1. epochs: 50 -> 30 (减少轮次)
# 2. batch: 16 -> 8 (减小Batch Size)
# 3. lr0: 0.01 -> 0.001 (减小初始学习率)
# 4. optimizer: auto -> AdamW (指定优化器)
model_custom.train(
    data=yaml_file,
    epochs=30,          
    batch=8,            
    lr0=0.001,          
    optimizer='AdamW',  
    imgsz=640,
    device=0,
    name='yolo_n_custom'
)

print("\n所有训练完成！结果保存在 runs/detect/ 目录下。")