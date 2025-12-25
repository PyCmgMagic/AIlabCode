# --- 验证代码示例 ---
from ultralytics import YOLO

# 加载你训练好的最佳模型 (路径根据实际生成的文件夹修改)
# 例如: runs/detect/yolo_n_default/weights/best.pt
model = YOLO('runs/detect/yolo_n_default/weights/best.pt')

# 在验证集上评估
metrics = model.val(data='distribution_box.yaml')

# 打印 Slide 8 中提到的指标
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}") # Mean Precision
print(f"Recall: {metrics.box.mr}")    # Mean Recall
# F1 Score 通常可以在 runs/detect/.../F1_curve.png 图表中找到，或者通过 P 和 R 计算
print(f"Fitness: {metrics.fitness}")  # 综合指标