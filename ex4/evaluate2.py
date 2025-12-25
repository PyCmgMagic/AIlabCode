from ultralytics import YOLO

# 定义训练生成的模型路径
models = [
    'runs/detect/yolo_n_default/weights/best.pt',
    'runs/detect/yolo_s_default/weights/best.pt',
    'runs/detect/yolo_n_custom/weights/best.pt'
]

print(f"{'模型':<20} | {'mAP50':<8} | {'mAP50-95':<8} | {'Precision':<8} | {'Recall':<8}")
print("-" * 65)

for model_path in models:
    try:
        model = YOLO(model_path)
        # 在验证集上评估
        res = model.val(data='distribution_box.yaml', verbose=False)
        
        print(f"{model_path.split('/')[2]:<20} | "
              f"{res.box.map50:.4f}   | "
              f"{res.box.map:.4f}     | "
              f"{res.box.mp:.4f}      | "
              f"{res.box.mr:.4f}")
    except:
        print(f"{model_path} 未找到，请检查训练是否完成。")