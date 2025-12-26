"""
快速实验脚本 - 仅训练前3个模型用于快速测试
作者：实验四
"""

from ultralytics import YOLO
import json
from datetime import datetime

# 数据集配置
DATA_CONFIG = 'distribution_box.yaml'

# 基础参数
BASE_PARAMS = {
    'data': DATA_CONFIG,
    'imgsz': 640,
    'device': 0,
}

print("="*70)
print("快速实验模式 - 仅训练前3个模型")
print("="*70)

experiment_log = []

def train_model(model_size, config_name, **train_params):
    print(f"\n{'='*70}")
    print(f"训练: {config_name}")
    print(f"{'='*70}\n")

    model = YOLO(f'yolo11{model_size}.pt')
    start_time = datetime.now()
    results = model.train(**train_params)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    experiment_info = {
        'config_name': config_name,
        'model_size': model_size,
        'duration_seconds': duration,
        'save_dir': str(results.save_dir)
    }
    experiment_log.append(experiment_info)

    print(f"\n{config_name} 完成！耗时: {duration:.2f}秒\n")
    return results


# 实验1: Nano基线
print("\n[实验1] Nano模型基线")
train_model(
    model_size='n',
    config_name='quick_yolo_n_baseline',
    **BASE_PARAMS,
    epochs=30,  # 减少epochs用于快速测试
    batch=16,
    name='quick_exp1_n_baseline'
)

# 实验2: Small基线
print("\n[实验2] Small模型基线")
train_model(
    model_size='s',
    config_name='quick_yolo_s_baseline',
    **BASE_PARAMS,
    epochs=30,
    batch=16,
    name='quick_exp2_s_baseline'
)

# 实验3: Nano with modified hyperparameters
print("\n[实验3] Nano模型 - 修改超参数")
train_model(
    model_size='n',
    config_name='quick_yolo_n_custom',
    **BASE_PARAMS,
    epochs=30,
    batch=8,
    lr0=0.001,
    optimizer='AdamW',
    name='quick_exp3_n_custom'
)

# 保存日志
with open('quick_experiment_log.json', 'w', encoding='utf-8') as f:
    json.dump(experiment_log, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print("快速实验完成！")
print("="*70)
print("\n下一步: 运行 validate_all_models.py 进行验证")
