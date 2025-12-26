"""
YOLO模型训练实验脚本
功能：训练多个不同配置的YOLO模型，用于对比实验
作者：实验四
"""

from ultralytics import YOLO
import json
import os
from datetime import datetime

# 创建实验记录文件
experiment_log = []

def train_model(model_size, config_name, **train_params):
    """
    训练YOLO模型并记录实验信息

    参数:
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        config_name: 配置名称，用于标识实验
        **train_params: 训练参数
    """
    print(f"\n{'='*70}")
    print(f"开始训练: {config_name}")
    print(f"模型大小: YOLO11{model_size}")
    print(f"训练参数: {train_params}")
    print(f"{'='*70}\n")

    # 加载模型
    model = YOLO(f'yolo11{model_size}.pt')

    # 记录开始时间
    start_time = datetime.now()

    # 训练模型
    results = model.train(**train_params)

    # 记录结束时间
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 记录实验信息
    experiment_info = {
        'config_name': config_name,
        'model_size': model_size,
        'train_params': train_params,
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': duration,
        'save_dir': str(results.save_dir)
    }
    experiment_log.append(experiment_info)

    print(f"\n{config_name} 训练完成！耗时: {duration:.2f}秒")
    print(f"结果保存在: {results.save_dir}\n")

    return results


# ==============================================================================
# 实验配置
# ==============================================================================

# 数据集配置文件
DATA_CONFIG = 'distribution_box.yaml'

# 基础训练参数（所有实验共用）
BASE_PARAMS = {
    'data': DATA_CONFIG,
    'imgsz': 640,
    'device': 0,  # 使用GPU 0，如果没有GPU改为 'cpu'
}

print("="*70)
print("YOLO模型训练实验 - 配电箱门未关闭检测")
print("="*70)

# ==============================================================================
# 实验1: 训练Nano模型（默认参数）
# ==============================================================================
print("\n[实验1] 训练Nano模型 - 默认参数基线")
train_model(
    model_size='n',
    config_name='yolo_n_baseline',
    **BASE_PARAMS,
    epochs=50,
    batch=16,
    name='exp1_yolo_n_baseline'
)

# ==============================================================================
# 实验2: 训练Small模型（默认参数）
# ==============================================================================
print("\n[实验2] 训练Small模型 - 默认参数基线")
train_model(
    model_size='s',
    config_name='yolo_s_baseline',
    **BASE_PARAMS,
    epochs=50,
    batch=16,
    name='exp2_yolo_s_baseline'
)

# ==============================================================================
# 实验3: 修改超参数实验 - 批次大小
# 超参数1: batch size (16 -> 8)
# ==============================================================================
print("\n[实验3] Nano模型 - 减小批次大小")
train_model(
    model_size='n',
    config_name='yolo_n_batch8',
    **BASE_PARAMS,
    epochs=50,
    batch=8,  # 修改点1: 减小批次大小
    name='exp3_yolo_n_batch8'
)

# ==============================================================================
# 实验4: 修改超参数实验 - 学习率
# 超参数2: learning rate (default 0.01 -> 0.001)
# ==============================================================================
print("\n[实验4] Nano模型 - 降低学习率")
train_model(
    model_size='n',
    config_name='yolo_n_lr0.001',
    **BASE_PARAMS,
    epochs=50,
    batch=16,
    lr0=0.001,  # 修改点2: 降低初始学习率
    name='exp4_yolo_n_lr0.001'
)

# ==============================================================================
# 实验5: 修改超参数实验 - 优化器
# 超参数3: optimizer (SGD -> AdamW)
# ==============================================================================
print("\n[实验5] Nano模型 - 使用AdamW优化器")
train_model(
    model_size='n',
    config_name='yolo_n_adamw',
    **BASE_PARAMS,
    epochs=50,
    batch=16,
    optimizer='AdamW',  # 修改点3: 更换优化器
    name='exp5_yolo_n_adamw'
)

# ==============================================================================
# 实验6: 修改超参数实验 - 图像增强
# 超参数4: 数据增强参数 (hsv_h, hsv_s, hsv_v, degrees)
# ==============================================================================
print("\n[实验6] Nano模型 - 增强数据增强")
train_model(
    model_size='n',
    config_name='yolo_n_augment',
    **BASE_PARAMS,
    epochs=50,
    batch=16,
    hsv_h=0.02,    # 修改点4a: 色调增强 (默认0.015)
    hsv_s=0.8,     # 修改点4b: 饱和度增强 (默认0.7)
    hsv_v=0.5,     # 修改点4c: 明度增强 (默认0.4)
    degrees=15.0,  # 修改点4d: 旋转角度 (默认0.0)
    name='exp6_yolo_n_augment'
)

# ==============================================================================
# 实验7: 组合超参数实验
# 综合应用多个较优的超参数设置
# ==============================================================================
print("\n[实验7] Nano模型 - 组合优化超参数")
train_model(
    model_size='n',
    config_name='yolo_n_combined',
    **BASE_PARAMS,
    epochs=50,
    batch=8,           # 小批次
    lr0=0.001,         # 低学习率
    optimizer='AdamW', # AdamW优化器
    hsv_h=0.02,        # 增强的数据增强
    hsv_s=0.8,
    name='exp7_yolo_n_combined'
)

# ==============================================================================
# 保存实验记录
# ==============================================================================
log_file = 'experiment_log.json'
with open(log_file, 'w', encoding='utf-8') as f:
    json.dump(experiment_log, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print("所有训练实验完成！")
print(f"实验记录已保存到: {log_file}")
print("="*70)

# 打印实验摘要
print("\n实验摘要:")
print(f"{'实验名称':<30} {'模型':<10} {'训练时长(秒)':<15}")
print("-"*70)
for exp in experiment_log:
    print(f"{exp['config_name']:<30} {exp['model_size']:<10} {exp['duration_seconds']:<15.2f}")

print("\n下一步: 运行 validate_all_models.py 进行模型验证")
