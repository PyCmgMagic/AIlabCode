"""
YOLO模型验证脚本
功能：验证所有训练好的模型，收集并保存评估指标
作者：实验四
"""

from ultralytics import YOLO
import json
import os
import glob
from pathlib import Path
import pandas as pd

# 数据集配置
DATA_CONFIG = 'distribution_box.yaml'

# 定义要验证的模型配置
EXPERIMENTS = [
    {'name': 'exp1_yolo_n_baseline', 'desc': 'Nano模型(基线)'},
    {'name': 'exp2_yolo_s_baseline', 'desc': 'Small模型(基线)'},
    {'name': 'exp3_yolo_n_batch8', 'desc': 'Nano-批次大小8'},
    {'name': 'exp4_yolo_n_lr0.001', 'desc': 'Nano-学习率0.001'},
    {'name': 'exp5_yolo_n_adamw', 'desc': 'Nano-AdamW优化器'},
    {'name': 'exp6_yolo_n_augment', 'desc': 'Nano-增强数据增强'},
    {'name': 'exp7_yolo_n_combined', 'desc': 'Nano-组合优化'},
]

def validate_model(model_path, exp_name, exp_desc):
    """
    验证单个模型并返回指标

    参数:
        model_path: 模型权重文件路径
        exp_name: 实验名称
        exp_desc: 实验描述

    返回:
        包含所有评估指标的字典
    """
    print(f"\n{'='*70}")
    print(f"验证模型: {exp_desc}")
    print(f"模型路径: {model_path}")
    print(f"{'='*70}")

    try:
        # 加载模型
        model = YOLO(model_path)

        # 在验证集上评估
        metrics = model.val(data=DATA_CONFIG, verbose=True)

        # 提取指标
        results = {
            'experiment': exp_name,
            'description': exp_desc,
            'model_path': str(model_path),
            # 主要指标
            'mAP50': float(metrics.box.map50),      # AP at IoU=0.5
            'mAP50-95': float(metrics.box.map),     # AP at IoU=0.5:0.95
            'mAP75': float(metrics.box.map75),      # AP at IoU=0.75
            'Precision': float(metrics.box.mp),      # Mean Precision
            'Recall': float(metrics.box.mr),         # Mean Recall
            'F1-Score': float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-6)),  # F1分数
            'Fitness': float(metrics.fitness),       # 综合适应度
            # 详细指标
            'P': metrics.box.p.tolist() if hasattr(metrics.box.p, 'tolist') else [],  # 每类精确率
            'R': metrics.box.r.tolist() if hasattr(metrics.box.r, 'tolist') else [],  # 每类召回率
            'F1': metrics.box.f1.tolist() if hasattr(metrics.box.f1, 'tolist') else [],  # 每类F1
            'all_ap': metrics.box.all_ap.tolist() if hasattr(metrics.box.all_ap, 'tolist') else [],  # 所有IoU阈值的AP
        }

        print(f"\n验证结果:")
        print(f"  mAP50:      {results['mAP50']:.4f}")
        print(f"  mAP50-95:   {results['mAP50-95']:.4f}")
        print(f"  mAP75:      {results['mAP75']:.4f}")
        print(f"  Precision:  {results['Precision']:.4f}")
        print(f"  Recall:     {results['Recall']:.4f}")
        print(f"  F1-Score:   {results['F1-Score']:.4f}")
        print(f"  Fitness:    {results['Fitness']:.4f}")

        return results

    except FileNotFoundError:
        print(f"❌ 模型文件未找到: {model_path}")
        print(f"   请确认已完成训练！")
        return None
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        return None


def main():
    """主函数：验证所有模型并保存结果"""
    print("="*70)
    print("YOLO模型验证 - 配电箱门未关闭检测")
    print("="*70)

    all_results = []

    # 验证每个实验的模型
    for exp in EXPERIMENTS:
        # 查找模型权重文件
        model_path = f"runs/detect/{exp['name']}/weights/best.pt"

        if not os.path.exists(model_path):
            print(f"\n⚠️  跳过 {exp['name']}: 模型文件不存在")
            print(f"   期望路径: {model_path}")
            continue

        # 验证模型
        result = validate_model(model_path, exp['name'], exp['desc'])
        if result:
            all_results.append(result)

    # 保存结果
    if all_results:
        # 保存为JSON
        json_file = 'validation_results.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 详细结果已保存到: {json_file}")

        # 保存为CSV（便于Excel打开）
        csv_file = 'validation_results.csv'
        df = pd.DataFrame([{
            '实验名称': r['experiment'],
            '描述': r['description'],
            'mAP50': r['mAP50'],
            'mAP50-95': r['mAP50-95'],
            'mAP75': r['mAP75'],
            'Precision': r['Precision'],
            'Recall': r['Recall'],
            'F1-Score': r['F1-Score'],
            'Fitness': r['Fitness']
        } for r in all_results])
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"✓ CSV结果已保存到: {csv_file}")

        # 打印对比表格
        print("\n" + "="*70)
        print("模型性能对比表")
        print("="*70)
        print(f"{'实验':<25} {'mAP50':<8} {'mAP50-95':<10} {'Precision':<10} {'Recall':<8} {'F1':<8}")
        print("-"*70)
        for r in all_results:
            exp_short = r['experiment'].replace('exp', 'E').replace('yolo_', '')[:24]
            print(f"{exp_short:<25} {r['mAP50']:<8.4f} {r['mAP50-95']:<10.4f} "
                  f"{r['Precision']:<10.4f} {r['Recall']:<8.4f} {r['F1-Score']:<8.4f}")

        print("\n" + "="*70)
        print("下一步: 运行 visualize_results.py 生成对比图表")
        print("="*70)
    else:
        print("\n❌ 没有成功验证任何模型！")
        print("   请先运行 train_all_experiments.py 完成训练")


if __name__ == '__main__':
    main()
