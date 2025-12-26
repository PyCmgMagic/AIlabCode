"""
实验结果可视化脚本
功能：生成各种对比图表，用于实验报告
作者：实验四
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
OUTPUT_DIR = 'visualization_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_validation_results():
    """加载验证结果"""
    try:
        with open('validation_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 未找到验证结果文件 validation_results.json")
        print("   请先运行 validate_all_models.py")
        return None


def plot_metrics_comparison(results):
    """
    绘制主要指标对比柱状图
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('YOLO模型性能指标对比', fontsize=16, fontweight='bold')

    # 提取数据
    experiments = [r['description'] for r in results]
    metrics = {
        'mAP50': [r['mAP50'] for r in results],
        'mAP50-95': [r['mAP50-95'] for r in results],
        'mAP75': [r['mAP75'] for r in results],
        'Precision': [r['Precision'] for r in results],
        'Recall': [r['Recall'] for r in results],
        'F1-Score': [r['F1-Score'] for r in results],
    }

    colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))

    # 绘制每个指标
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(range(len(experiments)), values, color=colors, edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(experiments)))
        ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, max(values) * 1.15])

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 指标对比图已保存: {output_path}")
    plt.close()


def plot_radar_chart(results):
    """
    绘制雷达图对比前3个模型
    """
    # 选择前3个模型
    num_models = min(3, len(results))
    selected_results = results[:num_models]

    # 指标名称
    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
    num_vars = len(metrics)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)

    # 绘制每个模型
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for idx, result in enumerate(selected_results):
        values = [result[m] for m in metrics]
        values += values[:1]  # 闭合

        ax.plot(angles, values, 'o-', linewidth=2, label=result['description'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title('模型性能雷达图对比', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)

    output_path = os.path.join(OUTPUT_DIR, 'radar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 雷达图已保存: {output_path}")
    plt.close()


def plot_model_size_comparison(results):
    """
    对比不同模型大小的性能
    """
    # 筛选baseline模型
    baseline_results = [r for r in results if 'baseline' in r['experiment'].lower()]

    if len(baseline_results) < 2:
        print("⚠️  跳过模型大小对比图：需要至少2个baseline模型")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    models = [r['description'] for r in baseline_results]
    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall']

    x = np.arange(len(models))
    width = 0.2

    for idx, metric in enumerate(metrics):
        values = [r[metric] for r in baseline_results]
        offset = width * (idx - len(metrics)/2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=metric)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('模型', fontsize=12, fontweight='bold')
    ax.set_ylabel('指标值', fontsize=12, fontweight='bold')
    ax.set_title('不同模型尺寸性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'model_size_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 模型大小对比图已保存: {output_path}")
    plt.close()


def plot_hyperparameter_impact(results):
    """
    超参数影响分析
    """
    # 找到baseline和超参数调整的实验
    baseline = next((r for r in results if 'baseline' in r['experiment'] and 'n' in r['experiment']), None)

    if not baseline:
        print("⚠️  跳过超参数影响图：未找到baseline模型")
        return

    # 筛选Nano模型的超参数实验
    param_experiments = [r for r in results if 'exp' in r['experiment'] and
                        r['experiment'] not in [baseline['experiment']] and
                        'n' in r['experiment'] and 's' not in r['experiment']]

    if len(param_experiments) == 0:
        print("⚠️  跳过超参数影响图：未找到超参数实验")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('超参数调整对模型性能的影响', fontsize=14, fontweight='bold')

    # 准备数据
    experiment_names = ['Baseline'] + [r['description'].replace('Nano-', '') for r in param_experiments]
    all_experiments = [baseline] + param_experiments

    # 图1: mAP对比
    ax1 = axes[0]
    map50_values = [r['mAP50'] for r in all_experiments]
    map50_95_values = [r['mAP50-95'] for r in all_experiments]

    x = np.arange(len(experiment_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, map50_values, width, label='mAP50', color='#FF6B6B', edgecolor='black')
    bars2 = ax1.bar(x + width/2, map50_95_values, width, label='mAP50-95', color='#4ECDC4', edgecolor='black')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax1.set_ylabel('mAP值', fontsize=11, fontweight='bold')
    ax1.set_title('mAP指标对比', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiment_names, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, max(map50_values) * 1.15])

    # 图2: Precision和Recall对比
    ax2 = axes[1]
    precision_values = [r['Precision'] for r in all_experiments]
    recall_values = [r['Recall'] for r in all_experiments]

    bars1 = ax2.bar(x - width/2, precision_values, width, label='Precision', color='#95E1D3', edgecolor='black')
    bars2 = ax2.bar(x + width/2, recall_values, width, label='Recall', color='#F38181', edgecolor='black')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax2.set_ylabel('指标值', fontsize=11, fontweight='bold')
    ax2.set_title('Precision和Recall对比', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(experiment_names, rotation=45, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, max(precision_values) * 1.15])

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'hyperparameter_impact.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 超参数影响图已保存: {output_path}")
    plt.close()


def plot_top_models(results):
    """
    绘制TOP模型排名
    """
    # 按mAP50排序
    sorted_results = sorted(results, key=lambda x: x['mAP50'], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    models = [r['description'] for r in sorted_results]
    map50_values = [r['mAP50'] for r in sorted_results]

    # 使用渐变色
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))

    bars = ax.barh(range(len(models)), map50_values, color=colors, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for idx, (bar, value) in enumerate(zip(bars, map50_values)):
        ax.text(value, bar.get_y() + bar.get_height()/2.,
               f'  {value:.4f}',
               ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel('mAP50', fontsize=12, fontweight='bold')
    ax.set_title('模型性能排名 (按mAP50排序)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim([0, max(map50_values) * 1.15])

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'top_models_ranking.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 模型排名图已保存: {output_path}")
    plt.close()


def create_summary_table(results):
    """
    创建汇总表格图片
    """
    fig, ax = plt.subplots(figsize=(16, len(results) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    table_data = []
    headers = ['实验', 'mAP50', 'mAP50-95', 'mAP75', 'Precision', 'Recall', 'F1-Score', 'Fitness']

    for r in results:
        row = [
            r['description'][:20],  # 截断长名称
            f"{r['mAP50']:.4f}",
            f"{r['mAP50-95']:.4f}",
            f"{r['mAP75']:.4f}",
            f"{r['Precision']:.4f}",
            f"{r['Recall']:.4f}",
            f"{r['F1-Score']:.4f}",
            f"{r['Fitness']:.4f}"
        ]
        table_data.append(row)

    # 创建表格
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置行颜色
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#F0F0F0')

    plt.title('实验结果汇总表', fontsize=14, fontweight='bold', pad=20)

    output_path = os.path.join(OUTPUT_DIR, 'summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 汇总表格已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("="*70)
    print("YOLO实验结果可视化")
    print("="*70)

    # 加载验证结果
    results = load_validation_results()
    if not results:
        return

    print(f"\n加载了 {len(results)} 个实验结果")
    print("\n开始生成图表...\n")

    # 生成各种图表
    plot_metrics_comparison(results)
    plot_radar_chart(results)
    plot_model_size_comparison(results)
    plot_hyperparameter_impact(results)
    plot_top_models(results)
    create_summary_table(results)

    print("\n" + "="*70)
    print(f"所有图表已保存到目录: {OUTPUT_DIR}")
    print("="*70)
    print("\n生成的图表包括:")
    print("  1. metrics_comparison.png     - 主要指标对比")
    print("  2. radar_chart.png           - 性能雷达图")
    print("  3. model_size_comparison.png - 模型大小对比")
    print("  4. hyperparameter_impact.png - 超参数影响分析")
    print("  5. top_models_ranking.png    - 模型性能排名")
    print("  6. summary_table.png         - 结果汇总表")
    print("\n这些图表可以直接用于实验报告！")


if __name__ == '__main__':
    main()
