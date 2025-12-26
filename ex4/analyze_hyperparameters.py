"""
超参数影响深度分析脚本
功能：详细分析各个超参数对模型性能的影响
作者：实验四
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
OUTPUT_DIR = 'hyperparameter_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_results():
    """加载验证结果"""
    try:
        with open('validation_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 未找到验证结果文件")
        print("   请先运行 validate_all_models.py")
        return None


def analyze_batch_size_impact(results):
    """
    分析批次大小的影响
    """
    # 筛选相关实验
    baseline = next((r for r in results if 'exp1' in r['experiment']), None)
    batch8 = next((r for r in results if 'exp3' in r['experiment']), None)

    if not baseline or not batch8:
        print("⚠️  跳过批次大小分析：缺少相关实验")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
    baseline_values = [baseline[m] for m in metrics]
    batch8_values = [batch8[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_values, width, label='Batch=16 (基线)', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, batch8_values, width, label='Batch=8', color='#4ECDC4')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    # 添加变化百分比
    for i, metric in enumerate(metrics):
        change = ((batch8_values[i] - baseline_values[i]) / baseline_values[i]) * 100
        color = 'green' if change > 0 else 'red'
        ax.text(i, max(baseline_values[i], batch8_values[i]) + 0.03,
               f'{change:+.1f}%',
               ha='center', fontsize=9, color=color, fontweight='bold')

    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('批次大小影响分析 (Batch Size: 16 vs 8)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.15])

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'batch_size_impact.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 批次大小分析图已保存: {output_path}")
    plt.close()


def analyze_learning_rate_impact(results):
    """
    分析学习率的影响
    """
    baseline = next((r for r in results if 'exp1' in r['experiment']), None)
    lr_low = next((r for r in results if 'exp4' in r['experiment']), None)

    if not baseline or not lr_low:
        print("⚠️  跳过学习率分析：缺少相关实验")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
    baseline_values = [baseline[m] for m in metrics]
    lr_values = [lr_low[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_values, width, label='LR=0.01 (默认)', color='#95E1D3')
    bars2 = ax.bar(x + width/2, lr_values, width, label='LR=0.001', color='#F38181')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    for i, metric in enumerate(metrics):
        change = ((lr_values[i] - baseline_values[i]) / baseline_values[i]) * 100
        color = 'green' if change > 0 else 'red'
        ax.text(i, max(baseline_values[i], lr_values[i]) + 0.03,
               f'{change:+.1f}%',
               ha='center', fontsize=9, color=color, fontweight='bold')

    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('学习率影响分析 (Learning Rate: 0.01 vs 0.001)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.15])

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'learning_rate_impact.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 学习率分析图已保存: {output_path}")
    plt.close()


def analyze_optimizer_impact(results):
    """
    分析优化器的影响
    """
    baseline = next((r for r in results if 'exp1' in r['experiment']), None)
    adamw = next((r for r in results if 'exp5' in r['experiment']), None)

    if not baseline or not adamw:
        print("⚠️  跳过优化器分析：缺少相关实验")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
    baseline_values = [baseline[m] for m in metrics]
    adamw_values = [adamw[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_values, width, label='SGD (默认)', color='#FFD93D')
    bars2 = ax.bar(x + width/2, adamw_values, width, label='AdamW', color='#6BCB77')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    for i, metric in enumerate(metrics):
        change = ((adamw_values[i] - baseline_values[i]) / baseline_values[i]) * 100
        color = 'green' if change > 0 else 'red'
        ax.text(i, max(baseline_values[i], adamw_values[i]) + 0.03,
               f'{change:+.1f}%',
               ha='center', fontsize=9, color=color, fontweight='bold')

    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('优化器影响分析 (SGD vs AdamW)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.15])

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'optimizer_impact.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 优化器分析图已保存: {output_path}")
    plt.close()


def analyze_augmentation_impact(results):
    """
    分析数据增强的影响
    """
    baseline = next((r for r in results if 'exp1' in r['experiment']), None)
    augment = next((r for r in results if 'exp6' in r['experiment']), None)

    if not baseline or not augment:
        print("⚠️  跳过数据增强分析：缺少相关实验")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
    baseline_values = [baseline[m] for m in metrics]
    augment_values = [augment[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_values, width, label='默认增强', color='#A8DADC')
    bars2 = ax.bar(x + width/2, augment_values, width, label='增强增强', color='#E63946')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    for i, metric in enumerate(metrics):
        change = ((augment_values[i] - baseline_values[i]) / baseline_values[i]) * 100
        color = 'green' if change > 0 else 'red'
        ax.text(i, max(baseline_values[i], augment_values[i]) + 0.03,
               f'{change:+.1f}%',
               ha='center', fontsize=9, color=color, fontweight='bold')

    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('数据增强影响分析', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.15])

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'augmentation_impact.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 数据增强分析图已保存: {output_path}")
    plt.close()


def create_impact_summary(results):
    """
    创建超参数影响汇总表
    """
    baseline = next((r for r in results if 'exp1' in r['experiment']), None)
    if not baseline:
        print("⚠️  未找到基线模型")
        return

    # 收集各超参数实验
    experiments = {
        'Batch=8': next((r for r in results if 'exp3' in r['experiment']), None),
        'LR=0.001': next((r for r in results if 'exp4' in r['experiment']), None),
        'AdamW': next((r for r in results if 'exp5' in r['experiment']), None),
        '数据增强': next((r for r in results if 'exp6' in r['experiment']), None),
        '组合优化': next((r for r in results if 'exp7' in r['experiment']), None),
    }

    # 计算变化百分比
    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
    impact_data = []

    for param_name, exp in experiments.items():
        if not exp:
            continue

        row = {'超参数': param_name}
        for metric in metrics:
            baseline_val = baseline[metric]
            exp_val = exp[metric]
            change_pct = ((exp_val - baseline_val) / baseline_val) * 100
            row[metric] = f"{change_pct:+.2f}%"

        impact_data.append(row)

    # 创建表格图
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    headers = ['超参数修改'] + metrics
    table_data = [[row['超参数']] + [row.get(m, 'N/A') for m in metrics]
                  for row in impact_data]

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置单元格颜色（根据正负值）
    for i in range(1, len(table_data) + 1):
        table[(i, 0)].set_facecolor('#F0F0F0')
        for j in range(1, len(headers)):
            cell_text = table_data[i-1][j]
            if cell_text != 'N/A':
                value = float(cell_text.rstrip('%'))
                if value > 0:
                    table[(i, j)].set_facecolor('#D4EDDA')  # 绿色
                elif value < 0:
                    table[(i, j)].set_facecolor('#F8D7DA')  # 红色
                else:
                    table[(i, j)].set_facecolor('#FFF3CD')  # 黄色

    plt.title('超参数影响汇总表 (相对基线的变化百分比)', fontsize=14, fontweight='bold', pad=20)

    output_path = os.path.join(OUTPUT_DIR, 'impact_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 影响汇总表已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("="*70)
    print("超参数影响深度分析")
    print("="*70)

    results = load_results()
    if not results:
        return

    print("\n开始分析各超参数影响...\n")

    analyze_batch_size_impact(results)
    analyze_learning_rate_impact(results)
    analyze_optimizer_impact(results)
    analyze_augmentation_impact(results)
    create_impact_summary(results)

    print("\n" + "="*70)
    print(f"所有分析图表已保存到: {OUTPUT_DIR}/")
    print("="*70)
    print("\n生成的图表:")
    print("  1. batch_size_impact.png     - 批次大小影响")
    print("  2. learning_rate_impact.png  - 学习率影响")
    print("  3. optimizer_impact.png      - 优化器影响")
    print("  4. augmentation_impact.png   - 数据增强影响")
    print("  5. impact_summary_table.png  - 影响汇总表")


if __name__ == '__main__':
    main()
