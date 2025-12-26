"""
参数影响详细可视化脚本
功能：生成高质量的参数调节效果图，适合实验报告使用
作者：实验四
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os
from matplotlib.patches import Rectangle
import seaborn as sns

# 设置中文字体和绘图风格
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 创建输出目录
OUTPUT_DIR = 'parameter_effects_visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义配色方案
COLORS = {
    'baseline': '#3498DB',
    'improved': '#2ECC71',
    'degraded': '#E74C3C',
    'neutral': '#95A5A6'
}


def load_results():
    """加载验证结果"""
    try:
        with open('validation_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 未找到验证结果文件 validation_results.json")
        print("   请先运行 validate_all_models.py")
        return None


def create_parameter_comparison_grid(results):
    """
    创建参数对比网格图 - 展示所有参数的影响
    """
    baseline = next((r for r in results if 'exp1' in r['experiment']), None)
    if not baseline:
        print("⚠️  未找到基线模型")
        return

    # 收集各参数实验
    experiments = {
        'Batch Size\n(16→8)': next((r for r in results if 'exp3' in r['experiment']), None),
        'Learning Rate\n(0.01→0.001)': next((r for r in results if 'exp4' in r['experiment']), None),
        'Optimizer\n(SGD→AdamW)': next((r for r in results if 'exp5' in r['experiment']), None),
        'Data Aug\n(Enhanced)': next((r for r in results if 'exp6' in r['experiment']), None),
    }

    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('超参数调整对模型性能的影响分析', fontsize=18, fontweight='bold', y=0.995)

    axes = axes.flatten()

    for idx, (param_name, exp) in enumerate(experiments.items()):
        if exp is None:
            continue

        ax = axes[idx]

        # 计算变化
        baseline_values = [baseline[m] for m in metrics]
        exp_values = [exp[m] for m in metrics]
        changes = [(exp[m] - baseline[m]) / baseline[m] * 100 for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        # 绘制基线和修改后的值
        bars1 = ax.bar(x - width/2, baseline_values, width,
                       label='基线配置', color=COLORS['baseline'],
                       edgecolor='black', linewidth=1.2, alpha=0.8)

        # 根据变化给bars着色
        bar_colors = [COLORS['improved'] if c > 0 else COLORS['degraded'] for c in changes]
        bars2 = ax.bar(x + width/2, exp_values, width,
                       label=param_name.replace('\n', ' '),
                       color=bar_colors, edgecolor='black',
                       linewidth=1.2, alpha=0.8)

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        for bar, change in zip(bars2, changes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 添加变化百分比箭头
        for i, (b1, b2, change) in enumerate(zip(baseline_values, exp_values, changes)):
            y_pos = max(b1, b2) + 0.08
            arrow_color = 'green' if change > 0 else 'red'
            arrow = '↑' if change > 0 else '↓'
            ax.text(i, y_pos, f'{arrow} {abs(change):.1f}%',
                   ha='center', fontsize=10, color=arrow_color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=arrow_color, alpha=0.7))

        ax.set_ylabel('指标值', fontsize=11, fontweight='bold')
        ax.set_title(f'参数调整: {param_name}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.2])

        # 添加背景色区分
        ax.axhspan(0, 0.3, alpha=0.05, color='red')
        ax.axhspan(0.3, 0.7, alpha=0.05, color='yellow')
        ax.axhspan(0.7, 1.0, alpha=0.05, color='green')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '1_parameter_comparison_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 参数对比网格图: {output_path}")
    plt.close()


def create_change_percentage_heatmap(results):
    """
    创建变化百分比热力图
    """
    baseline = next((r for r in results if 'exp1' in r['experiment']), None)
    if not baseline:
        return

    experiments = {
        'Batch=8': next((r for r in results if 'exp3' in r['experiment']), None),
        'LR=0.001': next((r for r in results if 'exp4' in r['experiment']), None),
        'AdamW': next((r for r in results if 'exp5' in r['experiment']), None),
        'Data Aug': next((r for r in results if 'exp6' in r['experiment']), None),
        'Combined': next((r for r in results if 'exp7' in r['experiment']), None),
    }

    metrics = ['mAP50', 'mAP50-95', 'mAP75', 'Precision', 'Recall', 'F1-Score']

    # 构建变化百分比矩阵
    change_matrix = []
    exp_names = []

    for name, exp in experiments.items():
        if exp:
            changes = [(exp[m] - baseline[m]) / baseline[m] * 100 for m in metrics]
            change_matrix.append(changes)
            exp_names.append(name)

    change_matrix = np.array(change_matrix)

    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 使用seaborn的热力图
    im = ax.imshow(change_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-10, vmax=10, interpolation='nearest')

    # 设置刻度
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(exp_names)))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_yticklabels(exp_names, fontsize=11)

    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 添加数值标注
    for i in range(len(exp_names)):
        for j in range(len(metrics)):
            value = change_matrix[i, j]
            color = 'white' if abs(value) > 5 else 'black'
            text = ax.text(j, i, f'{value:.1f}%',
                          ha="center", va="center", color=color,
                          fontsize=10, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('变化百分比 (%)', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

    ax.set_title('超参数调整对各指标的影响热力图\n(相对基线的变化百分比)',
                fontsize=14, fontweight='bold', pad=15)

    # 添加说明
    ax.text(0.5, -0.15, '绿色表示性能提升，红色表示性能下降',
           transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '2_change_percentage_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 变化百分比热力图: {output_path}")
    plt.close()


def create_parameter_sensitivity_analysis(results):
    """
    创建参数敏感性分析图
    """
    baseline = next((r for r in results if 'exp1' in r['experiment']), None)
    if not baseline:
        return

    experiments = {
        'Batch Size': next((r for r in results if 'exp3' in r['experiment']), None),
        'Learning Rate': next((r for r in results if 'exp4' in r['experiment']), None),
        'Optimizer': next((r for r in results if 'exp5' in r['experiment']), None),
        'Data Aug': next((r for r in results if 'exp6' in r['experiment']), None),
    }

    # 计算每个参数对各指标的平均影响
    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']

    fig, ax = plt.subplots(figsize=(12, 8))

    sensitivity_scores = {}
    for param_name, exp in experiments.items():
        if exp:
            changes = [abs((exp[m] - baseline[m]) / baseline[m] * 100) for m in metrics]
            avg_change = np.mean(changes)
            sensitivity_scores[param_name] = {
                'avg': avg_change,
                'changes': changes
            }

    # 按平均变化排序
    sorted_params = sorted(sensitivity_scores.items(), key=lambda x: x[1]['avg'], reverse=True)

    param_names = [p[0] for p in sorted_params]
    avg_changes = [p[1]['avg'] for p in sorted_params]

    # 创建水平条形图
    y_pos = np.arange(len(param_names))
    colors_list = plt.cm.viridis(np.linspace(0.3, 0.9, len(param_names)))

    bars = ax.barh(y_pos, avg_changes, color=colors_list,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, avg_changes)):
        ax.text(value + 0.1, bar.get_y() + bar.get_height()/2,
               f'{value:.2f}%',
               ha='left', va='center', fontsize=11, fontweight='bold')

        # 添加各指标的详细变化
        changes = sorted_params[i][1]['changes']
        detail_text = f"详细: {', '.join([f'{c:.1f}%' for c in changes])}"
        ax.text(value + 0.1, bar.get_y() + bar.get_height()/2 - 0.15,
               detail_text, ha='left', va='center', fontsize=8, style='italic')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names, fontsize=12, fontweight='bold')
    ax.set_xlabel('平均影响程度 (绝对变化百分比)', fontsize=12, fontweight='bold')
    ax.set_title('超参数敏感性分析\n(不同参数对模型性能的平均影响)',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 添加说明
    ax.text(0.98, 0.02, f'评估指标: {", ".join(metrics)}',
           transform=ax.transAxes, ha='right', va='bottom',
           fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '3_parameter_sensitivity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 参数敏感性分析图: {output_path}")
    plt.close()


def create_before_after_comparison(results):
    """
    创建修改前后对比的详细展示
    """
    baseline = next((r for r in results if 'exp1' in r['experiment']), None)
    if not baseline:
        return

    experiments = [
        ('Batch Size (16→8)', next((r for r in results if 'exp3' in r['experiment']), None)),
        ('Learning Rate (0.01→0.001)', next((r for r in results if 'exp4' in r['experiment']), None)),
        ('Optimizer (SGD→AdamW)', next((r for r in results if 'exp5' in r['experiment']), None)),
        ('Data Aug (Enhanced)', next((r for r in results if 'exp6' in r['experiment']), None)),
    ]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

    fig.suptitle('超参数修改前后性能对比详细分析', fontsize=16, fontweight='bold')

    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']

    for idx, (param_name, exp) in enumerate(experiments):
        if exp is None:
            continue

        # 上半部分：柱状图对比
        ax1 = fig.add_subplot(gs[0, idx])

        baseline_values = [baseline[m] for m in metrics]
        exp_values = [exp[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax1.bar(x - width/2, baseline_values, width,
                       label='修改前', color='lightblue',
                       edgecolor='navy', linewidth=1.2)
        bars2 = ax1.bar(x + width/2, exp_values, width,
                       label='修改后', color='lightcoral',
                       edgecolor='darkred', linewidth=1.2)

        ax1.set_ylabel('指标值', fontsize=9, fontweight='bold')
        ax1.set_title(param_name, fontsize=10, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, fontsize=8, rotation=45, ha='right')
        ax1.legend(fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])

        # 下半部分：变化百分比
        ax2 = fig.add_subplot(gs[1, idx])

        changes = [(exp[m] - baseline[m]) / baseline[m] * 100 for m in metrics]
        colors = ['green' if c > 0 else 'red' for c in changes]

        bars = ax2.bar(metrics, changes, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.2)

        # 添加数值标签
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            label_y = height + 0.5 if height > 0 else height - 0.5
            ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{change:+.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_ylabel('变化 (%)', fontsize=9, fontweight='bold')
        ax2.set_title('变化百分比', fontsize=10)
        ax2.set_xticklabels(metrics, fontsize=8, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        # 计算平均改善
        avg_change = np.mean(changes)
        ax2.text(0.95, 0.95, f'平均: {avg_change:+.1f}%',
                transform=ax2.transAxes, ha='right', va='top',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '4_before_after_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 修改前后对比图: {output_path}")
    plt.close()


def create_combined_effect_analysis(results):
    """
    创建组合效果分析图
    """
    baseline = next((r for r in results if 'exp1' in r['experiment']), None)
    combined = next((r for r in results if 'exp7' in r['experiment']), None)

    if not baseline or not combined:
        print("⚠️  跳过组合效果分析")
        return

    individual_exps = {
        'Batch=8': next((r for r in results if 'exp3' in r['experiment']), None),
        'LR=0.001': next((r for r in results if 'exp4' in r['experiment']), None),
        'AdamW': next((r for r in results if 'exp5' in r['experiment']), None),
        'Data Aug': next((r for r in results if 'exp6' in r['experiment']), None),
    }

    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('超参数组合优化效果分析', fontsize=16, fontweight='bold')

    # 左图：组合效果 vs 单个效果
    ax1 = axes[0]

    # 计算各配置的平均提升
    improvements = {'Baseline': 0}

    for name, exp in individual_exps.items():
        if exp:
            changes = [(exp[m] - baseline[m]) / baseline[m] * 100 for m in metrics]
            improvements[name] = np.mean(changes)

    combined_changes = [(combined[m] - baseline[m]) / baseline[m] * 100 for m in metrics]
    improvements['Combined\n(All)'] = np.mean(combined_changes)

    names = list(improvements.keys())
    values = list(improvements.values())
    colors_list = ['gray'] + ['lightblue']*4 + ['red']

    bars = ax1.barh(names, values, color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)

    # 添加数值标签
    for bar, value in zip(bars, values):
        label_x = value + 0.2 if value > 0 else value - 0.2
        ax1.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{value:+.2f}%',
                ha='left' if value > 0 else 'right', va='center',
                fontsize=11, fontweight='bold')

    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('平均性能提升 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('单独优化 vs 组合优化', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # 右图：组合优化在各指标上的表现
    ax2 = axes[1]

    baseline_values = [baseline[m] for m in metrics]
    combined_values = [combined[m] for m in metrics]
    changes = [(combined[m] - baseline[m]) / baseline[m] * 100 for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax2.bar(x - width/2, baseline_values, width,
                   label='基线配置', color='#3498DB',
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    bars2 = ax2.bar(x + width/2, combined_values, width,
                   label='组合优化', color='#E74C3C',
                   edgecolor='black', linewidth=1.2, alpha=0.8)

    # 添加数值
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    for bar, change in zip(bars2, changes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\n({change:+.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylabel('指标值', fontsize=12, fontweight='bold')
    ax2.set_title('组合优化各指标表现', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '5_combined_effect_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 组合效果分析图: {output_path}")
    plt.close()


def create_model_size_comparison(results):
    """
    创建模型大小对比的详细图
    """
    nano = next((r for r in results if 'exp1' in r['experiment']), None)
    small = next((r for r in results if 'exp2' in r['experiment']), None)

    if not nano or not small:
        print("⚠️  跳过模型大小对比")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('不同模型尺寸性能对比分析', fontsize=16, fontweight='bold')

    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']

    # 图1：直接对比
    ax1 = axes[0]
    nano_values = [nano[m] for m in metrics]
    small_values = [small[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, nano_values, width,
                   label='Nano (轻量)', color='#95E1D3',
                   edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, small_values, width,
                   label='Small (标准)', color='#F38181',
                   edgecolor='black', linewidth=1.2)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    ax1.set_ylabel('指标值', fontsize=11, fontweight='bold')
    ax1.set_title('各指标对比', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.1])

    # 图2：性能提升百分比
    ax2 = axes[1]
    improvements = [(small[m] - nano[m]) / nano[m] * 100 for m in metrics]
    colors = ['green' if i > 0 else 'red' for i in improvements]

    bars = ax2.bar(metrics, improvements, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.2)

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        label_y = height + 0.3 if height > 0 else height - 0.3
        ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{imp:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Small相对Nano的提升 (%)', fontsize=11, fontweight='bold')
    ax2.set_title('性能提升幅度', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(metrics, fontsize=10, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # 图3：雷达图对比
    ax3 = axes[2]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    nano_values_radar = nano_values + nano_values[:1]
    small_values_radar = small_values + small_values[:1]
    angles += angles[:1]

    ax3 = plt.subplot(133, projection='polar')
    ax3.plot(angles, nano_values_radar, 'o-', linewidth=2, label='Nano', color='#95E1D3')
    ax3.fill(angles, nano_values_radar, alpha=0.25, color='#95E1D3')
    ax3.plot(angles, small_values_radar, 'o-', linewidth=2, label='Small', color='#F38181')
    ax3.fill(angles, small_values_radar, alpha=0.25, color='#F38181')

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metrics, fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.set_title('雷达图对比', fontsize=12, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax3.grid(True)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '6_model_size_comparison_detailed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 模型尺寸对比详细图: {output_path}")
    plt.close()


def create_comprehensive_summary(results):
    """
    创建综合总结图表
    """
    baseline = next((r for r in results if 'exp1' in r['experiment']), None)
    if not baseline:
        return

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('超参数调整实验综合总结报告', fontsize=18, fontweight='bold', y=0.98)

    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']

    # 收集所有实验数据
    all_experiments = {
        'Baseline': baseline,
        'Batch=8': next((r for r in results if 'exp3' in r['experiment']), None),
        'LR=0.001': next((r for r in results if 'exp4' in r['experiment']), None),
        'AdamW': next((r for r in results if 'exp5' in r['experiment']), None),
        'Data Aug': next((r for r in results if 'exp6' in r['experiment']), None),
        'Combined': next((r for r in results if 'exp7' in r['experiment']), None),
    }

    # 子图1-5：各指标的对比
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])

        exp_names = []
        values = []

        for name, exp in all_experiments.items():
            if exp:
                exp_names.append(name)
                values.append(exp[metric])

        colors_list = ['#3498DB'] + ['#95A5A6']*4 + ['#E74C3C']
        bars = ax.bar(exp_names, values, color=colors_list[:len(values)],
                     edgecolor='black', linewidth=1.2, alpha=0.8)

        # 标注最佳值
        best_idx = np.argmax(values)
        bars[best_idx].set_color('#2ECC71')
        bars[best_idx].set_edgecolor('darkgreen')
        bars[best_idx].set_linewidth(2)

        # 添加数值
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., value,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} 对比', fontsize=12, fontweight='bold')
        ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, max(values) * 1.15])

    # 子图6：平均性能提升
    ax = fig.add_subplot(gs[1, 2])

    avg_improvements = {}
    for name, exp in all_experiments.items():
        if exp and name != 'Baseline':
            changes = [(exp[m] - baseline[m]) / baseline[m] * 100 for m in metrics]
            avg_improvements[name] = np.mean(changes)

    names = list(avg_improvements.keys())
    values = list(avg_improvements.values())
    colors = ['green' if v > 0 else 'red' for v in values]

    bars = ax.barh(names, values, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.2)

    for bar, value in zip(bars, values):
        ax.text(value + 0.1, bar.get_y() + bar.get_height()/2,
               f'{value:+.2f}%',
               ha='left', va='center', fontsize=10, fontweight='bold')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('平均提升 (%)', fontsize=11, fontweight='bold')
    ax.set_title('平均性能提升', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 子图7-9：总结文本
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')

    # 找出最佳配置
    best_config = max(avg_improvements.items(), key=lambda x: x[1])

    summary_text = f"""
    实验总结
    {'='*100}

    ✓ 总实验数: {len([e for e in all_experiments.values() if e])} 个配置

    ✓ 最佳配置: {best_config[0]} (平均提升: {best_config[1]:+.2f}%)

    ✓ 各超参数影响排序:
    """

    sorted_params = sorted(avg_improvements.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (name, value) in enumerate(sorted_params, 1):
        summary_text += f"\n      {i}. {name:<15} : {value:+.2f}% 平均变化"

    summary_text += f"\n\n    ✓ 评估指标: {', '.join(metrics)}"

    ax.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '7_comprehensive_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 综合总结图: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("="*70)
    print("参数影响详细可视化 - 实验报告专用")
    print("="*70)

    results = load_results()
    if not results:
        return

    print(f"\n加载了 {len(results)} 个实验结果")
    print("\n开始生成详细可视化图表...\n")

    # 生成所有图表
    create_parameter_comparison_grid(results)
    create_change_percentage_heatmap(results)
    create_parameter_sensitivity_analysis(results)
    create_before_after_comparison(results)
    create_combined_effect_analysis(results)
    create_model_size_comparison(results)
    create_comprehensive_summary(results)

    print("\n" + "="*70)
    print(f"所有图表已保存到: {OUTPUT_DIR}/")
    print("="*70)
    print("\n生成的7张高质量图表:")
    print("  1. parameter_comparison_grid.png        - 参数对比网格 (2x2)")
    print("  2. change_percentage_heatmap.png        - 变化百分比热力图")
    print("  3. parameter_sensitivity.png            - 参数敏感性分析")
    print("  4. before_after_comparison.png          - 修改前后详细对比")
    print("  5. combined_effect_analysis.png         - 组合优化效果")
    print("  6. model_size_comparison_detailed.png   - 模型尺寸详细对比")
    print("  7. comprehensive_summary.png            - 综合总结报告")
    print("\n这些图表适合直接用于实验报告！")


if __name__ == '__main__':
    main()
