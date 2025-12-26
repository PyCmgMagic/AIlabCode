"""
训练曲线分析脚本
功能：读取训练结果，绘制训练过程曲线
作者：实验四
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
from pathlib import Path

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
OUTPUT_DIR = 'training_curves'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_results_csv(exp_name):
    """查找训练结果CSV文件"""
    results_path = f"runs/detect/{exp_name}/results.csv"
    if os.path.exists(results_path):
        return results_path
    return None


def plot_training_curves(exp_name, exp_desc):
    """
    绘制单个实验的训练曲线
    """
    csv_path = find_results_csv(exp_name)

    if not csv_path:
        print(f"⚠️  未找到 {exp_name} 的训练结果")
        return None

    try:
        # 读取CSV
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # 去除列名空格

        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'训练曲线 - {exp_desc}', fontsize=16, fontweight='bold')

        # 图1: Loss曲线
        ax = axes[0, 0]
        if 'train/box_loss' in df.columns:
            ax.plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
        if 'train/cls_loss' in df.columns:
            ax.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', linewidth=2)
        if 'train/dfl_loss' in df.columns:
            ax.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('训练Loss')
        ax.legend()
        ax.grid(alpha=0.3)

        # 图2: mAP曲线
        ax = axes[0, 1]
        if 'metrics/mAP50(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=2, color='#FF6B6B')
        if 'metrics/mAP50-95(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2, color='#4ECDC4')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('mAP指标')
        ax.legend()
        ax.grid(alpha=0.3)

        # 图3: Precision和Recall
        ax = axes[0, 2]
        if 'metrics/precision(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='#95E1D3')
        if 'metrics/recall(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='#F38181')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision & Recall')
        ax.legend()
        ax.grid(alpha=0.3)

        # 图4: 学习率
        ax = axes[1, 0]
        if 'lr/pg0' in df.columns:
            ax.plot(df['epoch'], df['lr/pg0'], label='LR-pg0', linewidth=2, color='#FFA07A')
        if 'lr/pg1' in df.columns:
            ax.plot(df['epoch'], df['lr/pg1'], label='LR-pg1', linewidth=2, color='#98D8C8')
        if 'lr/pg2' in df.columns:
            ax.plot(df['epoch'], df['lr/pg2'], label='LR-pg2', linewidth=2, color='#F7DC6F')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('学习率变化')
        ax.legend()
        ax.grid(alpha=0.3)

        # 图5: Validation Loss
        ax = axes[1, 1]
        if 'val/box_loss' in df.columns:
            ax.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2, color='#E74C3C')
        if 'val/cls_loss' in df.columns:
            ax.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', linewidth=2, color='#3498DB')
        if 'val/dfl_loss' in df.columns:
            ax.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', linewidth=2, color='#2ECC71')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('验证Loss')
        ax.legend()
        ax.grid(alpha=0.3)

        # 图6: 统计信息
        ax = axes[1, 2]
        ax.axis('off')

        # 计算统计信息
        final_epoch = df['epoch'].iloc[-1]
        best_map50 = df['metrics/mAP50(B)'].max() if 'metrics/mAP50(B)' in df.columns else 0
        best_map50_95 = df['metrics/mAP50-95(B)'].max() if 'metrics/mAP50-95(B)' in df.columns else 0
        final_precision = df['metrics/precision(B)'].iloc[-1] if 'metrics/precision(B)' in df.columns else 0
        final_recall = df['metrics/recall(B)'].iloc[-1] if 'metrics/recall(B)' in df.columns else 0

        stats_text = f"""
        训练统计
        {'='*40}

        实验名称: {exp_desc}

        训练轮数: {int(final_epoch)}

        最佳mAP50: {best_map50:.4f}
        最佳mAP50-95: {best_map50_95:.4f}

        最终Precision: {final_precision:.4f}
        最终Recall: {final_recall:.4f}

        训练结果保存在:
        runs/detect/{exp_name}/
        """

        ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'{exp_name}_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 训练曲线已保存: {output_path}")
        plt.close()

        return df

    except Exception as e:
        print(f"❌ 绘制 {exp_name} 曲线时出错: {str(e)}")
        return None


def plot_comparison_curves(experiments):
    """
    绘制多个实验的对比曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('多实验训练曲线对比', fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(range(len(experiments)))

    for idx, (exp_name, exp_desc) in enumerate(experiments):
        csv_path = find_results_csv(exp_name)
        if not csv_path:
            continue

        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            # mAP50对比
            if 'metrics/mAP50(B)' in df.columns:
                axes[0, 0].plot(df['epoch'], df['metrics/mAP50(B)'],
                              label=exp_desc, linewidth=2, color=colors[idx])

            # mAP50-95对比
            if 'metrics/mAP50-95(B)' in df.columns:
                axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'],
                              label=exp_desc, linewidth=2, color=colors[idx])

            # Precision对比
            if 'metrics/precision(B)' in df.columns:
                axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'],
                              label=exp_desc, linewidth=2, color=colors[idx])

            # Recall对比
            if 'metrics/recall(B)' in df.columns:
                axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'],
                              label=exp_desc, linewidth=2, color=colors[idx])

        except Exception as e:
            print(f"⚠️  跳过 {exp_name}: {str(e)}")
            continue

    # 设置各子图
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('mAP50')
    axes[0, 0].set_title('mAP50对比')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP50-95')
    axes[0, 1].set_title('mAP50-95对比')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision对比')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall对比')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'comparison_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比曲线已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("="*70)
    print("训练曲线分析")
    print("="*70)

    # 定义实验列表
    EXPERIMENTS = [
        ('exp1_yolo_n_baseline', 'Nano基线'),
        ('exp2_yolo_s_baseline', 'Small基线'),
        ('exp3_yolo_n_batch8', 'Nano-批次8'),
        ('exp4_yolo_n_lr0.001', 'Nano-学习率0.001'),
        ('exp5_yolo_n_adamw', 'Nano-AdamW'),
        ('exp6_yolo_n_augment', 'Nano-数据增强'),
        ('exp7_yolo_n_combined', 'Nano-组合优化'),
    ]

    # 绘制每个实验的曲线
    print("\n绘制各实验训练曲线...")
    for exp_name, exp_desc in EXPERIMENTS:
        plot_training_curves(exp_name, exp_desc)

    # 绘制对比曲线
    print("\n绘制对比曲线...")
    plot_comparison_curves(EXPERIMENTS)

    print("\n" + "="*70)
    print(f"所有训练曲线已保存到: {OUTPUT_DIR}/")
    print("="*70)


if __name__ == '__main__':
    main()
