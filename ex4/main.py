"""
YOLO实验主控脚本
功能：一键运行训练、验证、可视化全流程
作者：实验四
"""

import os
import sys
import argparse

def print_banner():
    """打印标题"""
    print("="*70)
    print("    YOLO模型实验 - 配电箱门未关闭检测")
    print("    实验四：智能算力平台应用（二）")
    print("="*70)
    print()


def print_menu():
    """打印菜单"""
    print("\n请选择要执行的操作：")
    print("  1. 准备数据集（从NFS挂载点划分训练集和验证集）")
    print("  2. 训练所有模型（包含7个实验配置）")
    print("  3. 验证所有模型（评估性能指标）")
    print("  4. 生成基础可视化图表（6张）")
    print("  5. 生成参数影响详细图表（7张，推荐用于报告）⭐")
    print("  6. 生成所有可视化图表（基础+训练曲线+超参数分析）")
    print("  7. 运行完整流程（训练+验证+所有可视化）")
    print("  8. 快速实验（仅训练前3个模型）")
    print("  0. 退出")
    print()


def prepare_dataset():
    """准备数据集"""
    print("\n" + "="*70)
    print("步骤1: 准备数据集")
    print("="*70)
    print("从NFS挂载点划分训练集和验证集...")
    os.system(f"{sys.executable} prepare_data.py")


def train_all():
    """训练所有模型"""
    print("\n" + "="*70)
    print("步骤2: 训练所有模型")
    print("="*70)
    print("这将训练7个不同配置的模型，可能需要较长时间...")
    print("预计时间：2-4小时（取决于GPU性能）")
    input("按回车键继续，或Ctrl+C取消...")
    os.system(f"{sys.executable} train_all_experiments.py")


def train_quick():
    """快速训练（仅前3个模型）"""
    print("\n" + "="*70)
    print("快速实验模式")
    print("="*70)
    print("将只训练前3个模型进行快速测试...")
    os.system(f"{sys.executable} train_quick.py")


def validate_all():
    """验证所有模型"""
    print("\n" + "="*70)
    print("步骤3: 验证所有模型")
    print("="*70)
    print("在验证集上评估所有训练好的模型...")
    os.system(f"{sys.executable} validate_all_models.py")


def visualize():
    """生成基础可视化图表"""
    print("\n" + "="*70)
    print("生成基础可视化图表")
    print("="*70)
    print("生成6张基础对比图表...")
    os.system(f"{sys.executable} visualize_results.py")


def visualize_parameter_effects():
    """生成参数影响详细图表"""
    print("\n" + "="*70)
    print("生成参数影响详细图表")
    print("="*70)
    print("生成7张高质量参数影响分析图表...")
    os.system(f"{sys.executable} visualize_parameter_effects.py")


def visualize_all():
    """生成所有可视化图表"""
    print("\n" + "="*70)
    print("生成所有可视化图表")
    print("="*70)
    print("\n1/4 基础对比图表...")
    os.system(f"{sys.executable} visualize_results.py")

    print("\n2/4 参数影响详细图表...")
    os.system(f"{sys.executable} visualize_parameter_effects.py")

    print("\n3/4 训练曲线分析...")
    os.system(f"{sys.executable} analyze_training.py")

    print("\n4/4 超参数深度分析...")
    os.system(f"{sys.executable} analyze_hyperparameters.py")

    print("\n✓ 所有可视化图表生成完成！")


def run_full_pipeline():
    """运行完整流程"""
    print("\n" + "="*70)
    print("运行完整实验流程")
    print("="*70)
    print("将依次执行：训练 -> 验证 -> 所有可视化")
    input("按回车键继续，或Ctrl+C取消...")

    train_all()
    print("\n✓ 训练完成！继续验证...")
    validate_all()
    print("\n✓ 验证完成！生成所有图表...")
    visualize_all()

    print("\n" + "="*70)
    print("完整流程执行完毕！")
    print("="*70)
    print("\n实验结果：")
    print("  - 训练记录: experiment_log.json")
    print("  - 验证结果: validation_results.json, validation_results.csv")
    print("  - 基础图表: visualization_results/ (6张)")
    print("  - 参数影响: parameter_effects_visualization/ (7张)")
    print("  - 训练曲线: training_curves/ (8+张)")
    print("  - 超参数分析: hyperparameter_analysis/ (5张)")
    print("\n总计: 26+张图表，所有结果可直接用于实验报告！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO实验主控脚本')
    parser.add_argument('--auto', type=int, choices=[1,2,3,4,5,6,7,8],
                       help='自动运行指定步骤（1-8）')
    args = parser.parse_args()

    print_banner()

    # 自动模式
    if args.auto:
        choice = str(args.auto)
    else:
        # 交互模式
        while True:
            print_menu()
            choice = input("请输入选项 (0-8): ").strip()

            if choice not in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
                print("❌ 无效选项，请重新输入！")
                continue
            break

    # 执行相应操作
    try:
        if choice == '0':
            print("退出程序")
            return
        elif choice == '1':
            prepare_dataset()
        elif choice == '2':
            train_all()
        elif choice == '3':
            validate_all()
        elif choice == '4':
            visualize()
        elif choice == '5':
            visualize_parameter_effects()
        elif choice == '6':
            visualize_all()
        elif choice == '7':
            run_full_pipeline()
        elif choice == '8':
            train_quick()

    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        return
    except Exception as e:
        print(f"\n❌ 执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("\n操作完成！")


if __name__ == '__main__':
    main()
