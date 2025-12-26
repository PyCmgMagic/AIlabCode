# YOLO模型训练实验 - 配电箱门未关闭检测

实验四：智能算力平台应用（二）

## 实验目标

1. 使用YOLO模型训练配电箱门未关闭检测模型
2. 训练两个不同尺寸的模型（Nano和Small）
3. 修改至少4个超参数并对比其影响
4. 使用至少5个指标评价模型性能
5. 生成可视化图表用于实验报告

## 文件说明

### 核心脚本

- **main.py** - 主控脚本，提供交互式菜单
- **train_all_experiments.py** - 完整训练脚本（7个实验配置）
- **train_quick.py** - 快速训练脚本（3个实验配置，用于测试）
- **validate_all_models.py** - 验证所有模型，收集性能指标
- **visualize_results.py** - 生成可视化对比图表

### 辅助脚本

- **prepare_data.py** - 准备数据集（从NFS挂载点划分训练/验证集）
- **distribution_box.yaml** - 数据集配置文件

### 输出文件

- **experiment_log.json** - 训练实验记录
- **validation_results.json** - 验证结果（JSON格式）
- **validation_results.csv** - 验证结果（CSV格式，可用Excel打开）
- **visualization_results/** - 可视化图表目录

## 快速开始

### 方法1：使用主控脚本（推荐）

```bash
python main.py
```

然后按照菜单提示选择操作：
- 选项1：准备数据集
- 选项2：训练所有模型
- 选项3：验证所有模型
- 选项4：生成可视化图表
- 选项5：运行完整流程
- 选项6：快速实验（测试用）

### 方法2：自动化执行

```bash
# 运行完整流程
python main.py --auto 5

# 仅训练
python main.py --auto 2

# 仅验证
python main.py --auto 3

# 仅可视化
python main.py --auto 4
```

### 方法3：分步执行

```bash
# 步骤1: 准备数据集
python prepare_data.py

# 步骤2: 训练模型
python train_all_experiments.py

# 步骤3: 验证模型
python validate_all_models.py

# 步骤4: 生成图表
python visualize_results.py
```

## 实验配置说明

### 7个实验配置

| 实验编号 | 模型 | 配置说明 | 主要修改 |
|---------|------|---------|---------|
| 实验1 | Nano | 基线配置 | epochs=50, batch=16 |
| 实验2 | Small | 基线配置 | epochs=50, batch=16 |
| 实验3 | Nano | 减小批次 | **batch=8** |
| 实验4 | Nano | 降低学习率 | **lr0=0.001** |
| 实验5 | Nano | AdamW优化器 | **optimizer='AdamW'** |
| 实验6 | Nano | 增强数据增强 | **hsv_h/s/v, degrees** |
| 实验7 | Nano | 组合优化 | batch=8 + lr0=0.001 + AdamW + 数据增强 |

### 修改的超参数

1. **batch** (批次大小): 16 → 8
2. **lr0** (初始学习率): 0.01 → 0.001
3. **optimizer** (优化器): SGD → AdamW
4. **数据增强参数**: hsv_h, hsv_s, hsv_v, degrees

### 评估指标（5个以上）

1. **mAP50** - AP at IoU=0.5
2. **mAP50-95** - AP at IoU=0.5:0.95
3. **mAP75** - AP at IoU=0.75
4. **Precision** - 精确率
5. **Recall** - 召回率
6. **F1-Score** - F1分数
7. **Fitness** - 综合适应度

## 环境要求

```bash
# 安装依赖
pip install ultralytics
pip install pandas
pip install matplotlib

# 如果使用NFS数据集
sudo apt install nfs-common
sudo mkdir /mnt/nfsdir
sudo mount 172.26.1.21:/mnt/diska /mnt/nfsdir
```

## 数据集结构

```
my_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── ...
    └── val/
        ├── image1.txt
        └── ...
```

## 生成的图表

运行 `visualize_results.py` 后，将在 `visualization_results/` 目录生成以下图表：

1. **metrics_comparison.png** - 6个主要指标对比柱状图
2. **radar_chart.png** - 前3个模型的性能雷达图
3. **model_size_comparison.png** - 不同模型尺寸对比
4. **hyperparameter_impact.png** - 超参数影响分析
5. **top_models_ranking.png** - 模型性能排名
6. **summary_table.png** - 结果汇总表

这些图表可以直接插入到实验报告中！

## 实验报告建议

### 需要记录的内容

1. **实验环境**
   - GPU/CPU型号
   - CUDA版本
   - Ultralytics版本
   - 数据集规模

2. **训练过程**
   - 每个实验的训练时间
   - 训练过程中的loss曲线
   - 最佳模型出现的epoch

3. **实验结果**
   - 使用 `validation_results.csv` 中的数据
   - 插入 `visualization_results/` 中的图表
   - 分析不同超参数的影响

4. **问题与解决方案**（至少2个）
   - 例如：GPU内存不足 → 减小batch size
   - 例如：过拟合 → 增加数据增强
   - 例如：训练不稳定 → 降低学习率

5. **结论**
   - 哪个模型表现最好
   - 哪些超参数影响最大
   - 对实际应用的建议

## 常见问题

### Q1: GPU内存不足怎么办？

```python
# 在脚本中修改batch大小
batch=8  # 或更小，如 batch=4
```

### Q2: 没有GPU怎么办？

```python
# 修改device参数
device='cpu'  # 使用CPU训练（会比较慢）
```

### Q3: 训练时间太长怎么办？

```python
# 方案1: 减少epochs
epochs=30  # 替代默认的50

# 方案2: 使用快速实验模式
python train_quick.py
```

### Q4: 数据集路径错误怎么办？

修改 `distribution_box.yaml` 文件中的path参数：

```yaml
path: /your/actual/dataset/path
```

## 训练进度查看

训练过程中，可以实时查看：

```bash
# 查看训练日志
tensorboard --logdir runs/detect

# 访问 http://localhost:6006
```

## 提交清单

- [ ] 所有训练脚本代码
- [ ] 验证脚本代码
- [ ] 可视化脚本代码
- [ ] 实验记录文件 (experiment_log.json)
- [ ] 验证结果文件 (validation_results.csv)
- [ ] 可视化图表 (visualization_results/*.png)
- [ ] 实验报告（包含问题和解决方案）

## 联系方式

如有问题，请联系助教或在课程群中提问。

---

**祝实验顺利！**
