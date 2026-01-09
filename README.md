# RecSys-BiRetrieval

基于双塔对比学习的蛋白质-分子双向检索系统。

## 作者

- **赵桐** - [https://github.com/TongZhao1030](https://github.com/TongZhao1030)
- **杜伊诺** - [https://github.com/DDDeno](https://github.com/DDDeno)

## 简介

本项目实现了一个**双向检索系统**，支持：
- **Protein → Molecule**：给定靶点蛋白，检索候选结合分子
- **Molecule → Protein**：给定小分子药物，识别潜在靶点蛋白

## 方法

### 模型架构

采用双塔编码器架构：
- **蛋白质编码器**：[SaProt](https://github.com/westlake-repl/SaProt)（650M），基于结构感知的蛋白质语言模型
- **分子编码器**：[ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)，基于 SMILES 的分子语言模型
- **投影网络**：4层 MLP（1024 → 1024 → 512 → 256），带 LayerNorm、GELU 和 Dropout

### 训练策略

- 对比学习损失（InfoNCE）
- 可学习温度参数（范围 [0.03, 0.2]）
- 正样本定义：pIC50 ≥ 7.0（即 IC50 < 100nM）
- 批次内负采样 + 多标签处理
- Cosine 学习率调度 + Warmup
- 支持分布式训练与混合精度（FP16）

### 数据集

来自 Hugging Face 的 BindingDB 数据集：
- **IC50 数据集**：[vladak/bindingdb](https://huggingface.co/datasets/vladak/bindingdb)（主要使用）
  - 包含 pIC50 值作为结合亲和力标签
  - 划分为 train/test
- **Kd 数据集**：[amirhallaji/bindingdb_kd](https://huggingface.co/datasets/amirhallaji/bindingdb_kd)（备选）

预训练模型：
- **SaProt**：[westlake-repl/SaProt_650M_AF2](https://huggingface.co/westlake-repl/SaProt_650M_AF2)
- **ChemBERTa**：[seyonec/ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)

## 环境依赖

```
torch>=2.0
transformers
datasets
accelerate
gradio
matplotlib
seaborn
tqdm
```

## 使用方法

### 1. 下载预训练模型与数据集

```bash
python down_util.py
```

### 2. 训练模型

```bash
# 单卡训练
python train_ic50.py

# 多卡分布式训练
accelerate launch train_ic50.py
```

### 3. 评估检索性能

```bash
python inference_valid.py
```

评估指标：
- **Recall@K**：Top-K 检索结果中命中正样本的比例
- **MRR**：平均倒数排名

### 4. 可视化分析

```bash
python heat_map.py
```

生成：
- `diagnosis_heatmap.png`：蛋白质-分子相似度矩阵热力图
- `score_distribution.png`：正负样本分数分布直方图

### 5. 启动 Web Demo

```bash
python app.py
```

运行后会生成访问链接，支持：
- 蛋白质序列 → 分子检索
- SMILES → 蛋白质检索
- 单对相似度计算

## 演示视频

系统使用演示视频：[final_video_UI.mp4](final_video_UI.mp4)

该视频展示了完整的用户界面操作流程，包括双向检索功能和相似度计算功能的使用方法。

## 项目结构

```
├── train_ic50.py       # 训练脚本（基于 pIC50 亲和力）
├── inference_valid.py  # 全局检索评估（Recall@K, MRR）
├── heat_map.py         # 相似度热力图可视化
├── app.py              # Gradio Web Demo
├── down_util.py        # 模型与数据下载工具
├── models/             # 预训练模型存放目录
│   ├── SaProt_650M_AF2/
│   └── ChemBERTa-zinc-base-v1/
├── data/               # 数据集目录
│   └── vladak_bindingdb/
│       ├── train/
│       └── test/
└── outputs/            # 训练输出
    ├── model_best.pth
    ├── training_curves.png
    └── config.json
```

## 训练配置

| 参数 | 值 |
|------|-----|
| 批次大小 | 32 蛋白质 × 12 分子/蛋白质 |
| 梯度累积 | 4 |
| 蛋白质编码器学习率 | 3e-5 |
| 分子编码器学习率 | 2e-5 |
| 投影层学习率 | 2e-4 |
| 正样本阈值 | pIC50 ≥ 7.0 |
| 温度范围 | [0.03, 0.2] |
| 早停耐心 | 3 epochs |
