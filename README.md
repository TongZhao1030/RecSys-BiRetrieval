# RecSys-BiRetrieval

基于双塔对比学习的蛋白质-分子双向检索系统。

## 简介

本项目实现了一个**双向检索系统**，支持：
- **Protein → Molecule**：给定靶点蛋白，检索候选结合分子
- **Molecule → Protein**：给定小分子药物，识别潜在靶点蛋白

## 方法

采用双塔编码器架构：
- **蛋白质编码器**：[SaProt](https://github.com/westlake-repl/SaProt)（650M），基于结构感知的蛋白质语言模型
- **分子编码器**：[ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)，基于 SMILES 的分子语言模型

训练策略：
- 对称对比损失（InfoNCE）
- 可学习温度参数
- 批次内蛋白质去重采样，确保负样本有效性
- 支持分布式训练与混合精度

数据集：[BindingDB](https://www.bindingdb.org/)（蛋白质-配体结合亲和力数据）

## 环境依赖

```
torch>=2.6
transformers
datasets
accelerate
huggingface_hub
```

## 使用方法

**1. 下载预训练模型与数据集**

```bash
python down_util.py
```

**2. 训练模型**

```bash
# 单卡训练
python train.py

# 多卡分布式训练
accelerate launch train.py
```

**3. 评估检索性能**

```bash
python inference_valid.py
```

## 项目结构

```
├── train.py           # 训练脚本
├── inference_valid.py # 双向检索评估
├── down_util.py       # 模型与数据下载工具
└── heat_map.py        # 相似度热力图可视化
```
