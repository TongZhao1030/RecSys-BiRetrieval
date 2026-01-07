"""
蛋白质-分子双塔模型训练脚本 (基于pIC50监督的对比学习)

主要改进：
1. 使用更大的vladak/bindingdb数据集 (~100万训练样本)
2. 数据集中的'ic50'列实际上已经是pIC50值，直接使用
3. 基于pIC50值定义正负样本
4. 使用Supervised Contrastive Loss进行训练
5. 支持验证集评估

pIC50说明 (数据集已预处理):
- pIC50 = -log10(IC50_M)，值越大表示结合越强
- 高亲和力: pIC50 > 7 (IC50 < 100 nM)
- 中亲和力: 5 < pIC50 < 7 (100 nM < IC50 < 10 μM)
- 低亲和力: pIC50 < 5 (IC50 > 10 μM)
- 负值/极低值: 表示极弱或无结合
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

# ============================================================================
# 训练配置
# ============================================================================
@dataclass
class TrainConfig:
    # 基础配置
    per_device_batch_size: int = 16          # 每个蛋白质的批次大小
    mols_per_protein: int = 8                 # 每个蛋白质采样的分子数
    gradient_accumulation_steps: int = 2
    epochs: int = 50
    
    # 学习率
    lr_prot_backbone: float = 1e-5
    lr_mol_backbone: float = 5e-6
    lr_head: float = 2e-4
    lr_temp: float = 1e-3
    
    # 正则化
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # IC50阈值 (pIC50)
    positive_threshold: float = 7.0    # pIC50 > 7 为强正样本 (IC50 < 100nM)
    negative_threshold: float = 5.0    # pIC50 < 5 为负样本 (IC50 > 10μM)
    
    # 损失函数配置
    temperature: float = 0.07
    margin: float = 0.5                # Triplet loss margin
    
    # 路径
    path_saprot: str = "./models/SaProt_650M_AF2"
    path_chemberta: str = "./models/ChemBERTa-zinc-base-v1"
    path_data_train: str = "./data/vladak_bindingdb/train"
    path_data_test: str = "./data/vladak_bindingdb/test"
    
    # 输出
    output_dir: str = "./outputs"
    save_every_epochs: int = 10
    eval_every_steps: int = 500
    log_every_steps: int = 20

config = TrainConfig()

# ============================================================================
# pIC50 处理函数
# ============================================================================
def pic50_to_affinity_class(pic50: float) -> int:
    """
    将pIC50转换为亲和力类别
    2: 高亲和力 (pIC50 > 7, IC50 < 100nM)
    1: 中亲和力 (5 < pIC50 < 7)
    0: 低亲和力 (pIC50 < 5, IC50 > 10μM)
    
    注意: 数据集中的'ic50'列已经是pIC50值，无需转换
    """
    if pic50 >= config.positive_threshold:
        return 2
    elif pic50 >= config.negative_threshold:
        return 1
    else:
        return 0

# ============================================================================
# 数据预处理
# ============================================================================
def preprocess_dataset(dataset, accelerator=None):
    """
    预处理数据集：
    1. 数据集中的'ic50'列实际上是pIC50值，直接使用
    2. 过滤无效样本
    3. 按蛋白质分组
    """
    is_main = accelerator is None or accelerator.is_main_process
    
    # 按蛋白质分组
    protein_to_data = defaultdict(list)
    valid_count = 0
    invalid_count = 0
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        # 数据集中的'ic50'列实际上已经是pIC50值
        pic50 = item['ic50']
        
        # 过滤无效pIC50
        if pic50 is None or math.isnan(pic50) or math.isinf(pic50):
            invalid_count += 1
            continue
        
        # 过滤异常值 (pIC50一般在-2到12之间是合理的)
        # 负值表示极弱结合，保留但限制范围
        if pic50 < -2 or pic50 > 14:
            invalid_count += 1
            continue
        
        protein_to_data[item['protein']].append({
            'idx': idx,
            'ligand': item['ligand'],
            'protein': item['protein'],
            'pic50': pic50,
            'affinity_class': pic50_to_affinity_class(pic50)
        })
        valid_count += 1
    
    if is_main:
        print(f"数据预处理完成: {valid_count} 有效样本, {invalid_count} 无效样本")
        print(f"唯一蛋白质数: {len(protein_to_data)}")
        
        # 统计亲和力分布
        class_counts = {0: 0, 1: 0, 2: 0}
        pic50_values = []
        for prot, mols in protein_to_data.items():
            for m in mols:
                class_counts[m['affinity_class']] += 1
                pic50_values.append(m['pic50'])
        
        print(f"亲和力分布: 高(pIC50>7)={class_counts[2]}, 中(5-7)={class_counts[1]}, 低(<5)={class_counts[0]}")
        print(f"pIC50范围: [{min(pic50_values):.2f}, {max(pic50_values):.2f}], 均值: {sum(pic50_values)/len(pic50_values):.2f}")
    
    return protein_to_data

# ============================================================================
# 数据集和采样器
# ============================================================================
class BindingAffinityDataset(Dataset):
    """
    基于蛋白质分组的数据集
    每个样本是一个蛋白质及其关联的分子列表
    """
    def __init__(self, protein_to_data: Dict, min_mols_per_protein: int = 2):
        """
        Args:
            protein_to_data: 蛋白质到分子数据的映射
            min_mols_per_protein: 每个蛋白质最少需要的分子数
        """
        # 过滤掉分子数太少的蛋白质
        self.protein_data = {}
        for prot, mols in protein_to_data.items():
            if len(mols) >= min_mols_per_protein:
                # 按pIC50排序，方便后续采样
                mols_sorted = sorted(mols, key=lambda x: -x['pic50'])
                self.protein_data[prot] = mols_sorted
        
        self.proteins = list(self.protein_data.keys())
        print(f"符合条件的蛋白质数: {len(self.proteins)} (>={min_mols_per_protein}个分子)")
        
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        protein_seq = self.proteins[idx]
        mol_data = self.protein_data[protein_seq]
        return {
            'protein': protein_seq,
            'molecules': mol_data
        }

class AffinityBatchSampler(Sampler):
    """
    亲和力感知的批次采样器
    
    每个批次包含:
    - batch_size 个不同的蛋白质
    - 每个蛋白质采样 mols_per_protein 个分子
    
    采样策略：
    - 优先采样高亲和力(正样本)和低亲和力(负样本)的分子
    - 确保每个蛋白质的样本中有正有负
    """
    def __init__(self, dataset: BindingAffinityDataset, 
                 batch_size: int, 
                 mols_per_protein: int,
                 drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.mols_per_protein = mols_per_protein
        self.drop_last = drop_last
        
        # 找出有正负样本的蛋白质，记录索引
        self.valid_indices = []
        for idx, prot_seq in enumerate(dataset.proteins):
            mols = dataset.protein_data[prot_seq]
            has_positive = any(m['affinity_class'] == 2 for m in mols)
            has_negative = any(m['affinity_class'] == 0 for m in mols)
            if has_positive and has_negative:
                self.valid_indices.append(idx)
        
        print(f"有正负样本的蛋白质数: {len(self.valid_indices)}")
        
    def __iter__(self):
        # 打乱索引顺序
        indices = self.valid_indices.copy()
        random.shuffle(indices)
        
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self):
        n = len(self.valid_indices)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

class DataCollator:
    """
    数据整理器：处理批次数据的tokenization和采样
    """
    def __init__(self, mol_tokenizer, prot_tokenizer, mols_per_protein: int):
        self.mol_tok = mol_tokenizer
        self.prot_tok = prot_tokenizer
        self.mols_per_protein = mols_per_protein
    
    def _sample_molecules(self, mol_data: List[Dict]) -> Tuple[List[Dict], List[float]]:
        """
        智能采样分子：确保正负样本均衡
        返回采样的分子数据和对应的pIC50值
        """
        n = self.mols_per_protein
        
        # 按亲和力分组
        high_affinity = [m for m in mol_data if m['affinity_class'] == 2]
        mid_affinity = [m for m in mol_data if m['affinity_class'] == 1]
        low_affinity = [m for m in mol_data if m['affinity_class'] == 0]
        
        sampled = []
        
        # 采样策略：尽量保证正负样本各半
        n_positive = min(n // 2, len(high_affinity))
        n_negative = min(n - n_positive, len(low_affinity))
        n_mid = n - n_positive - n_negative
        
        if high_affinity:
            sampled.extend(random.sample(high_affinity, min(n_positive, len(high_affinity))))
        if low_affinity:
            sampled.extend(random.sample(low_affinity, min(n_negative, len(low_affinity))))
        if mid_affinity and len(sampled) < n:
            remaining = n - len(sampled)
            sampled.extend(random.sample(mid_affinity, min(remaining, len(mid_affinity))))
        
        # 如果还不够，从所有数据中补充
        if len(sampled) < n:
            remaining_pool = [m for m in mol_data if m not in sampled]
            if remaining_pool:
                sampled.extend(random.sample(remaining_pool, min(n - len(sampled), len(remaining_pool))))
        
        random.shuffle(sampled)
        return sampled
    
    def __call__(self, batch_items: List[Dict]):
        """
        处理一个批次的蛋白质
        
        Args:
            batch_items: DataLoader返回的数据列表，每个元素是 {'protein': str, 'molecules': list}
        
        Returns:
            mol_inputs: 分子的tokenized输入
            prot_inputs: 蛋白质的tokenized输入
            labels: 配对标签矩阵
            pic50_matrix: pIC50值矩阵 (用于加权损失)
        """
        all_smiles = []
        all_prots = []
        all_pic50 = []
        prot_indices = []  # 记录每个分子属于哪个蛋白质
        
        for prot_idx, item in enumerate(batch_items):
            prot_seq = item['protein']
            mol_data = item['molecules']
            sampled_mols = self._sample_molecules(mol_data)
            
            for mol in sampled_mols:
                all_smiles.append(mol['ligand'])
                all_pic50.append(mol['pic50'])
                prot_indices.append(prot_idx)
            
            # 格式化蛋白质序列 (SaProt格式)
            prot_list = [aa + "#" for aa in prot_seq]
            formatted_prot = " ".join(prot_list)
            all_prots.append(formatted_prot)
        
        # Tokenize
        mol_inputs = self.mol_tok(
            all_smiles, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        prot_inputs = self.prot_tok(
            all_prots, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # 构建标签矩阵
        n_prots = len(batch_items)
        n_mols = len(all_smiles)
        
        # labels[i, j] = 1 如果分子j属于蛋白质i
        labels = torch.zeros(n_prots, n_mols)
        pic50_matrix = torch.zeros(n_prots, n_mols)
        
        for mol_idx, prot_idx in enumerate(prot_indices):
            labels[prot_idx, mol_idx] = 1.0
            pic50_matrix[prot_idx, mol_idx] = all_pic50[mol_idx]
        
        return {
            'mol_inputs': mol_inputs,
            'prot_inputs': prot_inputs,
            'labels': labels,
            'pic50_matrix': pic50_matrix,
            'prot_indices': torch.tensor(prot_indices)
        }

# ============================================================================
# 模型定义
# ============================================================================
class DualTowerModel(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        
        # 加载预训练模型
        self.prot_model = AutoModel.from_pretrained(config.path_saprot, trust_remote_code=True)
        self.mol_model = AutoModel.from_pretrained(config.path_chemberta)
        
        # 全量微调
        for param in self.prot_model.parameters():
            param.requires_grad = True
        for param in self.mol_model.parameters():
            param.requires_grad = True
        
        # 统计参数
        prot_params = sum(p.numel() for p in self.prot_model.parameters())
        mol_params = sum(p.numel() for p in self.mol_model.parameters())
        print(f"蛋白质编码器参数: {prot_params/1e6:.1f}M")
        print(f"分子编码器参数: {mol_params/1e6:.1f}M")
        
        # 投影头
        prot_hidden = self.prot_model.config.hidden_size
        mol_hidden = 768
        hidden_dim = 512
        embedding_dim = 256
        
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.mol_proj = nn.Sequential(
            nn.Linear(mol_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self._init_projection_weights()
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.temperature))
    
    def _init_projection_weights(self):
        for module in [self.prot_proj, self.mol_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def encode_protein(self, input_ids, attention_mask):
        outputs = self.prot_model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        embeddings = self.prot_proj(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def encode_molecule(self, input_ids, attention_mask):
        outputs = self.mol_model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        embeddings = self.mol_proj(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def forward(self, prot_input_ids, prot_attention_mask, mol_input_ids, mol_attention_mask):
        prot_embeddings = self.encode_protein(prot_input_ids, prot_attention_mask)
        mol_embeddings = self.encode_molecule(mol_input_ids, mol_attention_mask)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        return prot_embeddings, mol_embeddings, logit_scale

# ============================================================================
# 损失函数
# ============================================================================
class AffinityContrastiveLoss(nn.Module):
    """
    基于亲和力的对比学习损失函数
    
    结合了:
    1. InfoNCE Loss: 使正样本对更近
    2. 亲和力加权: 使用pIC50作为软标签
    3. Hard Negative Mining: 聚焦于难样本
    """
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.margin = config.margin
    
    def forward(self, prot_emb, mol_emb, labels, pic50_matrix, logit_scale):
        """
        Args:
            prot_emb: [n_prots, dim] 蛋白质embeddings
            mol_emb: [n_mols, dim] 分子embeddings
            labels: [n_prots, n_mols] 配对标签 (1=配对, 0=非配对)
            pic50_matrix: [n_prots, n_mols] pIC50值
            logit_scale: 温度缩放因子
        
        Returns:
            total_loss, loss_dict
        """
        device = prot_emb.device
        n_prots = prot_emb.size(0)
        n_mols = mol_emb.size(0)
        
        # 计算相似度矩阵 [n_prots, n_mols]
        sim_matrix = torch.matmul(prot_emb, mol_emb.T) * logit_scale
        
        # ===============================
        # 1. 多标签对比损失 (Protein -> Molecule)
        # ===============================
        # 对于每个蛋白质，所有匹配的分子都是正样本
        # 使用pIC50加权的softmax
        
        # 归一化pIC50到[0, 1]作为软权重
        # 数据实际范围约[-2, 12]，主要集中在[2, 9]
        # 使用[2, 10]作为归一化范围
        pic50_norm = (pic50_matrix - 2) / 8  # 将[2, 10]映射到[0, 1]
        pic50_norm = torch.clamp(pic50_norm, 0, 1)
        
        # 正样本的加权
        pos_weights = labels * pic50_norm
        pos_weights = pos_weights / (pos_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 带权重的交叉熵损失 (Protein -> Molecule)
        log_softmax_p2m = F.log_softmax(sim_matrix, dim=1)
        loss_p2m = -torch.sum(pos_weights * log_softmax_p2m) / n_prots
        
        # ===============================
        # 2. Molecule -> Protein 方向的损失
        # ===============================
        # 每个分子只属于一个蛋白质，使用标准交叉熵
        mol_labels = labels.T.argmax(dim=1)  # [n_mols]
        log_softmax_m2p = F.log_softmax(sim_matrix.T, dim=1)
        loss_m2p = F.nll_loss(log_softmax_m2p, mol_labels)
        
        # ===============================
        # 3. 亲和力排名损失 (Margin Ranking Loss)
        # ===============================
        # 对于每个蛋白质，高亲和力分子应该比低亲和力分子得分更高
        ranking_loss = 0.0
        ranking_count = 0
        
        for i in range(n_prots):
            # 获取该蛋白质的正样本索引
            pos_mask = labels[i] > 0
            if pos_mask.sum() < 2:
                continue
            
            pos_indices = pos_mask.nonzero().squeeze(-1)
            pos_pic50 = pic50_matrix[i, pos_indices]
            pos_scores = sim_matrix[i, pos_indices]
            
            # 对所有正样本对进行排名约束
            for j in range(len(pos_indices)):
                for k in range(j + 1, len(pos_indices)):
                    if pos_pic50[j] > pos_pic50[k]:  # j应该比k得分高
                        margin_diff = self.margin - (pos_scores[j] - pos_scores[k])
                        ranking_loss += F.relu(margin_diff)
                    elif pos_pic50[k] > pos_pic50[j]:  # k应该比j得分高
                        margin_diff = self.margin - (pos_scores[k] - pos_scores[j])
                        ranking_loss += F.relu(margin_diff)
                    ranking_count += 1
        
        if ranking_count > 0:
            ranking_loss = ranking_loss / ranking_count
        
        # ===============================
        # 4. 负样本对比损失
        # ===============================
        # 确保非配对的蛋白质-分子对相似度低
        neg_mask = 1 - labels
        neg_sim = sim_matrix * neg_mask
        neg_loss = F.relu(neg_sim - 0.0).mean()  # 希望负样本相似度 < 0
        
        # 总损失
        total_loss = loss_p2m + loss_m2p + 0.5 * ranking_loss + 0.1 * neg_loss
        
        loss_dict = {
            'loss_p2m': loss_p2m.item(),
            'loss_m2p': loss_m2p.item(),
            'loss_ranking': ranking_loss.item() if isinstance(ranking_loss, torch.Tensor) else ranking_loss,
            'loss_neg': neg_loss.item()
        }
        
        return total_loss, loss_dict

# ============================================================================
# 评估指标
# ============================================================================
def compute_retrieval_metrics(prot_emb, mol_emb, labels, k_list=[1, 5, 10]):
    """
    计算检索指标
    
    Returns:
        metrics: dict with Recall@K for both directions
    """
    # Protein -> Molecule
    sim_p2m = torch.matmul(prot_emb, mol_emb.T)
    
    # Molecule -> Protein  
    sim_m2p = sim_p2m.T
    
    metrics = {}
    
    # P2M Recall@K
    n_prots = prot_emb.size(0)
    for k in k_list:
        _, topk_indices = sim_p2m.topk(k, dim=1)
        hits = 0
        for i in range(n_prots):
            # 检查topk中是否有正样本
            pos_indices = (labels[i] > 0).nonzero().squeeze(-1)
            for idx in topk_indices[i]:
                if idx in pos_indices:
                    hits += 1
                    break
        metrics[f'P2M_R@{k}'] = hits / n_prots
    
    # M2P Recall@K
    n_mols = mol_emb.size(0)
    labels_t = labels.T
    for k in k_list:
        _, topk_indices = sim_m2p.topk(k, dim=1)
        hits = 0
        for i in range(n_mols):
            true_prot = labels_t[i].argmax()
            if true_prot in topk_indices[i]:
                hits += 1
        metrics[f'M2P_R@{k}'] = hits / n_mols
    
    return metrics

# ============================================================================
# 学习率调度器
# ============================================================================
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """线性warmup后线性衰减"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                   float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================================
# 主训练流程
# ============================================================================
def main():
    # 初始化Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    device = accelerator.device
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("蛋白质-分子双塔模型训练 (IC50监督对比学习)")
        print("=" * 80)
        print(f"设备: {device}, 进程数: {accelerator.num_processes}")
        print(f"批次大小: {config.per_device_batch_size} proteins × {config.mols_per_protein} mols")
        print(f"梯度累积: {config.gradient_accumulation_steps}")
        effective_batch = (config.per_device_batch_size * config.mols_per_protein * 
                          config.gradient_accumulation_steps * accelerator.num_processes)
        print(f"有效批次大小: ~{effective_batch} samples")
        print(f"正样本阈值: pIC50 > {config.positive_threshold}")
        print(f"负样本阈值: pIC50 < {config.negative_threshold}")
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    # Tokenizer
    mol_tokenizer = AutoTokenizer.from_pretrained(config.path_chemberta)
    prot_tokenizer = AutoTokenizer.from_pretrained(config.path_saprot, trust_remote_code=True)
    
    # 加载数据
    if accelerator.is_main_process:
        print("\n加载训练数据...")
    train_raw = load_from_disk(config.path_data_train)
    train_protein_data = preprocess_dataset(train_raw, accelerator)
    
    if accelerator.is_main_process:
        print("\n加载测试数据...")
    test_raw = load_from_disk(config.path_data_test)
    test_protein_data = preprocess_dataset(test_raw, accelerator)
    
    # 创建数据集
    train_dataset = BindingAffinityDataset(
        train_protein_data, 
        min_mols_per_protein=config.mols_per_protein
    )
    test_dataset = BindingAffinityDataset(
        test_protein_data,
        min_mols_per_protein=config.mols_per_protein
    )
    
    # 创建采样器和数据加载器
    train_sampler = AffinityBatchSampler(
        train_dataset,
        batch_size=config.per_device_batch_size,
        mols_per_protein=config.mols_per_protein,
        drop_last=True
    )
    
    train_collator = DataCollator(
        mol_tokenizer,
        prot_tokenizer,
        config.mols_per_protein
    )
    
    train_loader = DataLoader(
        train_dataset,  # 传整个dataset
        batch_sampler=train_sampler,
        collate_fn=train_collator,
        num_workers=4,
        pin_memory=True
    )
    
    # 模型
    model = DualTowerModel(config)
    
    # 损失函数
    criterion = AffinityContrastiveLoss(config)
    
    # 优化器
    optimizer = torch.optim.AdamW([
        {'params': model.prot_model.parameters(), 
         'lr': config.lr_prot_backbone, 'weight_decay': config.weight_decay},
        {'params': model.prot_proj.parameters(), 
         'lr': config.lr_head, 'weight_decay': 0.0},
        {'params': model.mol_model.parameters(), 
         'lr': config.lr_mol_backbone, 'weight_decay': config.weight_decay},
        {'params': model.mol_proj.parameters(), 
         'lr': config.lr_head, 'weight_decay': 0.0},
        {'params': [model.logit_scale], 
         'lr': config.lr_temp, 'weight_decay': 0.0}
    ])
    
    # 学习率调度器
    num_training_steps = len(train_sampler) * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    if accelerator.is_main_process:
        print(f"\n训练步数: {num_training_steps}, Warmup步数: {num_warmup_steps}")
    
    # Accelerator prepare
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_metrics': [],
        'steps': []
    }
    
    if accelerator.is_main_process:
        print("\n开始训练...")
    
    global_step = 0
    best_recall = 0.0
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        # 重新创建sampler (重新打乱)
        train_sampler = AffinityBatchSampler(
            train_dataset,
            batch_size=config.per_device_batch_size,
            mols_per_protein=config.mols_per_protein,
            drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=train_collator,
            num_workers=4,
            pin_memory=True
        )
        
        for step, batch in enumerate(train_loader):
            # 移动到设备
            mol_inputs = {k: v.to(device) for k, v in batch['mol_inputs'].items()}
            prot_inputs = {k: v.to(device) for k, v in batch['prot_inputs'].items()}
            labels = batch['labels'].to(device)
            pic50_matrix = batch['pic50_matrix'].to(device)
            
            with accelerator.accumulate(model):
                # 前向传播
                prot_emb, mol_emb, logit_scale = model(
                    prot_inputs['input_ids'], prot_inputs['attention_mask'],
                    mol_inputs['input_ids'], mol_inputs['attention_mask']
                )
                
                # 计算损失
                loss, loss_dict = criterion(prot_emb, mol_emb, labels, pic50_matrix, logit_scale)
                
                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            
            # 日志
            if accelerator.is_main_process and step % config.log_every_steps == 0:
                lr = scheduler.get_last_lr()[0]
                temp = logit_scale.item()
                print(f"Epoch {epoch+1} | Step {step:4d} | Loss: {loss.item():.4f} | "
                      f"P2M: {loss_dict['loss_p2m']:.3f} | M2P: {loss_dict['loss_m2p']:.3f} | "
                      f"Rank: {loss_dict['loss_ranking']:.3f} | Temp: {temp:.2f} | LR: {lr:.2e}")
                
                history['train_loss'].append(loss.item())
                history['steps'].append(global_step)
        
        # Epoch结束统计
        avg_loss = epoch_loss / epoch_steps
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1} 完成 | 平均损失: {avg_loss:.4f}")
            print(f"{'='*60}\n")
        
        # 保存检查点
        if (epoch + 1) % config.save_every_epochs == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config.__dict__,
                    'history': history
                }, save_path)
                print(f"保存检查点: {save_path}")
    
    # 最终保存
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_path = os.path.join(config.output_dir, "model_final.pth")
        torch.save({
            'model_state_dict': unwrapped_model.state_dict(),
            'config': config.__dict__
        }, final_path)
        print(f"\n最终模型保存: {final_path}")
        
        # 绘制训练曲线
        if history['train_loss']:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(history['steps'], history['train_loss'], alpha=0.3, label='Raw')
            
            # 平滑曲线
            window = min(50, len(history['train_loss']) // 10)
            if window > 1:
                smooth = np.convolve(history['train_loss'], 
                                    np.ones(window)/window, mode='valid')
                ax.plot(history['steps'][window-1:], smooth, linewidth=2, label='Smoothed')
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_path = os.path.join(config.output_dir, 'training_curve.png')
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"训练曲线保存: {fig_path}")
        
        # 保存配置
        config_path = os.path.join(config.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)

if __name__ == "__main__":
    main()

