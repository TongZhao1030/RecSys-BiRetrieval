"""
蛋白质-分子双塔模型训练脚本 v2 (改进版)

主要改进：
1. 修复学习率调度器
2. 限制温度参数范围
3. 简化Loss设计，使用标准对比学习
4. 放宽数据筛选条件
5. 添加验证评估和早停
6. 使用Cosine退火学习率
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
    per_device_batch_size: int = 32          # 每个batch的蛋白质数
    mols_per_protein: int = 12                 # 每个蛋白质采样的分子数
    gradient_accumulation_steps: int = 4
    samples_per_epoch: int = 20               # 每个epoch内，每个蛋白质被采样的次数（提高数据利用率）
    epochs: int = 10                          # 减少epoch数（因为每个epoch更长了）
    
    # 学习率
    lr_prot_backbone: float = 3e-5            # 稍微提高backbone学习率
    lr_mol_backbone: float = 2e-5
    lr_head: float = 2e-4                     # 降低head学习率
    lr_temp: float = 1e-4                     # 大幅降低温度学习率
    
    # 正则化
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01                # 最小学习率为初始的1%
    
    # IC50阈值 (pIC50) - 放宽条件
    positive_threshold: float = 7.0           # 放宽正样本阈值 (IC50 < 1μM)
    negative_threshold: float = 5.0           # 负样本阈值
    min_mols_per_protein: int = 2             # 降低最小分子数要求
    
    # 损失函数配置
    temperature_init: float = 0.07
    temperature_min: float = 0.03             # 温度下限
    temperature_max: float = 0.2              # 温度上限 (比之前严格很多)
    
    # 路径
    path_saprot: str = "./models/SaProt_650M_AF2"
    path_chemberta: str = "./models/ChemBERTa-zinc-base-v1"
    path_data_train: str = "./data/vladak_bindingdb/train"
    path_data_test: str = "./data/vladak_bindingdb/test"
    
    # 输出
    output_dir: str = "./outputs"
    save_every_epochs: int = 5
    eval_every_epochs: int = 1                # 每5个epoch评估一次
    log_every_steps: int = 100
    
    # 早停
    patience: int = 3                        # 5个epoch没有提升就停止

config = TrainConfig()

# ============================================================================
# 数据预处理 (放宽条件)
# ============================================================================
def preprocess_dataset(dataset, accelerator=None, require_both_classes=False):
    """
    预处理数据集，放宽筛选条件
    
    返回:
        filtered_data: 蛋白质 -> 分子列表的映射
        ligand_to_proteins: 分子 -> 蛋白质集合的映射 (用于处理多标签)
    """
    is_main = accelerator is None or accelerator.is_main_process
    
    protein_to_data = defaultdict(list)
    ligand_to_proteins = defaultdict(set)  # 新增：记录每个分子对应的所有蛋白质
    valid_count = 0
    invalid_count = 0
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        pic50 = item['ic50']
        
        if pic50 is None or math.isnan(pic50) or math.isinf(pic50):
            invalid_count += 1
            continue
        
        if pic50 < -2 or pic50 > 14:
            invalid_count += 1
            continue
        
        # 分类：正样本(pIC50>=6)，负样本(pIC50<5)，中间样本(5<=pIC50<6)
        if pic50 >= config.positive_threshold:
            affinity_class = 2  # 正样本
        elif pic50 < config.negative_threshold:
            affinity_class = 0  # 负样本
        else:
            affinity_class = 1  # 中间样本
        
        protein_to_data[item['protein']].append({
            'idx': idx,
            'ligand': item['ligand'],
            'protein': item['protein'],
            'pic50': pic50,
            'affinity_class': affinity_class
        })
        
        # 记录分子-蛋白质关系 (只记录有一定亲和力的，pIC50 >= 4)
        if pic50 >= 4.0:
            ligand_to_proteins[item['ligand']].add(item['protein'])
        
        valid_count += 1
    
    # 如果不要求同时有正负样本，保留所有有足够分子的蛋白质
    filtered_data = {}
    for prot, mols in protein_to_data.items():
        if len(mols) >= config.min_mols_per_protein:
            if require_both_classes:
                # 严格模式：要求同时有正负样本
                has_positive = any(m['affinity_class'] == 2 for m in mols)
                has_negative = any(m['affinity_class'] == 0 for m in mols)
                if has_positive and has_negative:
                    filtered_data[prot] = sorted(mols, key=lambda x: -x['pic50'])
            else:
                # 宽松模式：只要有正样本就行
                has_positive = any(m['affinity_class'] == 2 for m in mols)
                if has_positive:
                    filtered_data[prot] = sorted(mols, key=lambda x: -x['pic50'])
    
    if is_main:
        print(f"数据预处理完成: {valid_count} 有效样本, {invalid_count} 无效样本")
        print(f"唯一蛋白质数: {len(protein_to_data)}")
        print(f"符合条件的蛋白质数: {len(filtered_data)}")
        
        # 统计亲和力分布
        class_counts = {0: 0, 1: 0, 2: 0}
        for prot, mols in filtered_data.items():
            for m in mols:
                class_counts[m['affinity_class']] += 1
        
        print(f"亲和力分布: 高(>={config.positive_threshold})={class_counts[2]}, "
              f"中={class_counts[1]}, 低(<{config.negative_threshold})={class_counts[0]}")
        
        # 统计多蛋白质分子
        multi_prot_ligands = sum(1 for prots in ligand_to_proteins.values() if len(prots) > 1)
        print(f"对应多个蛋白质的分子数: {multi_prot_ligands} ({multi_prot_ligands/len(ligand_to_proteins)*100:.1f}%)")
    
    return filtered_data, ligand_to_proteins

# ============================================================================
# 数据集和采样器
# ============================================================================
class BindingAffinityDataset(Dataset):
    def __init__(self, protein_data: Dict, ligand_to_proteins: Dict = None):
        self.protein_data = protein_data
        self.proteins = list(protein_data.keys())
        self.ligand_to_proteins = ligand_to_proteins or {}
        print(f"数据集蛋白质数: {len(self.proteins)}")
        
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        protein_seq = self.proteins[idx]
        mol_data = self.protein_data[protein_seq]
        return {
            'protein': protein_seq,
            'molecules': mol_data
        }

class SimpleBatchSampler(Sampler):
    """
    批次采样器，支持多轮采样以提高数据利用率
    
    每个epoch内，所有蛋白质会被采样 samples_per_epoch 次，
    每次采样时会随机选择不同的分子，从而充分利用数据。
    """
    def __init__(self, dataset: BindingAffinityDataset, batch_size: int, 
                 samples_per_epoch: int = 1, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch  # 每个蛋白质每epoch被采样的次数
        self.drop_last = drop_last
        self.base_indices = list(range(len(dataset)))
        
    def __iter__(self):
        # 将所有蛋白质索引重复 samples_per_epoch 次
        all_indices = self.base_indices * self.samples_per_epoch
        random.shuffle(all_indices)
        
        batch = []
        for idx in all_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self):
        n = len(self.base_indices) * self.samples_per_epoch
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

class DataCollator:
    """
    数据整理器 - 修复版
    
    核心修改：
    1. 只有高亲和力分子(affinity_class==2)才标记为正样本
    2. 采样策略只采样高亲和力分子，负样本来自batch内其他蛋白质的分子
    """
    def __init__(self, mol_tokenizer, prot_tokenizer, mols_per_protein: int, 
                 ligand_to_proteins: Dict = None, protein_to_idx: Dict = None):
        self.mol_tok = mol_tokenizer
        self.prot_tok = prot_tokenizer
        self.mols_per_protein = mols_per_protein
        self.ligand_to_proteins = ligand_to_proteins or {}
        self.protein_to_idx = protein_to_idx or {}
    
    def _sample_molecules(self, mol_data: List[Dict]) -> List[Dict]:
        """
        采样分子：只采样高亲和力分子作为正样本
        
        关键改动：不再采样低亲和力分子，负样本由batch内其他蛋白质的分子提供
        """
        n = self.mols_per_protein
        
        # 只采样高亲和力分子
        high = [m for m in mol_data if m['affinity_class'] == 2]
        
        if not high:
            # 如果没有高亲和力分子，返回空（这种情况理论上不应该发生，因为预处理已经筛选过）
            return []
        
        # 采样高亲和力分子（可以重复采样以达到数量要求）
        if len(high) >= n:
            sampled = random.sample(high, n)
        else:
            # 如果高亲和力分子不够，允许重复采样
            sampled = high.copy()
            while len(sampled) < n:
                sampled.append(random.choice(high))
        
        return sampled
    
    def __call__(self, batch_items: List[Dict]):
        all_smiles = []
        all_prots = []
        all_pic50 = []
        prot_indices = []
        affinity_classes = []
        
        # 建立batch内的蛋白质序列到索引的映射
        batch_prot_to_idx = {}
        for prot_idx, item in enumerate(batch_items):
            batch_prot_to_idx[item['protein']] = prot_idx
        
        for prot_idx, item in enumerate(batch_items):
            prot_seq = item['protein']
            mol_data = item['molecules']
            sampled_mols = self._sample_molecules(mol_data)
            
            for mol in sampled_mols:
                all_smiles.append(mol['ligand'])
                all_pic50.append(mol['pic50'])
                prot_indices.append(prot_idx)
                affinity_classes.append(mol['affinity_class'])
            
            # SaProt格式
            prot_list = [aa + "#" for aa in prot_seq]
            formatted_prot = " ".join(prot_list)
            all_prots.append(formatted_prot)
        
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
        
        n_prots = len(batch_items)
        n_mols = len(all_smiles)
        
        # 构建标签矩阵：只有高亲和力分子才是正样本
        labels = torch.zeros(n_prots, n_mols)
        pic50_matrix = torch.zeros(n_prots, n_mols)
        
        for mol_idx, (smiles, sampled_prot_idx, aff_class) in enumerate(
            zip(all_smiles, prot_indices, affinity_classes)
        ):
            # 只有高亲和力分子(affinity_class==2)才标记为正样本
            if aff_class == 2:
                labels[sampled_prot_idx, mol_idx] = 1.0
                pic50_matrix[sampled_prot_idx, mol_idx] = all_pic50[mol_idx]
            
            # 检查该分子是否还与batch内其他蛋白质有高亲和力结合关系
            if aff_class == 2 and smiles in self.ligand_to_proteins:
                other_proteins = self.ligand_to_proteins[smiles]
                for other_prot_seq in other_proteins:
                    if other_prot_seq in batch_prot_to_idx:
                        other_prot_idx = batch_prot_to_idx[other_prot_seq]
                        if other_prot_idx != sampled_prot_idx:
                            labels[other_prot_idx, mol_idx] = 1.0
        
        return {
            'mol_inputs': mol_inputs,
            'prot_inputs': prot_inputs,
            'labels': labels,
            'pic50_matrix': pic50_matrix,
            'prot_indices': torch.tensor(prot_indices),
            'affinity_classes': torch.tensor(affinity_classes)
        }

# ============================================================================
# 模型定义
# ============================================================================
class DualTowerModel(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        
        self.prot_model = AutoModel.from_pretrained(config.path_saprot, trust_remote_code=True)
        self.mol_model = AutoModel.from_pretrained(config.path_chemberta)
        
        for param in self.prot_model.parameters():
            param.requires_grad = True
        for param in self.mol_model.parameters():
            param.requires_grad = True
        
        prot_params = sum(p.numel() for p in self.prot_model.parameters())
        mol_params = sum(p.numel() for p in self.mol_model.parameters())
        print(f"蛋白质编码器参数: {prot_params/1e6:.1f}M")
        print(f"分子编码器参数: {mol_params/1e6:.1f}M")
        
        prot_hidden = self.prot_model.config.hidden_size
        mol_hidden = 768
        hidden_dim = 1024
        embedding_dim = 256
        
        # 4层投影网络（增加深度以提升表达能力）
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, embedding_dim)
        )
        
        self.mol_proj = nn.Sequential(
            nn.Linear(mol_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, embedding_dim)
        )
        
        self._init_projection_weights()
        
        # 可学习温度，但有严格上下限
        self.log_temperature = nn.Parameter(torch.tensor(math.log(config.temperature_init)))
    
    def _init_projection_weights(self):
        for module in [self.prot_proj, self.mol_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def get_temperature(self):
        """获取温度，带严格的上下限"""
        temp = self.log_temperature.exp()
        temp = torch.clamp(temp, min=self.config.temperature_min, max=self.config.temperature_max)
        return temp
    
    def encode_protein(self, input_ids, attention_mask):
        outputs = self.prot_model(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        embeddings = self.prot_proj(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def encode_molecule(self, input_ids, attention_mask):
        outputs = self.mol_model(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        embeddings = self.mol_proj(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def forward(self, prot_input_ids, prot_attention_mask, mol_input_ids, mol_attention_mask):
        prot_embeddings = self.encode_protein(prot_input_ids, prot_attention_mask)
        mol_embeddings = self.encode_molecule(mol_input_ids, mol_attention_mask)
        temperature = self.get_temperature()
        return prot_embeddings, mol_embeddings, temperature

# ============================================================================
# 简化的对比损失
# ============================================================================
class SimpleContrastiveLoss(nn.Module):
    """
    简化的对比学习损失，基于标准InfoNCE
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, prot_emb, mol_emb, labels, temperature):
        """
        Args:
            prot_emb: [n_prots, dim]
            mol_emb: [n_mols, dim]
            labels: [n_prots, n_mols] 配对标签
            temperature: 温度标量
        """
        # 计算相似度矩阵
        sim_matrix = torch.matmul(prot_emb, mol_emb.T) / temperature
        
        # ===============================
        # 1. Protein -> Molecule (多标签分类)
        # ===============================
        # 每个蛋白质对应多个正样本分子
        # 使用多标签softmax: sum of log-softmax for all positives
        
        n_prots = prot_emb.size(0)
        loss_p2m = 0.0
        
        for i in range(n_prots):
            pos_mask = labels[i] > 0
            if pos_mask.sum() == 0:
                continue
            
            # log_softmax over all molecules
            log_probs = F.log_softmax(sim_matrix[i], dim=0)
            
            # 平均正样本的log probability
            pos_log_probs = log_probs[pos_mask]
            loss_p2m -= pos_log_probs.mean()
        
        loss_p2m = loss_p2m / n_prots
        
        # ===============================
        # 2. Molecule -> Protein (多标签分类)
        # ===============================
        # 一个分子可能对应多个蛋白质！
        n_mols = mol_emb.size(0)
        loss_m2p = 0.0
        
        labels_t = labels.T  # [n_mols, n_prots]
        for j in range(n_mols):
            pos_mask = labels_t[j] > 0
            if pos_mask.sum() == 0:
                continue
            
            # log_softmax over all proteins
            log_probs = F.log_softmax(sim_matrix.T[j], dim=0)
            
            # 平均正样本的log probability
            pos_log_probs = log_probs[pos_mask]
            loss_m2p -= pos_log_probs.mean()
        
        loss_m2p = loss_m2p / n_mols
        
        # 总损失
        total_loss = (loss_p2m + loss_m2p) / 2
        
        return total_loss, {
            'loss_p2m': loss_p2m.item(),
            'loss_m2p': loss_m2p.item(),
            'temperature': temperature.item()
        }

# ============================================================================
# 评估指标
# ============================================================================
@torch.no_grad()
def evaluate(model, dataloader, device, accelerator=None):
    """评估模型"""
    model.eval()
    
    all_prot_emb = []
    all_mol_emb = []
    all_labels = []
    all_pic50 = []
    
    for batch in dataloader:
        mol_inputs = {k: v.to(device) for k, v in batch['mol_inputs'].items()}
        prot_inputs = {k: v.to(device) for k, v in batch['prot_inputs'].items()}
        
        prot_emb, mol_emb, _ = model(
            prot_inputs['input_ids'], prot_inputs['attention_mask'],
            mol_inputs['input_ids'], mol_inputs['attention_mask']
        )
        
        all_prot_emb.append(prot_emb)
        all_mol_emb.append(mol_emb)
        all_labels.append(batch['labels'].to(device))
        all_pic50.append(batch['pic50_matrix'].to(device))
    
    # 合并所有batch的embedding (注意这里简化处理，实际应该用all_gather)
    # 对于单进程评估足够
    
    metrics = {}
    total_p2m_correct = 0
    total_m2p_correct = 0
    total_prots = 0
    total_mols = 0
    
    for prot_emb, mol_emb, labels in zip(all_prot_emb, all_mol_emb, all_labels):
        sim_matrix = torch.matmul(prot_emb, mol_emb.T)
        
        n_prots = prot_emb.size(0)
        n_mols = mol_emb.size(0)
        
        # P2M: 对于每个蛋白质，检查top-1预测是否是正样本
        top1_indices = sim_matrix.argmax(dim=1)
        for i in range(n_prots):
            if labels[i, top1_indices[i]] > 0:
                total_p2m_correct += 1
        total_prots += n_prots
        
        # M2P: 对于每个分子，检查预测是否正确
        true_prots = labels.T.argmax(dim=1)
        pred_prots = sim_matrix.T.argmax(dim=1)
        total_m2p_correct += (pred_prots == true_prots).sum().item()
        total_mols += n_mols
    
    metrics['P2M_Acc@1'] = total_p2m_correct / total_prots if total_prots > 0 else 0
    metrics['M2P_Acc@1'] = total_m2p_correct / total_mols if total_mols > 0 else 0
    metrics['Avg_Acc'] = (metrics['P2M_Acc@1'] + metrics['M2P_Acc@1']) / 2
    
    model.train()
    return metrics

# ============================================================================
# Cosine学习率调度器
# ============================================================================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    """Cosine退火，最小LR为初始的min_lr_ratio"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)) * (1 - min_lr_ratio) + min_lr_ratio)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================================
# 主训练流程
# ============================================================================
def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    device = accelerator.device
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("蛋白质-分子双塔模型训练 v2 (改进版)")
        print("=" * 80)
        print(f"设备: {device}, 进程数: {accelerator.num_processes}")
        print(f"批次大小: {config.per_device_batch_size} proteins × {config.mols_per_protein} mols")
        print(f"梯度累积: {config.gradient_accumulation_steps}")
        print(f"温度范围: [{config.temperature_min}, {config.temperature_max}]")
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    # Tokenizer
    mol_tokenizer = AutoTokenizer.from_pretrained(config.path_chemberta)
    prot_tokenizer = AutoTokenizer.from_pretrained(config.path_saprot, trust_remote_code=True)
    
    # 加载数据 (使用宽松条件)
    if accelerator.is_main_process:
        print("\n加载训练数据...")
    train_raw = load_from_disk(config.path_data_train)
    train_protein_data, train_ligand_to_proteins = preprocess_dataset(train_raw, accelerator, require_both_classes=False)
    
    if accelerator.is_main_process:
        print("\n加载测试数据...")
    test_raw = load_from_disk(config.path_data_test)
    test_protein_data, test_ligand_to_proteins = preprocess_dataset(test_raw, accelerator, require_both_classes=False)
    
    # 合并分子-蛋白质映射（测试集的分子也可能在训练集出现过）
    all_ligand_to_proteins = defaultdict(set)
    for lig, prots in train_ligand_to_proteins.items():
        all_ligand_to_proteins[lig].update(prots)
    for lig, prots in test_ligand_to_proteins.items():
        all_ligand_to_proteins[lig].update(prots)
    
    if accelerator.is_main_process:
        print(f"\n合并后的分子-蛋白质映射: {len(all_ligand_to_proteins)} 个分子")
    
    # 创建数据集
    train_dataset = BindingAffinityDataset(train_protein_data, all_ligand_to_proteins)
    test_dataset = BindingAffinityDataset(test_protein_data, all_ligand_to_proteins)
    
    # 数据加载器 (传入ligand_to_proteins用于多标签处理)
    train_collator = DataCollator(mol_tokenizer, prot_tokenizer, config.mols_per_protein, 
                                   ligand_to_proteins=all_ligand_to_proteins)
    test_collator = DataCollator(mol_tokenizer, prot_tokenizer, config.mols_per_protein,
                                  ligand_to_proteins=all_ligand_to_proteins)
    
    # 模型
    model = DualTowerModel(config)
    
    # 损失函数
    criterion = SimpleContrastiveLoss()
    
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
        {'params': [model.log_temperature], 
         'lr': config.lr_temp, 'weight_decay': 0.0}
    ])
    
    # 计算总步数
    # 每个epoch的步数 = 蛋白质数 * 采样轮数 / batch_size
    steps_per_epoch = (len(train_dataset) * config.samples_per_epoch) // config.per_device_batch_size
    
    # 注意：scheduler.step() 在每个 forward step 调用，所以用 forward steps 计算
    # （而不是 optimization steps，因为 accelerate.prepare(scheduler) 后行为会自动适配）
    num_training_steps = steps_per_epoch * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    if accelerator.is_main_process:
        print(f"\n每epoch步数: {steps_per_epoch}")
        print(f"总训练步数: {num_training_steps}, Warmup步数: {num_warmup_steps}")
    
    # 学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, config.min_lr_ratio
    )
    
    # Accelerator prepare
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_metrics': [],
        'epochs': [],
        'lr': []
    }
    
    best_metric = 0.0
    patience_counter = 0
    global_step = 0
    
    if accelerator.is_main_process:
        print("\n开始训练...")
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        # 创建新的sampler和dataloader
        train_sampler = SimpleBatchSampler(
            train_dataset, 
            config.per_device_batch_size, 
            samples_per_epoch=config.samples_per_epoch,
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
            mol_inputs = {k: v.to(device) for k, v in batch['mol_inputs'].items()}
            prot_inputs = {k: v.to(device) for k, v in batch['prot_inputs'].items()}
            labels = batch['labels'].to(device)
            
            with accelerator.accumulate(model):
                prot_emb, mol_emb, temperature = model(
                    prot_inputs['input_ids'], prot_inputs['attention_mask'],
                    mol_inputs['input_ids'], mol_inputs['attention_mask']
                )
                
                loss, loss_dict = criterion(prot_emb, mol_emb, labels, temperature)
                
                accelerator.backward(loss)
                
                # 只在梯度同步时（即真正的优化步骤）更新
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    global_step += 1
                    scheduler.step()  # 移到这里，确保只在真正优化时调用
                
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            epoch_steps += 1
            
            # 日志
            if accelerator.is_main_process and step % config.log_every_steps == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1} | Step {step:4d} | Loss: {loss.item():.4f} | "
                      f"P2M: {loss_dict['loss_p2m']:.3f} | M2P: {loss_dict['loss_m2p']:.3f} | "
                      f"Temp: {loss_dict['temperature']:.4f} | LR: {lr:.2e}")
        
        avg_loss = epoch_loss / max(epoch_steps, 1)
        
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1} 完成 | 平均损失: {avg_loss:.4f}")
            
            history['train_loss'].append(avg_loss)
            history['epochs'].append(epoch + 1)
            history['lr'].append(scheduler.get_last_lr()[0])
        
        # 验证评估
        if (epoch + 1) % config.eval_every_epochs == 0:
            if accelerator.is_main_process:
                print("\n评估中...")
                
                # 创建测试dataloader
                test_sampler = SimpleBatchSampler(test_dataset, config.per_device_batch_size, drop_last=False)
                test_loader = DataLoader(
                    test_dataset,
                    batch_sampler=test_sampler,
                    collate_fn=test_collator,
                    num_workers=2,
                    pin_memory=True
                )
                
                metrics = evaluate(accelerator.unwrap_model(model), test_loader, device)
                history['val_metrics'].append(metrics)
                
                print(f"验证指标: P2M_Acc@1={metrics['P2M_Acc@1']:.4f}, "
                      f"M2P_Acc@1={metrics['M2P_Acc@1']:.4f}, Avg={metrics['Avg_Acc']:.4f}")
                
                # 早停检查
                current_metric = metrics['Avg_Acc']
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                    
                    # 保存最佳模型
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_path = os.path.join(config.output_dir, "model_best.pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': unwrapped_model.state_dict(),
                        'metrics': metrics,
                        'config': config.__dict__
                    }, best_path)
                    print(f"保存最佳模型: {best_path} (Avg_Acc={best_metric:.4f})")
                else:
                    patience_counter += 1
                    print(f"Early stopping counter: {patience_counter}/{config.patience}")
                    
                    if patience_counter >= config.patience:
                        print(f"\n早停触发! 最佳Avg_Acc: {best_metric:.4f}")
                        break
            
            print(f"{'='*60}\n")
        
        # 定期保存
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
                    'history': history
                }, save_path)
                print(f"保存检查点: {save_path}")
    
    # 最终保存
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 绘制训练曲线（只画Loss和验证准确率）
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss曲线
        axes[0].plot(history['epochs'], history['train_loss'], 'b-', linewidth=2, marker='o', markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # 验证指标
        if history['val_metrics']:
            val_epochs = list(range(config.eval_every_epochs, 
                                   len(history['val_metrics']) * config.eval_every_epochs + 1, 
                                   config.eval_every_epochs))
            p2m_acc = [m['P2M_Acc@1'] for m in history['val_metrics']]
            m2p_acc = [m['M2P_Acc@1'] for m in history['val_metrics']]
            avg_acc = [m['Avg_Acc'] for m in history['val_metrics']]
            axes[1].plot(val_epochs, p2m_acc, 'b-o', label='P2M Acc@1', linewidth=2)
            axes[1].plot(val_epochs, m2p_acc, 'r-o', label='M2P Acc@1', linewidth=2)
            axes[1].plot(val_epochs, avg_acc, 'g-s', label='Avg Acc', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title('Validation Metrics', fontsize=14)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(config.output_dir, 'training_curves.png')
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"\n训练曲线保存: {fig_path}")
        
        # 保存配置和历史
        with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        with open(os.path.join(config.output_dir, 'history.json'), 'w') as f:
            # 转换metrics为可序列化格式
            serializable_history = {
                'train_loss': history['train_loss'],
                'epochs': history['epochs'],
                'lr': history['lr'],
                'val_metrics': history['val_metrics']
            }
            json.dump(serializable_history, f, indent=2)
        
        print("\n" + "="*60)
        print(f"训练完成! 最佳Avg_Acc: {best_metric:.4f}")
        print("="*60)

if __name__ == "__main__":
    main()

