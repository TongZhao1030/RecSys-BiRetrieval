"""
蛋白质-分子双塔模型训练脚本

主要特性：
1. 数据去重处理 - 确保每个批次中蛋白质和分子都是唯一的
2. 使用分组采样器 - 避免同一蛋白质在同一批次中出现多次
3. 支持分布式训练和混合精度训练
"""

import os
import argparse
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

# 训练配置参数
SEED = 42
PER_DEVICE_BATCH_SIZE = 32  # 每设备批次大小
GRADIENT_ACCUMULATION_STEPS = 1  # 梯度累积步数
EPOCHS = 100

# 学习率配置
LR_PROT_BACKBONE = 2e-5   # 蛋白质编码器学习率
LR_MOL_BACKBONE = 1e-5    # 分子编码器学习率
LR_PROT_HEAD = 5e-4       # 蛋白质投影头学习率
LR_MOL_HEAD = 2e-4        # 分子投影头学习率
LR_TEMP = 1e-3            # 温度参数学习率

MAX_GRAD_NORM = 1.0

# SaProt 全量微调设置
SAPROT_FULL_FINETUNE = True

# 对比学习/监督信号增强
LABEL_SMOOTHING = 0.1  # 缓解“假负样本”带来的过强惩罚
USE_AFFINITY_WEIGHT = True  # 若数据里有亲和度列，则按强弱加权正样本
ENSURE_UNIQUE_MOLECULES_IN_BATCH = True  # 尽量避免同一分子在同一batch里出现多次
MASK_KNOWN_POSITIVES = True  # 若 (prot, mol) 在数据集中是已知正样本，则不要把它当作batch内负样本
AFFINITY_KEY_CANDIDATES = (
    # bindingdb_kd 数据集常见字段（优先用已经是 log-scale 且“越大越强”的）
    "Binding Affinity",
    "origin_affinity",
    "pKd", "pKI", "pKi", "pIC50", "pIC_50",
    "Kd (nM)", "Kd(nM)", "KD (nM)", "KD(nM)", "Kd", "KD",
    "Ki (nM)", "Ki(nM)", "KI (nM)", "KI(nM)", "Ki", "KI",
    "IC50 (nM)", "IC50(nM)", "IC_50 (nM)", "IC_50(nM)", "IC50", "IC_50", "ic50",
    "label", "Label", "affinity", "Affinity",
)

# 路径
PATH_SAPROT = "/share/home/zhangchiLab/duyinuo/models/westlake-repl_SaProt_650M_AF2"
PATH_CHEMBERTA = "/share/home/zhangchiLab/duyinuo/models/seyonec_ChemBERTa-zinc-base-v1"
PATH_DATA = "/share/home/zhangchiLab/duyinuo/data/vladak_bindingdb"

def resolve_split_dataset_path(data_path, split):
    """
    兼容两种磁盘结构：
    1) `load_from_disk(path)` 直接可读（Dataset/DatasetDict）
    2) path 下有 `train/` `test/` 子目录，每个子目录是一个 Dataset
    """
    if split:
        candidate = os.path.join(data_path, split)
        if os.path.isdir(candidate):
            return candidate
    for default_split in ("train", "test", "valid", "validation"):
        candidate = os.path.join(data_path, default_split)
        if os.path.isdir(candidate):
            return candidate
    return data_path

def infer_text_column(cols, *, kind):
    """
    kind: 'protein' or 'molecule'
    """
    cols = list(cols or [])
    if kind == "protein":
        preferred = ["Protein Sequence", "protein_sequence", "protein", "target_sequence", "target"]
        needles = ("protein", "target", "sequence")
    else:
        preferred = ["Molecule Sequence", "molecule_sequence", "smiles", "ligand", "molecule", "drug"]
        needles = ("smiles", "ligand", "molecule", "drug")

    for c in preferred:
        if c in cols:
            return c
    for c in cols:
        low = c.lower()
        if any(n in low for n in needles):
            return c
    return None

# 数据预处理 - 去重
def deduplicate_dataset(dataset):
    """
    创建去重后的数据集索引
    确保每个 (蛋白质, 分子) 对只出现一次
    """
    seen_pairs = set()
    unique_indices = []
    
    # 兼容旧/新数据集字段名（若找不到，会在后续训练阶段显式报错）
    protein_col = infer_text_column(getattr(dataset, "column_names", []), kind="protein") or "Protein Sequence"
    molecule_col = infer_text_column(getattr(dataset, "column_names", []), kind="molecule") or "Molecule Sequence"

    for idx in range(len(dataset)):
        prot = dataset[idx][protein_col]
        mol = dataset[idx][molecule_col]
        pair_key = (prot, mol)
        
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            unique_indices.append(idx)
    
    return unique_indices

# 分组采样器 - 确保批次内蛋白质唯一
class UniqueProteinBatchSampler(Sampler):
    """
    确保每个批次中的蛋白质都是唯一的
    这是对比学习成功的关键
    """
    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=True,
        *,
        prot_to_indices=None,
        protein_col="Protein Sequence",
        molecule_col="Molecule Sequence",
        seed=0,
        world_size=1,
        rank=0,
        ensure_unique_molecules_in_batch=False,
        max_molecule_dedup_tries=32,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.world_size = max(int(world_size), 1)
        self.rank = int(rank)
        self.ensure_unique_molecules_in_batch = ensure_unique_molecules_in_batch
        self.max_molecule_dedup_tries = int(max_molecule_dedup_tries)
        self.protein_col = protein_col
        self.molecule_col = molecule_col
        
        # 按蛋白质分组
        if prot_to_indices is None:
            self.prot_to_indices = defaultdict(list)
            for idx in range(len(dataset)):
                prot = dataset[idx][self.protein_col]
                self.prot_to_indices[prot].append(idx)
        else:
            self.prot_to_indices = prot_to_indices
        
        self.unique_proteins = list(self.prot_to_indices.keys())
        if self.rank == 0:
            print(f"数据集统计: {len(dataset)} 样本, {len(self.unique_proteins)} 唯一蛋白质")
    
    def __iter__(self):
        rng = random.Random(self.seed)

        # 打乱蛋白质顺序
        shuffled_prots = self.unique_proteins.copy()
        rng.shuffle(shuffled_prots)

        # 多进程训练时，按进程切分蛋白质，避免各卡重复跑同一批数据
        if self.world_size > 1:
            prots_per_proc = len(shuffled_prots) // self.world_size
            total = prots_per_proc * self.world_size
            shuffled_prots = shuffled_prots[:total]
            start = self.rank * prots_per_proc
            end = start + prots_per_proc
            shuffled_prots = shuffled_prots[start:end]
        
        batch = []
        used_molecules = set()
        for prot in shuffled_prots:
            # 从每个蛋白质的所有配对中选一个（可选：同batch分子去重）
            candidates = self.prot_to_indices[prot]
            if not self.ensure_unique_molecules_in_batch:
                idx = rng.choice(candidates)
            else:
                idx = None
                # 尽量挑一个没出现过的分子，避免同batch里出现“多靶点分子”导致假负样本
                for _ in range(min(self.max_molecule_dedup_tries, len(candidates))):
                    cand_idx = rng.choice(candidates)
                    mol = self.dataset[cand_idx][self.molecule_col]
                    if mol not in used_molecules:
                        idx = cand_idx
                        used_molecules.add(mol)
                        break
                if idx is None:
                    idx = rng.choice(candidates)
                    used_molecules.add(self.dataset[idx][self.molecule_col])
            batch.append(idx)
            
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                used_molecules.clear()
        
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self):
        prots_per_proc = len(self.unique_proteins)
        if self.world_size > 1:
            prots_per_proc = prots_per_proc // self.world_size
        if self.drop_last:
            return prots_per_proc // self.batch_size
        return (prots_per_proc + self.batch_size - 1) // self.batch_size

def infer_affinity_key(hf_dataset):
    cols = list(getattr(hf_dataset, "column_names", []) or [])
    for key in AFFINITY_KEY_CANDIDATES:
        if key in cols:
            return key
    for col in cols:
        low = col.lower()
        if any(tok in low for tok in ("affin", "kd", "ki", "ic50", "pki", "pkd")):
            return col
    return None

def affinity_to_strength(affinity, affinity_key):
    """
    将“亲和度值”转换为“越大越强”的强度。
    - 若字段名像 pKd/pKi：认为本身就是越大越强。
    - 否则默认是 (nM) 量纲：数值越小越强，转换为 -log10(nM)。
    """
    key = (affinity_key or "").lower()
    a = affinity.float()
    if "binding affinity" in key:
        return a
    if "pkd" in key or "pki" in key or "pic50" in key or "ic50" in key or "pchembl" in key:
        return a
    if "origin_affinity" in key or "nm" in key:
        # nM -> M: 9 - log10(nM)，常数项不影响排序但数值更直观
        return 9.0 - torch.log10(torch.clamp(a, min=1e-12))
    # 兜底：不猜单位，认为数值越大越强
    return a

def build_id_maps(hf_dataset, protein_col, molecule_col):
    """
    构建字符串 -> int 的映射，避免用长字符串做集合元素导致内存/速度问题。
    """
    prot2id = {}
    mol2id = {}
    for idx in range(len(hf_dataset)):
        row = hf_dataset[idx]
        prot = row[protein_col]
        mol = row[molecule_col]
        if prot not in prot2id:
            prot2id[prot] = len(prot2id)
        if mol not in mol2id:
            mol2id[mol] = len(mol2id)
    return prot2id, mol2id

def build_positive_maps(hf_dataset, prot2id, mol2id, protein_col, molecule_col):
    """
    构建:
      - prot_to_indices: prot_seq -> [row_idx...]
      - prot_id -> set(mol_id)
      - mol_id -> set(prot_id)
    """
    prot_to_indices = defaultdict(list)
    prot_to_mol_ids = defaultdict(set)
    mol_to_prot_ids = defaultdict(set)
    for idx in range(len(hf_dataset)):
        row = hf_dataset[idx]
        prot = row[protein_col]
        mol = row[molecule_col]
        prot_to_indices[prot].append(idx)
        pid = prot2id[prot]
        mid = mol2id[mol]
        prot_to_mol_ids[pid].add(mid)
        mol_to_prot_ids[mid].add(pid)
    return prot_to_indices, prot_to_mol_ids, mol_to_prot_ids

# 数据集和数据整理器
class BioDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        *,
        prot2id,
        mol2id,
        affinity_key,
        protein_col,
        molecule_col,
        indices=None,
    ):
        self.data = hf_dataset
        self.indices = indices if indices is not None else list(range(len(hf_dataset)))
        self.affinity_key = affinity_key
        self.prot2id = prot2id
        self.mol2id = mol2id
        self.protein_col = protein_col
        self.molecule_col = molecule_col
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # idx 是 sampler 给的原始索引，不需要转换
        if isinstance(idx, int) and idx < len(self.data):
            item = self.data[idx]
        else:
            item = self.data[self.indices[idx]]
        
        smiles = item[self.molecule_col]
        raw_prot = item[self.protein_col]
        
        prot_list = [aa + "#" for aa in raw_prot]
        formatted_prot = " ".join(prot_list)

        affinity = None
        if self.affinity_key is not None:
            try:
                affinity = float(item[self.affinity_key])
            except Exception:
                affinity = None

        prot_id = self.prot2id[raw_prot]
        mol_id = self.mol2id[smiles]
        
        return smiles, formatted_prot, affinity, prot_id, mol_id

class DataCollate:
    def __init__(self, mol_tok, prot_tok):
        self.mol_tok = mol_tok
        self.prot_tok = prot_tok

    def __call__(self, batch):
        smiles_list = [item[0] for item in batch]
        prot_list = [item[1] for item in batch]
        affinities = [item[2] for item in batch]
        prot_ids = [item[3] for item in batch]
        mol_ids = [item[4] for item in batch]
        
        mol_inputs = self.mol_tok(
            smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        prot_inputs = self.prot_tok(
            prot_list, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        
        # float tensor with NaN for missing values
        aff = torch.tensor(
            [a if a is not None else float("nan") for a in affinities],
            dtype=torch.float32,
        )
        prot_ids = torch.tensor(prot_ids, dtype=torch.long)
        mol_ids = torch.tensor(mol_ids, dtype=torch.long)
        return mol_inputs, prot_inputs, aff, prot_ids, mol_ids

# 模型定义
class DualTowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prot_model = AutoModel.from_pretrained(PATH_SAPROT, trust_remote_code=True)
        self.mol_model = AutoModel.from_pretrained(PATH_CHEMBERTA)
        
        # 微调策略设置
        if SAPROT_FULL_FINETUNE:
            for param in self.prot_model.parameters():
                param.requires_grad = True
            print(f"SaProt: 全量微调（所有参数可训练）")
        else:
            # 备用：冻结部分层
            for param in self.prot_model.parameters():
                param.requires_grad = False
            for layer in self.prot_model.encoder.layer[-8:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # 统计可训练参数
        prot_trainable = sum(p.numel() for p in self.prot_model.parameters() if p.requires_grad)
        prot_total = sum(p.numel() for p in self.prot_model.parameters())
        print(f"SaProt 可训练参数: {prot_trainable/1e6:.1f}M / {prot_total/1e6:.1f}M")

        for param in self.mol_model.parameters():
            param.requires_grad = True

        prot_hidden = self.prot_model.config.hidden_size
        mol_hidden = 768
        hidden_dim = 1024
        embedding_dim = 256

        self.prot_layernorm = nn.LayerNorm(prot_hidden)
        self.mol_layernorm = nn.LayerNorm(mol_hidden)

        self.prot_proj = nn.Sequential(
            nn.Linear(prot_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.mol_proj = nn.Sequential(
            nn.Linear(mol_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self._init_projection_weights()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _init_projection_weights(self):
        for module in [self.prot_proj, self.mol_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, prot_input_ids, prot_attention_mask, mol_input_ids, mol_attention_mask):
        # 蛋白质侧
        prot_out = self.prot_model(input_ids=prot_input_ids, attention_mask=prot_attention_mask)
        mask = prot_attention_mask.unsqueeze(-1).expand(prot_out.last_hidden_state.size()).float()
        prot_emb = torch.sum(prot_out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        prot_emb = self.prot_layernorm(prot_emb)
        prot_vec = self.prot_proj(prot_emb)

        # 分子侧
        mol_out = self.mol_model(input_ids=mol_input_ids, attention_mask=mol_attention_mask)
        mol_mask = mol_attention_mask.unsqueeze(-1).expand(mol_out.last_hidden_state.size()).float()
        mol_emb = torch.sum(mol_out.last_hidden_state * mol_mask, dim=1) / torch.clamp(mol_mask.sum(1), min=1e-9)
        mol_emb = self.mol_layernorm(mol_emb)
        mol_vec = self.mol_proj(mol_emb)

        # L2 归一化
        prot_vec = F.normalize(prot_vec, p=2, dim=1)
        mol_vec = F.normalize(mol_vec, p=2, dim=1)
        
        return prot_vec, mol_vec, self.logit_scale.exp()

# 主训练程序
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=PATH_DATA, help="数据集路径（支持含 train/test 子目录）")
    parser.add_argument("--split", type=str, default="train", help="若 data_path 下有 split 子目录，选择哪个 split")
    parser.add_argument("--protein_col", type=str, default=None, help="蛋白质列名（为空则自动推断）")
    parser.add_argument("--molecule_col", type=str, default=None, help="分子/SMILES 列名（为空则自动推断）")
    parser.add_argument("--affinity_col", type=str, default=None, help="亲和度列名（为空则自动推断）")
    args = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16", 
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print("=" * 70)
        print("蛋白质-分子双塔模型训练")
        print("=" * 70)
        print(f"设备: {device}, 进程数: {accelerator.num_processes}")
        print(f"批次大小: {PER_DEVICE_BATCH_SIZE}, 梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"有效批次大小: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * accelerator.num_processes}")
        print(f"SaProt 全量微调: {SAPROT_FULL_FINETUNE}")

    # Tokenizer
    mol_tokenizer = AutoTokenizer.from_pretrained(PATH_CHEMBERTA)
    prot_tokenizer = AutoTokenizer.from_pretrained(PATH_SAPROT, trust_remote_code=True)
    
    # 数据加载
    data_path = resolve_split_dataset_path(args.data_path, args.split)
    full_dataset = load_from_disk(data_path)

    protein_col = args.protein_col or infer_text_column(full_dataset.column_names, kind="protein")
    molecule_col = args.molecule_col or infer_text_column(full_dataset.column_names, kind="molecule")
    if protein_col is None or molecule_col is None:
        raise ValueError(
            f"无法推断列名，请显式传参 --protein_col/--molecule_col；现有列: {full_dataset.column_names}"
        )

    affinity_key = args.affinity_col or infer_affinity_key(full_dataset)
    if accelerator.is_main_process:
        print(f"数据路径: {data_path}")
        print(f"字段: protein_col={protein_col} | molecule_col={molecule_col}")
        print(f"亲和度字段: {affinity_key if affinity_key is not None else '未检测到（将退化为纯对比学习）'}")

    prot2id, mol2id = build_id_maps(full_dataset, protein_col, molecule_col)
    prot_to_indices, prot_to_mol_ids, mol_to_prot_ids = build_positive_maps(
        full_dataset, prot2id, mol2id, protein_col, molecule_col
    )
    
    if accelerator.is_main_process:
        print(f"\n原始数据集大小: {len(full_dataset)}")
        print(f"唯一蛋白质: {len(prot2id)} | 唯一分子: {len(mol2id)}")
    
    # 创建 Dataset（使用原始数据，采样器会处理去重）
    train_ds = BioDataset(
        full_dataset,
        prot2id=prot2id,
        mol2id=mol2id,
        affinity_key=affinity_key,
        protein_col=protein_col,
        molecule_col=molecule_col,
    )
    train_ds.prot_to_mol_ids = prot_to_mol_ids
    train_ds.mol_to_prot_ids = mol_to_prot_ids
    
    # 使用自定义采样器确保批次内蛋白质唯一
    batch_sampler = UniqueProteinBatchSampler(
        full_dataset, 
        batch_size=PER_DEVICE_BATCH_SIZE,
        drop_last=True,
        prot_to_indices=prot_to_indices,
        protein_col=protein_col,
        molecule_col=molecule_col,
        seed=SEED + accelerator.process_index,
        world_size=accelerator.num_processes,
        rank=accelerator.process_index,
        ensure_unique_molecules_in_batch=ENSURE_UNIQUE_MOLECULES_IN_BATCH,
    )
    
    collate_fn = DataCollate(mol_tokenizer, prot_tokenizer)
    
    # 注意：使用 batch_sampler 时不能设置 batch_size 和 shuffle
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    if accelerator.is_main_process:
        print(f"每个轮次的批次数: {len(batch_sampler)}")

    # 模型
    model = DualTowerModel()

    # 优化器
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.prot_model.named_parameters() if p.requires_grad], 
         'lr': LR_PROT_BACKBONE, 'weight_decay': 0.01},
        {'params': model.prot_layernorm.parameters(), 'lr': LR_PROT_HEAD},
        {'params': model.prot_proj.parameters(), 'lr': LR_PROT_HEAD},
        {'params': [p for n, p in model.mol_model.named_parameters() if p.requires_grad], 
         'lr': LR_MOL_BACKBONE, 'weight_decay': 0.01},
        {'params': model.mol_layernorm.parameters(), 'lr': LR_MOL_HEAD},
        {'params': model.mol_proj.parameters(), 'lr': LR_MOL_HEAD},
        {'params': [model.logit_scale], 'lr': LR_TEMP} 
    ])
    
    # 注意：使用自定义 batch_sampler 时，accelerator.prepare 可能会有问题
    # 我们需要手动处理
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # DataLoader 不通过 accelerator.prepare，因为我们使用自定义 batch_sampler
    # 但需要手动移动数据到设备

    loss_history = []
    acc_history = []

    if accelerator.is_main_process:
        print("\n开始训练...")

    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        steps_in_epoch = 0
        
        # 每个 epoch 重新创建 sampler（重新打乱）
        batch_sampler = UniqueProteinBatchSampler(
            full_dataset, 
            batch_size=PER_DEVICE_BATCH_SIZE,
            drop_last=True,
            prot_to_indices=prot_to_indices,
            protein_col=protein_col,
            molecule_col=molecule_col,
            seed=SEED + epoch * 1000 + accelerator.process_index,
            world_size=accelerator.num_processes,
            rank=accelerator.process_index,
            ensure_unique_molecules_in_batch=ENSURE_UNIQUE_MOLECULES_IN_BATCH,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        for step, (mol_inputs, prot_inputs, affinity, prot_ids, mol_ids) in enumerate(train_loader):
            # 手动移动到设备
            mol_inputs = {k: v.to(device) for k, v in mol_inputs.items()}
            prot_inputs = {k: v.to(device) for k, v in prot_inputs.items()}
            affinity = affinity.to(device)
            
            # 使用 accumulate 上下文管理器处理梯度累积
            with accelerator.accumulate(model):
                # 前向传播
                prot_vec, mol_vec, logit_scale = model(
                    prot_inputs['input_ids'], prot_inputs['attention_mask'],
                    mol_inputs['input_ids'], mol_inputs['attention_mask']
                )
                
                # Loss
                logit_scale = torch.clamp(logit_scale, max=100)
                logits = torch.matmul(prot_vec, mol_vec.T) * logit_scale
                
                batch_size = prot_vec.size(0)
                labels = torch.arange(batch_size).to(device)

                # 已知正样本去负：如果 (prot_i, mol_j) 在数据集中出现过，就别把它当batch内负样本
                if MASK_KNOWN_POSITIVES:
                    prot_ids_list = prot_ids.tolist()
                    mol_ids_list = mol_ids.tolist()

                    mask_p2m = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
                    for i, pid in enumerate(prot_ids_list):
                        known_mols = train_ds.prot_to_mol_ids.get(pid)
                        if not known_mols:
                            continue
                        for j, mid in enumerate(mol_ids_list):
                            if i == j:
                                continue
                            if mid in known_mols:
                                mask_p2m[i, j] = True
                    if mask_p2m.any():
                        logits = logits.masked_fill(mask_p2m, -1e4)

                # 亲和度加权：强结合的样本权重大，弱/噪声样本权重小
                weights = torch.ones(batch_size, device=device, dtype=torch.float32)
                if USE_AFFINITY_WEIGHT and torch.isfinite(affinity).any():
                    strength = affinity_to_strength(affinity, train_ds.affinity_key)
                    valid = torch.isfinite(strength)
                    if valid.any():
                        s = strength[valid]
                        center = s.median()
                        scale = s.std().clamp(min=1e-6)
                        weights_valid = torch.sigmoid((s - center) / scale)
                        weights[valid] = weights_valid

                loss_i_vec = F.cross_entropy(
                    logits, labels, reduction="none", label_smoothing=LABEL_SMOOTHING
                )
                logits_t = logits.T
                if MASK_KNOWN_POSITIVES:
                    prot_ids_list = prot_ids.tolist()
                    mol_ids_list = mol_ids.tolist()
                    mask_m2p = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
                    for i, mid in enumerate(mol_ids_list):
                        known_prots = train_ds.mol_to_prot_ids.get(mid)
                        if not known_prots:
                            continue
                        for j, pid in enumerate(prot_ids_list):
                            if i == j:
                                continue
                            if pid in known_prots:
                                mask_m2p[i, j] = True
                    if mask_m2p.any():
                        logits_t = logits_t.masked_fill(mask_m2p, -1e4)

                loss_t_vec = F.cross_entropy(
                    logits_t, labels, reduction="none", label_smoothing=LABEL_SMOOTHING
                )
                denom = weights.sum().clamp(min=1e-6)
                loss_i = (loss_i_vec * weights).sum() / denom
                loss_t = (loss_t_vec * weights).sum() / denom
                loss = (loss_i + loss_t) / 2

                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                optimizer.step()
                optimizer.zero_grad()
            
            # 统计
            current_loss = loss.item()
            total_loss += current_loss
            steps_in_epoch += 1
            
            # 计算 batch accuracy
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct = (pred == labels).sum().item()
                total_correct += correct
                total_samples += batch_size
            
            if accelerator.is_main_process and step % 20 == 0:
                acc = correct / batch_size * 100
                temp = logit_scale.item()
                print(f"Epoch {epoch+1} | Step {step:4d} | Loss: {current_loss:.4f} | "
                      f"Batch Acc: {acc:.1f}% | Temp: {temp:.2f}")
                loss_history.append(current_loss)
                acc_history.append(acc)

        # Epoch 结束
        avg_loss = total_loss / steps_in_epoch
        epoch_acc = total_correct / total_samples * 100
        
        if accelerator.is_main_process:
            print(f"\n轮次 {epoch+1} 完成")
            print(f"平均损失: {avg_loss:.4f}")
            print(f"轮次准确率: {epoch_acc:.2f}%\n")

        # 每隔50个轮次保存一次检查点
        if (epoch + 1) % 50 == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                torch.save(unwrapped_model.state_dict(), f"dual_tower_epoch_{epoch+1}.pth")
                print(f"已保存检查点: dual_tower_epoch_{epoch+1}.pth")

    # 最终保存
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        torch.save(unwrapped_model.state_dict(), "dual_tower_final.pth")
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        ax1.plot(loss_history, alpha=0.3)
        if len(loss_history) > 20:
            smooth = [np.mean(loss_history[max(0,i-20):i+1]) for i in range(len(loss_history))]
            ax1.plot(smooth, linewidth=2)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Step (x20)")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(acc_history, alpha=0.7)
        if len(acc_history) > 20:
            smooth_acc = [np.mean(acc_history[max(0,i-20):i+1]) for i in range(len(acc_history))]
            ax2.plot(smooth_acc, linewidth=2)
        ax2.set_title("Batch Accuracy")
        ax2.set_xlabel("Step (x20)")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_loss.png', dpi=150)
        plt.close()
        
        print("\n训练完成")
        print("模型文件: dual_tower_final.pth")
        print("训练曲线: training_loss.png")

if __name__ == "__main__":
    main()
