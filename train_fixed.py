"""
蛋白质-分子双塔模型训练脚本 v2

主要改进：
1. 基于 Binding Affinity (pKd) 的正负样本定义
2. 支持硬负样本挖掘 - 同一蛋白质的弱结合分子作为硬负样本
3. 亲和力加权的对比学习损失
4. 双向检索优化：蛋白质→分子，分子→蛋白质
"""

import os
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

# ============== 训练配置参数 ==============
PER_DEVICE_BATCH_SIZE = 24  # 每设备批次大小（包含 anchor + 正样本 + 硬负样本）
GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积步数
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

# ============== 亲和力阈值设置 ==============
# pKd 阈值 (pKd = -log10(Kd))
# pKd >= 7 表示 Kd <= 100nM，强结合
# pKd 5-7 表示 Kd 在 100nM - 10uM，中等结合  
# pKd < 5 表示 Kd > 10uM，弱结合/无结合
POSITIVE_THRESHOLD = 6.0   # 正样本阈值：pKd >= 6 (Kd <= 1uM)
HARD_NEGATIVE_MIN = 4.0    # 硬负样本下界：pKd >= 4
HARD_NEGATIVE_MAX = 5.5    # 硬负样本上界：pKd < 5.5

# 每个 anchor 的样本数
NUM_POSITIVES = 2          # 每个蛋白质选择的正样本数
NUM_HARD_NEGATIVES = 2     # 每个蛋白质的硬负样本数

# 路径
PATH_SAPROT = "./models/SaProt_650M_AF2"
PATH_CHEMBERTA = "./models/ChemBERTa-zinc-base-v1"
PATH_DATA = "./data/bindingdb_local"


class AffinityBasedDataset(Dataset):
    """
    基于亲和力的数据集
    
    为每个蛋白质构建：
    - 正样本集合：高亲和力配体 (pKd >= POSITIVE_THRESHOLD)
    - 硬负样本集合：低亲和力但有结合的配体 (HARD_NEGATIVE_MIN <= pKd < HARD_NEGATIVE_MAX)
    """
    def __init__(self, hf_dataset):
        self.data = hf_dataset
        
        # 按蛋白质分组，并记录亲和力
        self.prot_to_mols = defaultdict(list)  # prot -> [(mol, affinity, idx), ...]
        self.mol_to_prots = defaultdict(list)  # mol -> [(prot, affinity, idx), ...]
        
        for idx in range(len(hf_dataset)):
            prot = hf_dataset[idx]['Protein Sequence']
            mol = hf_dataset[idx]['Molecule Sequence']
            aff = hf_dataset[idx]['Binding Affinity']
            
            self.prot_to_mols[prot].append((mol, aff, idx))
            self.mol_to_prots[mol].append((prot, aff, idx))
        
        # 为每个蛋白质按亲和力排序
        for prot in self.prot_to_mols:
            self.prot_to_mols[prot].sort(key=lambda x: x[1], reverse=True)
        
        for mol in self.mol_to_prots:
            self.mol_to_prots[mol].sort(key=lambda x: x[1], reverse=True)
        
        # 筛选出有足够正样本的蛋白质作为训练锚点
        self.valid_proteins = []
        for prot, mols in self.prot_to_mols.items():
            positives = [m for m in mols if m[1] >= POSITIVE_THRESHOLD]
            if len(positives) >= NUM_POSITIVES:
                self.valid_proteins.append(prot)
        
        # 同样筛选有足够正样本的分子
        self.valid_molecules = []
        for mol, prots in self.mol_to_prots.items():
            positives = [p for p in prots if p[1] >= POSITIVE_THRESHOLD]
            if len(positives) >= 1:  # 分子至少有1个强结合蛋白质
                self.valid_molecules.append(mol)
        
        print(f"\n数据集统计:")
        print(f"  总样本数: {len(hf_dataset)}")
        print(f"  唯一蛋白质: {len(self.prot_to_mols)}")
        print(f"  唯一分子: {len(self.mol_to_prots)}")
        print(f"  有效蛋白质锚点 (>={NUM_POSITIVES}个正样本): {len(self.valid_proteins)}")
        print(f"  有效分子锚点: {len(self.valid_molecules)}")
        
    def __len__(self):
        return len(self.valid_proteins)
    
    def get_protein_samples(self, prot):
        """
        获取一个蛋白质的正样本和硬负样本分子
        返回: (正样本分子列表, 硬负样本分子列表, 亲和力列表)
        """
        mols = self.prot_to_mols[prot]
        
        # 正样本：高亲和力
        positives = [(m, a, i) for m, a, i in mols if a >= POSITIVE_THRESHOLD]
        
        # 硬负样本：中低亲和力（有结合但弱）
        hard_negatives = [(m, a, i) for m, a, i in mols 
                         if HARD_NEGATIVE_MIN <= a < HARD_NEGATIVE_MAX]
        
        # 随机选择
        selected_pos = random.sample(positives, min(NUM_POSITIVES, len(positives)))
        selected_neg = random.sample(hard_negatives, min(NUM_HARD_NEGATIVES, len(hard_negatives))) if hard_negatives else []
        
        return selected_pos, selected_neg
    
    def __getitem__(self, idx):
        """
        返回一个训练样本：
        - anchor 蛋白质
        - 正样本分子（高亲和力）
        - 硬负样本分子（低亲和力）
        """
        prot = self.valid_proteins[idx]
        pos_samples, hard_neg_samples = self.get_protein_samples(prot)
        
        # 格式化蛋白质序列（SaProt格式）
        prot_formatted = " ".join([aa + "#" for aa in prot])
        
        # 正样本分子和亲和力
        pos_mols = [s[0] for s in pos_samples]
        pos_affs = [s[1] for s in pos_samples]
        
        # 硬负样本分子和亲和力
        neg_mols = [s[0] for s in hard_neg_samples]
        neg_affs = [s[1] for s in hard_neg_samples]
        
        return {
            'protein': prot_formatted,
            'protein_raw': prot,
            'positive_mols': pos_mols,
            'positive_affs': pos_affs,
            'hard_negative_mols': neg_mols,
            'hard_negative_affs': neg_affs
        }


class AffinityDataCollator:
    """
    整理批次数据，处理不同数量的正负样本
    """
    def __init__(self, mol_tok, prot_tok):
        self.mol_tok = mol_tok
        self.prot_tok = prot_tok
    
    def __call__(self, batch):
        # 收集所有蛋白质
        proteins = [item['protein'] for item in batch]
        
        # 收集所有分子（正样本 + 硬负样本）
        all_mols = []
        mol_labels = []  # 记录每个分子属于哪个蛋白质
        mol_is_positive = []  # 记录是否是正样本
        mol_affinities = []  # 记录亲和力
        
        for batch_idx, item in enumerate(batch):
            # 正样本
            for mol, aff in zip(item['positive_mols'], item['positive_affs']):
                all_mols.append(mol)
                mol_labels.append(batch_idx)
                mol_is_positive.append(True)
                mol_affinities.append(aff)
            
            # 硬负样本
            for mol, aff in zip(item['hard_negative_mols'], item['hard_negative_affs']):
                all_mols.append(mol)
                mol_labels.append(batch_idx)
                mol_is_positive.append(False)
                mol_affinities.append(aff)
        
        # Tokenize
        prot_inputs = self.prot_tok(
            proteins, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        mol_inputs = self.mol_tok(
            all_mols, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        
        return {
            'prot_inputs': prot_inputs,
            'mol_inputs': mol_inputs,
            'mol_labels': torch.tensor(mol_labels),  # 每个分子对应的蛋白质索引
            'mol_is_positive': torch.tensor(mol_is_positive),  # 是否是正样本
            'mol_affinities': torch.tensor(mol_affinities, dtype=torch.float32),
            'batch_size': len(batch),
            'num_mols': len(all_mols)
        }


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

    def encode_protein(self, input_ids, attention_mask):
        """编码蛋白质"""
        prot_out = self.prot_model(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).expand(prot_out.last_hidden_state.size()).float()
        prot_emb = torch.sum(prot_out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        prot_emb = self.prot_layernorm(prot_emb)
        prot_vec = self.prot_proj(prot_emb)
        return F.normalize(prot_vec, p=2, dim=1)
    
    def encode_molecule(self, input_ids, attention_mask):
        """编码分子"""
        mol_out = self.mol_model(input_ids=input_ids, attention_mask=attention_mask)
        mol_mask = attention_mask.unsqueeze(-1).expand(mol_out.last_hidden_state.size()).float()
        mol_emb = torch.sum(mol_out.last_hidden_state * mol_mask, dim=1) / torch.clamp(mol_mask.sum(1), min=1e-9)
        mol_emb = self.mol_layernorm(mol_emb)
        mol_vec = self.mol_proj(mol_emb)
        return F.normalize(mol_vec, p=2, dim=1)

    def forward(self, prot_input_ids, prot_attention_mask, mol_input_ids, mol_attention_mask):
        prot_vec = self.encode_protein(prot_input_ids, prot_attention_mask)
        mol_vec = self.encode_molecule(mol_input_ids, mol_attention_mask)
        return prot_vec, mol_vec, self.logit_scale.exp()


def compute_affinity_weighted_loss(prot_vec, mol_vec, mol_labels, mol_is_positive, 
                                    mol_affinities, logit_scale, device):
    """
    计算亲和力加权的对比学习损失
    
    关键思想：
    1. 蛋白质→分子方向：正样本应该与蛋白质接近，负样本远离
    2. 分子→蛋白质方向：每个正样本分子应该与其对应蛋白质接近
    3. 硬负样本挖掘：批内其他蛋白质的分子作为负样本
    """
    batch_size = prot_vec.size(0)
    num_mols = mol_vec.size(0)
    
    # 计算相似度矩阵 [batch_size, num_mols]
    logit_scale = torch.clamp(logit_scale, max=100)
    sim_matrix = torch.matmul(prot_vec, mol_vec.T) * logit_scale
    
    # ========== 蛋白质 → 分子 损失 ==========
    # 对于每个蛋白质，正样本分子应该相似度高
    loss_p2m = 0.0
    
    for prot_idx in range(batch_size):
        # 找到属于这个蛋白质的分子
        mol_mask = (mol_labels == prot_idx)
        pos_mask = mol_mask & mol_is_positive
        
        if pos_mask.sum() == 0:
            continue
        
        # 获取相似度
        sims = sim_matrix[prot_idx]  # [num_mols]
        
        # 正样本的相似度
        pos_sims = sims[pos_mask]
        
        # 负样本：批内其他蛋白质的所有分子 + 当前蛋白质的硬负样本
        neg_mask = (mol_labels != prot_idx) | (~mol_is_positive & mol_mask)
        neg_sims = sims[neg_mask]
        
        if len(neg_sims) == 0:
            neg_sims = torch.zeros(1, device=device)
        
        # InfoNCE 损失变体：每个正样本 vs 所有负样本
        for pos_sim in pos_sims:
            # log-sum-exp trick for numerical stability
            all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            loss_p2m += -pos_sim + torch.logsumexp(all_sims, dim=0)
    
    loss_p2m = loss_p2m / max(batch_size, 1)
    
    # ========== 分子 → 蛋白质 损失 ==========
    # 对于每个正样本分子，它应该与其对应蛋白质相似
    loss_m2p = 0.0
    num_pos_mols = 0
    
    sim_matrix_t = sim_matrix.T  # [num_mols, batch_size]
    
    for mol_idx in range(num_mols):
        if not mol_is_positive[mol_idx]:
            continue
        
        num_pos_mols += 1
        prot_idx = mol_labels[mol_idx]
        
        # 该分子与所有蛋白质的相似度
        sims = sim_matrix_t[mol_idx]  # [batch_size]
        
        # 正样本蛋白质
        pos_sim = sims[prot_idx]
        
        # 负样本蛋白质
        neg_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        neg_mask[prot_idx] = False
        neg_sims = sims[neg_mask]
        
        if len(neg_sims) == 0:
            neg_sims = torch.zeros(1, device=device)
        
        # InfoNCE
        all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
        loss_m2p += -pos_sim + torch.logsumexp(all_sims, dim=0)
    
    loss_m2p = loss_m2p / max(num_pos_mols, 1)
    
    # 总损失
    total_loss = (loss_p2m + loss_m2p) / 2
    
    return total_loss, loss_p2m, loss_m2p


def compute_retrieval_accuracy(prot_vec, mol_vec, mol_labels, mol_is_positive, device):
    """
    计算检索准确率
    对于每个蛋白质，检查其正样本分子是否在 top-k 中
    """
    batch_size = prot_vec.size(0)
    sim_matrix = torch.matmul(prot_vec, mol_vec.T)
    
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0
    
    for prot_idx in range(batch_size):
        # 找到属于这个蛋白质的正样本分子
        pos_mask = (mol_labels == prot_idx) & mol_is_positive
        if pos_mask.sum() == 0:
            continue
        
        pos_indices = torch.where(pos_mask)[0]
        
        # 获取排序后的分子索引
        sims = sim_matrix[prot_idx]
        sorted_indices = torch.argsort(sims, descending=True)
        
        # 检查正样本是否在 top-k
        for pos_idx in pos_indices:
            total += 1
            rank = (sorted_indices == pos_idx).nonzero(as_tuple=True)[0].item()
            if rank == 0:
                correct_at_1 += 1
            if rank < 5:
                correct_at_5 += 1
    
    acc_at_1 = correct_at_1 / max(total, 1) * 100
    acc_at_5 = correct_at_5 / max(total, 1) * 100
    
    return acc_at_1, acc_at_5


def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16", 
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print("=" * 70)
        print("蛋白质-分子双塔模型训练 v2 - 基于亲和力的对比学习")
        print("=" * 70)
        print(f"设备: {device}, 进程数: {accelerator.num_processes}")
        print(f"批次大小: {PER_DEVICE_BATCH_SIZE}, 梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"有效批次大小: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * accelerator.num_processes}")
        print(f"正样本阈值 (pKd): >= {POSITIVE_THRESHOLD}")
        print(f"硬负样本范围 (pKd): [{HARD_NEGATIVE_MIN}, {HARD_NEGATIVE_MAX})")
        print(f"每蛋白质正样本数: {NUM_POSITIVES}, 硬负样本数: {NUM_HARD_NEGATIVES}")

    # Tokenizer
    mol_tokenizer = AutoTokenizer.from_pretrained(PATH_CHEMBERTA)
    prot_tokenizer = AutoTokenizer.from_pretrained(PATH_SAPROT, trust_remote_code=True)
    
    # 数据加载
    full_dataset = load_from_disk(PATH_DATA)
    
    if accelerator.is_main_process:
        print(f"\n原始数据集大小: {len(full_dataset)}")
    
    # 创建基于亲和力的数据集
    train_ds = AffinityBasedDataset(full_dataset)
    
    collate_fn = AffinityDataCollator(mol_tokenizer, prot_tokenizer)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=PER_DEVICE_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    if accelerator.is_main_process:
        print(f"每个轮次的批次数: {len(train_loader)}")

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
    
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    loss_history = []
    p2m_loss_history = []
    m2p_loss_history = []
    acc1_history = []
    acc5_history = []

    if accelerator.is_main_process:
        print("\n开始训练...")

    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_p2m = 0
        total_m2p = 0
        total_acc1 = 0
        total_acc5 = 0
        steps_in_epoch = 0
        
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                # 获取数据
                prot_inputs = batch['prot_inputs']
                mol_inputs = batch['mol_inputs']
                mol_labels = batch['mol_labels'].to(device)
                mol_is_positive = batch['mol_is_positive'].to(device)
                mol_affinities = batch['mol_affinities'].to(device)
                
                # 前向传播
                prot_vec, mol_vec, logit_scale = model(
                    prot_inputs['input_ids'], prot_inputs['attention_mask'],
                    mol_inputs['input_ids'], mol_inputs['attention_mask']
                )
                
                # 计算损失
                loss, loss_p2m, loss_m2p = compute_affinity_weighted_loss(
                    prot_vec, mol_vec, mol_labels, mol_is_positive,
                    mol_affinities, logit_scale, device
                )

                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                optimizer.step()
                optimizer.zero_grad()
            
            # 统计
            current_loss = loss.item()
            total_loss += current_loss
            total_p2m += loss_p2m.item()
            total_m2p += loss_m2p.item()
            steps_in_epoch += 1
            
            # 计算准确率
            with torch.no_grad():
                acc1, acc5 = compute_retrieval_accuracy(
                    prot_vec, mol_vec, mol_labels, mol_is_positive, device
                )
                total_acc1 += acc1
                total_acc5 += acc5
            
            if accelerator.is_main_process and step % 5 == 0:
                temp = logit_scale.item()
                print(f"Epoch {epoch+1} | Step {step:4d} | Loss: {current_loss:.4f} | "
                      f"P2M: {loss_p2m.item():.4f} | M2P: {loss_m2p.item():.4f} | "
                      f"Acc@1: {acc1:.1f}% | Acc@5: {acc5:.1f}% | Temp: {temp:.2f}")
                loss_history.append(current_loss)
                p2m_loss_history.append(loss_p2m.item())
                m2p_loss_history.append(loss_m2p.item())
                acc1_history.append(acc1)
                acc5_history.append(acc5)

        # Epoch 结束
        avg_loss = total_loss / steps_in_epoch
        avg_p2m = total_p2m / steps_in_epoch
        avg_m2p = total_m2p / steps_in_epoch
        avg_acc1 = total_acc1 / steps_in_epoch
        avg_acc5 = total_acc5 / steps_in_epoch
        
        if accelerator.is_main_process:
            print(f"\n轮次 {epoch+1} 完成")
            print(f"平均损失: {avg_loss:.4f} (P2M: {avg_p2m:.4f}, M2P: {avg_m2p:.4f})")
            print(f"平均 Acc@1: {avg_acc1:.2f}%, Acc@5: {avg_acc5:.2f}%\n")

        # 每隔一定轮次保存检查点
        if (epoch + 1) % 100 == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                torch.save(unwrapped_model.state_dict(), f"dual_tower_v2_epoch_{epoch+1}.pth")
                print(f"已保存检查点: dual_tower_v2_epoch_{epoch+1}.pth")

    # 最终保存
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        torch.save(unwrapped_model.state_dict(), "dual_tower_v2_final.pth")
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss 总览
        ax1 = axes[0, 0]
        ax1.plot(loss_history, alpha=0.3, label='Total Loss')
        if len(loss_history) > 20:
            smooth = [np.mean(loss_history[max(0,i-20):i+1]) for i in range(len(loss_history))]
            ax1.plot(smooth, linewidth=2, label='Smoothed')
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Step (x10)")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # P2M vs M2P Loss
        ax2 = axes[0, 1]
        ax2.plot(p2m_loss_history, alpha=0.5, label='P2M Loss')
        ax2.plot(m2p_loss_history, alpha=0.5, label='M2P Loss')
        ax2.set_title("Protein→Mol vs Mol→Protein Loss")
        ax2.set_xlabel("Step (x10)")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Acc@1
        ax3 = axes[1, 0]
        ax3.plot(acc1_history, alpha=0.5, label='Acc@1')
        if len(acc1_history) > 20:
            smooth_acc1 = [np.mean(acc1_history[max(0,i-20):i+1]) for i in range(len(acc1_history))]
            ax3.plot(smooth_acc1, linewidth=2, label='Smoothed')
        ax3.set_title("Retrieval Accuracy @1")
        ax3.set_xlabel("Step (x10)")
        ax3.set_ylabel("Accuracy (%)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Acc@5
        ax4 = axes[1, 1]
        ax4.plot(acc5_history, alpha=0.5, label='Acc@5')
        if len(acc5_history) > 20:
            smooth_acc5 = [np.mean(acc5_history[max(0,i-20):i+1]) for i in range(len(acc5_history))]
            ax4.plot(smooth_acc5, linewidth=2, label='Smoothed')
        ax4.set_title("Retrieval Accuracy @5")
        ax4.set_xlabel("Step (x10)")
        ax4.set_ylabel("Accuracy (%)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_loss_v2.png', dpi=150)
        plt.close()
        
        print("\n训练完成")
        print("模型文件: dual_tower_v2_final.pth")
        print("训练曲线: training_loss_v2.png")


if __name__ == "__main__":
    main()
