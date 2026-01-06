"""
蛋白质-分子双塔模型训练脚本

主要特性：
1. 数据去重处理 - 确保每个批次中蛋白质和分子都是唯一的
2. 使用分组采样器 - 避免同一蛋白质在同一批次中出现多次
3. 支持分布式训练和混合精度训练
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

# 训练配置参数
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

# 路径
PATH_SAPROT = "./models/SaProt_650M_AF2"
PATH_CHEMBERTA = "./models/ChemBERTa-zinc-base-v1"
PATH_DATA = "./data/bindingdb_local"

# 数据预处理 - 去重
def deduplicate_dataset(dataset):
    """
    创建去重后的数据集索引
    确保每个 (蛋白质, 分子) 对只出现一次
    """
    seen_pairs = set()
    unique_indices = []
    
    for idx in range(len(dataset)):
        prot = dataset[idx]['Protein Sequence']
        mol = dataset[idx]['Molecule Sequence']
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
    def __init__(self, dataset, batch_size, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # 按蛋白质分组
        self.prot_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            prot = dataset[idx]['Protein Sequence']
            self.prot_to_indices[prot].append(idx)
        
        self.unique_proteins = list(self.prot_to_indices.keys())
        print(f"数据集统计: {len(dataset)} 样本, {len(self.unique_proteins)} 唯一蛋白质")
    
    def __iter__(self):
        # 打乱蛋白质顺序
        shuffled_prots = self.unique_proteins.copy()
        random.shuffle(shuffled_prots)
        
        batch = []
        for prot in shuffled_prots:
            # 从每个蛋白质的所有配对中随机选一个
            idx = random.choice(self.prot_to_indices[prot])
            batch.append(idx)
            
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.unique_proteins) // self.batch_size
        return (len(self.unique_proteins) + self.batch_size - 1) // self.batch_size

# 数据集和数据整理器
class BioDataset(Dataset):
    def __init__(self, hf_dataset, indices=None):
        self.data = hf_dataset
        self.indices = indices if indices is not None else list(range(len(hf_dataset)))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # idx 是 sampler 给的原始索引，不需要转换
        if isinstance(idx, int) and idx < len(self.data):
            item = self.data[idx]
        else:
            item = self.data[self.indices[idx]]
        
        smiles = item['Molecule Sequence']
        raw_prot = item['Protein Sequence']
        
        prot_list = [aa + "#" for aa in raw_prot]
        formatted_prot = " ".join(prot_list)
        
        return smiles, formatted_prot

class DataCollate:
    def __init__(self, mol_tok, prot_tok):
        self.mol_tok = mol_tok
        self.prot_tok = prot_tok

    def __call__(self, batch):
        smiles_list = [item[0] for item in batch]
        prot_list = [item[1] for item in batch]
        
        mol_inputs = self.mol_tok(
            smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        prot_inputs = self.prot_tok(
            prot_list, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        
        return mol_inputs, prot_inputs

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
    full_dataset = load_from_disk(PATH_DATA)
    
    if accelerator.is_main_process:
        print(f"\n原始数据集大小: {len(full_dataset)}")
    
    # 创建 Dataset（使用原始数据，采样器会处理去重）
    train_ds = BioDataset(full_dataset)
    
    # 使用自定义采样器确保批次内蛋白质唯一
    batch_sampler = UniqueProteinBatchSampler(
        full_dataset, 
        batch_size=PER_DEVICE_BATCH_SIZE,
        drop_last=True
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
            drop_last=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        for step, (mol_inputs, prot_inputs) in enumerate(train_loader):
            # 手动移动到设备
            mol_inputs = {k: v.to(device) for k, v in mol_inputs.items()}
            prot_inputs = {k: v.to(device) for k, v in prot_inputs.items()}
            
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
                
                loss_i = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.T, labels)
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

