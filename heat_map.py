"""
双向检索评估脚本
"""

import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_MODEL = "dual_tower_final.pth"
PATH_SAPROT = "/share/home/zhangchiLab/duyinuo/models/westlake-repl_SaProt_650M_AF2"
PATH_CHEMBERTA = "/share/home/zhangchiLab/duyinuo/models/seyonec_ChemBERTa-zinc-base-v1"
PATH_DATA = "/share/home/zhangchiLab/duyinuo/data/vladak_bindingdb"
TEST_SIZE = 100 

print(f"Device: {DEVICE}")

def resolve_split_dataset_path(data_path, split):
    if split:
        candidate = os.path.join(data_path, split)
        if os.path.isdir(candidate):
            return candidate
    for default_split in ("test", "train", "valid", "validation"):
        candidate = os.path.join(data_path, default_split)
        if os.path.isdir(candidate):
            return candidate
    return data_path

def infer_text_column(cols, *, kind):
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

class DualTowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prot_model = AutoModel.from_pretrained(PATH_SAPROT, trust_remote_code=True)
        self.mol_model = AutoModel.from_pretrained(PATH_CHEMBERTA)
        
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
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, prot_input_ids, prot_attention_mask, mol_input_ids, mol_attention_mask):
        prot_out = self.prot_model(input_ids=prot_input_ids, attention_mask=prot_attention_mask)
        mask = prot_attention_mask.unsqueeze(-1).expand(prot_out.last_hidden_state.size()).float()
        prot_emb = torch.sum(prot_out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        prot_emb = self.prot_layernorm(prot_emb)
        prot_vec = self.prot_proj(prot_emb)

        mol_out = self.mol_model(input_ids=mol_input_ids, attention_mask=mol_attention_mask)
        mol_mask = mol_attention_mask.unsqueeze(-1).expand(mol_out.last_hidden_state.size()).float()
        mol_emb = torch.sum(mol_out.last_hidden_state * mol_mask, dim=1) / torch.clamp(mol_mask.sum(1), min=1e-9)
        mol_emb = self.mol_layernorm(mol_emb)
        mol_vec = self.mol_proj(mol_emb)

        prot_vec = F.normalize(prot_vec, p=2, dim=1)
        mol_vec = F.normalize(mol_vec, p=2, dim=1)
        
        return prot_vec, mol_vec, self.logit_scale.exp()

# 加载模型
print("加载模型...")
model = DualTowerModel().to(DEVICE)
try:
    state_dict = torch.load(PATH_MODEL, map_location=DEVICE, weights_only=False)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"模型加载成功, Temperature: {model.logit_scale.exp().item():.2f}")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# 准备测试数据
mol_tokenizer = AutoTokenizer.from_pretrained(PATH_CHEMBERTA)
prot_tokenizer = AutoTokenizer.from_pretrained(PATH_SAPROT, trust_remote_code=True)
dataset = load_from_disk(resolve_split_dataset_path(PATH_DATA, "test"))
protein_col = infer_text_column(dataset.column_names, kind="protein")
molecule_col = infer_text_column(dataset.column_names, kind="molecule")
if protein_col is None or molecule_col is None:
    raise ValueError(f"无法推断列名；现有列: {dataset.column_names}")

# 构建全局映射
from collections import defaultdict
prot_to_mols = defaultdict(set)
mol_to_prots = defaultdict(set)
for idx in range(len(dataset)):
    prot = dataset[idx][protein_col]
    mol = dataset[idx][molecule_col]
    prot_to_mols[prot].add(mol)
    mol_to_prots[mol].add(prot)

# 选择唯一配对数据
seen_prots = set()
seen_mols = set()
test_pairs = []
for idx in range(len(dataset)):
    prot = dataset[idx][protein_col]
    mol = dataset[idx][molecule_col]
    if prot not in seen_prots and mol not in seen_mols:
        seen_prots.add(prot)
        seen_mols.add(mol)
        test_pairs.append((prot, mol))
    if len(test_pairs) >= TEST_SIZE:
        break

prots = [p for p, m in test_pairs]
mols = [m for p, m in test_pairs]
prots_set = set(prots)
mols_set = set(mols)
print(f"测试样本: {len(test_pairs)} 对")

def encode_batch(p_list, m_list):
    p_fmt = [" ".join([aa + "#" for aa in seq]) for seq in p_list]
    
    p_in = prot_tokenizer(p_fmt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    m_in = mol_tokenizer(m_list, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    
    with torch.no_grad():
        pv, mv, _ = model(
            p_in['input_ids'], p_in['attention_mask'],
            m_in['input_ids'], m_in['attention_mask']
        )
    return pv, mv

print("计算向量...")
db_prot_vecs = []
db_mol_vecs = []

BATCH = 32
for i in range(0, TEST_SIZE, BATCH):
    p_batch = prots[i:i+BATCH]
    m_batch = mols[i:i+BATCH]
    pv, mv = encode_batch(p_batch, m_batch)
    db_prot_vecs.append(pv)
    db_mol_vecs.append(mv)

prot_vecs = torch.cat(db_prot_vecs, dim=0)
mol_vecs = torch.cat(db_mol_vecs, dim=0)

sim_matrix = torch.matmul(prot_vecs, mol_vecs.T).cpu().numpy()

print("\n=== 双向检索评估 ===")

def compute_metrics(sim_matrix, direction="P→M"):
    n = sim_matrix.shape[0]
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    ranks = []
    valid_queries = 0
    
    for i in range(n):
        if direction == "P→M":
            scores = sim_matrix[i]
            gt_set = prot_to_mols[prots[i]] & mols_set
        else:
            scores = sim_matrix.T[i]
            gt_set = mol_to_prots[mols[i]] & prots_set
        
        if len(gt_set) == 0:
            continue
        valid_queries += 1
        
        sorted_indices = np.argsort(scores)[::-1]
        target_list = mols if direction == "P→M" else prots
        first_hit_rank = None
        for rank, idx in enumerate(sorted_indices):
            if target_list[idx] in gt_set:
                if first_hit_rank is None:
                    first_hit_rank = rank
                break
        
        if first_hit_rank is not None:
            ranks.append(first_hit_rank)
            if first_hit_rank == 0:
                top1_hits += 1
            if first_hit_rank < 5:
                top5_hits += 1
            if first_hit_rank < 10:
                top10_hits += 1
    
    return {
        'top1': top1_hits / valid_queries * 100 if valid_queries > 0 else 0,
        'top5': top5_hits / valid_queries * 100 if valid_queries > 0 else 0,
        'top10': top10_hits / valid_queries * 100 if valid_queries > 0 else 0,
        'mean_rank': np.mean(ranks) if ranks else 0,
        'mrr': np.mean([1.0 / (r + 1) for r in ranks]) if ranks else 0,
        'valid_queries': valid_queries
    }

# Protein → Molecule
p2m = compute_metrics(sim_matrix, "P→M")
print(f"\nProtein → Molecule (有效查询: {p2m['valid_queries']})")
print(f"   Top-1: {p2m['top1']:.1f}%  Top-5: {p2m['top5']:.1f}%  Top-10: {p2m['top10']:.1f}%")
print(f"   Mean Rank: {p2m['mean_rank']:.2f}  MRR: {p2m['mrr']:.4f}")

# Molecule → Protein
m2p = compute_metrics(sim_matrix, "M→P")
print(f"\nMolecule → Protein (有效查询: {m2p['valid_queries']})")
print(f"   Top-1: {m2p['top1']:.1f}%  Top-5: {m2p['top5']:.1f}%  Top-10: {m2p['top10']:.1f}%")
print(f"   Mean Rank: {m2p['mean_rank']:.2f}  MRR: {m2p['mrr']:.4f}")

# 平均
print(f"\n平均: Top-1 {(p2m['top1'] + m2p['top1']) / 2:.1f}%  MRR {(p2m['mrr'] + m2p['mrr']) / 2:.4f}")

# 区分度分析
diag_scores = np.diag(sim_matrix)
mask = ~np.eye(TEST_SIZE, dtype=bool)
off_diag_scores = sim_matrix[mask]
margin = diag_scores.mean() - off_diag_scores.mean()

print(f"\n区分度: 正样本 {diag_scores.mean():.4f} / 负样本 {off_diag_scores.mean():.4f} / Margin {margin:.4f}")

# 生成热力图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 完整热力图
ax1 = axes[0]
sns.heatmap(sim_matrix, cmap="viridis", center=0, ax=ax1)
ax1.set_title(f"Similarity Matrix (N={TEST_SIZE})\nP→M: {p2m['top1']:.1f}% | M→P: {m2p['top1']:.1f}%")
ax1.set_xlabel("Molecule Index")
ax1.set_ylabel("Protein Index")

# 局部放大 (前20个)
ax2 = axes[1]
sub_size = min(20, TEST_SIZE)
sub_matrix = sim_matrix[:sub_size, :sub_size]
sns.heatmap(sub_matrix, cmap="viridis", center=0, ax=ax2, annot=True, fmt=".2f", annot_kws={"size": 6})
ax2.set_title(f"Zoomed View")
ax2.set_xlabel("Molecule Index")
ax2.set_ylabel("Protein Index")

plt.tight_layout()
plt.savefig("diagnosis_heatmap.png", dpi=150)
print(f"\n热力图已保存: diagnosis_heatmap.png")

# 不对称性检查
asymmetry = abs(p2m['top1'] - m2p['top1'])
if asymmetry > 20:
    weaker = "P→M" if p2m['top1'] < m2p['top1'] else "M→P"
    print(f"\n注意: 双向不对称 {asymmetry:.1f}%, {weaker} 较弱")

print("\n评估完成")
