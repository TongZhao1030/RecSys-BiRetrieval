"""
双向检索推理脚本

本脚本用于评估蛋白质-分子双向检索模型的性能，包括：
- Protein → Molecule 检索
- Molecule → Protein 检索
正确处理测试集内的真实标签
"""

import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# 配置参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_MODEL = "dual_tower_final.pth"
PATH_SAPROT = "/share/home/zhangchiLab/duyinuo/models/westlake-repl_SaProt_650M_AF2"
PATH_CHEMBERTA = "/share/home/zhangchiLab/duyinuo/models/seyonec_ChemBERTa-zinc-base-v1"
PATH_DATA = "/share/home/zhangchiLab/duyinuo/data/vladak_bindingdb"
TEST_SIZE = 100

print(f"启动双向检索验证，设备: {DEVICE}")

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

# 模型定义（与训练脚本完全一致）
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
print("正在加载模型...")
model = DualTowerModel().to(DEVICE)
try:
    state_dict = torch.load(PATH_MODEL, map_location=DEVICE, weights_only=False)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    temp = model.logit_scale.exp().item()
    print(f"模型加载成功，温度参数: {temp:.2f}")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

mol_tokenizer = AutoTokenizer.from_pretrained(PATH_CHEMBERTA)
prot_tokenizer = AutoTokenizer.from_pretrained(PATH_SAPROT, trust_remote_code=True)
dataset = load_from_disk(resolve_split_dataset_path(PATH_DATA, "test"))
protein_col = infer_text_column(dataset.column_names, kind="protein")
molecule_col = infer_text_column(dataset.column_names, kind="molecule")
if protein_col is None or molecule_col is None:
    raise ValueError(f"无法推断列名；现有列: {dataset.column_names}")

# 准备测试数据（选择配对数据）
print("\n正在准备测试数据...")

# 构建映射
prot_to_mols = defaultdict(set)
mol_to_prots = defaultdict(set)

for idx in range(len(dataset)):
    prot = dataset[idx][protein_col]
    mol = dataset[idx][molecule_col]
    prot_to_mols[prot].add(mol)
    mol_to_prots[mol].add(prot)

# 选择唯一蛋白质，并记录其配对分子
# 确保测试集中的分子和蛋白质是配对的
seen_prots = set()
seen_mols = set()
test_pairs = []  # [(prot, mol), ...]

for idx in range(len(dataset)):
    prot = dataset[idx][protein_col]
    mol = dataset[idx][molecule_col]
    
    # 确保蛋白质和分子都是唯一的
    if prot not in seen_prots and mol not in seen_mols:
        seen_prots.add(prot)
        seen_mols.add(mol)
        test_pairs.append((prot, mol))
    
    if len(test_pairs) >= TEST_SIZE:
        break

# 分离蛋白质和分子列表
unique_prots = [p for p, m in test_pairs]
unique_mols = [m for p, m in test_pairs]

# 转换为集合用于快速查找
unique_prots_set = set(unique_prots)
unique_mols_set = set(unique_mols)

print(f"测试配对数量: {len(test_pairs)}")
print(f"唯一蛋白质: {len(unique_prots)}")
print(f"唯一分子: {len(unique_mols)}")

# 计算测试集内的正确答案数量
avg_gt_in_testset = []
for prot in unique_prots:
    gt_mols_in_testset = prot_to_mols[prot] & unique_mols_set
    avg_gt_in_testset.append(len(gt_mols_in_testset))
print(f"平均每个蛋白质在测试集中有 {np.mean(avg_gt_in_testset):.2f} 个正确分子")

# 编码函数
def encode_proteins(prot_list):
    p_fmt = [" ".join([aa + "#" for aa in seq]) for seq in prot_list]
    p_in = prot_tokenizer(p_fmt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        prot_out = model.prot_model(input_ids=p_in['input_ids'], attention_mask=p_in['attention_mask'])
        mask = p_in['attention_mask'].unsqueeze(-1).expand(prot_out.last_hidden_state.size()).float()
        prot_emb = torch.sum(prot_out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        prot_emb = model.prot_layernorm(prot_emb)
        prot_vec = model.prot_proj(prot_emb)
        prot_vec = F.normalize(prot_vec, p=2, dim=1)
    return prot_vec

def encode_molecules(mol_list):
    m_in = mol_tokenizer(mol_list, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        mol_out = model.mol_model(input_ids=m_in['input_ids'], attention_mask=m_in['attention_mask'])
        mask = m_in['attention_mask'].unsqueeze(-1).expand(mol_out.last_hidden_state.size()).float()
        mol_emb = torch.sum(mol_out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        mol_emb = model.mol_layernorm(mol_emb)
        mol_vec = model.mol_proj(mol_emb)
        mol_vec = F.normalize(mol_vec, p=2, dim=1)
    return mol_vec

# 构建向量库
print("\n正在构建向量库...")

BATCH = 16
prot_vecs = []
mol_vecs = []

for i in range(0, len(unique_prots), BATCH):
    batch = unique_prots[i:i+BATCH]
    vecs = encode_proteins(batch)
    prot_vecs.append(vecs)

for i in range(0, len(unique_mols), BATCH):
    batch = unique_mols[i:i+BATCH]
    vecs = encode_molecules(batch)
    mol_vecs.append(vecs)

DB_PROT = torch.cat(prot_vecs, dim=0)
DB_MOL = torch.cat(mol_vecs, dim=0)

print(f"蛋白质向量库: {DB_PROT.shape}")
print(f"分子向量库: {DB_MOL.shape}")

# 评估：Protein → Molecule
print("\n" + "=" * 60)
print("任务 1: Protein → Molecule（给定蛋白质，推荐药物）")
print("=" * 60)

sim_matrix_p2m = torch.matmul(DB_PROT, DB_MOL.T)

recall_at_1 = 0
recall_at_5 = 0
recall_at_10 = 0
mrr_sum = 0
valid_queries = 0

for i, prot in enumerate(unique_prots):
    scores = sim_matrix_p2m[i].cpu().numpy()
    sorted_indices = np.argsort(scores)[::-1]
    
    # 只计算测试集内的正确答案
    gt_mols_in_testset = prot_to_mols[prot] & unique_mols_set
    
    if len(gt_mols_in_testset) == 0:
        continue  # 跳过没有正确答案的查询
    
    valid_queries += 1
    
    hit_at_1 = any(unique_mols[idx] in gt_mols_in_testset for idx in sorted_indices[:1])
    hit_at_5 = any(unique_mols[idx] in gt_mols_in_testset for idx in sorted_indices[:5])
    hit_at_10 = any(unique_mols[idx] in gt_mols_in_testset for idx in sorted_indices[:10])
    
    if hit_at_1: recall_at_1 += 1
    if hit_at_5: recall_at_5 += 1
    if hit_at_10: recall_at_10 += 1
    
    for rank, idx in enumerate(sorted_indices, 1):
        if unique_mols[idx] in gt_mols_in_testset:
            mrr_sum += 1 / rank
            break

print(f"\nProtein → Molecule 结果（有效查询: {valid_queries}）:")
print(f"   Recall@1:  {recall_at_1}/{valid_queries} ({recall_at_1/valid_queries*100:.1f}%)")
print(f"   Recall@5:  {recall_at_5}/{valid_queries} ({recall_at_5/valid_queries*100:.1f}%)")
print(f"   Recall@10: {recall_at_10}/{valid_queries} ({recall_at_10/valid_queries*100:.1f}%)")
print(f"   MRR:       {mrr_sum/valid_queries:.4f}")

# 评估：Molecule → Protein
print("\n" + "=" * 60)
print("任务 2: Molecule → Protein（给定药物，找靶点蛋白）")
print("=" * 60)

sim_matrix_m2p = sim_matrix_p2m.T

recall_at_1_m = 0
recall_at_5_m = 0
recall_at_10_m = 0
mrr_sum_m = 0
valid_queries_m = 0

for i, mol in enumerate(unique_mols):
    scores = sim_matrix_m2p[i].cpu().numpy()
    sorted_indices = np.argsort(scores)[::-1]
    
    # 只计算测试集内的正确答案
    gt_prots_in_testset = mol_to_prots[mol] & unique_prots_set
    
    if len(gt_prots_in_testset) == 0:
        continue
    
    valid_queries_m += 1
    
    hit_at_1 = any(unique_prots[idx] in gt_prots_in_testset for idx in sorted_indices[:1])
    hit_at_5 = any(unique_prots[idx] in gt_prots_in_testset for idx in sorted_indices[:5])
    hit_at_10 = any(unique_prots[idx] in gt_prots_in_testset for idx in sorted_indices[:10])
    
    if hit_at_1: recall_at_1_m += 1
    if hit_at_5: recall_at_5_m += 1
    if hit_at_10: recall_at_10_m += 1
    
    for rank, idx in enumerate(sorted_indices, 1):
        if unique_prots[idx] in gt_prots_in_testset:
            mrr_sum_m += 1 / rank
            break

print(f"\nMolecule → Protein 结果（有效查询: {valid_queries_m}）:")
print(f"   Recall@1:  {recall_at_1_m}/{valid_queries_m} ({recall_at_1_m/valid_queries_m*100:.1f}%)")
print(f"   Recall@5:  {recall_at_5_m}/{valid_queries_m} ({recall_at_5_m/valid_queries_m*100:.1f}%)")
print(f"   Recall@10: {recall_at_10_m}/{valid_queries_m} ({recall_at_10_m/valid_queries_m*100:.1f}%)")
print(f"   MRR:       {mrr_sum_m/valid_queries_m:.4f}")

# 汇总结果
print("\n" + "=" * 60)
print("汇总结果")
print("=" * 60)
p2m_r1 = recall_at_1/valid_queries*100
m2p_r1 = recall_at_1_m/valid_queries_m*100
p2m_mrr = mrr_sum/valid_queries
m2p_mrr = mrr_sum_m/valid_queries_m

print(f"\n   方向          | Recall@1 | Recall@5 | Recall@10 | MRR")
print(f"   --------------|----------|----------|-----------|-------")
print(f"   P → M         | {p2m_r1:6.1f}% | {recall_at_5/valid_queries*100:6.1f}% | {recall_at_10/valid_queries*100:7.1f}% | {p2m_mrr:.4f}")
print(f"   M → P         | {m2p_r1:6.1f}% | {recall_at_5_m/valid_queries_m*100:6.1f}% | {recall_at_10_m/valid_queries_m*100:7.1f}% | {m2p_mrr:.4f}")
print(f"   --------------|----------|----------|-----------|-------")
print(f"   平均          | {(p2m_r1+m2p_r1)/2:6.1f}% | {(recall_at_5/valid_queries+recall_at_5_m/valid_queries_m)*50:6.1f}% | {(recall_at_10/valid_queries+recall_at_10_m/valid_queries_m)*50:7.1f}% | {(p2m_mrr+m2p_mrr)/2:.4f}")

# 不对称性分析
asymmetry = abs(p2m_r1 - m2p_r1)
print(f"\n双向不对称度: {asymmetry:.1f}%")
if asymmetry < 10:
    print("对称性良好")
elif asymmetry < 20:
    print("存在一定不对称，但可接受")
else:
    print("严重不对称，需要关注")

# 示例查询
print("\n" + "=" * 60)
print("示例查询")
print("=" * 60)

# 示例 1: 蛋白质 → 分子
print("\n【示例 1】输入蛋白质，推荐药物")
query_prot = unique_prots[0]
gt_mols_in_testset = prot_to_mols[query_prot] & unique_mols_set
print(f"查询蛋白质: {query_prot[:50]}...")
print(f"测试集中的正确答案数: {len(gt_mols_in_testset)}")

scores = sim_matrix_p2m[0].cpu().numpy()
top5_indices = np.argsort(scores)[::-1][:5]
print("Top-5 推荐:")
for rank, idx in enumerate(top5_indices, 1):
    mol = unique_mols[idx]
    is_correct = "[***正确***]" if mol in gt_mols_in_testset else "[错误]"
    print(f"  {rank}. {mol[:40]}... (得分: {scores[idx]:.4f}) {is_correct}")

# 示例 2: 分子 → 蛋白质
print("\n【示例 2】输入药物，找靶点蛋白")
query_mol = unique_mols[0]
gt_prots_in_testset = mol_to_prots[query_mol] & unique_prots_set
print(f"查询药物: {query_mol[:50]}...")
print(f"测试集中的正确答案数: {len(gt_prots_in_testset)}")

scores = sim_matrix_m2p[0].cpu().numpy()
top5_indices = np.argsort(scores)[::-1][:5]
print("Top-5 推荐:")
for rank, idx in enumerate(top5_indices, 1):
    prot = unique_prots[idx]
    is_correct = "[***正确***]" if prot in gt_prots_in_testset else "[错误]"
    print(f"  {rank}. {prot[:40]}... (得分: {scores[idx]:.4f}) {is_correct}")

print("\n验证完成")
