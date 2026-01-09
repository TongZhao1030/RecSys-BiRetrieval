"""
蛋白质-分子双向检索验证脚本 (IC50版本)

正确的评估逻辑（全局检索）：
1. 构建全局蛋白质库和分子库
2. 构建正样本配对关系（pIC50 >= 7.0 的配对）
3. P→M：给定蛋白质，从全局分子库检索，检查 top-k 是否命中正样本
4. M→P：给定分子，从全局蛋白质库检索，检查 top-k 是否命中正样本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm

# ============================================================================
# 配置
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_MODEL = "./outputs/model_best.pth"
PATH_SAPROT = "./models/SaProt_650M_AF2"
PATH_CHEMBERTA = "./models/ChemBERTa-zinc-base-v1"
PATH_DATA_TEST = "./data/vladak_bindingdb/test"

# 评估配置（与训练对齐）
POSITIVE_THRESHOLD = 7.0       # pIC50 >= 7.0 视为正样本（与训练配置一致）
MAX_PROTEINS = 2000            # 全局蛋白质库大小
MAX_MOLECULES = 5000           # 全局分子库大小
TOP_K_LIST = [1, 5, 10, 50, 100]  # Recall@K 的 K 值

print(f"设备: {DEVICE}")

# ============================================================================
# 模型定义（与训练一致）
# ============================================================================
class DualTowerModel(nn.Module):
    def __init__(self, path_saprot, path_chemberta):
        super().__init__()
        self.prot_model = AutoModel.from_pretrained(path_saprot, trust_remote_code=True)
        self.mol_model = AutoModel.from_pretrained(path_chemberta)
        
        prot_hidden = self.prot_model.config.hidden_size
        mol_hidden = 768
        hidden_dim = 1024  # 与训练一致
        embedding_dim = 256
        
        # 4层投影网络（与训练一致）
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
        
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))
    
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
    
    def get_temperature(self):
        return self.log_temperature.exp()

# ============================================================================
# 数据预处理
# ============================================================================
def build_global_database(dataset, positive_threshold=POSITIVE_THRESHOLD):
    """
    构建全局数据库
    
    Returns:
        proteins: 唯一蛋白质列表
        molecules: 唯一分子列表
        prot_to_idx: 蛋白质 -> 索引
        mol_to_idx: 分子 -> 索引
        prot_positives: 蛋白质索引 -> 正样本分子索引集合
        mol_positives: 分子索引 -> 正样本蛋白质索引集合
    """
    print("构建全局数据库...")
    
    # 收集所有蛋白质-分子配对
    all_pairs = []  # (protein, molecule, is_positive)
    seen_prots = set()
    seen_mols = set()
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        pic50 = item['ic50']
        
        if pic50 is None or math.isnan(pic50) or math.isinf(pic50):
            continue
        if pic50 < -2 or pic50 > 14:
            continue
        
        prot = item['protein']
        mol = item['ligand']
        is_positive = pic50 >= positive_threshold
        
        all_pairs.append((prot, mol, is_positive))
        seen_prots.add(prot)
        seen_mols.add(mol)
    
    print(f"原始数据: {len(all_pairs)} 配对, {len(seen_prots)} 蛋白质, {len(seen_mols)} 分子")
    
    # 统计每个蛋白质的正样本数，优先选择有正样本的蛋白质
    prot_pos_count = defaultdict(int)
    mol_pos_count = defaultdict(int)
    for prot, mol, is_pos in all_pairs:
        if is_pos:
            prot_pos_count[prot] += 1
            mol_pos_count[mol] += 1
    
    # 选择有正样本的蛋白质和分子（优先）
    prots_with_pos = [p for p in seen_prots if prot_pos_count[p] > 0]
    mols_with_pos = [m for m in seen_mols if mol_pos_count[m] > 0]
    
    # 构建蛋白质库：优先选择有正样本的
    proteins = prots_with_pos[:MAX_PROTEINS]
    if len(proteins) < MAX_PROTEINS:
        # 补充没有正样本的蛋白质
        prots_without_pos = [p for p in seen_prots if prot_pos_count[p] == 0]
        proteins.extend(prots_without_pos[:MAX_PROTEINS - len(proteins)])
    
    # 构建分子库：优先选择有正样本的
    molecules = mols_with_pos[:MAX_MOLECULES]
    if len(molecules) < MAX_MOLECULES:
        mols_without_pos = [m for m in seen_mols if mol_pos_count[m] == 0]
        molecules.extend(mols_without_pos[:MAX_MOLECULES - len(molecules)])
    
    # 建立索引映射
    prot_to_idx = {p: i for i, p in enumerate(proteins)}
    mol_to_idx = {m: i for i, m in enumerate(molecules)}
    
    # 构建正样本关系（只考虑在库中的配对）
    prot_positives = defaultdict(set)  # prot_idx -> set of mol_idx
    mol_positives = defaultdict(set)   # mol_idx -> set of prot_idx
    
    for prot, mol, is_pos in all_pairs:
        if not is_pos:
            continue
        if prot not in prot_to_idx or mol not in mol_to_idx:
            continue
        
        prot_idx = prot_to_idx[prot]
        mol_idx = mol_to_idx[mol]
        prot_positives[prot_idx].add(mol_idx)
        mol_positives[mol_idx].add(prot_idx)
    
    # 统计
    prots_with_positives = sum(1 for p_idx in range(len(proteins)) if len(prot_positives[p_idx]) > 0)
    mols_with_positives = sum(1 for m_idx in range(len(molecules)) if len(mol_positives[m_idx]) > 0)
    
    print(f"\n全局数据库构建完成:")
    print(f"  蛋白质库: {len(proteins)} 个 (其中 {prots_with_positives} 个有正样本)")
    print(f"  分子库: {len(molecules)} 个 (其中 {mols_with_positives} 个有正样本)")
    print(f"  正样本配对数: {sum(len(v) for v in prot_positives.values())}")
    
    return proteins, molecules, prot_to_idx, mol_to_idx, prot_positives, mol_positives

# ============================================================================
# 主评估流程
# ============================================================================
def main():
    # 加载模型
    print("\n加载模型...")
    model = DualTowerModel(PATH_SAPROT, PATH_CHEMBERTA).to(DEVICE)
    
    checkpoint = torch.load(PATH_MODEL, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    temp = model.get_temperature().item()
    print(f"模型加载成功，温度参数: {temp:.4f}")
    
    # 加载 tokenizer
    mol_tokenizer = AutoTokenizer.from_pretrained(PATH_CHEMBERTA)
    prot_tokenizer = AutoTokenizer.from_pretrained(PATH_SAPROT, trust_remote_code=True)
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_dataset = load_from_disk(PATH_DATA_TEST)
    
    # 构建全局数据库
    proteins, molecules, prot_to_idx, mol_to_idx, prot_positives, mol_positives = \
        build_global_database(test_dataset, POSITIVE_THRESHOLD)
    
    # ========================================================================
    # 预计算所有向量
    # ========================================================================
    print("\n预计算所有向量...")
    
    def encode_proteins_batch(prot_list, batch_size=8):
        all_vecs = []
        for i in tqdm(range(0, len(prot_list), batch_size), desc="编码蛋白质"):
            batch = prot_list[i:i+batch_size]
            formatted = [" ".join([aa + "#" for aa in seq]) for seq in batch]
            inputs = prot_tokenizer(formatted, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                vecs = model.encode_protein(inputs['input_ids'], inputs['attention_mask'])
            all_vecs.append(vecs.cpu())
        return torch.cat(all_vecs, dim=0)
    
    def encode_molecules_batch(mol_list, batch_size=32):
        all_vecs = []
        for i in tqdm(range(0, len(mol_list), batch_size), desc="编码分子"):
            batch = mol_list[i:i+batch_size]
            inputs = mol_tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                vecs = model.encode_molecule(inputs['input_ids'], inputs['attention_mask'])
            all_vecs.append(vecs.cpu())
        return torch.cat(all_vecs, dim=0)
    
    prot_vecs = encode_proteins_batch(proteins)
    mol_vecs = encode_molecules_batch(molecules)
    
    print(f"蛋白质向量: {prot_vecs.shape}")
    print(f"分子向量: {mol_vecs.shape}")
    
    # ========================================================================
    # 评估：Protein → Molecule（全局检索）
    # ========================================================================
    print("\n" + "=" * 70)
    print("评估: Protein → Molecule (全局检索)")
    print(f"从 {len(molecules)} 个分子中检索")
    print("=" * 70)
    
    # 只评估有正样本的蛋白质
    query_prot_indices = [i for i in range(len(proteins)) if len(prot_positives[i]) > 0]
    print(f"有正样本的蛋白质数: {len(query_prot_indices)}")
    
    p2m_recall = {k: 0 for k in TOP_K_LIST}
    p2m_mrr = 0.0
    
    # 计算相似度矩阵（分块计算避免OOM）
    print("计算 P→M 相似度...")
    
    for prot_idx in tqdm(query_prot_indices, desc="P→M 检索"):
        prot_vec = prot_vecs[prot_idx:prot_idx+1]  # [1, dim]
        
        # 计算与所有分子的相似度
        scores = torch.matmul(prot_vec, mol_vecs.T).squeeze().numpy()  # [n_mols]
        
        # 获取正样本分子索引
        positive_mol_indices = prot_positives[prot_idx]
        
        # 排序
        sorted_indices = np.argsort(scores)[::-1]
        
        # 计算 Recall@K
        for k in TOP_K_LIST:
            top_k_indices = set(sorted_indices[:k])
            if len(top_k_indices & positive_mol_indices) > 0:
                p2m_recall[k] += 1
        
        # 计算 MRR
        for rank, idx in enumerate(sorted_indices, 1):
            if idx in positive_mol_indices:
                p2m_mrr += 1.0 / rank
                break
    
    n_p2m_queries = len(query_prot_indices)
    
    print(f"\nProtein → Molecule 结果 (n={n_p2m_queries}, 分子库={len(molecules)}):")
    for k in TOP_K_LIST:
        print(f"  Recall@{k}: {p2m_recall[k]/n_p2m_queries*100:.2f}%")
    print(f"  MRR: {p2m_mrr/n_p2m_queries:.4f}")
    
    # ========================================================================
    # 评估：Molecule → Protein（全局检索）
    # ========================================================================
    print("\n" + "=" * 70)
    print("评估: Molecule → Protein (全局检索)")
    print(f"从 {len(proteins)} 个蛋白质中检索")
    print("=" * 70)
    
    # 只评估有正样本的分子
    query_mol_indices = [i for i in range(len(molecules)) if len(mol_positives[i]) > 0]
    print(f"有正样本的分子数: {len(query_mol_indices)}")
    
    m2p_recall = {k: 0 for k in TOP_K_LIST}
    m2p_mrr = 0.0
    
    print("计算 M→P 相似度...")
    
    for mol_idx in tqdm(query_mol_indices, desc="M→P 检索"):
        mol_vec = mol_vecs[mol_idx:mol_idx+1]  # [1, dim]
        
        # 计算与所有蛋白质的相似度
        scores = torch.matmul(mol_vec, prot_vecs.T).squeeze().numpy()  # [n_prots]
        
        # 获取正样本蛋白质索引
        positive_prot_indices = mol_positives[mol_idx]
        
        # 排序
        sorted_indices = np.argsort(scores)[::-1]
        
        # 计算 Recall@K
        for k in TOP_K_LIST:
            top_k_indices = set(sorted_indices[:k])
            if len(top_k_indices & positive_prot_indices) > 0:
                m2p_recall[k] += 1
        
        # 计算 MRR
        for rank, idx in enumerate(sorted_indices, 1):
            if idx in positive_prot_indices:
                m2p_mrr += 1.0 / rank
                break
    
    n_m2p_queries = len(query_mol_indices)
    
    print(f"\nMolecule → Protein 结果 (n={n_m2p_queries}, 蛋白质库={len(proteins)}):")
    for k in TOP_K_LIST:
        print(f"  Recall@{k}: {m2p_recall[k]/n_m2p_queries*100:.2f}%")
    print(f"  MRR: {m2p_mrr/n_m2p_queries:.4f}")
    
    # ========================================================================
    # 汇总
    # ========================================================================
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    
    print(f"\n正样本阈值: pIC50 >= {POSITIVE_THRESHOLD}")
    print(f"蛋白质库大小: {len(proteins)}, 分子库大小: {len(molecules)}")
    
    print(f"\n        P→M       M→P       平均")
    print("-" * 45)
    for k in TOP_K_LIST:
        p2m = p2m_recall[k]/n_p2m_queries*100
        m2p = m2p_recall[k]/n_m2p_queries*100
        avg = (p2m + m2p) / 2
        print(f"R@{k:<3}   {p2m:5.2f}%    {m2p:5.2f}%    {avg:5.2f}%")
    
    p2m_mrr_val = p2m_mrr/n_p2m_queries
    m2p_mrr_val = m2p_mrr/n_m2p_queries
    avg_mrr = (p2m_mrr_val + m2p_mrr_val) / 2
    print(f"MRR     {p2m_mrr_val:.4f}    {m2p_mrr_val:.4f}    {avg_mrr:.4f}")
    
    # ========================================================================
    # 示例展示
    # ========================================================================
    print("\n" + "=" * 70)
    print("示例展示")
    print("=" * 70)
    
    # 选择一个有正样本的蛋白质
    example_prot_idx = query_prot_indices[0]
    example_prot = proteins[example_prot_idx]
    example_positives = prot_positives[example_prot_idx]
    
    print(f"\n【查询蛋白质】{example_prot[:60]}...")
    print(f"该蛋白质在分子库中有 {len(example_positives)} 个正样本分子")
    
    # 模型检索结果
    prot_vec = prot_vecs[example_prot_idx:example_prot_idx+1]
    scores = torch.matmul(prot_vec, mol_vecs.T).squeeze().numpy()
    sorted_indices = np.argsort(scores)[::-1]
    
    print("\n模型检索 Top-10:")
    for rank, mol_idx in enumerate(sorted_indices[:10], 1):
        mol = molecules[mol_idx]
        is_pos = mol_idx in example_positives
        label = "✓ 正样本" if is_pos else "✗ 负样本"
        print(f"  {rank:2d}. {mol[:50]}... | score={scores[mol_idx]:.4f} | {label}")
    
    # 正样本的排名
    print(f"\n正样本分子的排名:")
    pos_ranks = []
    for rank, mol_idx in enumerate(sorted_indices, 1):
        if mol_idx in example_positives:
            pos_ranks.append(rank)
            if len(pos_ranks) <= 5:
                mol = molecules[mol_idx]
                print(f"  排名 {rank}: {mol[:50]}... | score={scores[mol_idx]:.4f}")
    
    if pos_ranks:
        print(f"\n  最高排名: {min(pos_ranks)}, 平均排名: {np.mean(pos_ranks):.1f}")
    
    print("\n" + "=" * 70)
    print("评估完成")
    print("=" * 70)

if __name__ == "__main__":
    main()
