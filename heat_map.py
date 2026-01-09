"""
蛋白质-分子相似度热力图

正确的评估逻辑：
1. 选择 N 个蛋白质和 N 个分子（可以不是一一对应的配对）
2. 构建正样本矩阵（标记哪些 (i,j) 是已知的高亲和力配对）
3. 计算相似度矩阵
4. 评估：正样本位置的分数是否显著高于负样本位置
5. 热力图可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 配置
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_MODEL = "./outputs/model_best.pth"
PATH_SAPROT = "./models/SaProt_650M_AF2"
PATH_CHEMBERTA = "./models/ChemBERTa-zinc-base-v1"
PATH_DATA_TEST = "./data/vladak_bindingdb/test"

# 热力图配置
N_SAMPLES = 50  # 热力图大小 (N x N)
POSITIVE_THRESHOLD = 7.0  # 与训练配置一致

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
# 数据构建
# ============================================================================
def build_heatmap_data(dataset, n_samples=N_SAMPLES):
    """
    构建热力图数据
    
    策略：
    1. 收集所有正样本配对
    2. 选择 N 个蛋白质和 N 个分子
    3. 构建正样本矩阵（标记哪些 (i,j) 是正样本）
    
    Returns:
        proteins: N 个蛋白质
        molecules: N 个分子
        positive_matrix: N x N 的 0/1 矩阵，表示正样本位置
    """
    print("构建热力图数据...")
    
    # 收集所有正样本配对
    positive_pairs = set()  # (prot, mol)
    prot_to_mols = defaultdict(set)
    mol_to_prots = defaultdict(set)
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        pic50 = item['ic50']
        
        if pic50 is None or math.isnan(pic50) or math.isinf(pic50):
            continue
        if pic50 < POSITIVE_THRESHOLD:
            continue
        
        prot = item['protein']
        mol = item['ligand']
        
        positive_pairs.add((prot, mol))
        prot_to_mols[prot].add(mol)
        mol_to_prots[mol].add(prot)
    
    print(f"正样本配对数: {len(positive_pairs)}")
    print(f"有正样本的蛋白质数: {len(prot_to_mols)}")
    print(f"有正样本的分子数: {len(mol_to_prots)}")
    
    # 选择蛋白质：优先选择有多个正样本分子的
    sorted_prots = sorted(prot_to_mols.keys(), key=lambda p: -len(prot_to_mols[p]))
    proteins = sorted_prots[:n_samples]
    
    # 选择分子：优先选择与已选蛋白质有配对关系的
    selected_mols = set()
    for prot in proteins:
        selected_mols.update(prot_to_mols[prot])
    
    # 按配对数排序
    sorted_mols = sorted(selected_mols, key=lambda m: -len(mol_to_prots[m]))
    molecules = sorted_mols[:n_samples]
    
    # 如果不够，补充其他分子
    if len(molecules) < n_samples:
        other_mols = [m for m in mol_to_prots.keys() if m not in selected_mols]
        molecules.extend(other_mols[:n_samples - len(molecules)])
    
    print(f"\n选择的蛋白质数: {len(proteins)}")
    print(f"选择的分子数: {len(molecules)}")
    
    # 构建正样本矩阵
    prot_to_idx = {p: i for i, p in enumerate(proteins)}
    mol_to_idx = {m: i for i, m in enumerate(molecules)}
    
    positive_matrix = np.zeros((len(proteins), len(molecules)), dtype=np.int32)
    n_positives = 0
    
    for prot, mol in positive_pairs:
        if prot in prot_to_idx and mol in mol_to_idx:
            i = prot_to_idx[prot]
            j = mol_to_idx[mol]
            positive_matrix[i, j] = 1
            n_positives += 1
    
    print(f"热力图中的正样本数: {n_positives} / {len(proteins) * len(molecules)} ({n_positives / (len(proteins) * len(molecules)) * 100:.2f}%)")
    
    return proteins, molecules, positive_matrix

# ============================================================================
# 主函数
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
    print(f"模型加载成功，温度: {temp:.4f}")
    
    # 加载 tokenizer
    mol_tokenizer = AutoTokenizer.from_pretrained(PATH_CHEMBERTA)
    prot_tokenizer = AutoTokenizer.from_pretrained(PATH_SAPROT, trust_remote_code=True)
    
    # 加载数据并构建热力图数据
    print("\n加载数据...")
    test_dataset = load_from_disk(PATH_DATA_TEST)
    proteins, molecules, positive_matrix = build_heatmap_data(test_dataset, N_SAMPLES)
    
    n_prots = len(proteins)
    n_mols = len(molecules)
    
    # 编码
    print("\n编码蛋白质和分子...")
    
    def encode_proteins_batch(prot_list, batch_size=8):
        all_vecs = []
        for i in range(0, len(prot_list), batch_size):
            batch = prot_list[i:i+batch_size]
            formatted = [" ".join([aa + "#" for aa in seq]) for seq in batch]
            inputs = prot_tokenizer(formatted, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                vecs = model.encode_protein(inputs['input_ids'], inputs['attention_mask'])
            all_vecs.append(vecs)
        return torch.cat(all_vecs, dim=0)
    
    def encode_molecules_batch(mol_list, batch_size=16):
        all_vecs = []
        for i in range(0, len(mol_list), batch_size):
            batch = mol_list[i:i+batch_size]
            inputs = mol_tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                vecs = model.encode_molecule(inputs['input_ids'], inputs['attention_mask'])
            all_vecs.append(vecs)
        return torch.cat(all_vecs, dim=0)
    
    prot_vecs = encode_proteins_batch(proteins)
    mol_vecs = encode_molecules_batch(molecules)
    
    # 计算相似度矩阵
    sim_matrix = torch.matmul(prot_vecs, mol_vecs.T).cpu().numpy()
    
    # ========================================================================
    # 评估
    # ========================================================================
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    # 正样本 vs 负样本分数
    positive_mask = positive_matrix > 0
    negative_mask = ~positive_mask
    
    pos_scores = sim_matrix[positive_mask]
    neg_scores = sim_matrix[negative_mask]
    
    if len(pos_scores) > 0:
        print(f"\n正样本分数: {pos_scores.mean():.4f} ± {pos_scores.std():.4f} (n={len(pos_scores)})")
        print(f"负样本分数: {neg_scores.mean():.4f} ± {neg_scores.std():.4f} (n={len(neg_scores)})")
        print(f"Margin: {pos_scores.mean() - neg_scores.mean():.4f}")
    else:
        print("警告: 热力图中没有正样本！")
    
    # P2M Recall@K: 每个蛋白质的 top-k 预测中是否命中正样本
    print(f"\nP→M 检索评估 (在 {n_mols} 个分子中检索):")
    for k in [1, 5, 10]:
        recall = 0
        n_queries = 0
        for i in range(n_prots):
            if positive_matrix[i].sum() > 0:  # 该蛋白质有正样本
                n_queries += 1
                top_k_indices = np.argsort(sim_matrix[i])[::-1][:k]
                if any(positive_matrix[i, j] > 0 for j in top_k_indices):
                    recall += 1
        if n_queries > 0:
            print(f"  Recall@{k}: {recall/n_queries*100:.1f}% ({recall}/{n_queries})")
    
    # M2P Recall@K
    print(f"\nM→P 检索评估 (在 {n_prots} 个蛋白质中检索):")
    for k in [1, 5, 10]:
        recall = 0
        n_queries = 0
        for j in range(n_mols):
            if positive_matrix[:, j].sum() > 0:  # 该分子有正样本
                n_queries += 1
                top_k_indices = np.argsort(sim_matrix[:, j])[::-1][:k]
                if any(positive_matrix[i, j] > 0 for i in top_k_indices):
                    recall += 1
        if n_queries > 0:
            print(f"  Recall@{k}: {recall/n_queries*100:.1f}% ({recall}/{n_queries})")
    
    # ========================================================================
    # 绘制热力图
    # ========================================================================
    print("\n绘制热力图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：相似度热力图
    ax1 = axes[0]
    im1 = ax1.imshow(sim_matrix, cmap="viridis", aspect='auto')
    ax1.set_title(f"Similarity Matrix (N={n_prots}×{n_mols})\n"
                  f"Positive: {pos_scores.mean():.3f} | Negative: {neg_scores.mean():.3f}",
                  fontsize=11)
    ax1.set_xlabel("Molecule Index", fontsize=10)
    ax1.set_ylabel("Protein Index", fontsize=10)
    plt.colorbar(im1, ax=ax1, label='Similarity Score')
    
    # 标记正样本位置（用红色小点）
    pos_i, pos_j = np.where(positive_matrix > 0)
    ax1.scatter(pos_j, pos_i, c='red', s=5, alpha=0.7, marker='s', label='Positive pairs')
    ax1.legend(loc='upper right', fontsize=8)
    
    # 右图：正样本矩阵（Ground Truth）
    ax2 = axes[1]
    im2 = ax2.imshow(positive_matrix, cmap="Reds", aspect='auto')
    ax2.set_title(f"Ground Truth Positive Pairs\n"
                  f"{positive_matrix.sum()} positives ({positive_matrix.sum()/(n_prots*n_mols)*100:.1f}%)",
                  fontsize=11)
    ax2.set_xlabel("Molecule Index", fontsize=10)
    ax2.set_ylabel("Protein Index", fontsize=10)
    plt.colorbar(im2, ax=ax2, label='Is Positive')
    
    plt.tight_layout()
    plt.savefig("diagnosis_heatmap.png", dpi=150)
    print(f"\n热力图已保存: diagnosis_heatmap.png")
    
    # ========================================================================
    # 绘制分数分布直方图
    # ========================================================================
    if len(pos_scores) > 0:
        fig2, ax = plt.subplots(figsize=(8, 5))
        
        ax.hist(neg_scores, bins=50, alpha=0.6, label=f'Negative (n={len(neg_scores)})', color='blue')
        ax.hist(pos_scores, bins=50, alpha=0.6, label=f'Positive (n={len(pos_scores)})', color='red')
        ax.axvline(x=pos_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Pos mean: {pos_scores.mean():.3f}')
        ax.axvline(x=neg_scores.mean(), color='blue', linestyle='--', linewidth=2, label=f'Neg mean: {neg_scores.mean():.3f}')
        
        ax.set_xlabel('Similarity Score', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Score Distribution: Positive vs Negative Pairs', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("score_distribution.png", dpi=150)
        print(f"分数分布图已保存: score_distribution.png")
    
    # ========================================================================
    # 诊断信息
    # ========================================================================
    print("\n" + "=" * 60)
    print("诊断信息")
    print("=" * 60)
    
    # 找出分数最高的正样本和负样本
    if len(pos_scores) > 0:
        best_pos_idx = np.unravel_index(np.argmax(sim_matrix * positive_matrix), sim_matrix.shape)
        worst_pos_idx = np.unravel_index(np.argmin(sim_matrix + (1 - positive_matrix) * 10), sim_matrix.shape)
        
        print(f"\n最高正样本分数: {sim_matrix[best_pos_idx]:.4f} at ({best_pos_idx[0]}, {best_pos_idx[1]})")
        print(f"最低正样本分数: {sim_matrix[worst_pos_idx]:.4f} at ({worst_pos_idx[0]}, {worst_pos_idx[1]})")
    
    # 找出分数最高的负样本（错误的高分）
    neg_sim = sim_matrix * (1 - positive_matrix)
    best_neg_idx = np.unravel_index(np.argmax(neg_sim), sim_matrix.shape)
    print(f"最高负样本分数: {sim_matrix[best_neg_idx]:.4f} at ({best_neg_idx[0]}, {best_neg_idx[1]})")
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
