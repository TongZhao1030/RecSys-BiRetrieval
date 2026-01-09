"""
蛋白质-分子双向检索 Demo
Gradio Web UI

基于 IC50 训练的双塔对比学习模型
支持:
- Protein → Molecule: 给定靶点蛋白，检索候选结合分子
- Molecule → Protein: 给定小分子，识别潜在靶点蛋白
- 相似度计算: 评估蛋白质-分子配对的结合可能性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import gradio as gr
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
from collections import defaultdict

# ============================================================================
# 配置
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_MODEL = "./outputs/model_best.pth"
PATH_SAPROT = "./models/SaProt_650M_AF2"
PATH_CHEMBERTA = "./models/ChemBERTa-zinc-base-v1"
PATH_DATA = "./data/vladak_bindingdb/test"
TOP_K = 10
POSITIVE_THRESHOLD = 7.0  # 与训练一致

# 数据库大小限制
MAX_PROTEINS = 1000
MAX_MOLECULES = 3000

# ============================================================================
# 模型定义 (与训练脚本一致)
# ============================================================================
class DualTowerModel(nn.Module):
    def __init__(self, path_saprot, path_chemberta):
        super().__init__()
        self.prot_model = AutoModel.from_pretrained(path_saprot, trust_remote_code=True)
        self.mol_model = AutoModel.from_pretrained(path_chemberta)
        
        prot_hidden = self.prot_model.config.hidden_size
        mol_hidden = 768
        hidden_dim = 1024
        embedding_dim = 256
        
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
# 全局变量
# ============================================================================
model = None
mol_tokenizer = None
prot_tokenizer = None
mol_database = []
prot_database = []
mol_vectors = None
prot_vectors = None
pair_pic50 = {}  # (蛋白质, 分子) -> pIC50
prot_to_idx = {}  # 蛋白质 -> 索引
mol_to_idx = {}   # 分子 -> 索引

# 示例数据（从数据库中选择）
EXAMPLE_PROTEIN = ""
EXAMPLE_SMILES = ""

# ============================================================================
# 模型和数据加载
# ============================================================================
def load_model():
    """加载模型和数据"""
    global model, mol_tokenizer, prot_tokenizer
    global mol_database, prot_database, mol_vectors, prot_vectors
    global pair_pic50, prot_to_idx, mol_to_idx
    global EXAMPLE_PROTEIN, EXAMPLE_SMILES
    
    print("正在加载模型...")
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
    print(f"模型加载完成，设备: {DEVICE}, 温度参数: {temp:.4f}")
    
    mol_tokenizer = AutoTokenizer.from_pretrained(PATH_CHEMBERTA)
    prot_tokenizer = AutoTokenizer.from_pretrained(PATH_SAPROT, trust_remote_code=True)
    
    # ========================================================================
    # 构建数据库：以蛋白质为中心，确保正样本分子在库中
    # ========================================================================
    print("正在构建检索数据库...")
    try:
        dataset = load_from_disk(PATH_DATA)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 1. 收集所有配对关系
    print("收集配对关系...")
    prot_to_mols = defaultdict(list)  # prot -> [(mol, pic50), ...]
    mol_to_prots = defaultdict(list)  # mol -> [(prot, pic50), ...]
    all_pairs = []
    
    for idx in range(len(dataset)):
        try:
            prot = dataset[idx]['protein']
            mol = dataset[idx]['ligand']
            pic50 = dataset[idx].get('ic50', None)
            
            if pic50 is None or math.isnan(pic50) or math.isinf(pic50):
                continue
            if pic50 < -2 or pic50 > 14:
                continue
            
            prot_to_mols[prot].append((mol, pic50))
            mol_to_prots[mol].append((prot, pic50))
            all_pairs.append((prot, mol, pic50))
            
        except Exception as e:
            continue
    
    print(f"原始数据: {len(all_pairs)} 配对, {len(prot_to_mols)} 蛋白质, {len(mol_to_prots)} 分子")
    
    # 2. 选择有正样本的蛋白质，按正样本数排序
    prots_with_positives = []
    for prot, mols in prot_to_mols.items():
        positives = [(m, p) for m, p in mols if p >= POSITIVE_THRESHOLD]
        if len(positives) >= 2:  # 至少有2个正样本的蛋白质
            prots_with_positives.append((prot, len(positives), mols))
    
    # 按正样本数降序排序
    prots_with_positives.sort(key=lambda x: -x[1])
    print(f"有>=2个正样本的蛋白质数: {len(prots_with_positives)}")
    
    # 3. 选择蛋白质，并确保其正样本分子都在分子库中
    selected_prots = set()
    selected_mols = set()
    
    for prot, n_pos, mols in prots_with_positives[:MAX_PROTEINS]:
        selected_prots.add(prot)
        # 添加该蛋白质的所有正样本分子
        for mol, pic50 in mols:
            if pic50 >= POSITIVE_THRESHOLD:
                selected_mols.add(mol)
    
    print(f"选择蛋白质后，正样本分子数: {len(selected_mols)}")
    
    # 4. 补充更多分子（包括负样本，使检索更有挑战性）
    for prot, n_pos, mols in prots_with_positives[:MAX_PROTEINS]:
        for mol, pic50 in mols:
            selected_mols.add(mol)
            if len(selected_mols) >= MAX_MOLECULES:
                break
        if len(selected_mols) >= MAX_MOLECULES:
            break
    
    # 如果分子数还不够，从其他蛋白质补充
    if len(selected_mols) < MAX_MOLECULES:
        for mol in mol_to_prots.keys():
            if mol not in selected_mols:
                selected_mols.add(mol)
                if len(selected_mols) >= MAX_MOLECULES:
                    break
    
    # 5. 构建最终数据库
    prot_database = list(selected_prots)
    mol_database = list(selected_mols)
    
    prot_to_idx = {p: i for i, p in enumerate(prot_database)}
    mol_to_idx = {m: i for i, m in enumerate(mol_database)}
    
    # 6. 构建配对映射（只保留在库中的配对）
    for prot, mol, pic50 in all_pairs:
        if prot in selected_prots and mol in selected_mols:
            key = (prot, mol)
            if key not in pair_pic50 or pic50 > pair_pic50[key]:
                pair_pic50[key] = pic50
    
    # 统计
    n_positive_pairs = sum(1 for v in pair_pic50.values() if v >= POSITIVE_THRESHOLD)
    print(f"\n数据库构建完成:")
    print(f"  蛋白质库: {len(prot_database)}")
    print(f"  分子库: {len(mol_database)}")
    print(f"  已知配对数: {len(pair_pic50)} (其中正样本: {n_positive_pairs})")
    
    # 7. 选择好的示例（正样本数最多的蛋白质和它的一个正样本分子）
    if prots_with_positives:
        best_prot, n_pos, mols = prots_with_positives[0]
        EXAMPLE_PROTEIN = best_prot
        
        # 找该蛋白质 pIC50 最高的正样本分子
        best_mol = None
        best_pic50 = 0
        for mol, pic50 in mols:
            if pic50 >= POSITIVE_THRESHOLD and pic50 > best_pic50:
                best_mol = mol
                best_pic50 = pic50
        
        if best_mol:
            EXAMPLE_SMILES = best_mol
            print(f"\n示例蛋白质: {EXAMPLE_PROTEIN[:50]}... (正样本数: {n_pos})")
            print(f"示例分子: {EXAMPLE_SMILES} (pIC50: {best_pic50:.2f})")
    
    # 8. 预计算向量
    print("\n正在预计算向量...")
    mol_vectors = encode_molecules_batch(mol_database)
    prot_vectors = encode_proteins_batch(prot_database)
    print("向量计算完成")

def encode_protein(seq):
    """编码单个蛋白质"""
    formatted = " ".join([aa + "#" for aa in seq])
    inputs = prot_tokenizer(formatted, return_tensors="pt", padding=True, 
                            truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        vec = model.encode_protein(inputs['input_ids'], inputs['attention_mask'])
    return vec

def encode_molecule(smiles):
    """编码单个分子"""
    inputs = mol_tokenizer(smiles, return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        vec = model.encode_molecule(inputs['input_ids'], inputs['attention_mask'])
    return vec

def encode_proteins_batch(prot_list, batch_size=8):
    """批量编码蛋白质"""
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

def encode_molecules_batch(mol_list, batch_size=32):
    """批量编码分子"""
    all_vecs = []
    for i in range(0, len(mol_list), batch_size):
        batch = mol_list[i:i+batch_size]
        inputs = mol_tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            vecs = model.encode_molecule(inputs['input_ids'], inputs['attention_mask'])
        all_vecs.append(vecs)
    return torch.cat(all_vecs, dim=0)

# ============================================================================
# 检索功能
# ============================================================================
def search_molecules(protein_seq, top_k=TOP_K):
    """给定蛋白质，检索分子"""
    if not protein_seq.strip():
        return "请输入蛋白质序列"
    
    try:
        prot = protein_seq.strip().upper()
        query_vec = encode_protein(prot)
        scores = torch.matmul(query_vec, mol_vectors.T).squeeze().cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        hit_positive = 0
        
        for rank, idx in enumerate(top_indices, 1):
            mol = mol_database[idx]
            sim = scores[idx]
            
            # 查找该蛋白质与检索到的分子之间的实际 pIC50
            info_str = ""
            key = (prot, mol)
            if key in pair_pic50:
                pic50 = pair_pic50[key]
                is_positive = pic50 >= POSITIVE_THRESHOLD
                if is_positive:
                    hit_positive += 1
                    info_str = f" | pIC50: **{pic50:.2f}** ✓"
                else:
                    info_str = f" | pIC50: {pic50:.2f}"
            
            results.append(f"**{rank}.** `{mol}`  \n相似度: {sim:.4f}{info_str}\n")
        
        # 添加统计信息
        header = f"**Top-{top_k} 检索结果** (命中正样本: {hit_positive})\n\n"
        return header + "\n".join(results)
    except Exception as e:
        return f"错误: {str(e)}"

def search_proteins(smiles, top_k=TOP_K):
    """给定分子，检索蛋白质"""
    if not smiles.strip():
        return "请输入 SMILES"
    
    try:
        mol = smiles.strip()
        query_vec = encode_molecule(mol)
        scores = torch.matmul(query_vec, prot_vectors.T).squeeze().cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        hit_positive = 0
        
        for rank, idx in enumerate(top_indices, 1):
            prot_seq = prot_database[idx]
            sim = scores[idx]
            
            display_seq = prot_seq[:60] + "..." if len(prot_seq) > 60 else prot_seq
            
            # 查找检索到的蛋白质与该分子之间的实际 pIC50
            info_str = ""
            key = (prot_seq, mol)
            if key in pair_pic50:
                pic50 = pair_pic50[key]
                is_positive = pic50 >= POSITIVE_THRESHOLD
                if is_positive:
                    hit_positive += 1
                    info_str = f" | pIC50: **{pic50:.2f}** ✓"
                else:
                    info_str = f" | pIC50: {pic50:.2f}"
            
            results.append(f"**{rank}.** `{display_seq}`  \n相似度: {sim:.4f}{info_str}\n")
        
        header = f"**Top-{top_k} 检索结果** (命中正样本: {hit_positive})\n\n"
        return header + "\n".join(results)
    except Exception as e:
        return f"错误: {str(e)}"

def compute_similarity(protein_seq, smiles):
    """计算单对相似度"""
    if not protein_seq.strip() or not smiles.strip():
        return "请输入蛋白质序列和 SMILES"
    
    try:
        prot = protein_seq.strip().upper()
        mol = smiles.strip()
        prot_vec = encode_protein(prot)
        mol_vec = encode_molecule(mol)
        similarity = torch.matmul(prot_vec, mol_vec.T).item()
        
        # 解释相似度
        if similarity > 0.4:
            interpretation = "🟢 高相似度 - 可能存在较强的结合亲和力"
        elif similarity > 0.3:
            interpretation = "🟡 中等相似度 - 可能存在一定的结合能力"
        elif similarity > 0.2:
            interpretation = "🟠 低相似度 - 结合可能性较低"
        else:
            interpretation = "🔴 极低相似度 - 不太可能有结合"
        
        # 查找实际 pIC50（如果有记录）
        key = (prot, mol)
        ground_truth = ""
        if key in pair_pic50:
            pic50 = pair_pic50[key]
            is_positive = "✓ 正样本" if pic50 >= POSITIVE_THRESHOLD else "✗ 非正样本"
            ground_truth = f"\n\n**已知实际 pIC50: {pic50:.2f}** ({is_positive})"
        
        return f"## 相似度得分: {similarity:.4f}\n\n{interpretation}{ground_truth}"
    except Exception as e:
        return f"错误: {str(e)}"

def get_random_example():
    """获取随机示例（从数据库中选择有正样本配对的）"""
    import random
    
    # 找有正样本的蛋白质
    prots_with_pos = defaultdict(list)
    for (prot, mol), pic50 in pair_pic50.items():
        if pic50 >= POSITIVE_THRESHOLD:
            prots_with_pos[prot].append((mol, pic50))
    
    if not prots_with_pos:
        return EXAMPLE_PROTEIN, EXAMPLE_SMILES
    
    # 随机选一个蛋白质
    prot = random.choice(list(prots_with_pos.keys()))
    # 选它 pIC50 最高的分子
    mols = prots_with_pos[prot]
    mol, pic50 = max(mols, key=lambda x: x[1])
    
    return prot, mol

# ============================================================================
# Gradio 界面
# ============================================================================
def create_demo():
    with gr.Blocks(title="蛋白质-分子双向检索", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧬 蛋白质-分子双向检索系统
        
        基于双塔对比学习的检索模型（IC50 版本），支持：
        - **Protein → Molecule**：给定靶点蛋白，检索候选结合分子
        - **Molecule → Protein**：给定小分子，识别潜在靶点蛋白
        - **相似度计算**：评估蛋白质-分子配对的结合可能性
        
        > 💡 正样本定义: pIC50 ≥ 7.0 (IC50 < 100nM)
        """)
        
        with gr.Tab("🔬 蛋白质 → 分子"):
            gr.Markdown("输入蛋白质序列，检索可能结合的小分子（按相似度排序）")
            with gr.Row():
                with gr.Column():
                    prot_input = gr.Textbox(
                        label="蛋白质序列",
                        placeholder="输入氨基酸序列",
                        lines=4,
                        value=EXAMPLE_PROTEIN
                    )
                    with gr.Row():
                        search_mol_btn = gr.Button("🔍 检索分子", variant="primary")
                        random_prot_btn = gr.Button("🎲 随机示例")
                with gr.Column():
                    mol_output = gr.Markdown(label="检索结果")
            
            search_mol_btn.click(search_molecules, inputs=prot_input, outputs=mol_output)
            random_prot_btn.click(lambda: get_random_example()[0], outputs=prot_input)
        
        with gr.Tab("💊 分子 → 蛋白质"):
            gr.Markdown("输入分子 SMILES，检索可能的靶点蛋白（按相似度排序）")
            with gr.Row():
                with gr.Column():
                    mol_input = gr.Textbox(
                        label="分子 SMILES",
                        placeholder="输入 SMILES",
                        lines=2,
                        value=EXAMPLE_SMILES
                    )
                    with gr.Row():
                        search_prot_btn = gr.Button("🔍 检索蛋白质", variant="primary")
                        random_mol_btn = gr.Button("🎲 随机示例")
                with gr.Column():
                    prot_output = gr.Markdown(label="检索结果")
            
            search_prot_btn.click(search_proteins, inputs=mol_input, outputs=prot_output)
            random_mol_btn.click(lambda: get_random_example()[1], outputs=mol_input)
        
        with gr.Tab("⚡ 相似度计算"):
            gr.Markdown("计算单对蛋白质-分子的相似度得分")
            with gr.Row():
                with gr.Column():
                    pair_prot = gr.Textbox(label="蛋白质序列", lines=3, value=EXAMPLE_PROTEIN)
                    pair_mol = gr.Textbox(label="分子 SMILES", lines=1, value=EXAMPLE_SMILES)
                    with gr.Row():
                        calc_btn = gr.Button("⚡ 计算相似度", variant="primary")
                        random_pair_btn = gr.Button("🎲 随机配对")
                with gr.Column():
                    sim_output = gr.Markdown()
            
            calc_btn.click(compute_similarity, inputs=[pair_prot, pair_mol], outputs=sim_output)
            random_pair_btn.click(get_random_example, outputs=[pair_prot, pair_mol])
        
        with gr.Tab("📊 数据库统计"):
            gr.Markdown(f"""
            ### 检索数据库信息
            
            | 项目 | 数量 |
            |------|------|
            | 蛋白质库 | {len(prot_database)} |
            | 分子库 | {len(mol_database)} |
            | 已知配对 | {len(pair_pic50)} |
            | 正样本配对 | {sum(1 for v in pair_pic50.values() if v >= POSITIVE_THRESHOLD)} |
            
            **正样本阈值**: pIC50 ≥ {POSITIVE_THRESHOLD}
            """)
        
        gr.Markdown("""
        ---
        **模型信息**: SaProt (650M) + ChemBERTa | 训练数据: BindingDB
        """)
    
    return demo

# ============================================================================
# 主函数
# ============================================================================
if __name__ == "__main__":
    load_model()
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
