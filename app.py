"""
蛋白质-分子双向检索 Demo
Gradio Web UI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import gradio as gr
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_MODEL = "dual_tower_final.pth"
PATH_SAPROT = "/share/home/zhangchiLab/duyinuo/models/westlake-repl_SaProt_650M_AF2"
PATH_CHEMBERTA = "/share/home/zhangchiLab/duyinuo/models/seyonec_ChemBERTa-zinc-base-v1"
PATH_DATA = "/share/home/zhangchiLab/duyinuo/data/vladak_bindingdb"
TOP_K = 10

def resolve_split_dataset_path(data_path, split):
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

# 模型定义
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

# 全局变量
model = None
mol_tokenizer = None
prot_tokenizer = None
mol_database = []
prot_database = []
mol_vectors = None
prot_vectors = None

def load_model():
    """加载模型和数据"""
    global model, mol_tokenizer, prot_tokenizer
    global mol_database, prot_database, mol_vectors, prot_vectors
    
    print("正在加载模型...")
    model = DualTowerModel().to(DEVICE)
    state_dict = torch.load(PATH_MODEL, map_location=DEVICE, weights_only=False)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"模型加载完成，设备: {DEVICE}")
    
    mol_tokenizer = AutoTokenizer.from_pretrained(PATH_CHEMBERTA)
    prot_tokenizer = AutoTokenizer.from_pretrained(PATH_SAPROT, trust_remote_code=True)
    
    # 加载数据库
    print("正在构建检索数据库...")
    dataset = load_from_disk(resolve_split_dataset_path(PATH_DATA, "train"))
    protein_col = infer_text_column(dataset.column_names, kind="protein")
    molecule_col = infer_text_column(dataset.column_names, kind="molecule")
    if protein_col is None or molecule_col is None:
        raise ValueError(f"无法推断列名；现有列: {dataset.column_names}")
    
    seen_prots = set()
    seen_mols = set()
    
    for idx in range(min(len(dataset), 5000)):  # 限制数据库大小
        prot = dataset[idx][protein_col]
        mol = dataset[idx][molecule_col]
        
        if prot not in seen_prots:
            seen_prots.add(prot)
            prot_database.append(prot)
        
        if mol not in seen_mols:
            seen_mols.add(mol)
            mol_database.append(mol)
    
    print(f"数据库: {len(prot_database)} 蛋白质, {len(mol_database)} 分子")
    
    # 预计算向量
    print("正在预计算向量...")
    mol_vectors = encode_molecules_batch(mol_database)
    prot_vectors = encode_proteins_batch(prot_database)
    print("向量计算完成")

def encode_protein(seq):
    """编码单个蛋白质"""
    formatted = " ".join([aa + "#" for aa in seq])
    inputs = prot_tokenizer(formatted, return_tensors="pt", padding=True, 
                            truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.prot_model(input_ids=inputs['input_ids'], 
                               attention_mask=inputs['attention_mask'])
        mask = inputs['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        emb = torch.sum(out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        emb = model.prot_layernorm(emb)
        vec = model.prot_proj(emb)
        vec = F.normalize(vec, p=2, dim=1)
    return vec

def encode_molecule(smiles):
    """编码单个分子"""
    inputs = mol_tokenizer(smiles, return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = model.mol_model(input_ids=inputs['input_ids'],
                              attention_mask=inputs['attention_mask'])
        mask = inputs['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        emb = torch.sum(out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
        emb = model.mol_layernorm(emb)
        vec = model.mol_proj(emb)
        vec = F.normalize(vec, p=2, dim=1)
    return vec

def encode_proteins_batch(prot_list, batch_size=16):
    """批量编码蛋白质"""
    all_vecs = []
    for i in range(0, len(prot_list), batch_size):
        batch = prot_list[i:i+batch_size]
        formatted = [" ".join([aa + "#" for aa in seq]) for seq in batch]
        inputs = prot_tokenizer(formatted, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = model.prot_model(input_ids=inputs['input_ids'],
                                   attention_mask=inputs['attention_mask'])
            mask = inputs['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            emb = torch.sum(out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
            emb = model.prot_layernorm(emb)
            vec = model.prot_proj(emb)
            vec = F.normalize(vec, p=2, dim=1)
        all_vecs.append(vec)
    return torch.cat(all_vecs, dim=0)

def encode_molecules_batch(mol_list, batch_size=32):
    """批量编码分子"""
    all_vecs = []
    for i in range(0, len(mol_list), batch_size):
        batch = mol_list[i:i+batch_size]
        inputs = mol_tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model.mol_model(input_ids=inputs['input_ids'],
                                  attention_mask=inputs['attention_mask'])
            mask = inputs['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            emb = torch.sum(out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
            emb = model.mol_layernorm(emb)
            vec = model.mol_proj(emb)
            vec = F.normalize(vec, p=2, dim=1)
        all_vecs.append(vec)
    return torch.cat(all_vecs, dim=0)

def search_molecules(protein_seq, top_k=TOP_K):
    """给定蛋白质，检索分子"""
    if not protein_seq.strip():
        return "请输入蛋白质序列"
    
    try:
        query_vec = encode_protein(protein_seq.strip().upper())
        scores = torch.matmul(query_vec, mol_vectors.T).squeeze().cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append(f"**{rank}.** `{mol_database[idx]}`  \n相似度: {scores[idx]:.4f}\n")
        
        return "\n".join(results)
    except Exception as e:
        return f"错误: {str(e)}"

def search_proteins(smiles, top_k=TOP_K):
    """给定分子，检索蛋白质"""
    if not smiles.strip():
        return "请输入 SMILES"
    
    try:
        query_vec = encode_molecule(smiles.strip())
        scores = torch.matmul(query_vec, prot_vectors.T).squeeze().cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            prot_seq = prot_database[idx]
            display_seq = prot_seq[:60] + "..." if len(prot_seq) > 60 else prot_seq
            results.append(f"**{rank}.** `{display_seq}`  \n相似度: {scores[idx]:.4f}\n")
        
        return "\n".join(results)
    except Exception as e:
        return f"错误: {str(e)}"

def compute_similarity(protein_seq, smiles):
    """计算单对相似度"""
    if not protein_seq.strip() or not smiles.strip():
        return "请输入蛋白质序列和 SMILES"
    
    try:
        prot_vec = encode_protein(protein_seq.strip().upper())
        mol_vec = encode_molecule(smiles.strip())
        similarity = torch.matmul(prot_vec, mol_vec.T).item()
        return f"## 相似度得分: {similarity:.4f}"
    except Exception as e:
        return f"错误: {str(e)}"

# 示例数据
EXAMPLE_PROTEIN = "MIKSALLVLEDGTQFHGRAIGATGSAVGEVVFNTSMTGYQEILTDPSYSRQIVTLTYPHIGNVGTNDADEESSQVHAQGLVIRDLPLIASNFRNTEDLSSYLKRHNIVAIADIDTRKLTRLLREKGAQNGCIIAGDNPDAALALEKARAFPGLNGMDLAKEVTTAEAYSWTQGSWTLTGGLPEAKKEDELPFHVVAYDFGAKRNILRMLVDRGCRLTIVPAQTSAEDVLKMNPDGIFLSNGPGDPAPCDYAITAIQKFLETDIPVFGICLGHQLLALASGAKTVKMKFGHHGGNHPVKDVEKNVVMITAQNHGFAVDEATLPANLRVTHKSLFDGTLQGIHRTDKPAFSFQGHPEASPGPHDAAPLFDHFIELIEQYRKTAK"
EXAMPLE_SMILES = "O[C@@H]1[C@@H](COP(O)(O)=O)O[C@H]([C@@H]1O)n1cnc2c3nccn3cnc12"

# 构建界面
with gr.Blocks(title="蛋白质-分子双向检索", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧬 蛋白质-分子双向检索系统
    
    基于双塔对比学习的检索模型，支持：
    - **Protein → Molecule**：给定靶点蛋白，检索候选结合分子
    - **Molecule → Protein**：给定小分子，识别潜在靶点蛋白
    """)
    
    with gr.Tab("🔬 蛋白质 → 分子"):
        gr.Markdown("输入蛋白质序列，检索可能结合的小分子药物")
        with gr.Row():
            with gr.Column():
                prot_input = gr.Textbox(
                    label="蛋白质序列",
                    placeholder="输入氨基酸序列（如 MKTVRQ...）",
                    lines=4,
                    value=EXAMPLE_PROTEIN
                )
                search_mol_btn = gr.Button("🔍 检索分子", variant="primary")
            with gr.Column():
                mol_output = gr.Markdown(label="检索结果")
        search_mol_btn.click(search_molecules, inputs=prot_input, outputs=mol_output)
    
    with gr.Tab("💊 分子 → 蛋白质"):
        gr.Markdown("输入分子 SMILES，检索可能的靶点蛋白")
        with gr.Row():
            with gr.Column():
                mol_input = gr.Textbox(
                    label="分子 SMILES",
                    placeholder="输入 SMILES（如 CC(=O)Oc1ccccc1C(=O)O）",
                    lines=2,
                    value=EXAMPLE_SMILES
                )
                search_prot_btn = gr.Button("🔍 检索蛋白质", variant="primary")
            with gr.Column():
                prot_output = gr.Markdown(label="检索结果")
        search_prot_btn.click(search_proteins, inputs=mol_input, outputs=prot_output)
    
    with gr.Tab("⚡ 相似度计算"):
        gr.Markdown("计算单对蛋白质-分子的相似度得分")
        with gr.Row():
            with gr.Column():
                pair_prot = gr.Textbox(label="蛋白质序列", lines=3, value=EXAMPLE_PROTEIN)
                pair_mol = gr.Textbox(label="分子 SMILES", lines=1, value=EXAMPLE_SMILES)
                calc_btn = gr.Button("⚡ 计算相似度", variant="primary")
            with gr.Column():
                sim_output = gr.Markdown()
        calc_btn.click(compute_similarity, inputs=[pair_prot, pair_mol], outputs=sim_output)
    
    gr.Markdown("""
    ---
    **模型信息**：SaProt (650M) + ChemBERTa | 对比学习 | BindingDB 数据集
    """)

if __name__ == "__main__":
    load_model()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
