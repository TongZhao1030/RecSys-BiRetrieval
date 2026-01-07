import os

# 配置国内镜像加速
# 必须在导入 transformers/datasets 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
from datasets import load_dataset
import shutil

def download_models_and_data():
    """
    下载项目所需的预训练模型和数据集
    """
    print("开始下载项目所需资源（已启用国内镜像加速）...")
    
    # 创建存放目录
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    # 下载 SaProt 模型
    saprot_path = "./models/SaProt_650M_AF2"
    if os.path.exists(saprot_path) and os.listdir(saprot_path):
        print("\n[1/4] SaProt 模型已存在，跳过下载")
    else:
        print("\n[1/4] 正在下载 SaProt 模型 (westlake-repl/SaProt_650M_AF2)...")
        try:
            model_path = snapshot_download(
                repo_id="westlake-repl/SaProt_650M_AF2",
                local_dir=saprot_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"SaProt 模型下载完成，保存路径: {model_path}")
        except Exception as e:
            print(f"SaProt 模型下载失败: {e}")

    # 下载 ChemBERTa 模型
    chemberta_path = "./models/ChemBERTa-zinc-base-v1"
    if os.path.exists(chemberta_path) and os.listdir(chemberta_path):
        print("\n[2/4] ChemBERTa 模型已存在，跳过下载")
    else:
        print("\n[2/4] 正在下载 ChemBERTa 模型 (seyonec/ChemBERTa-zinc-base-v1)...")
        try:
            model_path = snapshot_download(
                repo_id="seyonec/ChemBERTa-zinc-base-v1",
                local_dir=chemberta_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"ChemBERTa 模型下载完成，保存路径: {model_path}")
        except Exception as e:
            print(f"ChemBERTa 模型下载失败: {e}")

    # 下载 BindingDB 数据集 (amirhallaji/bindingdb_kd)
    bindingdb_kd_path = "./data/bindingdb_local"
    if os.path.exists(bindingdb_kd_path) and os.listdir(bindingdb_kd_path):
        print("\n[3/4] BindingDB Kd 数据集已存在，跳过下载")
    else:
        print("\n[3/4] 正在下载 BindingDB 数据集 (amirhallaji/bindingdb_kd)...")
        try:
            dataset = load_dataset("amirhallaji/bindingdb_kd", split="train")
            dataset.save_to_disk(bindingdb_kd_path)
            print(f"数据集下载完成，保存路径: {bindingdb_kd_path}")
            print(f"数据集大小: {len(dataset)} 条记录")
        except Exception as e:
            print(f"数据集下载失败: {e}")

    # 下载 vladak/bindingdb 数据集 (包含 IC50 数据)
    vladak_train_path = "./data/vladak_bindingdb/train"
    vladak_test_path = "./data/vladak_bindingdb/test"
    
    if os.path.exists(vladak_train_path) and os.listdir(vladak_train_path) and \
       os.path.exists(vladak_test_path) and os.listdir(vladak_test_path):
        print("\n[4/4] BindingDB IC50 数据集已存在，跳过下载")
    else:
        print("\n[4/4] 正在下载 BindingDB IC50 数据集 (vladak/bindingdb)...")
        try:
            # 下载训练集
            if not (os.path.exists(vladak_train_path) and os.listdir(vladak_train_path)):
                train_dataset = load_dataset("vladak/bindingdb", split="train")
                train_dataset.save_to_disk(vladak_train_path)
                print(f"训练集下载完成，大小: {len(train_dataset)} 条记录")
            else:
                print("训练集已存在，跳过")
            
            # 下载测试集
            if not (os.path.exists(vladak_test_path) and os.listdir(vladak_test_path)):
                test_dataset = load_dataset("vladak/bindingdb", split="test")
                test_dataset.save_to_disk(vladak_test_path)
                print(f"测试集下载完成，大小: {len(test_dataset)} 条记录")
            else:
                print("测试集已存在，跳过")
            
            print(f"vladak/bindingdb 数据集下载完成，保存路径: ./data/vladak_bindingdb")
        except Exception as e:
            print(f"vladak/bindingdb 数据集下载失败: {e}")

    print("\n资源下载完成")
    print("目录结构:")
    print(" - SaProt 模型:         ./models/SaProt_650M_AF2")
    print(" - ChemBERTa 模型:      ./models/ChemBERTa-zinc-base-v1")
    print(" - BindingDB Kd 数据集: ./data/bindingdb_local")
    print(" - BindingDB IC50 数据集: ./data/vladak_bindingdb")

if __name__ == "__main__":
    download_models_and_data()