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
    # print("\n[1/3] 正在下载 SaProt 模型 (westlake-repl/SaProt_650M_AF2)...")
    # try:
    #     model_path = snapshot_download(
    #         repo_id="westlake-repl/SaProt_650M_AF2",
    #         local_dir="./models/SaProt_650M_AF2",
    #         local_dir_use_symlinks=False,
    #         resume_download=True
    #     )
    #     print(f"SaProt 模型下载完成，保存路径: {model_path}")
    # except Exception as e:
    #     print(f"SaProt 模型下载失败: {e}")

    # # 下载 ChemBERTa 模型
    # print("\n[2/3] 正在下载 ChemBERTa 模型 (seyonec/ChemBERTa-zinc-base-v1)...")
    # try:
    #     model_path = snapshot_download(
    #         repo_id="seyonec/ChemBERTa-zinc-base-v1",
    #         local_dir="./models/ChemBERTa-zinc-base-v1",
    #         local_dir_use_symlinks=False,
    #         resume_download=True
    #     )
    #     print(f"ChemBERTa 模型下载完成，保存路径: {model_path}")
    # except Exception as e:
    #     print(f"ChemBERTa 模型下载失败: {e}")

    # 下载 BindingDB 数据集
    # print("\n[3/3] 正在下载 BindingDB 数据集 (amirhallaji/bindingdb_kd)...")
    # try:
    #     dataset = load_dataset("amirhallaji/bindingdb_kd", split="train")
    #     dataset.save_to_disk("./data/bindingdb_local")
    #     print(f"数据集下载完成，保存路径: ./data/bindingdb_local")
    #     print(f"数据集大小: {len(dataset)} 条记录")
    # except Exception as e:
    #     print(f"数据集下载失败: {e}")
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

    # print("\n资源下载完成")
    # print("目录结构:")
    # # print(" - SaProt 模型:    ./models/SaProt_650M_AF2")
    # # print(" - ChemBERTa 模型: ./models/ChemBERTa-zinc-base-v1")
    # print(" - 数据集:         ./data/bindingdb_local")

if __name__ == "__main__":
    download_models_and_data()