# 修改数据集名称
repo_id = "Caltech-UCSD-Birds-200-2011"  # 正确名称

# 完整下载代码
from huggingface_hub import snapshot_download
from datasets import load_dataset
import os

# 设置镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 登录HuggingFace（可选）
from huggingface_hub import login

login(token="your_hf_token")  # 前往 https://huggingface.co/settings/tokens 获取

# 分步下载
try:
    # 第一步：下载元数据
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="*.parquet",
        local_dir="./bird_data",
        resume_download=True
    )

    # 第二步：下载图片
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="image/*",
        local_dir="./bird_data",
        resume_download=True,
        max_workers=4  # 增加下载线程
    )

    # 验证加载
    dataset = load_dataset(repo_id, cache_dir="./bird_data")
    print(f"成功加载数据集，样本数：{len(dataset['train'])}")
except Exception as e:
    print(f"下载失败：{str(e)}")