"""
阶段二：构建 FAISS 索引
用法：
  python build_faiss_index.py

功能：
  1. 扫描 data/library_embeddings/ 下所有 .pt 文件
  2. 提取每首歌的均值向量，构建 FAISS IVF 索引（粗检索）
  3. 同时生成 song_id → 音频路径 的映射文件

输出：
  data/faiss_index/coarse.index   — FAISS 索引文件
  data/faiss_index/song_meta.pt   — {song_id: int → path: str, emb_file: str} 映射
"""

import os
import argparse
import time
import torch
import numpy as np
import faiss
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--emb_dir", type=str, default="/home/zjw524/projects/clews_data/library_embeddings")
parser.add_argument("--output_dir", type=str, default="/home/zjw524/projects/clews_data/faiss_index")
parser.add_argument("--nlist", type=int, default=4096, help="IVF 聚类数量（建议 sqrt(N) 附近）")
parser.add_argument("--use_gpu", action="store_true", default=False, help="使用 GPU 训练索引（当 GPU 被其他进程占用时不建议使用）")
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

###############################################################################
# 1. 扫描所有 .pt 文件，加载均值向量
###############################################################################

print("Scanning embedding files...")
pt_files = sorted([f for f in os.listdir(args.emb_dir) if f.endswith(".pt")])
print(f"Found {len(pt_files)} embedding files.")

# 分批加载，避免内存爆炸
all_means = []
song_meta = {}  # song_id → {"path": str, "emb_file": str}

print("Loading mean vectors...")
for song_id, pt_file in enumerate(tqdm(pt_files, ascii=True, ncols=100, desc="Load")):
    pt_path = os.path.join(args.emb_dir, pt_file)
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        mean_vec = data["mean"].float().numpy()  # (C,)
        all_means.append(mean_vec)
        song_meta[song_id] = {
            "path": data["path"],
            "emb_file": pt_file,
        }
    except Exception as e:
        print(f"  Error loading {pt_file}: {e}")
        continue

all_means = np.stack(all_means, axis=0).astype(np.float32)  # (N, C)
n_songs, dim = all_means.shape
print(f"Loaded {n_songs} songs, embedding dim = {dim}")
print(f"Memory: {all_means.nbytes / 1024**3:.2f} GB")

###############################################################################
# 2. 构建 FAISS IVF 索引
###############################################################################

# 调整 nlist（不能超过样本数）
nlist = min(args.nlist, n_songs // 40)
print(f"Building IVFFlat index with nlist={nlist}...")

t_start = time.time()

# 使用 L2 距离（与模型的 nsqeuc 一致，只差一个常数缩放，不影响排序）
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)

if args.use_gpu:
    print(f"  Training on GPU {args.gpu_id}...")
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, args.gpu_id, index)
    gpu_index.train(all_means)
    gpu_index.add(all_means)
    # 转回 CPU 用于保存
    index = faiss.index_gpu_to_cpu(gpu_index)
    del gpu_index
else:
    print("  Training on CPU...")
    index.train(all_means)
    index.add(all_means)

elapsed = time.time() - t_start
print(f"Index built in {elapsed:.1f}s, total vectors: {index.ntotal}")

###############################################################################
# 3. 保存
###############################################################################

index_path = os.path.join(args.output_dir, "coarse.index")
meta_path = os.path.join(args.output_dir, "song_meta.pt")

print(f"Saving index to {index_path}...")
faiss.write_index(index, index_path)

print(f"Saving metadata to {meta_path}...")
torch.save(song_meta, meta_path)

print("Done!")
print(f"  Index: {os.path.getsize(index_path) / 1024**3:.2f} GB")
print(f"  Songs: {len(song_meta)}")
