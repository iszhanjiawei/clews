"""
阶段三：在线查询检索
用法：
  python search_query.py --query_audio /path/to/query.mp3
  python search_query.py --query_audio /path/to/query.mp3 --top_k 20 --gpu_id 0

功能：
  1. 加载查询音频，提取向量
  2. FAISS 粗检索 Top-K 候选
  3. 加载候选歌曲的 shingle 向量，用模型做精排
  4. 输出最终匹配结果

输出：
  打印 Top-K 匹配结果（歌曲路径 + 距离分数）
"""

import sys
import os
import argparse
import importlib
import time
import torch
import numpy as np
import faiss
from omegaconf import OmegaConf

from utils import pytorch_utils, audio_utils
from lib import tensor_ops as tops

parser = argparse.ArgumentParser()
parser.add_argument("--query_audio", type=str, required=True, help="查询音频文件路径")
parser.add_argument("--checkpoint", type=str, default="pretrained_models/dvi-clews/checkpoint_best.ckpt")
parser.add_argument("--index_path", type=str, default="/home/zjw524/projects/clews_data/faiss_index/coarse.index")
parser.add_argument("--meta_path", type=str, default="/home/zjw524/projects/clews_data/faiss_index/song_meta.pt")
parser.add_argument("--emb_dir", type=str, default="/home/zjw524/projects/clews_data/library_embeddings")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--coarse_top_k", type=int, default=100, help="FAISS 粗检索候选数")
parser.add_argument("--final_top_k", type=int, default=10, help="最终输出 Top-K")
parser.add_argument("--nprobe", type=int, default=64, help="FAISS 搜索时探查的聚类数")
parser.add_argument("--query_hop", type=float, default=5.0, help="查询音频 shingle hop (秒)")
parser.add_argument("--query_win", type=float, default=-1, help="查询音频 shingle 长度 (-1=模型默认)")
parser.add_argument("--max_audio_len", type=float, default=600, help="最大音频长度（秒）")
args = parser.parse_args()

if args.query_win <= 0:
    args.query_win = None

device = torch.device(f"cuda:{args.gpu_id}")
torch.cuda.set_device(device)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

###############################################################################
# 1. 加载模型
###############################################################################

print("=" * 80)
print("Loading model...")
path_checkpoint, _ = os.path.split(args.checkpoint)
conf = OmegaConf.load(os.path.join(path_checkpoint, "configuration.yaml"))

module = importlib.import_module("models." + conf.model.name)
model = module.Model(conf.model, sr=conf.data.samplerate).to(device)

state_dict = torch.load(args.checkpoint, map_location=device)
if "model" in state_dict:
    model_state = state_dict["model"]
else:
    model_state = state_dict
cleaned_state = {}
for k, v in model_state.items():
    new_key = k.replace("_forward_module.", "")
    cleaned_state[new_key] = v
model.load_state_dict(cleaned_state, strict=False)
model.eval()
print("Model loaded.")

###############################################################################
# 2. 提取查询音频的向量
###############################################################################

print(f"Processing query: {args.query_audio}")
t0 = time.time()

x = audio_utils.load_audio(args.query_audio, sample_rate=model.sr, n_channels=1)
if x is None:
    print("ERROR: Could not load query audio.")
    sys.exit(1)

max_samples = int(args.max_audio_len * model.sr)
if x.size(1) > max_samples:
    x = x[:, :max_samples]

x = x.to(device)
with torch.inference_mode():
    query_z = model(x, shingle_hop=args.query_hop, shingle_len=args.query_win)
    # query_z: (1, S_q, C)

query_z_cpu = query_z.squeeze(0).cpu()  # (S_q, C)
query_mean = query_z_cpu.mean(dim=0).float().numpy().reshape(1, -1)  # (1, C)

print(f"  Query shingles: {query_z_cpu.shape[0]}, dim: {query_z_cpu.shape[1]}")
print(f"  Feature extraction: {time.time() - t0:.2f}s")

###############################################################################
# 3. FAISS 粗检索
###############################################################################

print(f"Loading FAISS index: {args.index_path}")
index = faiss.read_index(args.index_path)
index.nprobe = args.nprobe
print(f"  Index size: {index.ntotal} vectors, nprobe={args.nprobe}")

print(f"Loading metadata: {args.meta_path}")
song_meta = torch.load(args.meta_path, map_location="cpu", weights_only=False)

print("Coarse search...")
t1 = time.time()
distances_coarse, indices_coarse = index.search(query_mean, args.coarse_top_k)
distances_coarse = distances_coarse[0]  # (coarse_top_k,)
indices_coarse = indices_coarse[0]      # (coarse_top_k,)
print(f"  Coarse search: {time.time() - t1:.4f}s, top-{args.coarse_top_k} candidates found")

###############################################################################
# 4. 精排：加载候选歌曲的 shingle 向量，用模型 distances() 计算
###############################################################################

print("Fine-grained re-ranking...")
t2 = time.time()

results = []
with torch.inference_mode():
    for rank, (coarse_dist, song_id) in enumerate(zip(distances_coarse, indices_coarse)):
        if song_id < 0:  # FAISS 返回 -1 表示未找到
            continue

        meta = song_meta[int(song_id)]
        emb_path = os.path.join(args.emb_dir, meta["emb_file"])

        try:
            data = torch.load(emb_path, map_location="cpu", weights_only=False)
            cand_shingles = data["shingles"].float()  # (S_c, C)
        except Exception as e:
            print(f"  Warning: could not load {emb_path}: {e}")
            continue

        # 用模型的 distances() 做精排
        # query_z: (1, S_q, C), cand: (1, S_c, C)
        q = query_z.float()  # (1, S_q, C) 已在 GPU
        c = cand_shingles.unsqueeze(0).to(device)  # (1, S_c, C)

        dist = model.distances(q, c)  # (1, 1) → scalar
        dist_val = dist.item()

        results.append({
            "song_id": int(song_id),
            "path": meta["path"],
            "fine_dist": dist_val,
            "coarse_dist": float(coarse_dist),
        })

# 按精排距离排序（越小越相似）
results.sort(key=lambda r: r["fine_dist"])

print(f"  Re-ranking: {time.time() - t2:.2f}s")

###############################################################################
# 5. 输出结果
###############################################################################

total_time = time.time() - t0
print("=" * 80)
print(f"Query: {args.query_audio}")
print(f"Total search time: {total_time:.2f}s")
print("=" * 80)
print(f"Top-{args.final_top_k} Results:")
print("-" * 80)

for i, r in enumerate(results[:args.final_top_k]):
    # 优先使用元数据中的歌名，否则从路径提取
    song_id = r["song_id"]
    if song_id in song_meta and "song_name" in song_meta[song_id]:
        song_name = song_meta[song_id]["song_name"]
        artist = song_meta[song_id].get("artist", "")
        display_name = f"{song_name} - {artist}" if artist else song_name
    else:
        display_name = os.path.splitext(os.path.basename(r["path"]))[0]
    print(f"  #{i+1} | dist={r['fine_dist']:.6f} | {display_name}")
    print(f"       | path: {r['path']}")

print("=" * 80)
