#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量检索翻唱歌曲
功能：
  1. 加载所有worker提取的查询向量
  2. 对每首歌进行FAISS检索（Top5）
  3. 保存检索结果供后续评估
"""

import os
import sys
import argparse
import time
import json
import torch
import numpy as np
import faiss
import importlib
from tqdm import tqdm
from omegaconf import OmegaConf

from utils import pytorch_utils
from lib import tensor_ops as tops

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="pretrained_models/dvi-clews/checkpoint_best.ckpt")
parser.add_argument("--index_path", type=str, default="/home/zjw524/projects/clews_data/faiss_index/coarse.index")
parser.add_argument("--meta_path", type=str, default="/home/zjw524/projects/clews_data/faiss_index/song_meta.pt")
parser.add_argument("--emb_dir", type=str, default="/home/zjw524/projects/clews_data/library_embeddings")
parser.add_argument("--query_dir", type=str, default="/home/zjw524/projects/clews_data/batch_test_results")
parser.add_argument("--output_file", type=str, default="/home/zjw524/projects/clews_data/batch_test_results/search_results.json")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--top_k", type=int, default=5, help="返回Top-K结果")
parser.add_argument("--nprobe", type=int, default=64)
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_id}")

###############################################################################
# 1. 加载模型
###############################################################################

print("加载模型...")
path_checkpoint, _ = os.path.split(args.checkpoint)
conf = OmegaConf.load(os.path.join(path_checkpoint, "configuration.yaml"))

module = importlib.import_module("models." + conf.model.name)
model = module.Model(conf.model, sr=conf.data.samplerate).to(device)

state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
print("模型加载完成")

###############################################################################
# 2. 加载FAISS索引和元数据
###############################################################################

print(f"加载FAISS索引: {args.index_path}")
index = faiss.read_index(args.index_path)
index.nprobe = args.nprobe
print(f"  索引大小: {index.ntotal} 向量")

print(f"加载元数据: {args.meta_path}")
song_meta = torch.load(args.meta_path, map_location="cpu", weights_only=False)
print(f"  歌曲数量: {len(song_meta)}")

###############################################################################
# 3. 合并所有worker的查询向量
###############################################################################

print(f"加载查询向量: {args.query_dir}")
all_queries = {}

for worker_file in sorted(os.listdir(args.query_dir)):
    if worker_file.startswith("query_embeddings_worker") and worker_file.endswith(".pt"):
        file_path = os.path.join(args.query_dir, worker_file)
        queries = torch.load(file_path, map_location="cpu", weights_only=False)
        all_queries.update(queries)
        print(f"  {worker_file}: {len(queries)} 个查询")

print(f"总查询数量: {len(all_queries)}")

###############################################################################
# 4. 批量检索
###############################################################################

print(f"开始批量检索（Top-{args.top_k}）...")
search_results = []

for task_id, query_data in tqdm(all_queries.items(), desc="搜索", ascii=True):
    query_shingles = torch.from_numpy(query_data["shingles"]).to(device)  # (S, C)
    query_mean = torch.from_numpy(query_data["mean"]).to(device).unsqueeze(0)  # (1, C)
    
    # FAISS粗检索（使用均值向量）
    coarse_k = 100
    D, I = index.search(query_mean.cpu().numpy(), coarse_k)
    candidate_ids = I[0].tolist()
    
    # 精排：加载候选歌曲的shingle向量，用模型计算距离
    fine_results = []
    for cand_id in candidate_ids:
        if cand_id == -1:
            continue
        
        # 加载候选歌曲的向量
        cand_meta = song_meta[cand_id]
        cand_emb_file = os.path.join(args.emb_dir, cand_meta["emb_file"])
        
        try:
            cand_data = torch.load(cand_emb_file, map_location="cpu", weights_only=False)
            cand_shingles = cand_data["shingles"]
            if isinstance(cand_shingles, np.ndarray):
                cand_shingles = torch.from_numpy(cand_shingles)
            cand_shingles = cand_shingles.float().to(device)  # (S', C) 转换为float32
            
            # 计算距离
            with torch.inference_mode():
                dist = model.distances(query_shingles.unsqueeze(0), cand_shingles.unsqueeze(0))
                dist = dist.item()
            
            fine_results.append({
                "song_id": cand_id,
                "distance": dist,
                "path": cand_data["path"],
            })
        except Exception as e:
            # 静默失败，继续处理下一个候选
            continue
    
    # 按距离排序，取Top-K
    fine_results.sort(key=lambda x: x["distance"])
    top_k_results = fine_results[:args.top_k]
    
    # 提取歌曲名（优先使用元数据，否则从路径提取）
    for r in top_k_results:
        song_id = r["song_id"]
        if song_id in song_meta and "song_name" in song_meta[song_id]:
            song_name = song_meta[song_id]["song_name"]
            artist = song_meta[song_id].get("artist", "")
            r["song_name"] = f"{song_name} - {artist}" if artist else song_name
        else:
            r["song_name"] = os.path.splitext(os.path.basename(r["path"]))[0]
    
    # 保存结果
    search_results.append({
        "task_id": task_id,
        "query_song": query_data["song_name"],
        "query_artist": query_data["artist"],
        "query_path": query_data["audio_path"],
        "top_k_results": top_k_results,
    })

###############################################################################
# 5. 保存检索结果
###############################################################################

print(f"保存检索结果: {args.output_file}")
with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(search_results, f, ensure_ascii=False, indent=2)

print(f"检索完成！共 {len(search_results)} 个查询")
