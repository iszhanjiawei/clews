#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验：测试查询音频最低时长对检索精度的影响
对已知能成功匹配的歌曲，截取不同长度（10s-180s），分别进行检索，
统计各时长的命中率，找到最低可靠时长阈值。
"""

import os
import sys
import json
import torch
import numpy as np
import faiss
import importlib
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import audio_utils

# ===================== 配置 =====================
CHECKPOINT = "pretrained_models/dvi-clews/checkpoint_best.ckpt"
INDEX_PATH = "/home/zjw524/projects/clews_data/faiss_index/coarse.index"
META_PATH = "/home/zjw524/projects/clews_data/faiss_index/song_meta.pt"
EMB_DIR = "/home/zjw524/projects/clews_data/library_embeddings"
SEARCH_RESULTS = "/home/zjw524/projects/clews_data/batch_test_results/search_results.json"
OUTPUT_DIR = "/home/zjw524/projects/clews_data/min_duration_experiment"

GPU_ID = 0
COARSE_TOP_K = 100
NPROBE = 64
QUERY_HOP = 5.0  # 和 search_query.py 默认一致

# 测试时长（秒）
TEST_DURATIONS = [10, 15, 20, 25, 30, 45, 60, 90, 120, 180]

# 从已成功匹配的歌曲中随机抽取进行实验
NUM_SONGS = 50  # 抽取50首进行实验

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("查询音频最低时长实验")
print("=" * 80)

# ===================== 加载模型 =====================
print("\n加载模型...")
device = torch.device(f"cuda:{GPU_ID}")
torch.cuda.set_device(device)

path_checkpoint, _ = os.path.split(CHECKPOINT)
conf = OmegaConf.load(os.path.join(path_checkpoint, "configuration.yaml"))
module = importlib.import_module("models." + conf.model.name)
model = module.Model(conf.model, sr=conf.data.samplerate).to(device)

state_dict = torch.load(CHECKPOINT, map_location=device, weights_only=False)
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

print(f"  模型 shingle_len: {conf.model.shingling.len} 秒")
print(f"  模型 shingle_hop: {conf.model.shingling.hop} 秒")
print(f"  查询 query_hop: {QUERY_HOP} 秒")
print(f"  采样率: {model.sr} Hz")

# ===================== 加载索引和元数据 =====================
print("\n加载 FAISS 索引...")
index = faiss.read_index(INDEX_PATH)
index.nprobe = NPROBE
print(f"  索引大小: {index.ntotal}")

print("加载元数据...")
song_meta = torch.load(META_PATH, map_location="cpu", weights_only=False)

# ===================== 选取实验歌曲 =====================
print(f"\n从已匹配的歌曲中选取 {NUM_SONGS} 首进行实验...")

# 加载之前的 search_results.json，选取 Top-1 命中的歌曲
with open(SEARCH_RESULTS, 'r', encoding='utf-8') as f:
    all_results = json.load(f)

# 找到 Top-1 命中的歌曲
import re
def normalize_song_name(name):
    if not name:
        return ""
    name = re.sub(r'\s*\([^)]*\)\s*', '', name)
    name = re.sub(r'\s*（[^）]*）\s*', '', name)
    name = re.sub(r'\s*\[[^\]]*\]\s*', '', name)
    return name.strip().lower()

hit1_songs = []
for result in all_results:
    query_song = result["query_song"]
    top_k = result["top_k_results"]
    if len(top_k) > 0:
        result_song = top_k[0]["song_name"]
        if " - " in result_song:
            result_song_name = result_song.split(" - ")[0].strip()
        else:
            result_song_name = result_song
        
        q_norm = normalize_song_name(query_song)
        r_norm = normalize_song_name(result_song_name)
        
        if q_norm and r_norm and (q_norm in r_norm or r_norm in q_norm):
            hit1_songs.append({
                "query_song": query_song,
                "query_path": result["query_path"],
                "original_song_name": result_song,
                "original_distance": top_k[0]["distance"],
            })

print(f"  Top-1 命中歌曲总数: {len(hit1_songs)}")

# 随机抽取，优先选择时长较长的（保证能截取到180秒）
np.random.seed(42)
np.random.shuffle(hit1_songs)

# 筛选时长足够长的歌曲（> 180秒）
selected_songs = []
for song in hit1_songs:
    if len(selected_songs) >= NUM_SONGS:
        break
    
    path = song["query_path"]
    if not os.path.exists(path):
        continue
    
    try:
        x = audio_utils.load_audio(path, sample_rate=model.sr, n_channels=1)
        if x is None:
            continue
        duration = x.size(1) / model.sr
        if duration >= 180:  # 至少 180 秒
            song["full_duration"] = duration
            song["audio_tensor"] = x
            selected_songs.append(song)
    except:
        continue

print(f"  筛选后（时长>=180秒）: {len(selected_songs)} 首")

# 如果不够50首，降低时长要求
if len(selected_songs) < NUM_SONGS:
    for song in hit1_songs:
        if len(selected_songs) >= NUM_SONGS:
            break
        if song in selected_songs:
            continue
        
        path = song["query_path"]
        if not os.path.exists(path):
            continue
        
        try:
            x = audio_utils.load_audio(path, sample_rate=model.sr, n_channels=1)
            if x is None:
                continue
            duration = x.size(1) / model.sr
            if duration >= 120:  # 至少 120 秒
                song["full_duration"] = duration
                song["audio_tensor"] = x
                selected_songs.append(song)
        except:
            continue

print(f"  最终选取: {len(selected_songs)} 首")

# ===================== 开始实验 =====================
print(f"\n开始实验：{len(TEST_DURATIONS)} 个时长 × {len(selected_songs)} 首歌曲")
print(f"测试时长: {TEST_DURATIONS}")
print("=" * 80)

# 结果存储
# key: duration, value: list of {song_name, hit_rank, distance, num_shingles}
results_by_duration = {d: [] for d in TEST_DURATIONS}

for song_idx, song in enumerate(selected_songs):
    query_song = song["query_song"]
    full_audio = song["audio_tensor"]
    full_duration = song["full_duration"]
    
    print(f"\n[{song_idx+1}/{len(selected_songs)}] {query_song} (原始时长: {full_duration:.0f}秒)")
    
    for dur in TEST_DURATIONS:
        if dur > full_duration:
            # 歌曲不够长，跳过
            results_by_duration[dur].append({
                "song": query_song,
                "hit_rank": -2,  # 时长不足跳过
                "distance": -1,
                "num_shingles": 0,
            })
            continue
        
        # 截取指定时长
        max_samples = int(dur * model.sr)
        x_truncated = full_audio[:, :max_samples].to(device)
        
        # 提取特征
        with torch.inference_mode():
            z = model(x_truncated, shingle_hop=QUERY_HOP, shingle_len=None)
            query_shingles = z.squeeze(0)  # (S, C)
            query_mean = query_shingles.mean(dim=0).cpu().float().numpy().reshape(1, -1)
        
        num_shingles = query_shingles.shape[0]
        
        # FAISS 粗检索
        D, I = index.search(query_mean, COARSE_TOP_K)
        
        # 精排
        fine_results = []
        for cand_id in I[0]:
            if cand_id < 0:
                continue
            
            meta = song_meta[int(cand_id)]
            emb_path = os.path.join(EMB_DIR, meta["emb_file"])
            
            try:
                data = torch.load(emb_path, map_location="cpu", weights_only=False)
                cand_shingles = data["shingles"]
                if isinstance(cand_shingles, np.ndarray):
                    cand_shingles = torch.from_numpy(cand_shingles)
                cand_shingles = cand_shingles.float().to(device)
                
                with torch.inference_mode():
                    dist = model.distances(
                        query_shingles.unsqueeze(0),
                        cand_shingles.unsqueeze(0)
                    )
                    dist_val = dist.item()
                
                fine_results.append({
                    "song_id": int(cand_id),
                    "song_name": meta.get("song_name", ""),
                    "distance": dist_val,
                })
            except:
                continue
        
        fine_results.sort(key=lambda x: x["distance"])
        
        # 检查是否命中原唱
        hit_rank = -1  # 未命中
        hit_distance = -1
        
        q_norm = normalize_song_name(query_song)
        for rank, r in enumerate(fine_results[:5], 1):
            r_name = r["song_name"]
            if " - " in r_name:
                r_name = r_name.split(" - ")[0].strip()
            r_norm = normalize_song_name(r_name)
            
            if q_norm and r_norm and (q_norm in r_norm or r_norm in q_norm):
                hit_rank = rank
                hit_distance = r["distance"]
                break
        
        results_by_duration[dur].append({
            "song": query_song,
            "hit_rank": hit_rank,
            "distance": hit_distance,
            "num_shingles": num_shingles,
        })
        
        status = f"✅ Rank-{hit_rank}" if hit_rank > 0 else "❌ 未命中"
        print(f"  {dur:>4}秒 | shingles={num_shingles:>3} | {status} | dist={hit_distance:.4f}" if hit_rank > 0 else f"  {dur:>4}秒 | shingles={num_shingles:>3} | {status}")

# ===================== 统计分析 =====================
print("\n\n" + "=" * 80)
print("实验结果统计")
print("=" * 80)

summary = []
for dur in TEST_DURATIONS:
    results = results_by_duration[dur]
    
    # 过滤掉时长不足跳过的
    valid = [r for r in results if r["hit_rank"] != -2]
    
    if not valid:
        continue
    
    total = len(valid)
    hit1 = sum(1 for r in valid if r["hit_rank"] == 1)
    hit3 = sum(1 for r in valid if 1 <= r["hit_rank"] <= 3)
    hit5 = sum(1 for r in valid if 1 <= r["hit_rank"] <= 5)
    no_hit = sum(1 for r in valid if r["hit_rank"] == -1)
    
    avg_shingles = np.mean([r["num_shingles"] for r in valid])
    
    # 命中的平均距离
    hit_dists = [r["distance"] for r in valid if r["hit_rank"] > 0]
    avg_dist = np.mean(hit_dists) if hit_dists else -1
    
    row = {
        "时长(秒)": dur,
        "有效歌曲数": total,
        "平均shingles": f"{avg_shingles:.1f}",
        "Hit@1": f"{hit1}/{total} ({hit1/total*100:.1f}%)",
        "Hit@3": f"{hit3}/{total} ({hit3/total*100:.1f}%)",
        "Hit@5": f"{hit5}/{total} ({hit5/total*100:.1f}%)",
        "未命中": f"{no_hit}/{total} ({no_hit/total*100:.1f}%)",
        "平均距离": f"{avg_dist:.4f}" if avg_dist > 0 else "N/A",
    }
    summary.append(row)
    
    print(f"\n时长 = {dur} 秒:")
    print(f"  有效歌曲数: {total}")
    print(f"  平均 shingles: {avg_shingles:.1f}")
    print(f"  Hit@1: {hit1}/{total} = {hit1/total*100:.1f}%")
    print(f"  Hit@3: {hit3}/{total} = {hit3/total*100:.1f}%")
    print(f"  Hit@5: {hit5}/{total} = {hit5/total*100:.1f}%")
    print(f"  未命中: {no_hit}/{total} = {no_hit/total*100:.1f}%")
    if avg_dist > 0:
        print(f"  平均匹配距离: {avg_dist:.4f}")

# ===================== 保存结果 =====================
print("\n\n保存结果...")

# 保存汇总
summary_df = pd.DataFrame(summary)
summary_file = os.path.join(OUTPUT_DIR, "时长vs命中率汇总.xlsx")
summary_df.to_excel(summary_file, index=False, engine='openpyxl')
print(f"  汇总: {summary_file}")

# 保存详细结果
detail_data = []
for dur in TEST_DURATIONS:
    for r in results_by_duration[dur]:
        if r["hit_rank"] == -2:
            continue
        detail_data.append({
            "时长(秒)": dur,
            "歌曲": r["song"],
            "shingles数": r["num_shingles"],
            "命中排名": r["hit_rank"],
            "距离": r["distance"],
        })

detail_df = pd.DataFrame(detail_data)
detail_file = os.path.join(OUTPUT_DIR, "时长实验详细数据.xlsx")
detail_df.to_excel(detail_file, index=False, engine='openpyxl')
print(f"  详细: {detail_file}")

# 保存JSON
json_file = os.path.join(OUTPUT_DIR, "experiment_results.json")
json_data = {
    "test_durations": TEST_DURATIONS,
    "num_songs": len(selected_songs),
    "query_hop": QUERY_HOP,
    "shingle_len": conf.model.shingling.len,
    "results": {str(d): results_by_duration[d] for d in TEST_DURATIONS}
}
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)
print(f"  JSON: {json_file}")

print("\n" + "=" * 80)
print("实验完成！")
print("=" * 80)
