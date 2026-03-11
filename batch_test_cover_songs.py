#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量测试翻唱歌曲检索系统
功能：
  1. 从xlsx随机抽取500首翻唱歌曲
  2. 双GPU多进程并行提取查询向量
  3. 对每首歌检索Top5结果
  4. 生成详细测试报告（准确率、召回率等）
"""

import os
import sys
import argparse
import time
import json
import importlib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

from utils import pytorch_utils, audio_utils
from lib import tensor_ops as tops

parser = argparse.ArgumentParser()
parser.add_argument("--xlsx_path", type=str, default="data/exact_matched_cover_songs_existing_files.xlsx")
parser.add_argument("--sample_size", type=int, default=500, help="随机抽取样本数量")
parser.add_argument("--checkpoint", type=str, default="pretrained_models/dvi-clews/checkpoint_best.ckpt")
parser.add_argument("--gpu_id", type=int, default=0, help="当前进程使用的GPU")
parser.add_argument("--worker_id", type=int, default=0, help="当前worker编号")
parser.add_argument("--total_workers", type=int, default=1, help="总worker数量")
parser.add_argument("--output_dir", type=str, default="/home/zjw524/projects/clews_data/batch_test_results")
parser.add_argument("--query_hop", type=float, default=5.0)
parser.add_argument("--max_audio_len", type=float, default=600)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device(f"cuda:{args.gpu_id}")

###############################################################################
# 1. 读取并抽样测试数据（只在worker 0执行，避免重复抽样）
###############################################################################

if args.worker_id == 0:
    print(f"[Worker-{args.worker_id}] 读取xlsx文件: {args.xlsx_path}")
    df = pd.read_excel(args.xlsx_path)
    print(f"  总记录数: {len(df)}")
    
    # 随机抽取500首
    if len(df) > args.sample_size:
        df_sample = df.sample(n=args.sample_size, random_state=42)
    else:
        df_sample = df
    
    print(f"  抽取样本数: {len(df_sample)}")
    
    # 保存抽样结果
    sample_file = os.path.join(args.output_dir, "test_samples.csv")
    df_sample.to_csv(sample_file, index=False, encoding='utf-8')
    print(f"  样本已保存: {sample_file}")
    
    # 创建测试任务列表
    test_tasks = []
    for idx, row in df_sample.iterrows():
        test_tasks.append({
            "task_id": len(test_tasks),
            "song_name": row["歌名"],
            "artist": row["歌手"],
            "audio_path": row["服务器样本路径"],
        })
    
    tasks_file = os.path.join(args.output_dir, "test_tasks.json")
    with open(tasks_file, 'w', encoding='utf-8') as f:
        json.dump(test_tasks, f, ensure_ascii=False, indent=2)
    print(f"  测试任务已保存: {tasks_file}")
    print(f"  等待5秒让其他worker准备...")
    time.sleep(5)

# 其他worker等待worker 0准备好
if args.worker_id > 0:
    time.sleep(8)

###############################################################################
# 2. 加载测试任务并分配给各worker
###############################################################################

tasks_file = os.path.join(args.output_dir, "test_tasks.json")
with open(tasks_file, 'r', encoding='utf-8') as f:
    all_tasks = json.load(f)

# 分配任务（交错分配）
my_tasks = [t for i, t in enumerate(all_tasks) if i % args.total_workers == args.worker_id]
print(f"[Worker-{args.worker_id}] 分配到 {len(my_tasks)} 个测试任务")

###############################################################################
# 3. 加载模型
###############################################################################

print(f"[Worker-{args.worker_id}] 加载模型...")
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

print(f"[Worker-{args.worker_id}] 模型加载完成")

###############################################################################
# 4. 提取查询向量
###############################################################################

print(f"[Worker-{args.worker_id}] 开始提取查询向量...")
query_embeddings = {}
failed_tasks = []

for task in tqdm(my_tasks, desc=f"[Worker-{args.worker_id}] Extract", ascii=True):
    task_id = task["task_id"]
    audio_path = task["audio_path"]
    
    if not os.path.exists(audio_path):
        failed_tasks.append({"task_id": task_id, "reason": "文件不存在"})
        continue
    
    try:
        # 加载音频
        x = audio_utils.load_audio(audio_path, sample_rate=model.sr, n_channels=1)
        if x is None:
            failed_tasks.append({"task_id": task_id, "reason": "加载音频失败"})
            continue
        
        # 裁剪到最大长度
        max_samples = int(args.max_audio_len * model.sr)
        if x.size(1) > max_samples:
            x = x[:, :max_samples]
        
        x = x.to(device)
        
        # 提取向量
        with torch.inference_mode():
            z = model(x, shingle_hop=args.query_hop, shingle_len=None)  # (1, S, C)
            shingles = z.squeeze(0).cpu()  # (S, C)
            mean_vec = shingles.mean(dim=0)  # (C,)
        
        query_embeddings[task_id] = {
            "shingles": shingles.numpy(),
            "mean": mean_vec.numpy(),
            "song_name": task["song_name"],
            "artist": task["artist"],
            "audio_path": audio_path,
        }
        
    except Exception as e:
        failed_tasks.append({"task_id": task_id, "reason": str(e)})

# 保存查询向量
emb_file = os.path.join(args.output_dir, f"query_embeddings_worker{args.worker_id}.pt")
torch.save(query_embeddings, emb_file)
print(f"[Worker-{args.worker_id}] 查询向量已保存: {emb_file}")
print(f"  成功: {len(query_embeddings)}, 失败: {len(failed_tasks)}")

if failed_tasks:
    fail_file = os.path.join(args.output_dir, f"failed_tasks_worker{args.worker_id}.json")
    with open(fail_file, 'w', encoding='utf-8') as f:
        json.dump(failed_tasks, f, ensure_ascii=False, indent=2)

print(f"[Worker-{args.worker_id}] 完成!")
