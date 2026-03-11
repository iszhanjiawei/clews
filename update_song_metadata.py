#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新 song_meta.pt 中的歌曲名和歌手信息
从 Excel 文件中读取正确的歌名，替换从路径提取的不准确歌名
"""

import os
import torch
import pandas as pd
from tqdm import tqdm

# 配置
EXCEL_PATH = "百万原始歌曲曲库_去重后_正式版.xlsx"
SONG_META_PATH = "/home/zjw524/projects/clews_data/faiss_index/song_meta.pt"
OUTPUT_PATH = "/home/zjw524/projects/clews_data/faiss_index/song_meta_updated.pt"

print("=" * 80)
print("更新歌曲元数据")
print("=" * 80)

###############################################################################
# 1. 加载 Excel 文件，建立路径→歌名映射
###############################################################################

print(f"\n读取 Excel 文件: {EXCEL_PATH}")
df = pd.read_excel(EXCEL_PATH)
print(f"  总记录数: {len(df)}")
print(f"  列名: {list(df.columns)}")

# 建立路径映射字典
path_to_info = {}
for idx, row in tqdm(df.iterrows(), total=len(df), desc="建立路径映射", ascii=True):
    path = row["服务器样本路径"]
    song_name = row["歌名"]
    artist = row["歌手"]
    
    # 处理可能的 NaN 值
    if pd.isna(song_name):
        song_name = ""
    if pd.isna(artist):
        artist = ""
    
    path_to_info[path] = {
        "song_name": str(song_name).strip(),
        "artist": str(artist).strip(),
    }

print(f"\n建立映射完成: {len(path_to_info)} 条记录")

###############################################################################
# 2. 加载原始 song_meta.pt
###############################################################################

print(f"\n加载原始元数据: {SONG_META_PATH}")
song_meta = torch.load(SONG_META_PATH, map_location="cpu", weights_only=False)
print(f"  歌曲数量: {len(song_meta)}")

###############################################################################
# 3. 更新元数据
###############################################################################

print("\n更新元数据...")
matched_count = 0
unmatched_count = 0
unmatched_paths = []

for song_id, meta in tqdm(song_meta.items(), desc="更新", ascii=True):
    path = meta["path"]
    
    if path in path_to_info:
        # 找到匹配，添加正确的歌名和歌手
        info = path_to_info[path]
        meta["song_name"] = info["song_name"]
        meta["artist"] = info["artist"]
        matched_count += 1
    else:
        # 未找到匹配，使用路径提取的歌名
        meta["song_name"] = os.path.splitext(os.path.basename(path))[0]
        meta["artist"] = ""
        unmatched_count += 1
        unmatched_paths.append(path)

print(f"\n更新统计:")
print(f"  匹配成功: {matched_count} / {len(song_meta)} ({matched_count/len(song_meta)*100:.2f}%)")
print(f"  未匹配: {unmatched_count}")

if unmatched_count > 0:
    print(f"\n未匹配路径示例（前10个）:")
    for i, path in enumerate(unmatched_paths[:10], 1):
        print(f"  {i}. {path}")

###############################################################################
# 4. 保存更新后的元数据
###############################################################################

print(f"\n保存更新后的元数据: {OUTPUT_PATH}")
torch.save(song_meta, OUTPUT_PATH)

# 同时备份原文件并替换
backup_path = SONG_META_PATH + ".backup"
if not os.path.exists(backup_path):
    print(f"备份原文件: {backup_path}")
    torch.save(song_meta, backup_path)

print(f"替换原文件: {SONG_META_PATH}")
torch.save(song_meta, SONG_META_PATH)

print("\n" + "=" * 80)
print("更新完成！")
print("=" * 80)

# 验证更新结果
print("\n更新后数据示例（前5个）:")
for song_id in list(song_meta.keys())[:5]:
    meta = song_meta[song_id]
    print(f"\nSong ID {song_id}:")
    print(f"  歌名: {meta.get('song_name', 'N/A')}")
    print(f"  歌手: {meta.get('artist', 'N/A')}")
    print(f"  路径: {meta['path'][:80]}...")
