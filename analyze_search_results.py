#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析批量检索结果，保存各种分类数据
"""

import os
import json
import pandas as pd
from collections import defaultdict

# 配置
RESULTS_FILE = "/home/zjw524/projects/clews_data/batch_test_results/search_results.json"
OUTPUT_DIR = "/home/zjw524/projects/clews_data/batch_test_results/detailed_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("详细分析检索结果")
print("=" * 80)

###############################################################################
# 1. 加载检索结果
###############################################################################

print(f"\n加载检索结果: {RESULTS_FILE}")
with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
    results = json.load(f)

print(f"  总查询数: {len(results)}")

###############################################################################
# 2. 保存所有查询的 Top-5 结果
###############################################################################

print("\n保存所有查询的 Top-5 结果...")

# 准备 Excel 数据
all_top5_data = []

for result in results:
    task_id = result["task_id"]
    query_song = result["query_song"]
    query_artist = result["query_artist"]
    query_path = result["query_path"]
    top_k_results = result["top_k_results"]
    
    # 为每个查询创建一行，包含 Top-5 结果
    row = {
        "任务ID": task_id,
        "查询歌曲": query_song,
        "查询歌手": query_artist,
        "查询路径": query_path,
    }
    
    # 添加 Top-5 结果
    for i in range(5):
        if i < len(top_k_results):
            r = top_k_results[i]
            row[f"Top-{i+1}_歌名"] = r["song_name"]
            row[f"Top-{i+1}_距离"] = r["distance"]
            row[f"Top-{i+1}_路径"] = r["path"]
        else:
            row[f"Top-{i+1}_歌名"] = ""
            row[f"Top-{i+1}_距离"] = ""
            row[f"Top-{i+1}_路径"] = ""
    
    all_top5_data.append(row)

# 保存为 Excel
df_all_top5 = pd.DataFrame(all_top5_data)
output_file = os.path.join(OUTPUT_DIR, "所有查询的Top5结果.xlsx")
df_all_top5.to_excel(output_file, index=False, engine='openpyxl')
print(f"  已保存: {output_file}")

# 同时保存为 JSON
output_json = os.path.join(OUTPUT_DIR, "所有查询的Top5结果.json")
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(all_top5_data, f, ensure_ascii=False, indent=2)
print(f"  已保存: {output_json}")

###############################################################################
# 3. 检查匹配情况（按排名分类）
###############################################################################

print("\n分析匹配情况...")

def normalize_song_name(name):
    """标准化歌名用于比较"""
    if not name:
        return ""
    # 移除常见的括号内容、空格等
    import re
    name = re.sub(r'\s*\([^)]*\)\s*', '', name)  # 移除括号
    name = re.sub(r'\s*（[^）]*）\s*', '', name)  # 移除中文括号
    name = re.sub(r'\s*\[[^\]]*\]\s*', '', name)  # 移除方括号
    name = name.strip().lower()
    return name

# 分类存储
rank1_hits = []  # Top-1 命中
rank2_hits = []  # Top-2 命中
rank3_hits = []  # Top-3 命中
rank4_hits = []  # Top-4 命中
rank5_hits = []  # Top-5 命中
no_hits = []     # 前5名都未命中

for result in results:
    task_id = result["task_id"]
    query_song = result["query_song"]
    query_artist = result["query_artist"]
    query_path = result["query_path"]
    top_k_results = result["top_k_results"]
    
    # 标准化查询歌名
    query_normalized = normalize_song_name(query_song)
    
    # 检查每个排名是否匹配
    matched_rank = None
    matched_result = None
    
    for rank, r in enumerate(top_k_results, 1):
        result_song = r["song_name"]
        # 从 "歌名 - 歌手" 格式中提取歌名
        if " - " in result_song:
            result_song_name = result_song.split(" - ")[0].strip()
        else:
            result_song_name = result_song
        
        result_normalized = normalize_song_name(result_song_name)
        
        # 检查是否匹配
        if query_normalized and result_normalized and query_normalized in result_normalized or result_normalized in query_normalized:
            matched_rank = rank
            matched_result = r
            break
    
    # 创建记录
    record = {
        "任务ID": task_id,
        "查询歌曲": query_song,
        "查询歌手": query_artist,
        "查询路径": query_path,
    }
    
    if matched_rank:
        record["匹配排名"] = matched_rank
        record["匹配歌名"] = matched_result["song_name"]
        record["匹配距离"] = matched_result["distance"]
        record["匹配路径"] = matched_result["path"]
        
        # 添加到对应排名列表
        if matched_rank == 1:
            rank1_hits.append(record)
        elif matched_rank == 2:
            rank2_hits.append(record)
        elif matched_rank == 3:
            rank3_hits.append(record)
        elif matched_rank == 4:
            rank4_hits.append(record)
        elif matched_rank == 5:
            rank5_hits.append(record)
    else:
        # 未命中，添加 Top-5 结果供参考
        for i in range(min(5, len(top_k_results))):
            r = top_k_results[i]
            record[f"Top-{i+1}_歌名"] = r["song_name"]
            record[f"Top-{i+1}_距离"] = r["distance"]
        
        no_hits.append(record)

print(f"\n匹配统计:")
print(f"  Top-1 命中: {len(rank1_hits)}")
print(f"  Top-2 命中: {len(rank2_hits)}")
print(f"  Top-3 命中: {len(rank3_hits)}")
print(f"  Top-4 命中: {len(rank4_hits)}")
print(f"  Top-5 命中: {len(rank5_hits)}")
print(f"  未命中: {len(no_hits)}")
print(f"  总计: {len(rank1_hits) + len(rank2_hits) + len(rank3_hits) + len(rank4_hits) + len(rank5_hits) + len(no_hits)}")

###############################################################################
# 4. 保存各类别结果
###############################################################################

print("\n保存各类别结果...")

categories = [
    ("Top1命中的歌曲", rank1_hits),
    ("Top2命中的歌曲", rank2_hits),
    ("Top3命中的歌曲", rank3_hits),
    ("Top4命中的歌曲", rank4_hits),
    ("Top5命中的歌曲", rank5_hits),
    ("前5名都未命中的歌曲", no_hits),
]

for name, data in categories:
    if data:
        # 保存为 Excel
        df = pd.DataFrame(data)
        excel_file = os.path.join(OUTPUT_DIR, f"{name}.xlsx")
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"  已保存: {excel_file} ({len(data)} 条)")
        
        # 保存为 JSON
        json_file = os.path.join(OUTPUT_DIR, f"{name}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

###############################################################################
# 5. 生成汇总报告
###############################################################################

print("\n生成汇总报告...")

summary = {
    "总查询数": len(results),
    "Top-1命中": len(rank1_hits),
    "Top-2命中": len(rank2_hits),
    "Top-3命中": len(rank3_hits),
    "Top-4命中": len(rank4_hits),
    "Top-5命中": len(rank5_hits),
    "未命中": len(no_hits),
    "Top-1命中率": f"{len(rank1_hits)/len(results)*100:.2f}%",
    "Top-5命中率": f"{(len(rank1_hits)+len(rank2_hits)+len(rank3_hits)+len(rank4_hits)+len(rank5_hits))/len(results)*100:.2f}%",
}

summary_file = os.path.join(OUTPUT_DIR, "汇总统计.json")
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"  已保存: {summary_file}")

# 打印汇总
print("\n" + "=" * 80)
print("汇总统计")
print("=" * 80)
for key, value in summary.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 80)
print(f"所有结果已保存到: {OUTPUT_DIR}")
print("=" * 80)
