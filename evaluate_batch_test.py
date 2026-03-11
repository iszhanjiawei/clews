#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估批量测试结果
生成详细的测试报告，包括：
  - Precision@K（精确率）
  - Recall@K（召回率）
  - MRR（平均倒数排名）
  - Hit@K（命中率）
  - 分距离段统计
  - 详细的匹配/未匹配案例
"""

import os
import json
import pandas as pd
from collections import defaultdict
import numpy as np

def normalize_song_name(name):
    """
    归一化歌曲名，用于匹配比较
    去除空格、下划线、连字符等
    """
    import re
    # 去除空格、下划线、连字符
    name = re.sub(r'[\s_\-]+', '', name.lower())
    # 去除括号内容
    name = re.sub(r'[\(（].*?[\)）]', '', name)
    return name.strip()

def check_match(query_song, result_song):
    """
    检查结果歌曲名是否包含查询歌曲名
    """
    query_norm = normalize_song_name(query_song)
    result_norm = normalize_song_name(result_song)
    
    # 精确匹配或包含关系
    return query_norm in result_norm or result_norm in query_norm

def evaluate_results(results_file, output_dir):
    """
    评估检索结果
    """
    print("=" * 80)
    print("批量测试评估报告")
    print("=" * 80)
    
    # 加载检索结果
    print(f"\n加载检索结果: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    total_queries = len(results)
    print(f"总查询数: {total_queries}")
    
    # 统计指标
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    mrr_scores = []
    
    # 距离统计
    distance_stats = defaultdict(list)
    
    # 详细结果
    matched_cases = []
    unmatched_cases = []
    
    for query in results:
        query_song = query["query_song"]
        top_k = query["top_k_results"]
        
        # 检查每个结果是否匹配
        match_found = False
        match_rank = -1
        
        for rank, result in enumerate(top_k, start=1):
            result_song = result["song_name"]
            distance = result["distance"]
            
            # 记录距离
            distance_stats[f"Rank-{rank}"].append(distance)
            
            # 检查是否匹配
            if check_match(query_song, result_song):
                if not match_found:
                    match_found = True
                    match_rank = rank
                    
                    # 计算MRR
                    mrr_scores.append(1.0 / rank)
                    
                    # 更新Hit@K
                    if rank == 1:
                        hit_at_1 += 1
                    if rank <= 3:
                        hit_at_3 += 1
                    if rank <= 5:
                        hit_at_5 += 1
                    
                    # 记录匹配案例
                    matched_cases.append({
                        "query_song": query_song,
                        "query_artist": query["query_artist"],
                        "matched_rank": rank,
                        "matched_song": result_song,
                        "distance": distance,
                        "all_top5": [r["song_name"] for r in top_k],
                    })
                    break
        
        # 如果没找到匹配
        if not match_found:
            mrr_scores.append(0.0)
            unmatched_cases.append({
                "query_song": query_song,
                "query_artist": query["query_artist"],
                "query_path": query["query_path"],
                "top5_results": [
                    {"rank": i+1, "song": r["song_name"], "distance": r["distance"]}
                    for i, r in enumerate(top_k)
                ],
            })
    
    # 计算最终指标
    precision_at_1 = hit_at_1 / total_queries
    precision_at_3 = hit_at_3 / total_queries
    precision_at_5 = hit_at_5 / total_queries
    mrr = np.mean(mrr_scores)
    
    # 打印报告
    print("\n" + "=" * 80)
    print("评估指标")
    print("=" * 80)
    print(f"总查询数: {total_queries}")
    print(f"\nHit@1 (Top-1命中率):  {hit_at_1:4d} / {total_queries} = {precision_at_1:.2%}")
    print(f"Hit@3 (Top-3命中率):  {hit_at_3:4d} / {total_queries} = {precision_at_3:.2%}")
    print(f"Hit@5 (Top-5命中率):  {hit_at_5:4d} / {total_queries} = {precision_at_5:.2%}")
    print(f"\nMRR (平均倒数排名): {mrr:.4f}")
    print(f"未命中数: {len(unmatched_cases)}")
    
    # 距离统计
    print("\n" + "=" * 80)
    print("距离统计（匹配歌曲的排名分布）")
    print("=" * 80)
    for case in matched_cases:
        rank = case["matched_rank"]
        print(f"Rank-{rank}: {case['query_song']} → {case['matched_song']} (dist={case['distance']:.4f})")
    
    # 按排名统计距离
    print("\n平均距离（按排名）:")
    for rank in range(1, 6):
        if f"Rank-{rank}" in distance_stats:
            distances = distance_stats[f"Rank-{rank}"]
            print(f"  Rank-{rank}: mean={np.mean(distances):.4f}, std={np.std(distances):.4f}, min={np.min(distances):.4f}, max={np.max(distances):.4f}")
    
    # 保存详细结果
    report_data = {
        "summary": {
            "total_queries": total_queries,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "precision_at_1": precision_at_1,
            "precision_at_3": precision_at_3,
            "precision_at_5": precision_at_5,
            "mrr": mrr,
            "unmatched_count": len(unmatched_cases),
        },
        "matched_cases": matched_cases,
        "unmatched_cases": unmatched_cases,
    }
    
    report_file = os.path.join(output_dir, "evaluation_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已保存: {report_file}")
    
    # 保存为Excel便于查看
    excel_file = os.path.join(output_dir, "evaluation_report.xlsx")
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # 汇总sheet
        summary_df = pd.DataFrame([report_data["summary"]])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 匹配案例
        if matched_cases:
            matched_df = pd.DataFrame(matched_cases)
            matched_df.to_excel(writer, sheet_name='Matched', index=False)
        
        # 未匹配案例
        if unmatched_cases:
            unmatched_df = pd.DataFrame(unmatched_cases)
            unmatched_df.to_excel(writer, sheet_name='Unmatched', index=False)
    
    print(f"Excel报告已保存: {excel_file}")
    
    # 打印未匹配案例示例
    if unmatched_cases:
        print("\n" + "=" * 80)
        print(f"未匹配案例示例（前10个）")
        print("=" * 80)
        for i, case in enumerate(unmatched_cases[:10], 1):
            print(f"\n{i}. 查询歌曲: {case['query_song']} - {case['query_artist']}")
            print(f"   Top-5结果:")
            for r in case["top5_results"]:
                print(f"     #{r['rank']} {r['song']} (dist={r['distance']:.4f})")
    
    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, 
                       default="/home/zjw524/projects/clews_data/batch_test_results/search_results.json")
    parser.add_argument("--output_dir", type=str,
                       default="/home/zjw524/projects/clews_data/batch_test_results")
    args = parser.parse_args()
    
    evaluate_results(args.results_file, args.output_dir)
