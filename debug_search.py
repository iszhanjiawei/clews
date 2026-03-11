#!/usr/bin/env python3
import os
import torch
import numpy as np
import faiss
import importlib
from omegaconf import OmegaConf

# 加载模型
checkpoint_path = "pretrained_models/dvi-clews/checkpoint_best.ckpt"
path_checkpoint, _ = os.path.split(checkpoint_path)
conf = OmegaConf.load(os.path.join(path_checkpoint, "configuration.yaml"))

module = importlib.import_module("models." + conf.model.name)
model = module.Model(conf.model, sr=conf.data.samplerate).cuda()

state_dict = torch.load(checkpoint_path, map_location='cuda:0', weights_only=False)
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

# 加载FAISS索引
index = faiss.read_index("/home/zjw524/projects/clews_data/faiss_index/coarse.index")
index.nprobe = 64
song_meta = torch.load("/home/zjw524/projects/clews_data/faiss_index/song_meta.pt", map_location="cpu", weights_only=False)
print(f"FAISS索引加载完成: {index.ntotal} 向量")

# 加载一个查询
query_file = "/home/zjw524/projects/clews_data/batch_test_results/query_embeddings_worker0.pt"
queries = torch.load(query_file, map_location='cpu', weights_only=False)
task_id = list(queries.keys())[0]
query_data = queries[task_id]

print(f"\n查询歌曲: {query_data['song_name']} - {query_data['artist']}")
print(f"Shingles shape: {query_data['shingles'].shape}")

query_shingles = torch.from_numpy(query_data["shingles"]).cuda()
query_mean = torch.from_numpy(query_data["mean"]).unsqueeze(0)

# FAISS粗检索
print("\n执行FAISS粗检索...")
D, I = index.search(query_mean.numpy(), 100)
print(f"粗检索返回: {len(I[0])} 个候选")
print(f"候选ID前10个: {I[0][:10]}")
print(f"距离前10个: {D[0][:10]}")

# 精排前3个候选
print("\n开始精排前3个候选...")
for i, cand_id in enumerate(I[0][:3]):
    if cand_id == -1:
        continue
    
    print(f"\n候选 #{i+1} (ID={cand_id}):")
    cand_meta = song_meta[int(cand_id)]
    print(f"  文件: {cand_meta['emb_file']}")
    
    emb_file = os.path.join("/home/zjw524/projects/clews_data/library_embeddings", cand_meta['emb_file'])
    cand_data = torch.load(emb_file, map_location='cpu', weights_only=False)
    print(f"  歌曲路径: {cand_data['path']}")
    
    cand_shingles = cand_data["shingles"]
    if isinstance(cand_shingles, np.ndarray):
        cand_shingles = torch.from_numpy(cand_shingles)
    cand_shingles = cand_shingles.cuda()
    
    print(f"  候选 shingles shape: {cand_shingles.shape}")
    
    # 计算距离
    try:
        with torch.inference_mode():
            dist = model.distances(query_shingles.unsqueeze(0), cand_shingles.unsqueeze(0))
            print(f"  distances() 返回: {dist}")
            print(f"  距离值: {dist.item()}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
