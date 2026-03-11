"""
阶段一：提取歌曲库向量（支持多进程并行）
用法：
  python extract_library_embeddings.py --gpu_id 0 --worker_id 0 --workers_per_gpu 12 --total_gpus 2

  通常由 run_extract.sh 自动启动所有进程，无需手动调用。

参数说明：
  --gpu_id:          使用哪张 GPU (0 或 1)
  --worker_id:       当前进程在该 GPU 上的编号 (0 ~ workers_per_gpu-1)
  --workers_per_gpu: 每张 GPU 上的并行进程数
  --total_gpus:      总共使用几张 GPU

输出：
  clews_data/library_embeddings/ 目录下，每首歌一个 .pt 文件
  每个 .pt 文件包含 dict: {"shingles": (S, C), "mean": (C,), "path": str}
"""

import sys
import os
import hashlib
import argparse
import importlib
import time
import torch
from omegaconf import OmegaConf

from utils import pytorch_utils, audio_utils

CLEWS_DATA = "/home/zjw524/projects/clews_data"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="pretrained_models/dvi-clews/checkpoint_best.ckpt")
parser.add_argument("--file_list", type=str, default="data/library_paths.txt")
parser.add_argument("--output_dir", type=str, default=os.path.join(CLEWS_DATA, "library_embeddings"))
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID (0 or 1)")
parser.add_argument("--worker_id", type=int, default=0, help="Worker ID on this GPU (0 ~ workers_per_gpu-1)")
parser.add_argument("--workers_per_gpu", type=int, default=12, help="Number of parallel workers per GPU")
parser.add_argument("--total_gpus", type=int, default=2, help="Total number of GPUs")
parser.add_argument("--hop_size", type=float, default=20.0, help="Shingle hop in seconds")
parser.add_argument("--win_len", type=float, default=-1, help="Shingle length (-1=model default)")
parser.add_argument("--max_audio_len", type=float, default=600, help="Max audio length in seconds")
parser.add_argument("--log_interval", type=int, default=100, help="Log interval (number of files)")
args = parser.parse_args()

if args.win_len <= 0:
    args.win_len = None

# 全局进程编号
global_worker_id = args.gpu_id * args.workers_per_gpu + args.worker_id
total_workers = args.total_gpus * args.workers_per_gpu
tag = f"[GPU{args.gpu_id}-W{args.worker_id}]"

###############################################################################
# 读取文件列表并分片（每个进程只处理属于自己的那部分）
###############################################################################

print(f"{tag} Reading file list: {args.file_list}", flush=True)
with open(args.file_list, "r") as f:
    all_paths = [line.strip() for line in f if line.strip()]

total_files = len(all_paths)
# 交错分片：worker 0 取 idx 0, 24, 48, ...; worker 1 取 idx 1, 25, 49, ...
# 这样每个 worker 的文件分散在列表各处，避免某些 worker 集中处理长音频
my_paths = all_paths[global_worker_id::total_workers]
print(f"{tag} Total files: {total_files}, my share: {len(my_paths)} "
      f"(global_worker={global_worker_id}/{total_workers})", flush=True)

###############################################################################
# 加载模型
###############################################################################

device = torch.device(f"cuda:{args.gpu_id}")
torch.cuda.set_device(device)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

print(f"{tag} Loading model...", flush=True)
path_checkpoint, _ = os.path.split(args.checkpoint)
conf = OmegaConf.load(os.path.join(path_checkpoint, "configuration.yaml"))

module = importlib.import_module("models." + conf.model.name)
model = module.Model(conf.model, sr=conf.data.samplerate).to(device)

# 加载权重
state_dict = torch.load(args.checkpoint, map_location=device)
# 兼容 Fabric 保存的 checkpoint 格式
if "model" in state_dict:
    model_state = state_dict["model"]
else:
    model_state = state_dict
# 去掉可能的 '_forward_module.' 前缀
cleaned_state = {}
for k, v in model_state.items():
    new_key = k.replace("_forward_module.", "")
    cleaned_state[new_key] = v
model.load_state_dict(cleaned_state, strict=False)
model.eval()
print(f"{tag} Model loaded. GPU mem: {torch.cuda.memory_allocated(device)/1024**2:.0f} MB", flush=True)

###############################################################################
# 提取向量
###############################################################################

os.makedirs(args.output_dir, exist_ok=True)


def get_output_path(audio_path, output_dir):
    """根据音频路径生成唯一的输出 .pt 文件路径"""
    path_hash = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    return os.path.join(output_dir, f"{path_hash}.pt")


success_count = 0
skip_count = 0
fail_count = 0
t_start = time.time()

with torch.inference_mode():
    for idx, audio_path in enumerate(my_paths):
        out_path = get_output_path(audio_path, args.output_dir)

        # 跳过已提取的文件（支持断点续提）
        if os.path.exists(out_path):
            skip_count += 1
            continue

        try:
            # 加载音频
            x = audio_utils.load_audio(audio_path, sample_rate=model.sr, n_channels=1)
            if x is None:
                fail_count += 1
                continue

            # 限制最大长度
            max_samples = int(args.max_audio_len * model.sr)
            if x.size(1) > max_samples:
                x = x[:, :max_samples]

            # 送入 GPU
            x = x.to(device)

            # 提取 shingle 向量 (1, S, C)
            z = model(x, shingle_hop=args.hop_size, shingle_len=args.win_len)
            z = z.squeeze(0).cpu()  # (S, C)

            # 计算均值向量
            z_mean = z.mean(dim=0)  # (C,)

            # 保存
            torch.save({
                "shingles": z.half(),   # 用 fp16 节省空间
                "mean": z_mean.half(),  # 均值向量
                "path": audio_path,
            }, out_path)

            success_count += 1

        except Exception as e:
            fail_count += 1
            if fail_count <= 10:
                print(f"{tag} Error processing {audio_path}: {e}", flush=True)

        # 定期打印统计
        processed = success_count + skip_count + fail_count
        if processed > 0 and processed % args.log_interval == 0:
            elapsed = time.time() - t_start
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = len(my_paths) - (idx + 1)
            eta_h = remaining / max(speed, 0.01) / 3600
            print(
                f"{tag} {idx+1}/{len(my_paths)} | "
                f"OK:{success_count} Skip:{skip_count} Fail:{fail_count} | "
                f"{speed:.1f} files/s | ETA: {eta_h:.1f}h",
                flush=True,
            )

elapsed = time.time() - t_start
print(f"{tag} Done! OK:{success_count} Skip:{skip_count} Fail:{fail_count} | "
      f"Time: {elapsed/3600:.2f}h", flush=True)
