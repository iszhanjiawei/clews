#!/bin/bash
# ============================================================================
# 双GPU × 多进程 并行提取歌曲库向量
# 用法：bash run_extract.sh
#
# 配置：2张GPU，每卡10个进程，共20个进程并行（保守配置，避免显存溢出）
# 每个进程独立日志文件，方便实时查看
# ============================================================================

PYTHON="/home/zjw524/anaconda3/envs/clews/bin/python"
PROJECT_DIR="/home/zjw524/projects/clews"
LOG_DIR="/home/zjw524/projects/clews_data/logs/extract"
WORKERS_PER_GPU=10
TOTAL_GPUS=2

mkdir -p ${LOG_DIR}

echo "============================================================"
echo "  Embedding Extraction - ${TOTAL_GPUS} GPUs × ${WORKERS_PER_GPU} workers"
echo "  Total parallel processes: $((TOTAL_GPUS * WORKERS_PER_GPU))"
echo "  Log dir: ${LOG_DIR}"
echo "============================================================"

# 启动所有进程
for gpu_id in $(seq 0 $((TOTAL_GPUS - 1))); do
    for worker_id in $(seq 0 $((WORKERS_PER_GPU - 1))); do
        LOG_FILE="${LOG_DIR}/gpu${gpu_id}_w${worker_id}.log"

        nohup ${PYTHON} -u ${PROJECT_DIR}/extract_library_embeddings.py \
            --gpu_id ${gpu_id} \
            --worker_id ${worker_id} \
            --workers_per_gpu ${WORKERS_PER_GPU} \
            --total_gpus ${TOTAL_GPUS} \
            --checkpoint pretrained_models/dvi-clews/checkpoint_best.ckpt \
            --file_list data/library_paths.txt \
            --hop_size 20.0 \
            --max_audio_len 600 \
            --log_interval 100 \
            > ${LOG_FILE} 2>&1 &

        echo "  Started GPU${gpu_id}-W${worker_id} (PID=$!) -> ${LOG_FILE}"
    done
done

echo "============================================================"
echo "  All $((TOTAL_GPUS * WORKERS_PER_GPU)) processes launched!"
echo ""
echo "  Monitor commands:"
echo "    # 查看所有进程状态"
echo "    ps aux | grep extract_library | grep -v grep"
echo ""
echo "    # 实时查看某个进程日志（例如 GPU0 的 worker 0）"
echo "    tail -f ${LOG_DIR}/gpu0_w0.log"
echo ""
echo "    # 一次性查看所有进程的最新进度"
echo "    tail -n1 ${LOG_DIR}/gpu*.log"
echo ""
echo "    # 查看已提取的文件总数"
echo "    ls /home/zjw524/projects/clews_data/library_embeddings/ | wc -l"
echo ""
echo "    # 查看 GPU 显存占用"
echo "    nvidia-smi"
echo "============================================================"
