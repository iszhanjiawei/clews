#!/bin/bash
# ============================================================================
# 批量测试翻唱歌曲检索系统
# 功能：
#   1. 从xlsx随机抽取500首翻唱歌曲
#   2. 双GPU多进程并行提取查询向量（每卡5个进程，共10进程）
#   3. 批量检索Top5结果
#   4. 生成详细评估报告
# ============================================================================

PYTHON="/home/zjw524/anaconda3/envs/clews/bin/python"
PROJECT_DIR="/home/zjw524/projects/clews"
LOG_DIR="/home/zjw524/projects/clews_data/batch_test_results/logs"
WORKERS_PER_GPU=5
TOTAL_GPUS=2
TOTAL_WORKERS=$((TOTAL_GPUS * WORKERS_PER_GPU))

mkdir -p ${LOG_DIR}

echo "============================================================"
echo "  批量测试翻唱歌曲检索系统"
echo "  测试样本: 500首翻唱歌曲"
echo "  提取向量: ${TOTAL_GPUS} GPUs × ${WORKERS_PER_GPU} workers = ${TOTAL_WORKERS} 并行进程"
echo "============================================================"

# ============================================================================
# 阶段1: 多进程并行提取查询向量
# ============================================================================

echo ""
echo "阶段1: 提取查询向量（${TOTAL_WORKERS}进程并行）"
echo "------------------------------------------------------------"

for gpu_id in $(seq 0 $((TOTAL_GPUS - 1))); do
    for worker_id in $(seq 0 $((WORKERS_PER_GPU - 1))); do
        global_worker_id=$((gpu_id * WORKERS_PER_GPU + worker_id))
        LOG_FILE="${LOG_DIR}/extract_gpu${gpu_id}_w${worker_id}.log"

        nohup ${PYTHON} -u ${PROJECT_DIR}/batch_test_cover_songs.py \
            --gpu_id ${gpu_id} \
            --worker_id ${global_worker_id} \
            --total_workers ${TOTAL_WORKERS} \
            --sample_size 500 \
            > ${LOG_FILE} 2>&1 &

        echo "  启动 GPU${gpu_id}-Worker${worker_id} (PID=$!) -> ${LOG_FILE}"
    done
done

echo ""
echo "等待所有提取进程完成..."
wait

echo "查询向量提取完成!"

# ============================================================================
# 阶段2: 批量检索
# ============================================================================

echo ""
echo "阶段2: 批量检索（Top-5）"
echo "------------------------------------------------------------"

${PYTHON} -u ${PROJECT_DIR}/batch_search_cover_songs.py \
    --gpu_id 0 \
    --top_k 5 \
    --nprobe 64 \
    > ${LOG_DIR}/search.log 2>&1

echo "检索完成!"

# ============================================================================
# 阶段3: 生成评估报告
# ============================================================================

echo ""
echo "阶段3: 生成评估报告"
echo "------------------------------------------------------------"

${PYTHON} -u ${PROJECT_DIR}/evaluate_batch_test.py \
    > ${LOG_DIR}/evaluate.log 2>&1

echo ""
echo "============================================================"
echo "  批量测试完成!"
echo "  结果目录: /home/zjw524/projects/clews_data/batch_test_results/"
echo "  查看报告:"
echo "    - evaluation_report.json  (详细JSON报告)"
echo "    - evaluation_report.xlsx  (Excel报告)"
echo "    - test_samples.csv        (测试样本列表)"
echo "    - search_results.json     (所有检索结果)"
echo "============================================================"
