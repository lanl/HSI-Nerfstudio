#!/bin/bash

# === Setup ===
# May need to manually activate your environment
# conda activate hsi-nerf

DATA_PATH="data/complex_facility/transforms.json"
OUTPUT_BASE="benchmark_logs"
MODELS=("hsi_mipnerf_L2" "hsi_mipnerf" "hsi_mipnerf_MD" "hsi_mipnerf_MD_GR")

mkdir -p "$OUTPUT_BASE"

# === Loop over model variants ===
for MODEL in "${MODELS[@]}"; do
  echo "🚀 Starting benchmark for: $MODEL"
  
  LOG_DIR="${OUTPUT_BASE}/${MODEL}"
  mkdir -p "$LOG_DIR"
  LOG_FILE="${LOG_DIR}/train.log"
  GPU_LOG="${LOG_DIR}/gpu_mem.log"
  METRICS_FILE="${LOG_DIR}/summary.txt"

  # Start GPU memory logging (GPU 0 only)
  nvidia-smi --id=0 --query-compute-apps=used_memory --format=csv,nounits -l 1 > "$GPU_LOG" &
  GPU_LOG_PID=$!

  # Start timer
  START=$(date +%s)

  # Run training
  CUDA_VISIBLE_DEVICES=0 hsi-train "$MODEL" \
    --data "$DATA_PATH" \
    --max_num_iterations 30000 \
    --timestamp benchmark \
    hsi-data \
    --train_split_fraction 50 \
    > "$LOG_FILE" 2>&1

  END=$(date +%s)
  DURATION=$((END - START))

  # Stop GPU logger
  kill $GPU_LOG_PID
  wait $GPU_LOG_PID 2>/dev/null

  # Extract max memory in MiB
  PEAK_MEM=$(grep -o '[0-9]\+' "$GPU_LOG" | sort -nr | head -n1)

  # Log final metrics
  {
    echo "Model: $MODEL"
    echo "Duration (sec): $DURATION"
    echo "Peak GPU 0 Memory (MiB): $PEAK_MEM"
  } | tee "$METRICS_FILE"

  echo "✅ Completed: $MODEL — ${DURATION}s, Peak GPU 0: ${PEAK_MEM} MiB"
  echo "----------------------------------------"
done

echo "🎉 All models completed."
