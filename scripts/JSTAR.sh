#!/bin/bash

# === Config ===
conda activate hsi-nerf

DATA_PATH="/data/complex_facility/transforms.json"
OUTPUT_BASE="./outputs/"


METHODS=("hsi_mipnerf_L2" "hsi_mipnerf" "hsi_mipnerf_MD" "hsi_mipnerf_GR" "hsi_mipnerf_MD_GR" "hsi_mipnerf_MD_GR_L2")
SPLITS=(20 30 40 50 75 100)
REPEATS=(0 1 2 3 4)
GPU_LIST=(0 1 2 4 5)
CORES_PER_GPU=42

JOB_FILE="/tmp/job_queue_$$.txt"
LOCK_FILE="/tmp/job_queue_$$.lock"

# === Cleanup ===
trap "echo '🛑 Ctrl+C received. Killing all...'; kill 0; rm -f $JOB_FILE $LOCK_FILE; exit 1" SIGINT
rm -f "$JOB_FILE" "$LOCK_FILE"

# === Build job list ===
for METHOD in "${METHODS[@]}"; do
  for SPLIT in "${SPLITS[@]}"; do
    for SEED in "${REPEATS[@]}"; do
      echo "$METHOD|$SPLIT|$SEED" >> "$JOB_FILE"
    done
  done
done

echo "📋 Job file created at $JOB_FILE with $(wc -l < "$JOB_FILE") jobs"
echo "🚀 Launching ${#GPU_LIST[@]} GPU workers..."

gpu_worker() {
  local GPU_ID=$1
  local DELAY=$((GPU_ID * 2))  # startup stagger
  sleep "$DELAY"

  local CPU_START=$((GPU_ID * CORES_PER_GPU))
  local CPU_END=$((CPU_START + CORES_PER_GPU - 1))
  local CPU_RANGE="${CPU_START}-${CPU_END}"

  while true; do
    local JOB=""
    
    # Lock and claim a job
    {
      flock -x 200

      if [[ -s "$JOB_FILE" ]]; then
        JOB=$(tail -n 1 "$JOB_FILE")
        head -n -1 "$JOB_FILE" > "${JOB_FILE}.tmp" && mv "${JOB_FILE}.tmp" "$JOB_FILE"
        echo "$(date) 🧠 [GPU $GPU_ID] Claimed job: $JOB"
      fi
    } 200>"$LOCK_FILE"

    # === No job left ===
    if [[ -z "$JOB" ]]; then
      echo "$(date) 💤 [GPU $GPU_ID] No more jobs. Exiting..."
      break
    fi

    IFS="|" read -r METHOD SPLIT SEED <<< "$JOB"
    TIMESTAMP="${SPLIT}_${SEED}"
    EXP_DIR="${OUTPUT_BASE}/complex_facility/${METHOD}/${TIMESTAMP}"
    METRICS_PATH="${EXP_DIR}/hsi-eval/best/metrics.json"
    LOG_FILE="${EXP_DIR}/train.log"

    # Skip if already done
    if [[ -f "$METRICS_PATH" ]]; then
      echo "$(date) ✅ [GPU $GPU_ID] Already complete: $METHOD $SPLIT $SEED"
      continue
    fi

    mkdir -p "$EXP_DIR"

    {
      echo "========== 🟢 $(date) START TRAINING =========="

      CUDA_VISIBLE_DEVICES=$GPU_ID taskset -c "$CPU_RANGE" hsi-train "$METHOD" \
        --data "$DATA_PATH" \
        --timestamp "$TIMESTAMP" \
        --output-dir "$OUTPUT_BASE" \
        hsi-data \
        --train_split_fraction "$SPLIT" \
        --train_seed "$SEED"

      if [[ $? -ne 0 ]]; then
        echo "❌ Training failed for: $METHOD $SPLIT $SEED"
        continue
      fi

      echo "========== 🔵 $(date) START EVAL =========="

      CUDA_VISIBLE_DEVICES=$GPU_ID taskset -c "$CPU_RANGE" hsi-eval \
        --base-dir "$EXP_DIR"

      if [[ $? -eq 0 ]]; then
        echo "✅ Evaluation complete: $METHOD $SPLIT $SEED"
      else
        echo "❌ Evaluation failed: $METHOD $SPLIT $SEED"
      fi
    } >> "$LOG_FILE" 2>&1
  done
}


# === Launch all GPU workers ===
for GPU in "${GPU_LIST[@]}"; do
  gpu_worker "$GPU" &
done

wait
rm -f "$JOB_FILE" "$LOCK_FILE"
echo "🎉 All ablation jobs completed!"
