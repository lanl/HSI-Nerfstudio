#!/usr/bin/env bash
set -euo pipefail
set +H  # Disable history expansion (!)
trap "echo '🛑 Caught SIGINT. Killing all jobs...'; kill 0; exit 1" SIGINT

############################################
# User-configurable paths
############################################
BASE_CONFIG_DIR="./outputs/complex_facility"
OUT_DIR="./renders"
CAM_PATH="./data/complex_facility/camera_paths/Drone_Path_48fov.json"

# GPUs you said you have access to
GPUS=(0 1 2 3 4 5)

# Naming suffix to match your example filename pattern
NAME_SUFFIX="128x48fov"

# Command prefix (edit if needed)
RENDER_BIN="hsi-render"

############################################
# Seeds (per model, per size)
# Order corresponds to sizes: 100, 50, 40, 20
############################################
SIZES=(100 50 40 20)

# hsi_mipnerf_MD_GR best seeds: 3,4,0,0
# hsi_mipnerf_L2    best seeds: 0,4,4,3
declare -A SEED_MD_GR=(
  [100]=3
  [50]=4
  [40]=0
  [20]=0
)
declare -A SEED_L2=(
  [100]=0
  [50]=4
  [40]=4
  [20]=3
)

############################################
# Helpers
############################################
mkdir -p "$OUT_DIR"

run_render_pair() {
  local gpu="$1"
  local model="$2"
  local size="$3"
  local seed="$4"

  local cfg="${BASE_CONFIG_DIR}/${model}/${size}_${seed}/config.yml"

  local out_false="${OUT_DIR}/${model}_${size}_${seed}_${NAME_SUFFIX}_Falsecolor.mp4"
  local out_ace="${OUT_DIR}/${model}_${size}_${seed}_${NAME_SUFFIX}_ACE.mp4"

  echo "[$(date +'%F %T')] GPU ${gpu} :: ${model} ${size}_${seed} :: Falsecolor -> ${out_false}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${RENDER_BIN}" camera-path \
      --load-config "${cfg}" \
      --rendered-output-names Falsecolor \
      --output-path "${out_false}" \
      --camera-path-filename "${CAM_PATH}" \
      --use-best-model False

  echo "[$(date +'%F %T')] GPU ${gpu} :: ${model} ${size}_${seed} :: ACE -> ${out_ace}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${RENDER_BIN}" camera-path \
      --load-config "${cfg}" \
      --rendered-output-names ACE \
      --output-path "${out_ace}" \
      --camera-path-filename "${CAM_PATH}" \
      --use-best-model False
}

############################################
# Build task list (16 combos = 2 models * 4 sizes * 2 seeds? -> actually 2 models * 4 sizes = 8;
# you said 16 combinations, which implies 2 renders per combo (Falsecolor + ACE) -> total 16 renders.
# This script treats each (model,size,seed) as a "combo" and runs 2 renders for it.
############################################
TASKS=()
for size in "${SIZES[@]}"; do
  TASKS+=("hsi_mipnerf_L2,${size},${SEED_L2[$size]}")
done
for size in "${SIZES[@]}"; do
  TASKS+=("hsi_mipnerf_MD_GR,${size},${SEED_MD_GR[$size]}")
done

############################################
# Simple GPU scheduler: max one combo per GPU at a time
# Requires bash >= 5.1 for `wait -n -p`.
############################################
declare -A PID_TO_GPU=()
FREE_GPUS=("${GPUS[@]}")

on_launch() {
  local pid="$1"
  local gpu="$2"
  PID_TO_GPU["$pid"]="$gpu"
}

on_finish_one() {
  local finished_pid
  wait -n -p finished_pid
  local gpu="${PID_TO_GPU[$finished_pid]}"
  unset PID_TO_GPU["$finished_pid"]
  FREE_GPUS+=("$gpu")
}

for t in "${TASKS[@]}"; do
  IFS=',' read -r model size seed <<<"$t"

  # If no GPU is free, wait for one job to finish
  while ((${#FREE_GPUS[@]} == 0)); do
    on_finish_one
  done

  # Pop one GPU from FREE_GPUS
  gpu="${FREE_GPUS[0]}"
  FREE_GPUS=("${FREE_GPUS[@]:1}")

  # Launch this combo (Falsecolor then ACE) on that GPU
  (
    run_render_pair "$gpu" "$model" "$size" "$seed"
  ) &
  on_launch "$!" "$gpu"
done

# Wait for remaining jobs
while ((${#PID_TO_GPU[@]} > 0)); do
  on_finish_one
done

echo "[$(date +'%F %T')] All renders completed."
