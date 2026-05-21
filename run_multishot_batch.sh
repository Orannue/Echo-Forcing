#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="checkpoints/self_forcing_dmd.pt"
WAN_MODEL_PATH="../models/Wan2.1-T2V-1.3B"
CONFIG_PATH="configs/self_forcing_dmd.yaml"
PROMPTS_PATH="prompts/eval_caption_multishot_t2v_100_echo.txt"
METADATA_PATH="prompts/eval_caption_multishot_t2v_100_echo_meta.json"
OUTPUT_ROOT="output/multishot_eval"

HF_REPO_ID="Orannue/Baseline_results"
HF_REPO_TYPE="dataset"
HF_PATH_IN_REPO="eval_caption_multishot_t2v_100/echo_forcing"
# Set HF_TOKEN in your shell before running, or paste a token here if you prefer.
HF_TOKEN="${HF_TOKEN:-}"

START_IDX=0
END_IDX=100
SEED=0
NUM_GPUS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_multishot_batch.py \
  --model_path "$CHECKPOINT_PATH" \
  --wan_model_path "$WAN_MODEL_PATH" \
  --config_path "$CONFIG_PATH" \
  --prompts_path "$PROMPTS_PATH" \
  --metadata_path "$METADATA_PATH" \
  --output_root "$OUTPUT_ROOT" \
  --start_idx "$START_IDX" \
  --end_idx "$END_IDX" \
  --seed "$SEED" \
  --num_gpus "$NUM_GPUS" \
  --hf_repo_id "$HF_REPO_ID" \
  --hf_repo_type "$HF_REPO_TYPE" \
  --hf_path_in_repo "$HF_PATH_IN_REPO" \
  --hf_token "$HF_TOKEN" \
  --upload
