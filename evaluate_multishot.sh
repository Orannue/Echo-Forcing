set -euo pipefail

# Change these paths to your real generated videos root, manifest, and model cache.
RESULT_ROOT="demo/infer/eval_caption_multishot_t2v_100"
MANIFEST="../FAR-Dev2/assets/data/meta/vbench/Vbench_multishot_manifest.json"
OUTPUT_DIR="demo/infer/eval_caption_multishot_t2v_100"
DEVICE="cuda"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_CACHE_DIR="${VBENCH_CACHE_DIR:-$SCRIPT_DIR/../FAR-Dev2/experiments/pretrained_models/vbench}"
export VBENCH_CACHE_DIR="$(cd "$MODEL_CACHE_DIR" && pwd)"

echo "VBENCH_CACHE_DIR=$VBENCH_CACHE_DIR"
ls "$VBENCH_CACHE_DIR"

python evaluate_multishot.py \
  --result-root "$RESULT_ROOT" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --metrics overall_quality shot_structure intra_shot_quality inter_shot_quality \
  --text-alignment-metric overall_consistency \
  --overall-quality-dimensions aesthetic_quality dynamic_degree \
  --intra-shot-quality-dimensions subject_consistency background_consistency \
  --sca-detector transnetv2 \
  --sca-threshold 0.5 \
  --sca-min-gap-sec 0.35 \
  --character-frame-strategy middle \
  --continue-on-error \
  --cache-dir "$VBENCH_CACHE_DIR" \
  --load-ckpt-from-local \
  --transnetv2-weights "$VBENCH_CACHE_DIR/transnetv2/transnetv2-pytorch-weights.pth"