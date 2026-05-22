import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Echo-Forcing multishot prompts and save full/shot videos by source JSON index."
    )
    parser.add_argument("--checkpoint_path", default=None, help="Model checkpoint path passed to inference.py")
    parser.add_argument("--model_path", default=None, help="Alias for --checkpoint_path")
    parser.add_argument(
        "--wan_model_path",
        default="wan_models/Wan2.1-T2V-1.3B",
        help="Path to Wan2.1-T2V-1.3B base model directory",
    )
    parser.add_argument("--config_path", default="configs/self_forcing_dmd.yaml", help="Config path passed to inference.py")
    parser.add_argument(
        "--prompts_path",
        default="prompts/eval_caption_multishot_t2v_100_echo.txt",
        help="Prompt txt file readable by inference.py",
    )
    parser.add_argument(
        "--metadata_path",
        default="prompts/eval_caption_multishot_t2v_100_echo_meta.json",
        help="Metadata mapping prompt rows to original JSON indexes and shot durations",
    )
    parser.add_argument("--output_root", default="output/multishot_t2v_100", help="Final output root")
    parser.add_argument("--raw_output_folder", default=None, help="Raw inference output folder")
    parser.add_argument("--start_idx", type=int, default=0, help="Prompt row start index, inclusive")
    parser.add_argument("--end_idx", type=int, default=None, help="Prompt row end index, exclusive")
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to inference.py")
    parser.add_argument("--fps", type=float, default=16.0, help="Fallback fps for writing split clips")
    parser.add_argument("--generation_fps", type=float, default=16.0, help="FPS used by Echo-Forcing duration parsing")
    parser.add_argument("--temporal_compression", type=int, default=4, help="Latent-to-video temporal compression")
    parser.add_argument("--num_frame_per_block", type=int, default=3, help="Echo-Forcing latent frames per generation block")
    parser.add_argument("--python_executable", default=sys.executable, help="Python executable used for inference.py")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for inference. Uses torchrun when > 1")
    parser.add_argument("--torchrun_executable", default="torchrun", help="torchrun executable used when --num_gpus > 1")
    parser.add_argument("--skip_inference", action="store_true", help="Only organize and split existing raw videos")
    parser.add_argument("--keep_raw", action="store_true", help="Keep raw inference files after organizing")
    parser.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload output_root to Hugging Face after generation",
    )
    parser.add_argument("--hf_repo_id", default="Orannue/multishot_long_video", help="Hugging Face repo id")
    parser.add_argument("--hf_repo_type", default="model", choices=["model", "dataset", "space"], help="Hugging Face repo type")
    parser.add_argument(
        "--hf_path_in_repo",
        default=None,
        help="Destination path inside the Hugging Face repo. Defaults to output_root name.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token. If omitted, uses HF_TOKEN/HUGGINGFACE_HUB_TOKEN or local login.",
    )
    args = parser.parse_args()
    args.checkpoint_path = args.checkpoint_path or args.model_path
    if not args.checkpoint_path and not args.skip_inference:
        parser.error("--checkpoint_path or --model_path is required unless --skip_inference is used")
    if args.num_gpus < 1:
        parser.error("--num_gpus must be >= 1")
    return args


def load_metadata(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        metadata = json.load(f)
    return {int(item["line_idx"]): item for item in metadata}


def run_inference(args, raw_output_folder, inference_end_idx):
    env = os.environ.copy()
    env["WAN_MODEL_PATH"] = args.wan_model_path
    launcher = (
        [args.torchrun_executable, "--standalone", "--nproc_per_node", str(args.num_gpus)]
        if args.num_gpus > 1
        else [args.python_executable]
    )
    cmd = launcher + [
        "inference.py",
        "--config_path",
        args.config_path,
        "--checkpoint_path",
        args.checkpoint_path,
        "--output_folder",
        str(raw_output_folder),
        "--data_path",
        args.prompts_path,
        "--use_ema",
        "--save_with_index",
        "--seed",
        str(args.seed),
        "--start_idx",
        str(args.start_idx),
        "--end_idx",
        str(inference_end_idx),
    ]

    print("Running:", " ".join(cmd), flush=True)
    print(f"WAN_MODEL_PATH={args.wan_model_path}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def upload_outputs(args, output_root):
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import HfApi

    path_in_repo = args.hf_path_in_repo or output_root.name
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print(
        f"Uploading {output_root} to hf://{args.hf_repo_id}/{path_in_repo}",
        flush=True,
    )
    api = HfApi()
    api.upload_folder(
        folder_path=str(output_root),
        path_in_repo=path_in_repo,
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        token=token,
    )
    print("Upload completed", flush=True)


def find_raw_video(raw_output_folder, line_idx, seed):
    expected = raw_output_folder / f"{line_idx}-{seed}_ema.mp4"
    if expected.exists():
        return expected

    candidates = sorted(raw_output_folder.glob(f"{line_idx}-*_ema.mp4"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Could not find raw video for prompt row {line_idx} in {raw_output_folder}")


def seconds_to_blocks(duration_seconds, generation_fps, temporal_compression, num_frame_per_block):
    block_seconds = temporal_compression * num_frame_per_block
    return max(1, int((float(duration_seconds) * generation_fps) / block_seconds))


def total_latent_frames_from_duration(total_duration, generation_fps, temporal_compression, num_frame_per_block):
    total_output_frames = int(float(total_duration) * generation_fps)
    base_latent_frames = total_output_frames // temporal_compression
    return max(num_frame_per_block, math.ceil(base_latent_frames / num_frame_per_block) * num_frame_per_block)


def actual_switch_latent_frames(durations, generation_fps, temporal_compression, num_frame_per_block):
    block_counts = [
        seconds_to_blocks(duration, generation_fps, temporal_compression, num_frame_per_block)
        for duration in durations
    ]
    switch_latents = []
    cumulative_blocks = 0
    for block_count in block_counts[:-1]:
        cumulative_blocks += block_count
        switch_latents.append(cumulative_blocks * num_frame_per_block)

    total_latents = total_latent_frames_from_duration(
        sum(durations),
        generation_fps,
        temporal_compression,
        num_frame_per_block,
    )
    return switch_latents, total_latents, block_counts


def split_video(full_path, target_dir, durations, args):
    from torchvision.io import read_video, write_video

    video, _, info = read_video(str(full_path), pts_unit="sec")
    if video.numel() == 0:
        raise RuntimeError(f"Could not read frames from {full_path}")

    fps = float(info.get("video_fps") or args.fps)
    write_fps = int(round(fps))
    total_frames = int(video.shape[0])
    if sum(durations) <= 0:
        raise ValueError(f"Invalid shot durations for {full_path}: {durations}")

    switch_latents, total_latents, block_counts = actual_switch_latent_frames(
        durations,
        args.generation_fps,
        args.temporal_compression,
        args.num_frame_per_block,
    )

    boundaries = [0]
    for latent_frame in switch_latents:
        boundaries.append(round((latent_frame / total_latents) * total_frames))
    boundaries.append(total_frames)

    for shot_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        start = max(0, min(start, total_frames - 1))
        end = max(start + 1, min(end, total_frames))
        write_video(str(target_dir / f"shot{shot_idx}.mp4"), video[start:end], fps=write_fps)

    split_info = {
        "full_video": str(full_path),
        "video_total_frames": total_frames,
        "video_fps": fps,
        "write_fps": write_fps,
        "durations_seconds": durations,
        "block_counts_from_code": block_counts,
        "switch_latent_frames_from_code": switch_latents,
        "total_latent_frames_from_code": total_latents,
        "video_frame_boundaries": boundaries,
    }
    with (target_dir / "split_points.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    raw_output_folder = Path(args.raw_output_folder) if args.raw_output_folder else output_root / "_raw"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_output_folder.mkdir(parents=True, exist_ok=True)

    metadata_by_line = load_metadata(args.metadata_path)
    line_indexes = sorted(metadata_by_line)
    selected = [idx for idx in line_indexes if idx >= args.start_idx and (args.end_idx is None or idx < args.end_idx)]
    inference_end_idx = args.end_idx if args.end_idx is not None else (max(line_indexes) + 1)

    if not args.skip_inference:
        run_inference(args, raw_output_folder, inference_end_idx)

    for line_idx in selected:
        item = metadata_by_line[line_idx]
        json_index = int(item["json_index"])
        target_dir = output_root / f"video{json_index}"
        target_dir.mkdir(parents=True, exist_ok=True)

        raw_video = find_raw_video(raw_output_folder, line_idx, args.seed)
        full_path = target_dir / "full.mp4"
        shutil.copy2(raw_video, full_path)
        split_video(full_path, target_dir, item["durations_seconds"], args)
        print(f"Saved {target_dir}", flush=True)

    if not args.keep_raw and not args.skip_inference:
        shutil.rmtree(raw_output_folder, ignore_errors=True)

    if args.upload:
        upload_outputs(args, output_root)


if __name__ == "__main__":
    main()
