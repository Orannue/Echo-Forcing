#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any



DEFAULT_METRICS = [
    "overall_quality",
    "shot_structure",
    "intra_shot_quality",
    "inter_shot_quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VBench multi-shot results and print cross-sample metric averages."
    )
    parser.add_argument(
        "--result-root",
        required=True,
        help="Root directory containing generated video folders, e.g. demo/infer/eval_caption_multishot_t2v_100.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional multi-shot manifest JSON with shot captions, characters, and target boundaries.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results/multishot",
        help="Directory for raw and summary JSON outputs.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, usually cuda or cpu.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional model cache root. Sets VBENCH_CACHE_DIR before loading VBench models.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        choices=DEFAULT_METRICS,
        help="Metric sections to evaluate.",
    )
    parser.add_argument(
        "--text-alignment-metric",
        default="overall_consistency",
        choices=["overall_consistency", "clip_score"],
        help="VBench metric used for per-shot text alignment.",
    )
    parser.add_argument(
        "--overall-quality-dimensions",
        nargs="+",
        default=["aesthetic_quality", "dynamic_degree"],
        help="VBench dimensions included under overall_quality.",
    )
    parser.add_argument(
        "--intra-shot-quality-dimensions",
        nargs="+",
        default=["subject_consistency", "background_consistency"],
        help="VBench dimensions included under intra_shot_quality.",
    )
    parser.add_argument("--load-ckpt-from-local", action="store_true")
    parser.add_argument("--read-frame", action="store_true")
    parser.add_argument("--keep-vbench-meta", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--sca-detector", default="transnetv2", choices=["transnetv2", "opencv", "scenedetect"])
    parser.add_argument("--sca-threshold", type=float, default=0.5)
    parser.add_argument("--sca-min-gap-sec", type=float, default=0.35)
    parser.add_argument("--sca-tolerance-sec", type=float, default=None)
    parser.add_argument("--sca-unmatched-penalty-frames", type=float, default=None)
    parser.add_argument("--transnetv2-path", default=None)
    parser.add_argument("--transnetv2-weights", default=None)
    parser.add_argument(
        "--character-frame-strategy",
        default="middle",
        choices=["first", "middle", "last"],
    )
    return parser.parse_args()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def add_metric(buckets: dict[str, list[float]], name: str, value: Any) -> None:
    if is_number(value):
        buckets[name].append(float(value))


def collect_metric_averages(results: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[float]] = defaultdict(list)

    for sample_result in results.values():
        if not isinstance(sample_result, dict):
            continue

        overall = sample_result.get("overall_quality", {})
        if isinstance(overall, dict):
            for metric_name, metric_result in overall.items():
                if isinstance(metric_result, dict):
                    add_metric(
                        buckets,
                        f"overall_quality/{metric_name}",
                        metric_result.get("average"),
                    )

        shot_structure = sample_result.get("shot_structure", {})
        if isinstance(shot_structure, dict):
            for metric_name in [
                "sca",
                "boundary_match_rate",
                "cut_precision",
                "cut_recall",
                "cut_count_accuracy",
                "mean_boundary_timing_error_frames",
                "mean_boundary_timing_error_sec",
            ]:
                add_metric(
                    buckets,
                    f"shot_structure/{metric_name}",
                    shot_structure.get(metric_name),
                )

        intra = sample_result.get("intra_shot_quality", {})
        if isinstance(intra, dict):
            for metric_name, metric_result in intra.items():
                if isinstance(metric_result, dict):
                    add_metric(
                        buckets,
                        f"intra_shot_quality/{metric_name}",
                        metric_result.get("average"),
                    )

        inter = sample_result.get("inter_shot_quality", {})
        if isinstance(inter, dict):
            add_metric(
                buckets,
                "inter_shot_quality/character_subject_consistency",
                inter.get("average"),
            )

    return {
        name: {
            "mean": sum(values) / len(values),
            "count": len(values),
        }
        for name, values in sorted(buckets.items())
        if values
    }


def main() -> None:
    args = parse_args()
    if args.cache_dir:
        os.environ["VBENCH_CACHE_DIR"] = str(Path(args.cache_dir).expanduser().resolve())

    from vbench_multishot import VBenchMultishot

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = VBenchMultishot(
        device=args.device,
        output_dir=output_dir,
        text_alignment_metric=args.text_alignment_metric,
        overall_quality_dimensions=args.overall_quality_dimensions,
        intra_shot_quality_dimensions=args.intra_shot_quality_dimensions,
        load_ckpt_from_local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        keep_vbench_meta=args.keep_vbench_meta,
        sca_detector=args.sca_detector,
        sca_tolerance_sec=args.sca_tolerance_sec,
        sca_threshold=args.sca_threshold,
        sca_min_gap_sec=args.sca_min_gap_sec,
        sca_unmatched_penalty_frames=args.sca_unmatched_penalty_frames,
        transnetv2_path=args.transnetv2_path,
        transnetv2_weights=args.transnetv2_weights,
        character_frame_strategy=args.character_frame_strategy,
        continue_on_error=args.continue_on_error,
    )

    results = evaluator.evaluate(
        result_root=args.result_root,
        manifest=args.manifest,
        metrics=args.metrics,
        save_json=True,
    )

    summary = collect_metric_averages(results)
    summary_path = output_dir / "multishot_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Raw results saved to: {output_dir / 'multishot_eval_results.json'}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
