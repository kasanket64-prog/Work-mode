"""Order-balanced scheduling comparison using the real video CLI in fresh processes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs, reference = [], None
    for index, interval in enumerate((1, 3, 3, 1), 1):
        output = args.output / f"run-{index}-every-{interval}.jsonl"
        command = [
            sys.executable,
            "-m",
            "camera_system.video_cli",
            "--video",
            str(args.video),
            "--detector-config",
            "configs/detector-office.toml",
            "--config",
            "configs/camera.toml",
            "--stride",
            "7",
            "--max-frames",
            "12",
            "--relation-every",
            str(interval),
            "--include-preview",
            "--analyze-people",
            "--output",
            str(output),
        ]
        with (args.output / f"run-{index}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        frames = [r for r in rows if r["type"] == "frame"]
        if rows[-1]["status"] != "completed" or len(frames) != 12:
            raise RuntimeError("incomplete benchmark run")
        signature = [(f["frame_index"], f["regions"]) for f in frames]
        if reference is None:
            reference = signature
        for i, frame in enumerate(frames):
            if i % interval and (
                frame["relation_skip_reason"] != "scheduled_skip"
                or frame["relations"]
                or frame["confirmed_relations"]
            ):
                raise RuntimeError("scheduled frame incorrectly contains relationship evidence")
        result = {
            "run": index,
            "relation_every": interval,
            "mean_frame_seconds": mean(f["inference_seconds"] for f in frames),
            "warm_mean_seconds": mean(f["inference_seconds"] for f in frames[3:]),
            "total_seconds_including_startup": rows[-1]["elapsed_seconds"],
            "identical_detections_and_tracks": signature == reference,
            "relation_calls": sum(f["relation_skip_reason"] is None for f in frames),
        }
        runs.append(result)
        print(json.dumps(result), flush=True)
    averages = {
        n: mean(r["warm_mean_seconds"] for r in runs if r["relation_every"] == n) for n in (1, 3)
    }
    result = {
        "runs": runs,
        "warm_means": averages,
        "warm_time_reduction_fraction": 1 - averages[3] / averages[1],
        "scope": "One office clip; 12 sampled frames per run; order 1,3,3,1; first 3 frames excluded from warm averages. No accuracy benchmark.",
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Office scheduling benchmark",
        "",
        result["scope"],
        "",
        "| Run | Relationship interval | Warm seconds/frame | Same detection/tracking |",
        "| --- | --- | --- | --- |",
    ]
    for r in runs:
        lines.append(
            f"| {r['run']} | {r['relation_every']} | {r['warm_mean_seconds']:.3f} | {r['identical_detections_and_tracks']} |"
        )
    lines += [
        "",
        f"Mean warm time: every frame {averages[1]:.3f}s; every third frame {averages[3]:.3f}s.",
        f"Measured warm-time reduction: {result['warm_time_reduction_fraction']:.1%}.",
        "Short, single-machine experiment. Scheduling reduces temporal coverage; unchanged boxes do not prove unchanged event accuracy.",
    ]
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
