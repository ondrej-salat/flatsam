"""Run the refinement pass of FlatSAM on a dataset.

Every frame is rectified with its first-pass pose and both branches run again (paper Sec. 3.4). Reads the first
pass of ``run.py <dataset> --save-masks`` from ``<output dir>/<config>/<dataset>/`` and writes FlatSAM's output to
``<output dir>/warp/<dataset>/<sequence>.txt``. Sequences that are already done are skipped.
"""

import argparse
import sys
import traceback
from pathlib import Path

from flatsam.config import compute_output_dir
from flatsam.datasets import CLI_CHOICES, DATASETS
from flatsam.path_config import DEFAULT_CONFIG, paths

sys.path.insert(0, str(Path(__file__).resolve().parent / "subset_viewer" / "rerun"))
import run_warp_pipeline  # noqa: E402  (the script of the paper's runs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset of the first pass")
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Specific sequences to refine (if not specified, every sequence of the first pass)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Dataset base directory (uses path_config.py default if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(paths.OUTPUT_DIRECTORY),
        help="Top-level output directory, the same as for run.py",
    )
    args = parser.parse_args()

    info = DATASETS[args.dataset]
    first_pass = compute_output_dir(args.output_dir.expanduser(), args.dataset, DEFAULT_CONFIG)
    sequences = args.sequences or sorted(p.name[: -len("_debug.npz")] for p in first_pass.glob("*_debug.npz"))
    if not sequences:
        print(f"ERROR: no first pass in {first_pass}; run `python run.py {args.dataset} --save-masks` first")
        return 1

    # The paper's settings: 3x crop window around the target, never enlarged (Sec. 3.4).
    refine_args = run_warp_pipeline.build_parser().parse_args(
        ["--outroot", str(first_pass.parent.parent / "warp"), "--mode", "canvas", "--max-scale", "1",
         "--first-pass-dir", str(first_pass.parent)]
    )
    failed = []
    for i, seq in enumerate(sequences, 1):
        print(f"[{i}/{len(sequences)}] {info.label}/{seq}")
        try:
            run_warp_pipeline.refine_sequence(refine_args, info.key, info.label, seq, args.base_dir)
        except Exception:
            traceback.print_exc()
            failed.append(seq)
    if failed:
        print(f"FAILED: {len(failed)} sequence(s): {' '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
