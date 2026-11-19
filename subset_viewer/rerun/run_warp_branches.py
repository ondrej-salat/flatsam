#!/usr/bin/env python3
"""Warped rerun of the two branches only (written 2026-11-13).

Differs from ``run_rerun_variant.py`` in what the second pass sees:

* SAM3 is **not** run on the warped frames. The *original* run's SAM3 mask of every frame (from the pose-gate
  replay) is warped with the same transform as the frame and handed to the tracker (``step_with_mask``). The
  keypoint branch and the mask-line branch are the only things rerun.
* Sampling stays consistent with the original run. The original dilates the SAM3 mask by 10 *pixels*
  (``loma.init_mask_inflate_px`` / ``current_mask_inflate_px``) before sampling LoMa keypoints. Here the current-frame
  mask is dilated by 10 px **in the original frame** and then warped, so the sampling region is exactly the original's.
  The frame-0 mask is the exact GT quad (no 10% expansion, as in the original), and its sampling dilation is 10
  frame-0 pixels expressed in canvas pixels (``round(10 * s)``).
* The mask-line branch gets the warped, undilated mask.

--mode full   : same canvas as the frame, target at its frame-0 size (scale 1).
--mode canvas : square window of WINDOW x the initial target's long side around the initial target, resampled with
                scale s = min(TARGET_LONG / long side, max_scale), so the initial target's long side is TARGET_LONG
                px unless ``--max-scale`` caps it (``--max-scale 1`` never enlarges the target).
The baseline pose and the masks come from the same replay (``POSEGATE/<dataset>``) so they are consistent.
"""
from __future__ import annotations
import argparse, csv, json, traceback
from pathlib import Path
import cv2
import numpy as np

from flatsam2.config import load_config_from_file
from flatsam2.datasets import DATASETS, corners_to_mask, make_gt_loader, make_image_paths, warp_corners
from flatsam2.path_config import DEFAULT_CONFIG, paths
from flatsam2.tracker import FlatSAMTracker

CONFIG = DEFAULT_CONFIG  # demo: the final method (upstream: an experiment config in the author's checkout)
POSEGATE = Path("$WORK/flatsam2/results/experiments/pose_gate_8dataset_20261110/tracker/loma_samh_matchnaming")
COND_LIMIT = 1e10
TARGET_LONG = 384.0
WINDOW = 3.0
INFLATE_PX = 10          # the original run's sampling dilation (pixels)


class NoSegmenter:
    """Stands in for SAM3: the second pass must not segment, so any call is a bug."""
    def init(self, *a, **k): raise RuntimeError("SAM3 must not be initialised in the branch-only rerun")
    def track(self, *a, **k): raise RuntimeError("SAM3 must not run in the branch-only rerun")
    def correct(self, *a, **k): raise RuntimeError("SAM3 correction must not fire in the branch-only rerun")


def load_pose(path):
    v = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if v.shape[1] != 8: raise ValueError(f"Expected eight coordinates per row in {path}")
    return v.reshape(-1, 4, 2).transpose(0, 2, 1)

def corner_error(a, b): return float(np.sqrt(np.sum((a - b) ** 2) / 4.0))

def safe_homography(init, pred):
    if not np.isfinite(pred).all(): return None
    H = cv2.getPerspectiveTransform(init.T.astype(np.float32), pred.T.astype(np.float32)).astype(np.float64)
    if not np.isfinite(H).all() or abs(float(np.linalg.det(H))) <= 1e-10: return None
    cond = float(np.linalg.cond(H))
    if not np.isfinite(cond) or cond >= COND_LIMIT: return None
    proj = cv2.perspectiveTransform(init.T[None].astype(np.float32), H)[0]
    if not np.isfinite(proj).all() or abs(float(cv2.contourArea(proj.astype(np.float32)))) < 16: return None
    return H / H[2, 2] if abs(float(H[2, 2])) > 1e-12 else H

def dilate(mask_u8, px):
    """Same square-kernel dilation as flatsam2.tracking.common._inflate_mask."""
    return cv2.dilate(mask_u8, np.ones((2 * px + 1, 2 * px + 1), np.uint8), iterations=1) if px > 0 else mask_u8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--task-index", type=int, required=True)
    ap.add_argument("--outroot", required=True); ap.add_argument("--mode", choices=["full", "canvas"], required=True)
    ap.add_argument("--max-scale", type=float, default=None, help="canvas only: cap on the target scale (1 = never enlarge)")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    task = list(csv.DictReader(open(args.tasks, newline="")))[args.task_index]
    dataset, label, seq = task["dataset_key"], task["dataset"], task["sequence"]
    outdir = Path(args.outroot) / label; outdir.mkdir(parents=True, exist_ok=True)
    pose_out, sidecar, summary_out = outdir / f"{seq}.txt", outdir / f"{seq}_rectification.npz", outdir / f"{seq}_summary.json"
    if pose_out.exists() and sidecar.exists() and summary_out.exists() and not args.dry_run:
        print(f"Already complete: {label}/{seq}"); return
    info = DATASETS[dataset]
    pg = POSEGATE / label
    baseline = load_pose(pg / f"{seq}.txt")
    mdir = pg / "masks" / seq
    if not mdir.is_dir(): raise ValueError(f"No original SAM3 masks for {label}/{seq}: {mdir}")
    initfix = paths.PT_INITFIX_DIR if dataset == "planartrack-initfix" else None
    gt_loader = make_gt_loader(dataset, initfix_dir=initfix)
    bd = args.base_dir or task.get("base_dir") or None
    base_dir = Path(bd) if bd else Path(getattr(paths, info.base_dir_attr))
    gt, gt_valid = gt_loader(base_dir, seq)
    image_paths = make_image_paths(dataset)(base_dir, seq)
    count = min(len(image_paths), len(gt), len(gt_valid), len(baseline))
    if count < 1 or not np.isfinite(gt[0]).all(): raise ValueError("missing initial GT/prediction")
    init = gt[0].astype(np.float64)
    center = init.mean(axis=1, keepdims=True)
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None: raise ValueError(f"Cannot read initial image {image_paths[0]}")
    h0, w0 = first.shape[:2]
    if args.mode == "canvas":
        long_side = max(float(np.ptp(init[0])), float(np.ptp(init[1])))
        s = TARGET_LONG / long_side
        if args.max_scale is not None: s = min(s, args.max_scale)
        L = WINDOW * long_side
        ox, oy = float(center[0, 0]) - L / 2, float(center[1, 0]) - L / 2
        C = int(round(L * s))
        A = np.array([[s, 0, -s * ox], [0, s, -s * oy], [0, 0, 1.0]]); canvas = (C, C)
    else:
        s = 1.0; A = np.eye(3); canvas = (w0, h0)
    Ainv = np.linalg.inv(A)
    init_c = warp_corners(A, init)
    init_mask = corners_to_mask(init_c, (canvas[1], canvas[0]))           # exact GT quad, no expansion (as the original)
    c2r = np.repeat(np.eye(3)[None], count, axis=0)
    warp_valid = np.zeros(count, bool); warp_valid[0] = True
    metric_valid = np.asarray(gt_valid[:count], bool).copy(); metric_valid[0] = True
    for i in range(1, count):
        H = safe_homography(init, baseline[i])
        if H is None: warp_valid[i] = False; metric_valid[i] = False; continue
        c2r[i] = np.linalg.inv(H); warp_valid[i] = True
        if not (metric_valid[i] and np.isfinite(gt[i]).all()): metric_valid[i] = False
    if args.dry_run:
        print(f"{label}/{seq}: mode={args.mode} frames={count} warp_valid={int(warp_valid.sum())} scale={s:.3f} canvas={canvas} "
              f"init_inflate_px={int(round(INFLATE_PX * s))} masks={len(list(mdir.glob('*.png')))}"); return
    cfg = load_config_from_file(CONFIG)
    # Sampling dilation is applied by hand in the original frame (current frames); frame 0 uses 10 frame-0 pixels in canvas px.
    cfg = cfg.model_copy(update={"loma": cfg.loma.model_copy(update={"init_mask_inflate_px": int(round(INFLATE_PX * s)), "current_mask_inflate_px": 0})})
    tracker = FlatSAMTracker(cfg, fail_hard=True, segmenter=NoSegmenter())
    rect_pred = np.zeros((count, 2, 4)); mapped = np.zeros((count, 2, 4)); rows = []
    for i in range(count):
        frame = cv2.imread(str(image_paths[i]), cv2.IMREAD_COLOR)
        if frame is None: raise ValueError(f"Cannot read frame {i}: {image_paths[i]}")
        M = A if i == 0 else (A @ c2r[i] if warp_valid[i] else A)
        warped = cv2.warpPerspective(frame, M, canvas, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if i == 0:
            tracker.init(warped, init_mask=init_mask, init_corners=init_c, init_segmenter=False); rect_pred[i] = init
        else:
            m = cv2.imread(str(mdir / f"{i + 1:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is None: raise ValueError(f"Missing original SAM3 mask for frame {i}: {mdir / f'{i + 1:05d}.png'}")
            if m.shape != frame.shape[:2]: m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = (m > 0).astype(np.uint8)
            wm = cv2.warpPerspective(m, M, canvas, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            wd = cv2.warpPerspective(dilate(m, INFLATE_PX), M, canvas, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            tracker._loma_densify_mask = wd                      # LoMa samples where the original run sampled
            res = tracker.step_with_mask(warped, wm)             # the mask branch gets the warped, undilated mask
            rect_pred[i] = warp_corners(Ainv, warp_corners(res.homography, init_c))
        forward = np.linalg.inv(c2r[i]) if warp_valid[i] else np.eye(3)
        mapped[i] = warp_corners(forward, rect_pred[i])
        if metric_valid[i] and np.isfinite(gt[i]).all() and warp_valid[i]:
            be, re_ = corner_error(baseline[i], gt[i]), corner_error(mapped[i], gt[i])
            rows.append({"dataset": label, "tier": task["tier"], "sequence": seq, "frame_idx": i, "baseline_pose_error_px": be,
                         "rerun_mapped_pose_error_px": re_, "baseline_p_at_15": int(be <= 15), "rerun_mapped_p_at_15": int(re_ <= 15)})
    np.savetxt(pose_out, mapped.transpose(0, 2, 1).reshape(count, 8), fmt="%.6f")
    np.savez_compressed(sidecar, dataset=np.array(label), sequence=np.array(seq), mode=np.array(args.mode + "_masks"), scale=np.array(s), canvas=np.array(canvas),
                        current_to_reference=c2r, gt_corners=gt[:count], gt_valid=metric_valid, warp_valid=warp_valid,
                        baseline_pose=baseline[:count], rectified_predictions=rect_pred, mapped_predictions=mapped)
    if rows:
        with (outdir / f"{seq}_frame_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    avg = lambda k: float(np.mean([r[k] for r in rows])) if rows else None
    summary = {"dataset": label, "sequence": seq, "tier": task["tier"], "mode": args.mode + "_masks", "scale": s, "canvas": list(canvas), "frames": len(rows),
               "baseline_p_at_15_percent": 100 * avg("baseline_p_at_15") if rows else None,
               "rerun_mapped_p_at_15_percent": 100 * avg("rerun_mapped_p_at_15") if rows else None,
               "warp_valid_frames": int(warp_valid.sum())}
    summary_out.write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary))

if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
