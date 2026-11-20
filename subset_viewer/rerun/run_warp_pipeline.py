#!/usr/bin/env python3
"""Two-stage warped pipeline, branch-only rerun (written 2026-11-13).

Stage 1, first pass (offline from the stored replay): the per-frame choice between the keypoint (LoMa) pose and the
mask (Hough) pose is made by ``flatsam.gate.select_pose``:
  n >= 35 -> LoMa;  8 < n < 35 and the LoMa quad agrees with the Hough quad on 2 corners within the RANSAC threshold (5 px) -> LoMa;
  else the mask branch.
The two branches are independent of this choice (decoupled), so the stored ``loma_H`` / ``samh_H`` / ``n_inliers`` of the
replay are enough to apply any such rule exactly.

Stage 2, rerun: every frame with a usable first-pass pose is rectified with that pose (the mask pose where stage 1 did not
choose LoMa, e.g. n <= 8, in the hope of more matches in the rerun) and rerun; ``--rectify-with loma`` restricts this to LoMa frames. The rerun runs the two branches only (no SAM3): the original SAM3
mask is warped with the frame and handed to the tracker, the LoMa sampling region is the original mask dilated by 10 px in
the original frame (as in the original run), and the tracker applies the same decision to its own inliers.

With ``--first-pass-dir`` (as ``refine.py`` runs it), stage 1 is the finished first pass of ``run.py`` instead of the replay.
Outputs per sequence: ``<seq>.txt`` (rerun pose where it ran, first-pass pose elsewhere), ``<seq>_rectification.npz``
(first-pass pose, rerun pose, which frames ran, the decision labels and inlier counts of both passes), ``<seq>_summary.json``.
"""
from __future__ import annotations
import argparse, csv, json, time, traceback
from pathlib import Path
import cv2
import numpy as np

from flatsam import gate, runinfo
from flatsam.profiling import profiler
from flatsam.config import load_config_from_file
from flatsam.datasets import DATASETS, corners_to_mask, make_gt_loader, make_image_paths, warp_corners
from flatsam.path_config import paths
from flatsam.tracker import FlatSAMTracker
from run_warp_branches import (CONFIG, INFLATE_PX, POSEGATE, TARGET_LONG, WINDOW, NoSegmenter, corner_error, dilate, load_pose,
                               safe_homography)


def first_pass(dbg_path, init, count, args):
    """Per-frame stage-1 decision from the replay's stored branch homographies.

    Returns the first-pass poses (count, 2, 4), the chosen labels (count,) and the LoMa inlier counts (count,).
    """
    d = np.load(dbg_path, allow_pickle=True)
    idx, lh, sh, ninl = d["frame_idx"].astype(int), d["loma_H"], d["samh_H"], d["n_inliers"].astype(int)
    pose = np.repeat(init[None], count, axis=0); label = np.array(["samh_H"] * count, dtype=object); n_in = np.full(count, -1)
    pose[0] = init; label[0] = "init"
    last = init.copy()
    for j, f in enumerate(idx):
        if not 0 < f < count: continue
        loma_H = lh[j] if np.isfinite(lh[j]).all() else None
        if not np.isfinite(sh[j]).all():                       # no mask pose either: keep the last pose
            pose[f] = last; label[f] = "last"; n_in[f] = ninl[j]; continue
        lab, H = gate.select_pose(loma_H, int(ninl[j]), sh[j], init, inlier_threshold=args.inlier_threshold,
                                  agree_min_inliers=args.agree_min_inliers, agree_min_corners=args.agree_min_corners,
                                  agree_tol_px=args.agree_tol_px)
        pose[f] = warp_corners(H, init); label[f] = lab; n_in[f] = ninl[j]; last = pose[f]
    return pose, label, n_in


def first_pass_from_run(run_dir, seq, init, count):
    """Stage-1 result taken from a finished first-pass run (``flatsam_final``): its output poses and stored decisions."""
    pose = load_pose(run_dir / f"{seq}.txt")
    d = np.load(run_dir / f"{seq}_debug.npz", allow_pickle=True)
    n = min(count, len(pose)); pose = pose[:n]
    if n < count: pose = np.concatenate([pose, np.repeat(pose[-1:], count - n, axis=0)])
    label = np.array(["samh_H"] * count, dtype=object); n_in = np.full(count, -1)
    label[0] = "init"
    for j, f in enumerate(d["frame_idx"].astype(int)):
        if 0 < f < count: label[f] = str(d["selected"][j]); n_in[f] = int(d["n_inliers"][j])
    tmpl = d["template_corners"].astype(np.float64) if "template_corners" in d.files else init
    return pose, label, n_in, tmpl


def quad_area(q):
    x, y = q[0], q[1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def local_rectifier(init, pose):
    """Homography (frame -> canvas) that gives the object its frame-0 shape but keeps its current size and position.

    The target quad is the frame-0 quad scaled about its centre so that its area equals the current pose's area, and moved
    to the current pose's centre. Only perspective and rotation are removed; the object is not resized.
    """
    if not (np.isfinite(init).all() and np.isfinite(pose).all()): return None
    a0, ap = quad_area(init), quad_area(pose)
    if a0 <= 0 or ap < 16: return None
    k = float(np.sqrt(ap / a0))
    target = pose.mean(axis=1, keepdims=True) + k * (init - init.mean(axis=1, keepdims=True))
    R = cv2.getPerspectiveTransform(pose.T.astype(np.float32), target.T.astype(np.float32)).astype(np.float64)
    if not np.isfinite(R).all() or abs(float(np.linalg.det(R))) <= 1e-10: return None
    cond = float(np.linalg.cond(R))
    return R / R[2, 2] if np.isfinite(cond) and cond < 1e10 and abs(float(R[2, 2])) > 1e-12 else None


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks"); ap.add_argument("--task-index", type=int)
    ap.add_argument("--outroot", required=True); ap.add_argument("--mode", choices=["full", "canvas", "local"], required=True,
                    help="full: frame-sized canvas, object mapped to its frame-0 size; canvas: 3x crop window, object mapped to its frame-0 size (scaled by s); "
                         "local: frame-sized canvas, perspective and rotation removed but the object keeps its current size and position")
    ap.add_argument("--max-scale", type=float, default=None)
    ap.add_argument("--inlier-threshold", type=int, default=35)
    ap.add_argument("--agree-min-inliers", type=int, default=8)
    ap.add_argument("--agree-min-corners", type=int, default=2)
    ap.add_argument("--agree-tol-px", type=float, default=5.0, help="corner tolerance (px); the RANSAC reprojection threshold")
    ap.add_argument("--rectify-with", choices=["all", "loma"], default="all",
                    help="all: every frame is rectified with the first-pass pose (the mask pose where LoMa was not chosen); loma: only LoMa frames")
    ap.add_argument("--first-pass-dir", default=None, help="root of a finished first-pass run (contains <dataset label>/<seq>.txt, "
                    "<seq>_debug.npz, masks/<seq>/); uses its poses, decisions, masks and frame-0 template instead of the replay")
    ap.add_argument("--config", default=None, help="tracker config for the rerun (default: the replay config)")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main():
    ap = build_parser(); args = ap.parse_args()
    if args.tasks is None or args.task_index is None: ap.error("--tasks and --task-index are required")
    task = list(csv.DictReader(open(args.tasks, newline="")))[args.task_index]
    refine_sequence(args, task["dataset_key"], task["dataset"], task["sequence"], args.base_dir or task.get("base_dir") or None)


def refine_sequence(args, dataset, label, seq, bd):
    """Rectify every frame of one sequence with its first-pass pose and rerun both branches; ``bd``: dataset dir or None."""
    t_main = time.perf_counter()
    outdir = Path(args.outroot) / label; outdir.mkdir(parents=True, exist_ok=True)
    pose_out, sidecar, summary_out = outdir / f"{seq}.txt", outdir / f"{seq}_rectification.npz", outdir / f"{seq}_summary.json"
    if pose_out.exists() and sidecar.exists() and summary_out.exists() and not args.dry_run:
        print(f"Already complete: {label}/{seq}"); return
    info = DATASETS[dataset]
    pg = Path(args.first_pass_dir) / label if args.first_pass_dir else POSEGATE / label
    mdir = pg / "masks" / seq
    if not mdir.is_dir(): raise ValueError(f"No original SAM3 masks for {label}/{seq}: {mdir}")
    initfix = paths.PT_INITFIX_DIR if dataset == "planartrack-initfix" else None
    gt_loader = make_gt_loader(dataset, initfix_dir=initfix)
    base_dir = Path(bd) if bd else Path(getattr(paths, info.base_dir_attr))
    gt, gt_valid = gt_loader(base_dir, seq)
    image_paths = make_image_paths(dataset)(base_dir, seq)
    count = min(len(image_paths), len(gt), len(gt_valid), 10 ** 9)
    if count < 1 or not np.isfinite(gt[0]).all(): raise ValueError("missing initial GT")
    init = gt[0].astype(np.float64)
    if args.first_pass_dir:
        pose_fp, lab_fp, n_fp, template = first_pass_from_run(pg, seq, init, count)
    else:
        pose_fp, lab_fp, n_fp = first_pass(pg / f"{seq}_debug.npz", init, count, args); template = init
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
    else:                       # "full" and "local": canvas = the frame, no resampling of the frame as a whole
        s = 1.0; A = np.eye(3); canvas = (w0, h0)
    Ainv = np.linalg.inv(A)
    init_c = warp_corners(A, init)
    template_c = warp_corners(A, template)
    if args.first_pass_dir:      # the first pass's own frame-0 mask (the full object when it was accepted), warped into the canvas
        m0 = cv2.imread(str(mdir / "00001.png"), cv2.IMREAD_GRAYSCALE)
        if m0 is None: raise ValueError(f"Missing frame-0 mask {mdir / '00001.png'}")
        if m0.shape != first.shape[:2]: m0 = cv2.resize(m0, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_NEAREST)
        init_mask = cv2.warpPerspective((m0 > 0).astype(np.uint8), A, canvas, flags=cv2.INTER_NEAREST)
    else:
        init_mask = corners_to_mask(init_c, (canvas[1], canvas[0]))
    c2r = np.repeat(np.eye(3)[None], count, axis=0)
    run = np.zeros(count, bool)                             # frames that are rectified and rerun
    metric_valid = np.asarray(gt_valid[:count], bool).copy(); metric_valid[0] = True
    for i in range(1, count):
        if args.rectify_with == "loma" and lab_fp[i] != "loma_H": continue
        if args.mode == "local":
            R = local_rectifier(init, pose_fp[i])
            if R is None: continue
            c2r[i] = R; run[i] = True
            continue
        H = safe_homography(init, pose_fp[i])
        if H is None: continue
        c2r[i] = np.linalg.inv(H); run[i] = True
    if args.dry_run:
        print(f"{label}/{seq}: frames={count} loma={int((lab_fp == 'loma_H').sum())} mask={int((lab_fp == 'samh_H').sum())} "
              f"rerun_frames={int(run.sum())} scale={s:.3f}"); return
    cfg = load_config_from_file(Path(args.config) if args.config else CONFIG)
    cfg = cfg.model_copy(update={"full_object": cfg.full_object.model_copy(update={"enabled": False})})   # the template is given
    cfg = cfg.model_copy(update={"loma": cfg.loma.model_copy(update={
        "init_mask_inflate_px": int(round(INFLATE_PX * s)), "current_mask_inflate_px": 0,
        "inlier_threshold": args.inlier_threshold, "agree_min_inliers": args.agree_min_inliers,
        "agree_min_corners": args.agree_min_corners, "agree_tol_px": args.agree_tol_px})})
    t_load = time.perf_counter()
    tracker = FlatSAMTracker(cfg, fail_hard=True, segmenter=NoSegmenter())
    model_load_s = time.perf_counter() - t_load
    profiler.enable(sync_cuda=True); profiler.reset(); t_loop = time.perf_counter(); t_rect = 0.0
    rerun_pose = pose_fp.copy(); rect_pred = np.zeros((count, 2, 4)); rerun_label = np.array([""] * count, dtype=object)
    rerun_n = np.full(count, -1); rows = []
    branch_pose = {k: np.full((count, 2, 4), np.nan) for k in ("loma", "samh")}     # rerun branch quads, original frame
    for i in range(count):
        if i > 0 and not run[i]: continue
        t_i = time.perf_counter()
        frame = cv2.imread(str(image_paths[i]), cv2.IMREAD_COLOR)
        if frame is None: raise ValueError(f"Cannot read frame {i}: {image_paths[i]}")
        M = A if i == 0 else A @ c2r[i]
        warped = cv2.warpPerspective(frame, M, canvas, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if i == 0:
            tracker.init(warped, init_mask=init_mask, init_corners=template_c, init_segmenter=False); continue
        m = cv2.imread(str(mdir / f"{i + 1:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None: raise ValueError(f"Missing original SAM3 mask for frame {i}")
        if m.shape != frame.shape[:2]: m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
        wm = cv2.warpPerspective(m, M, canvas, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wd = cv2.warpPerspective(dilate(m, INFLATE_PX), M, canvas, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        tracker._loma_densify_mask = wd
        t_rect += time.perf_counter() - t_i
        with profiler.section("rerun_branches"):
            res = tracker.step_with_mask(warped, wm)
        rect_pred[i] = warp_corners(Ainv, warp_corners(res.homography, init_c))
        rerun_pose[i] = warp_corners(np.linalg.inv(c2r[i]), rect_pred[i])
        if res.debug is not None:
            rerun_label[i] = res.debug.selected; rerun_n[i] = int(res.debug.n_inliers)
            for k, cand in (("loma", res.debug.loma_H), ("samh", res.debug.samh_H)):
                if cand.H is not None and np.isfinite(cand.H).all():
                    branch_pose[k][i] = warp_corners(np.linalg.inv(c2r[i]), warp_corners(Ainv, warp_corners(cand.H, init_c)))
        if metric_valid[i] and np.isfinite(gt[i]).all():
            rows.append({"frame_idx": i, "first_pass_error_px": corner_error(pose_fp[i], gt[i]), "rerun_error_px": corner_error(rerun_pose[i], gt[i])})
    np.savetxt(pose_out, rerun_pose.transpose(0, 2, 1).reshape(count, 8), fmt="%.6f")
    np.savez_compressed(sidecar, dataset=np.array(label), sequence=np.array(seq), scale=np.array(s), canvas=np.array(canvas),
                        first_pass_pose=pose_fp, first_pass_label=lab_fp.astype(str), first_pass_inliers=n_fp, rerun_ran=run,
                        rerun_pose=rerun_pose, rerun_label=rerun_label.astype(str), rerun_inliers=rerun_n,
                        current_to_reference=c2r, gt_corners=gt[:count], gt_valid=metric_valid, template_corners=template,
                        rerun_loma_pose=branch_pose["loma"], rerun_samh_pose=branch_pose["samh"])
    if rows:
        with (outdir / f"{seq}_frame_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    allv = [i for i in range(1, count) if metric_valid[i] and np.isfinite(gt[i]).all()]
    p15 = lambda P: 100 * float(np.mean([corner_error(P[i], gt[i]) <= 15 for i in allv])) if allv else None
    summary = {"dataset": label, "sequence": seq, "scale": s, "canvas": list(canvas), "scored_frames": len(allv),
               "first_pass_loma_frames": int((lab_fp == "loma_H").sum()), "rerun_frames": int(run.sum()),
               "first_pass_p_at_15_percent": p15(pose_fp), "rerun_pose_p_at_15_percent": p15(rerun_pose)}
    wall = time.perf_counter() - t_main
    summary["timing"] = {"stage": "warp_rerun", "mode": args.mode, "wall_s": round(wall, 3), "model_load_s": round(model_load_s, 2),
                         "loop_s": round(time.perf_counter() - t_loop, 3), "read_and_warp_s": round(t_rect, 3), "frames": count,
                         "frames_rerun": int(run.sum()), "sections": runinfo.sections(profiler), "env": runinfo.environment(Path(__file__).resolve().parents[2])}
    summary_out.write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps({k: v for k, v in summary.items() if k != "timing"}))

if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
