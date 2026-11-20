"""Add a sequence to the FlatSAM viewer demo (or replace it): frames, per-frame outlines and scores.

Needs only numpy and opencv-python:   pip install numpy opencv-python-headless

Example (run from the viewer_demo folder; start_demo.py shows the result):

    python3 tools/add_sequence.py --dataset MyData --sequence clip1 \
        --frames /path/to/clip1_frames \
        --refinement clip1_FLATSAM.txt \
        --first-pass clip1_first_pass.txt \
        --gt clip1_gt.txt --woftsam clip1_woftsam.txt --samh clip1_samh.txt

Every prediction file has one row per frame with eight numbers (x1 y1 x2 y2 x3 y3 x4 y4, the four corners in the
same order as the annotation); row 0 is the frame the tracker was initialised on. Rows with NaN or missing rows mean
"no pose in this frame". This is the format the tracker of this repository writes (<sequence>_FLATSAM.txt).
Only --refinement is required (it is the FlatSAM output). Layers you do not give are shown as unavailable.
Scores (P@5/10/15, in pixels and relative to the target size, and IoU) are computed against --gt when it is given;
they use the corner-RMS error and the same definitions as the paper.
Use --video clip.mp4 instead of --frames to cut the frames from a video.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

LAYER_FILES = {  # command line option -> viewer layer id
    "first_pass": "original", "refinement": "rerun", "woftsam": "woft", "samh": "samhb",
}
LAYER_NAMES = {"original": "First pass", "rerun": "Refinement pass", "woft": "WOFTSAM", "samhb": "SAM-H",
               "o_loma": "Keypoint branch (first pass)", "o_samh": "Mask branch (first pass)",
               "n_loma": "Keypoint branch (refinement pass)", "n_samh": "Mask branch (refinement pass)"}


def load_poses(path: Path, count: int) -> list[np.ndarray | None]:
    """Rows of eight numbers -> list of (2, 4) arrays (x row, y row) or None."""
    values = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if values.shape[1] != 8:
        sys.exit(f"{path}: expected eight numbers per row, got {values.shape[1]}")
    out: list[np.ndarray | None] = [None] * count
    for i in range(min(count, len(values))):
        q = values[i].reshape(4, 2).T
        if np.isfinite(q).all() and np.abs(q).max() < 1e6:
            out[i] = q
    return out


def corner_error(a, b):
    return None if a is None or b is None else float(np.sqrt(np.sum((a - b) ** 2) / 4.0))


def poly_mask(q, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(q.T).astype(np.int32)], 1)
    return m > 0


def polygon_iou(a, b, shape):
    if a is None or b is None:
        return None
    ma, mb = poly_mask(a, shape), poly_mask(b, shape)
    union = np.count_nonzero(ma | mb)
    return float(np.count_nonzero(ma & mb) / union) if union else None


def target_size(q):
    x, y = q[0], q[1]
    return float(np.sqrt(abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))))


def r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def quad_json(q):
    return None if q is None else [[round(float(q[0, k]), 1), round(float(q[1, k]), 1)] for k in range(4)]


def read_frames(args) -> list[Path]:
    if args.video:
        tmp = Path(tempfile.mkdtemp(prefix="flatsam_frames_"))
        cap, n = cv2.VideoCapture(str(args.video)), 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            cv2.imwrite(str(tmp / f"{n:06d}.png"), img)
            n += 1
        if n == 0:
            sys.exit(f"cannot read frames from {args.video}")
        return sorted(tmp.glob("*.png"))
    files = sorted(p for p in Path(args.frames).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not files:
        sys.exit(f"no images in {args.frames}")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viewer", type=Path, default=Path(__file__).resolve().parent.parent, help="viewer_demo folder (default: the one next to tools/)")
    ap.add_argument("--dataset", required=True, help='group name in the list, e.g. "MyData" (not "PlanarTrack": the page hides that name)')
    ap.add_argument("--sequence", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", type=Path, help="folder with the frames in order (jpg/png)")
    src.add_argument("--video", type=Path)
    ap.add_argument("--refinement", type=Path, required=True, help="FlatSAM output poses (refinement pass)")
    ap.add_argument("--first-pass", type=Path, dest="first_pass")
    ap.add_argument("--gt", type=Path, help="ground-truth poses; frames without a valid row are not scored")
    ap.add_argument("--woftsam", type=Path)
    ap.add_argument("--samh", type=Path)
    ap.add_argument("--width", type=int, default=416, help="width of the stored frames in pixels (smaller = smaller folder)")
    ap.add_argument("--quality", type=int, default=46, help="JPEG quality of the stored frames")
    ap.add_argument("--no-rotate", action="store_true", help="do not turn portrait frames by 90 degrees")
    a = ap.parse_args()
    if a.dataset == "PlanarTrack":
        sys.exit('the page hides the dataset name "PlanarTrack"; use another name')

    imgs = read_frames(a)
    first_img = cv2.imread(str(imgs[0]), cv2.IMREAD_COLOR)
    H0, W0 = first_img.shape[:2]
    count = min(len(imgs), len(np.atleast_2d(np.loadtxt(a.refinement))))
    imgs = imgs[:count]
    rot = H0 > W0 and not a.no_rotate     # portrait frames are shown turned 90 degrees clockwise, like in the full viewer

    quads: dict[str, list | None] = {k: None for k in ("original", "rerun", "gated", "woft", "samhb", "o_loma", "o_samh", "n_loma", "n_samh")}
    for opt, lid in LAYER_FILES.items():
        path = getattr(a, opt)
        if path is not None:
            quads[lid] = load_poses(path, count)
    quads["gated"] = quads["rerun"]            # the page always shows the refinement pass as the output
    gt = load_poses(a.gt, count) if a.gt else [None] * count
    init = gt[0] if gt[0] is not None else quads["rerun"][0]
    if init is None:
        sys.exit("row 0 of --refinement (or --gt) must be a valid pose: it is the initialisation")
    gt_ok = [g is not None for g in gt]
    gt_ok[0] = False                            # frame 0 is the initialisation and is never scored
    shape = (H0, W0)

    def rq(q):
        """(x, y) -> (H0 - 1 - y, x) for the turned frame."""
        if not rot or q is None:
            return q
        return np.stack([(H0 - 1) - q[1], q[0]])

    err, err_rel, iou, stats = {}, {}, {}, {}
    sizes = [target_size(gt[i]) if gt_ok[i] else None for i in range(count)]
    for name, qs in quads.items():
        if qs is None or name == "gated":
            continue
        e = [None] + [r(corner_error(qs[i], gt[i]), 2) if gt_ok[i] else None for i in range(1, count)]
        err[name] = e
        err_rel[name] = [None if (v is None or sizes[i] is None or sizes[i] < 1.0) else r(100.0 * v / sizes[i], 2) for i, v in enumerate(e)]
        iou[name] = [None] + [r(polygon_iou(qs[i], gt[i], shape)) if (gt_ok[i] and qs[i] is not None) else None for i in range(1, count)]
        ev, er, iv = [v for v in e if v is not None], [v for v in err_rel[name] if v is not None], [v for v in iou[name] if v is not None]
        st = {"gt_iou": r(float(np.mean(iv))) if iv else None, "n": len(ev)}
        for t in (5, 10, 15):
            st[f"p{t}"] = r(100.0 * float(np.mean([v <= t for v in ev])), 2) if ev else None
            st[f"p{t}_rel"] = r(100.0 * float(np.mean([v <= t for v in er])), 2) if er else None
        stats[name] = st
    for d_ in (err, err_rel, iou):
        if "rerun" in d_:
            d_["gated"] = d_["rerun"]
    if "rerun" in stats:
        stats["gated"] = stats["rerun"]

    ovr = [None] * count                        # IoU of first pass and refinement pass
    if quads["original"] is not None:
        ovr = [None] + [r(polygon_iou(quads["original"][i], quads["rerun"][i], shape)) for i in range(1, count)]
    vals = [v for v in ovr if v is not None]

    notes = {k: "Not provided for this sequence." for k in LAYER_NAMES if quads.get(k) is None}
    notes["mask_o"] = "No SAM3 masks for this sequence."

    out = a.viewer
    fd = out / "frames" / a.dataset / a.sequence
    fd.mkdir(parents=True, exist_ok=True)
    for old in fd.glob("*.jpg"):
        old.unlink()
    for i, p in enumerate(imgs):
        img = first_img if i == 0 else cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"cannot read {p}")
        if rot:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
        if w > a.width:
            img = cv2.resize(img, (a.width, max(1, round(h * a.width / w))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(fd / f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
    ini = cv2.rotate(first_img, cv2.ROTATE_90_CLOCKWISE) if rot else first_img
    if ini.shape[1] > 480:
        ini = cv2.resize(ini, (480, max(1, round(ini.shape[0] * 480 / ini.shape[1]))), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", ini, [cv2.IMWRITE_JPEG_QUALITY, 85])

    layers = {"gt": [quad_json(rq(gt[i])) if (gt_ok[i] or i == 0) else None for i in range(count)]}
    for name, qs in quads.items():
        layers[name] = None if qs is None else [quad_json(rq(q)) for q in qs]
    none = [None] * count
    payload = {
        "dataset": a.dataset, "sequence": a.sequence, "tier": "added", "frames": count,
        "width": H0 if rot else W0, "height": W0 if rot else H0, "rotated": bool(rot),
        "frame_dir": f"frames/{a.dataset}/{a.sequence}", "init_image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
        "init_quad": quad_json(rq(init)), "layers": layers, "layer_notes": notes,
        "gt_note": "GT: provided with the sequence" if a.gt else "No ground truth for this sequence (no scores)",
        "gt_src": [0] * count, "mask_o": None, "mask_n": None, "mask_o_iou": None, "mask_n_iou": None, "mask_n_iou_native": None,
        "err": err, "err_rel": err_rel, "gt_iou": iou, "o_selected": none, "n_selected": none, "o_inliers": none, "n_inliers": none,
        "warp_valid": [True] * count, "gate_used": [False] * count, "iou": ovr, "layer_stats": stats, "agreement": {"original": None, "rerun": None},
        "stats": {"mean_iou": r(float(np.mean(vals))) if vals else None, "max_iou": r(float(np.max(vals))) if vals else None,
                  "min_iou": r(float(np.min(vals))) if vals else None,
                  "argmax_iou": int(np.nanargmax([np.nan if v is None else v for v in ovr])) if vals else None,
                  "argmin_iou": int(np.nanargmin([np.nan if v is None else v for v in ovr])) if vals else None, "iou_frames": len(vals)},
    }
    dd = out / "data" / a.dataset
    dd.mkdir(parents=True, exist_ok=True)
    (dd / f"{a.sequence}.js").write_text("window.__flatsamSeq(" + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ");\n")

    # manifest row: same columns as the other rows, the ones that cannot be computed stay empty
    mpath = out / "data" / "manifest.js"
    text = mpath.read_text()
    man = json.loads(text[text.index("(") + 1:text.rindex(")")])
    row = {k: None for k in (man["sequences"][0] if man["sequences"] else {})}
    g = lambda layer, k: stats.get(layer, {}).get(k)
    row.update({"dataset": a.dataset, "sequence": a.sequence, "tier": "added", "frames": count, "has_branches": False, "has_masks": False,
                "has_woft": quads["woft"] is not None, "has_samh_baseline": quads["samhb"] is not None, "rotated": bool(rot),
                "has_rerun": True, "has_gated": True, "mean_iou": payload["stats"]["mean_iou"], "max_iou": payload["stats"]["max_iou"],
                "layers_present": sum(quads[k] is not None for k in ("original", "rerun", "woft", "samhb")), "layers_total": 9,
                "woft_p15": g("woft", "p15"), "woft_p15_rel": g("woft", "p15_rel"), "samhb_p15": g("samhb", "p15"), "samhb_p15_rel": g("samhb", "p15_rel"),
                "reused": False, "lookalike": None, "accepted_share": None})
    for t in (5, 10, 15):
        for sfx in ("", "_rel"):
            k = f"p{t}{sfx}"
            o, w = g("original", k), g("rerun", k)
            row[f"original_{k}"], row[f"warp_{k}"], row[f"final_{k}"], row[f"rerun_{k}"] = o, w, w, w
            row[f"delta_{k}"] = r(w - o, 2) if o is not None and w is not None else None
    row["original_gt_iou"], row["rerun_gt_iou"] = g("original", "gt_iou"), g("rerun", "gt_iou")
    row["delta_gt_iou"] = r(row["rerun_gt_iou"] - row["original_gt_iou"]) if row["rerun_gt_iou"] is not None and row["original_gt_iou"] is not None else None
    man["sequences"] = [q for q in man["sequences"] if not (q["dataset"] == a.dataset and q["sequence"] == a.sequence)] + [row]
    mpath.write_text("window.__flatsamManifest(" + json.dumps(man, separators=(",", ":"), allow_nan=False) + ");\n")
    print(f"added {a.dataset}/{a.sequence}: {count} frames{' (turned 90 degrees)' if rot else ''}; "
          f"refinement-pass P@15 {g('rerun', 'p15')} px / {g('rerun', 'p15_rel')} rel. Reload the page.")


if __name__ == "__main__":
    main()
