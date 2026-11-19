"""First decision of the pipeline: keypoint pose or mask (Hough) pose.

Per frame, from the LoMa homography, its RANSAC inlier count ``n`` and the SAM-H (mask/Hough) homography:

1. ``n >= inlier_threshold`` (35)                               -> LoMa.
2. ``agree_min_inliers < n < inlier_threshold`` and the LoMa quad agrees with the Hough quad on at least
   ``agree_min_corners`` corners (each within ``agree_tol_px`` pixels, the RANSAC reprojection threshold)  -> LoMa.
3. otherwise                                                    -> the mask branch.

Corner agreement is checked up to a cyclic relabelling of one quad, because the Hough quad's corner order
can be wrong while its position is right. The rule needs only the two branches' homographies of the same
frame, so it can be applied online (the tracker) or offline from a run's stored ``loma_H`` / ``samh_H``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from flatsam2.datasets import warp_corners

LOMA = "loma_H"
SAMH = "samh_H"


def quads_agree(a: np.ndarray, b: np.ndarray, min_corners: int, tol_px: float) -> bool:
    """Whether at least ``min_corners`` corners of ``a`` lie within ``tol_px`` of ``b``'s, up to a cyclic shift.

    Args:
        a: (2, 4) quad.
        b: (2, 4) quad.
        min_corners: Number of corners that must agree.
        tol_px: Distance tolerance in pixels.

    Returns:
        True if some cyclic relabelling of ``b`` puts ``min_corners`` corners within ``tol_px``.
    """
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or tol_px <= 0:
        return False
    for shift in range(4):
        dist = np.linalg.norm(np.roll(b, shift, axis=1) - a, axis=0)
        if int((dist <= tol_px).sum()) >= min_corners:
            return True
    return False


def select_pose(
    loma_H: Optional[np.ndarray],
    n_inliers: int,
    samh_H: np.ndarray,
    init_corners: np.ndarray,
    inlier_threshold: int = 35,
    agree_min_inliers: int = 0,
    agree_min_corners: int = 2,
    agree_tol_px: float = 5.0,
) -> Tuple[str, np.ndarray]:
    """Choose the keypoint or the mask pose for one frame.

    Args:
        loma_H: LoMa homography (3, 3) (template -> frame), or None.
        n_inliers: LoMa RANSAC inlier count.
        samh_H: SAM-H homography (3, 3).
        init_corners: (2, 4) template corners, to compare the two quads.
        inlier_threshold: Inlier count from which LoMa is used outright.
        agree_min_inliers: Lower bound of the agreement band; 0 switches the band off (LoMa only at
            ``n >= inlier_threshold``, the original rule).
        agree_min_corners: Corners that must agree.
        agree_tol_px: Corner tolerance in pixels; the tracker passes its RANSAC reprojection threshold.

    Returns:
        ``(label, H)`` with ``label`` ``"loma_H"`` or ``"samh_H"``.
    """
    if loma_H is not None and n_inliers >= inlier_threshold:
        return LOMA, loma_H
    if loma_H is not None and agree_min_inliers > 0 and n_inliers > agree_min_inliers and init_corners is not None:
        quad_l = warp_corners(loma_H, init_corners)
        quad_s = warp_corners(samh_H, init_corners)
        if np.isfinite(quad_l).all() and quads_agree(quad_l, quad_s, agree_min_corners, agree_tol_px):
            return LOMA, loma_H
    return SAMH, samh_H
