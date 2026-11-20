"""Full-object template: use the whole planar object a GT region lies on.

The GT may mark only part of a flat object (the picture on a printed sheet). Every point of
that object shares one plane homography, so LoMa can sample keypoints on the whole object
— more texture for the same homography — while ``run.py`` still reports
``warp_corners(H, gt_corners)``.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from einops import rearrange

from flatsam.config import FullObjectCfg, HoughConfig
from flatsam.datasets import warp_corners
from flatsam.geometry import warp_points
from flatsam.hough_homography import _signed_area, mask_to_homography
from flatsam.match_naming import homography_from_corners

# GT centre, then the centres of its four quarters, as (x, y) in the GT's unit square
_CLICKS_UNIT = np.array([[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])


def click_points(gt_corners: np.ndarray) -> np.ndarray:
    """Click positions inside the GT: its center and the centers of its four quarters.

    Args:
        gt_corners: (2, 4) GT corners ``[[x0..x3], [y0..y3]]``.

    Returns:
        (5, 2) click pixels, the center first.

    Raises:
        ValueError: If the GT corners are degenerate.
    """
    unit = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    H = homography_from_corners(unit, gt_corners)
    if H is None:
        raise ValueError(f"Degenerate GT corners: {gt_corners.tolist()}")
    return warp_points(H, _CLICKS_UNIT)


def sam3_point_masks(segmenter: object, frame: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    """SAM3's multimask answer to one positive click (subpart / part / whole).

    Reuses the SAM3 model the streaming tracker has already loaded, so no second model is
    loaded.

    Args:
        segmenter: ``SAM3StreamingTracker``; its ``predictor`` is the SAM3 tracker model.
        frame: (H, W, 3) uint8 BGR image.
        point_xy: (2,) click position in pixels.

    Returns:
        (3, H, W) bool candidate masks.
    """
    from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor

    predictor = SAM3InteractiveImagePredictor(segmenter.predictor)
    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    masks, _, _ = predictor.predict(
        point_coords=point_xy[None].astype(np.float32),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    return masks > 0


def mask_stats(mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float, bool]:
    """How much bigger than the GT a mask is, how much of the GT it covers, cut off or not.

    Args:
        mask: (H, W) bool mask.
        gt_mask: (H, W) bool GT region.

    Returns:
        ``(area_ratio, gt_coverage, touches_border)``: mask area / GT area, fraction of GT
        pixels inside the mask, and whether the mask reaches the image border.
    """
    gt_area = gt_mask.sum()
    border = mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    return mask.sum() / gt_area, (mask & gt_mask).sum() / gt_area, bool(border)


def merged_middle_mask(click_masks: np.ndarray) -> np.ndarray:
    """Pixel majority vote over the middle-sized mask of every click.

    Each click gives three masks in no fixed order, so they are sorted by area and the
    middle one is taken. A pixel is kept when more than half of the clicks' middle masks
    contain it (3 of 5 clicks).

    Args:
        click_masks: (K, 3, H, W) bool, three masks per click.

    Returns:
        (H, W) bool merged mask.
    """
    areas = click_masks.sum(axis=(2, 3))
    middle_idx = np.argsort(areas, axis=1)[:, 1]
    middle = click_masks[np.arange(len(click_masks)), middle_idx]
    votes = middle.sum(axis=0)
    return votes * 2 > len(middle)  # more than half of the clicks


def quad_from_mask(mask: np.ndarray, gt_corners: np.ndarray) -> Optional[np.ndarray]:
    """Four corners of a mask's outline, named like the GT.

    Simplifies the convex hull of the largest blob to four vertices, then orders them in
    the GT's winding with corner 0 nearest GT corner 0.

    Args:
        mask: (H, W) bool mask.
        gt_corners: (2, 4) GT corners ``[[x0..x3], [y0..y3]]``.

    Returns:
        (2, 4) float64 corners, or None when the outline is not four-sided.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    perimeter = cv2.arcLength(hull, True)
    for fraction in np.linspace(0.005, 0.1, 20):
        approx = cv2.approxPolyDP(hull, fraction * perimeter, True)
        if len(approx) <= 4:
            break
    if len(approx) != 4:
        return None
    quad = rearrange(approx, "corners 1 xy -> xy corners").astype(np.float64)
    if np.sign(_signed_area(quad)) != np.sign(_signed_area(gt_corners)):
        quad = quad[:, ::-1]
    first = int(np.argmin(np.linalg.norm(quad - gt_corners[:, :1], axis=0)))
    return np.roll(quad, -first, axis=1)


def snap_quad(mask: np.ndarray, quad: np.ndarray, hough: HoughConfig) -> Optional[np.ndarray]:
    """Move a mask's hull quad onto the Hough line intersections the SAM-H branch measures.

    Same estimator as every later frame, so frame-0 and frame-t corners agree.

    Args:
        mask: (H, W) bool mask.
        quad: (2, 4) hull quad of that mask (:func:`quad_from_mask`).
        hough: the tracker's Hough settings.

    Returns:
        (2, 4) snapped corners, or None unless all four corners are line intersections
        (a partial fit can distort).
    """
    H, n_corr = mask_to_homography(
        mask,
        quad,
        np.eye(3),
        gradient_step=hough.gradient_step,
        topk_lines=hough.topk_lines,
        corner_max_distance_px=hough.corner_max_distance_px,
        corner_max_distance_image_fraction=hough.corner_max_distance_image_fraction,
        min_intersection_angle_deg=hough.min_intersection_angle_deg,
        line_refinement_method=hough.line_refinement_method,
    )
    return warp_corners(H, quad) if n_corr == 4 else None


def template_features(
    mask: np.ndarray,
    gt_mask: np.ndarray,
    hull: Optional[np.ndarray],
    snapped: Optional[np.ndarray],
) -> np.ndarray:
    """Decision inputs of the merged mask.

    Size and GT coverage are measured on the hull quad, not the mask: on a 3D object the mask
    also takes the side face, the four-corner quad drops it. ``snapped`` is recorded for
    analysis only; the decision does not use it.

    Args:
        mask: (H, W) bool merged mask.
        gt_mask: (H, W) bool GT region.
        hull: (2, 4) quad fitted to ``mask``, None when it is not four-sided.
        snapped: (2, 4) ``hull`` snapped to Hough intersections, None unless all 4 snap.

    Returns:
        (5,) float64 ``[has_quad, area_ratio, gt_coverage, touches_border, snapped]``,
        flags as 0/1; ratio and coverage 0 without a quad.
    """
    border = mask_stats(mask, gt_mask)[2]
    if hull is None:
        return np.array([0.0, 0.0, 0.0, border, 0.0])
    quad = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(quad, [rearrange(hull, "xy corners -> corners 1 xy").astype(np.int32)], 1)
    ratio, coverage, _ = mask_stats(quad > 0, gt_mask)
    return np.array([1.0, ratio, coverage, border, snapped is not None], dtype=np.float64)


def accept_full_object(features: np.ndarray, cfg: FullObjectCfg) -> bool:
    """The decision: use the merged mask (True) or keep the GT (False).

    Four conditions: a four-corner quad, ``min_area_ratio`` <= area ratio <=
    ``max_area_ratio``, GT coverage >= ``min_gt_coverage``, not on the image border.

    Args:
        features: (5,) row of :func:`template_features`
        cfg: the thresholds.

    Returns:
        Whether to use the merged mask.
    """
    has_quad, ratio, coverage, border = features[:4]
    return bool(
        has_quad
        and cfg.min_area_ratio <= ratio
        and (cfg.max_area_ratio is None or ratio <= cfg.max_area_ratio)
        and coverage >= cfg.min_gt_coverage
        and (cfg.allow_border or not border)
    )
