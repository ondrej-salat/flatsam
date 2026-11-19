"""Final FlatSAM method: one tracker on the full object, orientation resolved at the gate.

``loma_samh_matchnaming`` with:

* full-object template: on frame 0 SAM3 is clicked inside the GT and, when the merged mask passes
  ``flatsam2.full_object.accept_full_object``, the whole planar object replaces the GT as the template
  of both branches; the output is still the GT corners warped by the tracked plane homography;
* no match-support corner naming in the mask branch (``symmetry`` off): the mask branch gets nothing from LoMa;
* the keypoint/mask decision has the Hough agreement band (8 < n < 35 and LoMa agrees with the Hough quad on
  2 corners within the RANSAC threshold (5 px) -> LoMa; see ``flatsam2.gate``);
* when the gate returns the mask pose, its corners are shifted by the cyclic shift last measured against a
  LoMa pose with more than 8 inliers (``loma.gate_orientation_min_inliers``), carried over frames without one.
"""

import importlib.util
from pathlib import Path

from flatsam2.config import FullObjectCfg


_base_path = Path(__file__).with_name("loma_samh_matchnaming.py")
_spec = importlib.util.spec_from_file_location("loma_samh_matchnaming_config", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "full_object": FullObjectCfg(
            enabled=True,
            min_area_ratio=1.3,
            max_area_ratio=2.0,
            min_gt_coverage=0.9,
            allow_border=False,
        ),
        "symmetry": _module.config.symmetry.model_copy(update={"enabled": False}),
        "loma": _module.config.loma.model_copy(
            update={
                "agree_min_inliers": 8,
                "agree_min_corners": 2,
                "agree_tol_px": None,
                "gate_orientation_min_inliers": 8,
            }
        ),
    }
)
