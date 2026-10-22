"""Classical OpenCV page dewarping with editable boundary points."""
from .boundaries import BoundaryPoints, detect_boundaries
from .transform import piecewise_homography, shading_correction

__all__ = ["BoundaryPoints", "detect_boundaries", "piecewise_homography", "shading_correction"]
