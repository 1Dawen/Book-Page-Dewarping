"""Heuristic top/bottom page edge estimation, inspired by the source notebook."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline


@dataclass
class BoundaryPoints:
    top: np.ndarray
    bottom: np.ndarray

    def __post_init__(self):
        self.top = np.asarray(self.top, dtype=np.float64)
        self.bottom = np.asarray(self.bottom, dtype=np.float64)
        if self.top.shape != self.bottom.shape or self.top.ndim != 2 or self.top.shape[1] != 2:
            raise ValueError("Top and bottom must have matching Nx2 coordinates")
        if len(self.top) < 2 or not np.isfinite(self.top).all() or not np.isfinite(self.bottom).all():
            raise ValueError("At least two finite coordinate pairs are required")
        if np.any(np.diff(self.top[:, 0]) <= 0) or np.any(np.diff(self.bottom[:, 0]) <= 0):
            raise ValueError("Boundary x coordinates must increase left to right")
        if np.any(self.bottom[:, 1] - self.top[:, 1] <= 2):
            raise ValueError("Bottom points must lie below the top edge")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"top": self.top.tolist(), "bottom": self.bottom.tolist()}, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BoundaryPoints":
        item = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(item["top"], item["bottom"])


def preprocess_mask(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("Expected a grayscale input image")
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    mask = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 55, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def _smooth(positions: np.ndarray, heights: np.ndarray) -> np.ndarray:
    if len(positions) < 4:
        return heights.astype(float)
    return UnivariateSpline(positions, heights, k=3, s=len(positions) * 5)(positions)


def detect_boundaries(image: np.ndarray, *, num_points: int = 35,
                      left_fraction: float = 0.12, right_fraction: float = 0.88) -> BoundaryPoints:
    """Estimate top and bottom boundaries, without fitting on an external dataset.

    Heuristic: upper-edge first hit, lower-half dominant horizontal-edge band.
    Text-heavy or complex backgrounds may produce incorrect boundaries: inspect
    points and use --interactive or --points before accepting the output.
    """
    if image is None or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Expected a nonempty grayscale or color image")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Expected BGR image with three channels")
    if not 0 <= left_fraction < right_fraction <= 1:
        raise ValueError("Invalid horizontal detection range")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    if height < 40 or width < 40:
        raise ValueError("Image too small for boundary detection")
    xs = np.linspace(max(1, round(width * left_fraction)),
                     min(width - 2, round(width * right_fraction)), num_points)
    cols = xs.astype(int)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 125)
    # First, try isolating a bright page from a darker surrounding scene.
    # This usually works for documents photographed on a contrasting surface,
    # but a full-page crop or a white desk will require manual correction.
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros_like(gray, dtype=np.uint8)
    page_found = False
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        area = cv2.contourArea(largest) / float(width * height)
        if 0.18 < area < 0.92 and bw > width * 0.45 and bh > height * 0.35:
            cv2.drawContours(region, [largest], -1, 255, thickness=cv2.FILLED)
            page_found = all(np.count_nonzero(region[:, x]) > 0 for x in cols)
    mid = height // 2
    lower_activity = np.count_nonzero(edges[mid:], axis=1)
    peak = int(np.argmax(lower_activity)) + mid
    lower_start = max(mid, peak - max(15, height // 28))
    lower_end = min(height, peak + max(15, height // 28) + 1)
    if cv2.countNonZero(edges) == 0 and not page_found:
        raise ValueError("No usable edges detected. Supply an image with visible page boundaries")
    top_y, bottom_y = [], []
    for x in cols:
        if page_found:
            included = np.flatnonzero(region[:, x])
            near_top, near_bottom = included[0], included[-1]
            # Snap the mask boundaries to nearby Canny pixels, if present.
            t_candidates = np.flatnonzero(edges[max(0, near_top - 8):min(height, near_top + 9), x])
            b_candidates = np.flatnonzero(edges[max(0, near_bottom - 8):min(height, near_bottom + 9), x])
            top_y.append(float(max(0, near_top - 8) + t_candidates[0]) if len(t_candidates) else float(near_top))
            bottom_y.append(float(max(0, near_bottom - 8) + b_candidates[-1]) if len(b_candidates) else float(near_bottom))
        else:
            top_candidates = np.flatnonzero(edges[:mid, x])
            bottom_candidates = np.flatnonzero(edges[lower_start:lower_end, x])
            top_y.append(float(top_candidates[0]) if len(top_candidates) else float(height * 0.12))
            bottom_y.append(float(lower_start + bottom_candidates[0]) if len(bottom_candidates)
                            else float(height * 0.88))
    top = np.column_stack([xs, _smooth(xs, np.asarray(top_y))])
    bottom = np.column_stack([xs, _smooth(xs, np.asarray(bottom_y))])
    top[:, 1] = np.clip(top[:, 1], 0, height - 4)
    bottom[:, 1] = np.clip(bottom[:, 1], 3, height - 1)
    return BoundaryPoints(top, bottom)


def overlay_points(image: np.ndarray, boundaries: BoundaryPoints) -> np.ndarray:
    view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for a, b in zip(boundaries.top, boundaries.bottom):
        axy = tuple(np.rint(a).astype(int))
        bxy = tuple(np.rint(b).astype(int))
        cv2.line(view, axy, bxy, (50, 170, 50), 1)
        cv2.circle(view, axy, 3, (30, 30, 230), -1)
        cv2.circle(view, bxy, 3, (220, 70, 30), -1)
    return view
