"""Memory-conscious per-cell planar homography and grayscale postprocessing."""
from __future__ import annotations
import cv2
import numpy as np
from .boundaries import BoundaryPoints


def piecewise_homography(image: np.ndarray, boundaries: BoundaryPoints, *,
                         grid_rows: int = 30, output_size: tuple[int, int] | None = None) -> np.ndarray:
    """Map a ruled page to a flat rectangle using local perspective transforms.

    Each adjacent pair of top/bottom guide points defines a strip. Subdivide
    vertically; warp each source quadrilateral into its destination cell.
    Unlike the historical notebook, only warp each small ROI rather than an
    entire image for every cell. Gaps may occur with imperfect boundaries.
    """
    if image is None or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Expected a nonempty image")
    if grid_rows < 1:
        raise ValueError("grid_rows must be >= 1")
    width, height = output_size or (image.shape[1], image.shape[0])
    if width < len(boundaries.top) or height < grid_rows:
        raise ValueError("Output dimensions too small for the control-point grid")
    result = np.zeros((height, width) + (() if image.ndim == 2 else (image.shape[2],)), dtype=image.dtype)
    xs = np.rint(np.linspace(0, width, len(boundaries.top))).astype(int)
    ys = np.rint(np.linspace(0, height, grid_rows + 1)).astype(int)
    for strip in range(len(boundaries.top) - 1):
        t0, t1 = boundaries.top[strip:strip + 2]
        b0, b1 = boundaries.bottom[strip:strip + 2]
        x0, x1 = int(xs[strip]), int(xs[strip + 1])
        for row in range(grid_rows):
            y0, y1 = int(ys[row]), int(ys[row + 1])
            pw, ph = x1 - x0, y1 - y0
            if pw < 2 or ph < 2:
                raise ValueError("Output cells must be at least 2x2 pixels")
            a0, a1 = row / grid_rows, (row + 1) / grid_rows
            srcq = np.float32([
                (1 - a0) * t0 + a0 * b0,
                (1 - a0) * t1 + a0 * b1,
                (1 - a1) * t1 + a1 * b1,
                (1 - a1) * t0 + a1 * b0,
            ])
            dstq = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
            matrix = cv2.getPerspectiveTransform(srcq, dstq)
            result[y0:y1, x0:x1] = cv2.warpPerspective(
                image, matrix, (pw, ph), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE)
    return result


def shading_correction(image: np.ndarray, *, blur_sigma: float = 24.0) -> np.ndarray:
    """Suppress slow lighting variations by dividing grayscale by its blur."""
    if blur_sigma <= 0:
        raise ValueError("blur_sigma must be positive")
    if image is None or image.size == 0:
        raise ValueError("Expected a nonempty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = np.asarray(gray, dtype=np.uint8)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
    return cv2.divide(gray, np.maximum(bg, 1), scale=255)
