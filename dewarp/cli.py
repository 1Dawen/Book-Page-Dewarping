"""Command line interface for reproducible page dewarping."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from .boundaries import BoundaryPoints, detect_boundaries, overlay_points
from .transform import piecewise_homography, shading_correction


def main(argv=None):
    p = argparse.ArgumentParser(description="Piecewise-homography book-page dewarping")
    p.add_argument("--input", type=Path, required=True, help="Input page photograph")
    p.add_argument("--output", type=Path, default=Path("outputs/flattened.png"))
    p.add_argument("--points", type=Path, default=None, help="Use saved JSON points rather than auto detection")
    p.add_argument("--save-points", type=Path, default=None)
    p.add_argument("--overlay", type=Path, default=None, help="Save an image annotated with detection points")
    p.add_argument("--num-points", type=int, default=35)
    p.add_argument("--grid-rows", type=int, default=30)
    p.add_argument("--interactive", action="store_true", help="Open a desktop GUI to adjust points")
    p.add_argument("--no-shading", action="store_true")
    a = p.parse_args(argv)
    image = cv2.imread(str(a.input), cv2.IMREAD_COLOR)
    if image is None:
        p.error(f"Could not load input image: {a.input}")
    bounds = BoundaryPoints.load(a.points) if a.points else detect_boundaries(image, num_points=a.num_points)
    if a.interactive:
        from .editor import edit_boundaries
        bounds = edit_boundaries(image, bounds)
    if a.save_points:
        a.save_points.parent.mkdir(parents=True, exist_ok=True)
        bounds.save(a.save_points)
    if a.overlay:
        a.overlay.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(a.overlay), overlay_points(image, bounds)):
            raise OSError(f"Could not save overlay to {a.overlay}")
    final = piecewise_homography(image, bounds, grid_rows=a.grid_rows)
    if not a.no_shading:
        final = shading_correction(final)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(a.output), final):
        raise OSError(f"Could not write output to {a.output}")
    print(f"Saved {a.output} ({final.shape[1]}x{final.shape[0]})")
    if not a.points and not a.interactive:
        print("Auto boundaries are heuristic: inspect the overlay and correct points when necessary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
