"""Optional OpenCV desktop point editor. Not needed for headless CLI/testing."""
from __future__ import annotations
import cv2
import numpy as np
from .boundaries import BoundaryPoints


def edit_boundaries(image: np.ndarray, boundaries: BoundaryPoints, *, width: int = 900, height: int = 760) -> BoundaryPoints:
    """Left-drag a point; right-drag to pan; wheel to zoom; Enter saves; R resets; Esc cancels."""
    original_top, original_bottom = boundaries.top.copy(), boundaries.bottom.copy()
    points = np.vstack([original_top.copy(), original_bottom.copy()])
    view_h, view_w = image.shape[:2]
    state = {"zoom": min(width / view_w, height / view_h), "pan": [0.0, 0.0],
             "selected": None, "panning": False, "last": (0, 0)}

    def callback(event, x, y, flags, userdata):
        zoom = state["zoom"]
        ix = (x + state["pan"][0]) / zoom
        iy = (y + state["pan"][1]) / zoom
        if event == cv2.EVENT_LBUTTONDOWN:
            distances = np.linalg.norm(points - np.array([ix, iy]), axis=1)
            near = int(np.argmin(distances))
            if distances[near] <= 12 / zoom:
                state["selected"] = near
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["panning"] = True
            state["last"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if state["selected"] is not None:
                idx = state["selected"]
                points[idx] = [np.clip(ix, 0, view_w-1), np.clip(iy, 0, view_h-1)]
            elif state["panning"]:
                dx, dy = x-state["last"][0], y-state["last"][1]
                state["pan"][0] -= dx
                state["pan"][1] -= dy
                state["last"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["selected"] = None
        elif event == cv2.EVENT_RBUTTONUP:
            state["panning"] = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            scale = 1.1 if flags > 0 else 1/1.1
            new_zoom = np.clip(zoom * scale, 0.15, 6.0)
            state["pan"][0] = ix * new_zoom - x
            state["pan"][1] = iy * new_zoom - y
            state["zoom"] = new_zoom

    cv2.namedWindow("Edit page boundaries", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Edit page boundaries", width, height)
    cv2.setMouseCallback("Edit page boundaries", callback)
    print("Left drag: adjust | Right drag: pan | Wheel: zoom | R: reset | Enter: accept | Esc: cancel")
    while True:
        n = len(original_top)
        displayed = image.copy()
        # Draw the potentially-invalid intermediate points without prematurely validating.
        if displayed.ndim == 2:
            displayed = cv2.cvtColor(displayed, cv2.COLOR_GRAY2BGR)
        for a, b in zip(points[:n], points[n:]):
            aa, bb = tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int))
            cv2.line(displayed, aa, bb, (40, 160, 40), 1)
            cv2.circle(displayed, aa, 4, (25, 25, 230), -1)
            cv2.circle(displayed, bb, 4, (230, 90, 25), -1)
        zoom = state["zoom"]
        scaled = cv2.resize(displayed, None, fx=zoom, fy=zoom)
        sx = max(0, int(state["pan"][0])); sy = max(0, int(state["pan"][1]))
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        section = scaled[sy:sy+height, sx:sx+width]
        if section.size:
            canvas[:section.shape[0], :section.shape[1]] = section
        cv2.imshow("Edit page boundaries", canvas)
        key = cv2.waitKey(25) & 0xFF
        if key in (13, 10):
            break
        if key == 27:
            cv2.destroyWindow("Edit page boundaries")
            raise KeyboardInterrupt("Editing cancelled")
        if key == ord("r"):
            points = np.vstack([original_top.copy(), original_bottom.copy()])
    cv2.destroyWindow("Edit page boundaries")
    return BoundaryPoints(points[:len(original_top)], points[len(original_top):])
