"""Generate artificial curved-page data for code demonstration, not a real book scan."""
from pathlib import Path
import cv2
import numpy as np


def generate(output: Path):
    w, h = 640, 520
    rng = np.random.default_rng(3)
    canvas = np.full((h, w, 3), (78, 82, 88), dtype=np.uint8)
    left, right = 55, 585
    xx = np.arange(left, right + 1)
    top = 55 + 9 * np.sin((xx-left)/(right-left)*np.pi)
    bottom = 453 + 15 * np.sin((xx-left)/(right-left)*np.pi)
    polygon = np.array(list(zip(xx, top.astype(int))) + list(zip(xx[::-1], bottom[::-1].astype(int))), dtype=np.int32)
    cv2.fillPoly(canvas, [polygon], (239, 239, 239))
    # Artificial title and text rules distorted by the page's curved boundaries.
    for row in [0.11, 0.14, 0.22, *np.linspace(.30, .87, 12)]:
        start, stop = (90, 435) if row < .2 else (82, int(rng.integers(430, 560)))
        for x in range(start, stop, 2):
            relative = (x - left) / (right - left)
            a = np.interp(x, xx, top);b = np.interp(x, xx, bottom)
            y = round((1-row)*a + row*b)
            cv2.line(canvas, (x,y), (min(x+2,stop),y), (75,75,75), 1)
    pts1 = np.column_stack([xx, top]).astype(np.int32)
    pts2 = np.column_stack([xx, bottom]).astype(np.int32)
    cv2.polylines(canvas,[pts1],False,(22,22,22),3)
    cv2.polylines(canvas,[pts2],False,(22,22,22),3)
    speckle = rng.normal(0, 1.3, size=(h,w,1))
    canvas = np.uint8(np.clip(canvas.astype(np.float32) + speckle,0,255))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    return output


if __name__ == "__main__":
    print(generate(Path("data/demo/synthetic_curved_page.png")))
