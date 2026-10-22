"""Create an honestly labeled synthetic 3-panel README preview."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from dewarp import detect_boundaries, piecewise_homography, shading_correction
from dewarp.boundaries import overlay_points
from scripts.generate_demo import generate


def build(out: Path = Path('docs/demo_comparison.png')):
    source = generate(Path('data/demo/synthetic_curved_page.png'))
    image = cv2.imread(str(source))
    bounds = detect_boundaries(image)
    annot = overlay_points(image, bounds)
    final = shading_correction(piecewise_homography(image, bounds))
    final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    images = [image, annot, final]
    labels = ['Synthetic input', 'Estimated boundaries', 'Local homography output']
    scale = 0.62
    panels = []
    for label, img in zip(labels, images):
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        panel = np.full((resized.shape[0]+48, resized.shape[1],3),255,dtype=np.uint8)
        panel[48:] = resized
        cv2.putText(panel,label,(12,31),cv2.FONT_HERSHEY_SIMPLEX,0.57,(30,30,30),1,cv2.LINE_AA)
        panels.append(panel)
    combined = cv2.hconcat(panels)
    out.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out),combined)
    return out

if __name__ == '__main__':
    print(build())
