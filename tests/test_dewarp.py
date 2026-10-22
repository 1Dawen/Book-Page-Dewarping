import numpy as np
import cv2
import pytest
from dewarp import BoundaryPoints, detect_boundaries, piecewise_homography, shading_correction
from dewarp.boundaries import overlay_points, preprocess_mask
from scripts.generate_demo import generate


def test_points_roundtrip(tmp_path):
    b = BoundaryPoints([[0, 1], [50, 4]], [[0, 51], [50, 53]])
    b.save(tmp_path / "points.json")
    other = BoundaryPoints.load(tmp_path / "points.json")
    assert np.allclose(other.top, b.top)
    assert np.allclose(other.bottom, b.bottom)


@pytest.mark.parametrize("top,bottom", [([[0, 10], [50, 15]], [[0, 9], [50, 100]]),
                                          ([[0, 10], [0, 15]], [[0, 90], [0, 100]])])
def test_bad_points_rejected(top, bottom):
    with pytest.raises(ValueError):
        BoundaryPoints(top, bottom)


def test_piecewise_maps_flat_page():
    h, w = 120, 150
    image = np.zeros((h,w,3), dtype=np.uint8)
    image[:] = (245,245,245)
    cv2.rectangle(image, (25,25), (125,95), (22,22,22), -1)
    x = np.linspace(0, w-1, 11)
    b = BoundaryPoints(np.column_stack([x, np.zeros_like(x)]),
                       np.column_stack([x, np.full_like(x,h-1)]))
    out = piecewise_homography(image, b, grid_rows=6)
    assert out.shape == image.shape
    assert np.mean(np.abs(out.astype(float)-image.astype(float))) < 5.5


def test_deterministic_demo(tmp_path):
    img = cv2.imread(str(generate(tmp_path / "synthetic.png")))
    assert img.shape == (520,640,3)
    assert preprocess_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).shape == img.shape[:2]
    b1 = detect_boundaries(img, num_points=15)
    b2 = detect_boundaries(img, num_points=15)
    assert np.allclose(b1.top, b2.top)
    assert np.allclose(b1.bottom, b2.bottom)
    assert overlay_points(img, b1).shape == img.shape
    result = piecewise_homography(img,b1,grid_rows=12)
    assert result.shape == img.shape
    assert np.isfinite(result).all()
    assert shading_correction(result).dtype == np.uint8


def test_invalid_parameters():
    sample = np.zeros((60,90,3),dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_boundaries(sample,num_points=1)
    b = BoundaryPoints([[0,0],[89,0]],[[0,59],[89,59]])
    with pytest.raises(ValueError):
        piecewise_homography(sample,b,grid_rows=61)
    with pytest.raises(ValueError):
        shading_correction(sample,blur_sigma=0)
