import cv2
import pytest
from scripts.generate_demo import generate
from dewarp.cli import main
from dewarp import BoundaryPoints


def test_cli_roundtrip(tmp_path):
    source = generate(tmp_path / 'source.png')
    output = tmp_path / 'result.png'
    points = tmp_path / 'points.json'
    overlay = tmp_path / 'overlay.png'
    assert main(['--input', str(source), '--output', str(output), '--save-points', str(points), '--overlay', str(overlay)]) == 0
    assert cv2.imread(str(output), cv2.IMREAD_UNCHANGED).ndim == 2
    assert overlay.exists() and points.exists()
    assert len(BoundaryPoints.load(points).top) == 35
    assert main(['--input', str(source), '--output', str(tmp_path/'repeat.png'), '--points', str(points), '--no-shading']) == 0


def test_blank_image_fails_detection():
    from dewarp import detect_boundaries
    import numpy as np
    with pytest.raises(ValueError, match='No usable edges'):
        detect_boundaries(np.full((128,128,3), 255, dtype=np.uint8))
