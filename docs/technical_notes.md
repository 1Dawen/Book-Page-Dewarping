# Technical notes and limitations

## Source and refactoring

The original Computer Graphics midterm project consisted of a single notebook (`mid.ipynb`), a report, and one transformed JPEG. Its photographed source `DewarpBook.jpg` was not present in the archived project. The original method is described as multiple-line (piecewise) homography. It used CLAHE/adaptive thresholding, Canny edge detection, spline smoothing, an OpenCV mouse editor, independent per-cell perspective warps, and grayscale shading correction.

This repository reorganizes that method into a testable Python package, replaces the expensive full-image-per-cell warp with local patch warps, and aligns top/bottom control points at the same horizontal sample positions. The result is a revised implementation, **not an independently verified reproduction of the original notebook output**.

## Limitations

- The edge detector is heuristic. The original project explicitly required manual correction of control points. This remains necessary for many real photos, especially multiple-page images, shadows, text-heavy backgrounds, and folded pages.
- The demonstration image contains artificially drawn curves and text strokes. It tests implementation behavior, not the quality of dewarping on photographed pages.
- Per-cell homography may leave seams or distortion when source boundaries are inaccurate or when columns are not a sufficient model of the actual page deformation.
- Shading correction divides the warped grayscale by a Gaussian-blurred background. It can suppress broad shadows, but it is not a calibrated white-balance correction and can amplify noise.
- The original report includes examples from a textbook page. It is not included in this public package pending a check of rights to redistribute those images. The authored report and original transformed JPEG remain in the supplied private archive.

## Method

1. Estimate top and bottom rows of page-boundary control points using a contrast-based foreground mask when the page can be isolated, with a Canny edge heuristic as fallback. Smooth the samples with splines.
2. Display an overlay and optionally adjust points in the desktop GUI. Points can be exported as JSON.
3. Divide the page into adjacent strips and a chosen number of horizontal cells.
4. Derive the four source corners of each cell by interpolating between top and bottom control points.
5. Compute a separate projective transform for each cell and resample only its destination ROI.
6. Optionally apply grayscale background/shading correction.

This workflow follows the original report's organization, but removes discarded mask/corner computations that did not affect the actual transform in the original notebook.
