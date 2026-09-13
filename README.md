# SIH 26166 — Multi-modal, Sun-angle & Scale-invariant Image Correspondence

Full pipeline (this is your system architecture diagram)

Ingestion — pull PDS4 zip bundles from PRADAN/ISSDC, parse XML labels, convert to GeoTIFF/NumPy via GDAL, cache to Google Drive in a fixed folder structure.

Radiometric preprocessing — per-instrument calibration (already in "Calibrated" product level), then contrast-limited adaptive histogram equalization (CLAHE) or illumination normalization (e.g., homomorphic filtering) to reduce sun-angle-driven contrast differences before feature extraction.

Geometric preprocessing — reproject all three to a common lunar reference frame (use TMC-2 DEM/Ortho + available SPICE geometry kernels for rough initial alignment); this removes gross viewpoint offset before fine matching, so the matcher only has to resolve residual/local distortion.

Scale-space construction — build image pyramids per source so the matcher gets candidate patches at comparable effective GSD (this is what actually solves "scale invariance," not just hoping the network generalizes).

Correspondence generation — run chosen matcher(s) hierarchically (OHRC→TMC2→IIRS) per §4.

Outlier rejection & transform estimation — MAGSAC++ (more robust than plain RANSAC) to fit the geometric model; use affine/homography for locally flat patches, and a higher-order (polynomial / thin-plate-spline / piecewise-affine) model if the DEM shows meaningful relief across the tile, since lunar terrain isn't flat.

Sub-pixel refinement — after coarse matches + transform, do local normalized cross-correlation (NCC) or phase-correlation refinement around each inlier to push accuracy below 1 pixel.

Match distribution / uniformity enforcement — grid the image into an N×N cell layout, cap matches per cell (or resample so every cell contributes at least one inlier), and report a coverage/uniformity score — this is explicitly asked for and easy to lose points on if skipped.

Warping & product generation — warp the source image onto the reference grid, export as GeoTIFF (registered product), export match points as CSV/GeoJSON/shapefile with pixel + map coordinates.

Evaluation & reporting — RMSE (on held-out checkpoints), inlier count, inlier ratio, uniformity score, and a checkerboard/blend visualization for qualitative proof.
