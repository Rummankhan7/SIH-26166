# 🌕 Lunar Multi-Sensor Image Registration Pipeline

**Automated, memory-safe co-registration of Chandrayaan-2 OHRC, TMC-2, and IIRS imagery — from raw PDS4 archives to sub-pixel-accurate, geometrically registered products.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![GDAL](https://img.shields.io/badge/GDAL-Rasterio-2E7D32?logo=gdal&logoColor=white)](https://rasterio.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Runtime-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

---

## 📖 Overview

Chandrayaan-2 carries three imaging instruments with wildly different spatial resolutions and no shared reference frame out of the box:

| Instrument | Role | Native GSD | Character |
|---|---|---|---|
| **OHRC** (Orbiter High Resolution Camera) | Finest detail | ~0.3 m/px | Narrow, very high-resolution swath |
| **TMC-2** (Terrain Mapping Camera-2) | Mid-resolution bridge | ~5 m/px | Wider panchromatic strip, GCP-georeferenced |
| **IIRS** (Imaging Infrared Spectrometer) | Coarsest, hyperspectral | ~80 m/px | 256-band hyperspectral cube, **no per-pixel ground control** |

This project implements a full **8-stage, production-style geospatial pipeline** that ingests raw PDS4 lunar data and produces sub-pixel co-registered image pairs (**OHRC↔TMC-2** and **TMC-2↔IIRS**), along with quantitative accuracy metrics (RMSE, inlier ratio, spatial coverage) — the deliverables required for a rigorous image-registration evaluation.

The pipeline is deliberately engineered around one hard constraint: **every stage must run inside a memory-limited Google Colab runtime**, even though OHRC strips and IIRS hyperspectral cubes are far too large to load into RAM in full. This shaped almost every architectural decision below — streamed I/O, windowed reads, decimated matching, and GDAL-native warping instead of in-Python resampling.

---

## 🧭 Problem Statement Summary

> Given raw, unregistered multi-sensor lunar imagery (OHRC, TMC-2, IIRS) at three different resolutions and with inconsistent georeferencing, produce:
> 1. A **registered image product** aligning the finer sensor onto the coarser sensor's grid,
> 2. The **match points** used to compute the alignment,
> 3. A **quantitative evaluation** (accuracy + robustness) of the registration.

The core technical challenges this pipeline solves:

- **No common format** — OHRC/TMC-2 arrive as PDS4-labeled `.img` files, IIRS as ENVI `.qub`/`.hdr` cubes.
- **No common radiometry** — cross-sensor illumination and dynamic range differ enormously.
- **No common ground truth** — only OHRC and TMC-2 ship with real per-pixel GCPs; IIRS has none.
- **No common scale** — a ~250:1 resolution gap between OHRC and IIRS.
- **No common footprint** — each strip's true geographic overlap is a small fraction of the full scene.
- **No common memory budget** — full-resolution arrays for these products can exceed a Colab session's RAM outright.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RAW PDS4 ARCHIVE                                    │
│           OHRC (.img)     TMC-2 (.img)     IIRS (.qub + .hdr)                   │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                     ▼
                    ① INGEST  (01_ingest_pds4.ipynb)
        GDAL PDS4 driver → ENVI driver → manual XML parser fallback
        Streamed band-by-band write for the 256-band IIRS cube
                                     ▼
                    ② RADIOMETRIC NORMALIZATION  (02_radiometric_preprocess.ipynb)
        Percentile stretch → blockwise CLAHE → streaming PCA (IIRS band reduction)
                                     ▼
                    ③ GEOMETRIC PREPROCESSING  (03_geometric_preprocess.ipynb)
        Real GCP-based georeferencing (OHRC, TMC-2) on a spherical Moon datum
        Measured ground-sampling-distance (GSD) derived from GCP geometry
                                     ▼
                    ③b COMMON-FOOTPRINT CROP  (03b_common_footprint_crop.ipynb)
        Geographic bbox intersection → orientation-agnostic windowed crop
        IIRS: label-derived approximate footprint, else full-frame fallback
                                     ▼
                    ④ SCALE-SPACE CONSTRUCTION  (04_scale_space.ipynb)
        Gaussian pyramid (cv2.pyrDown) driven by REAL measured GSD ratios
        Aligns fine imagery to the coarse sensor's effective resolution
                                     ▼
                    ⑤ CORRESPONDENCE MATCHING  (05_correspondence_matching.ipynb)
     ┌───────────────┬──────────────────────────┬─────────────────┬──────────────┐
     │     SIFT       │  SuperPoint + LightGlue  │       RoMa       │    MINIMA    │
     │  (classical)   │   (learned, sparse)      │  (dense, robust) │ (modality-inv)│
     └───────────────┴──────────────────────────┴─────────────────┴──────────────┘
        Decimated-read matching + coordinate rescale-back to full resolution
                                     ▼
                    ⑥ OUTLIER REJECTION & WARPING  (06_transform_and_warp.ipynb)
        MAGSAC/RANSAC homography fit → best-of-3-methods selection
        Uniform spatial-coverage enforcement (8×8 grid)
        Sub-pixel refinement (windowed phase correlation)
        gdalwarp-based resampling (block-wise, memory-safe)
                                     ▼
                    ⑦ EVALUATION & REPORT  (07_evaluate.ipynb)
        Held-out RMSE • inlier ratio • coverage score • checkerboard QA
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             FINAL DELIVERABLES                                  │
│   registered.tif   •   match_points.csv   •   evaluation_report.json /csv       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Design principle throughout:** every notebook reads the manifest produced by the previous one and writes both an output artifact *and* an updated manifest — the pipeline is a chain of small, independently re-runnable, disk-backed stages rather than one monolithic in-memory script.

---

## 🛠️ Tech Stack — What & Why

| Technology | Used For | Why This Choice |
|---|---|---|
| **Rasterio / GDAL** | Reading PDS4/ENVI rasters, windowed I/O, GCP-based georeferencing, format conversion | The only libraries with native **PDS4** and **ENVI** drivers, plus windowed/decimated reads that avoid materializing full-resolution arrays in RAM — essential given OHRC/IIRS file sizes |
| **`gdalwarp` / `gdal_translate` (CLI, via `subprocess`)** | Final image resampling onto the reference grid | GDAL warps **block-by-block internally** in C, unlike `cv2.warpPerspective`, which requires the entire source array in memory — the only viable way to resample a full-resolution OHRC strip in Colab |
| **OpenCV (`cv2`)** | CLAHE contrast enhancement, SIFT detection, homomorphic filtering, RANSAC/MAGSAC transform fitting, phase-correlation refinement | Mature, fast, battle-tested implementations of classical CV primitives; `USAC_MAGSAC` gives a more robust fit than plain RANSAC when available, with automatic fallback |
| **PyTorch** | Backend runtime for the learned matchers (SuperPoint, LightGlue, RoMa) | Required by all three learned-matching models; GPU acceleration is essential for RoMa's dense matching to run in reasonable time |
| **LightGlue + SuperPoint** (`cvg/LightGlue`) | Fast, sparse, learned feature matching | State-of-the-art learned matcher that is far more robust to illumination/viewpoint differences than SIFT, while remaining lightweight enough to run per-pair in Colab |
| **RoMa** (`Parskatt/RoMa`) | Dense, robust correspondence estimation | Handles the large scale/viewpoint gaps between sensors (especially TMC-2↔IIRS) better than sparse matchers by estimating a dense warp field rather than discrete keypoints |
| **MINIMA** (`LSXI7/MINIMA`) | Modality-invariant matching scaffold | Included as an extensible fourth matcher for cross-modal pairs (e.g. panchromatic vs. hyperspectral) where appearance differs more than geometry |
| **NumPy** | Array manipulation, GCP math, transform application | The common numerical substrate every other library above builds on |
| **Matplotlib** | Footprint plots, before/after visual QA, match-line diagrams, checkerboard blends | Fast, dependency-light plotting sufficient for inline Colab visual verification at every stage |
| **PROJ (via `+proj=longlat +a=1737400 +b=1737400`)** | Custom spherical **lunar** coordinate reference system | Avoids depending on whether the runtime's PROJ database ships IAU planetary authority codes — a self-contained CRS definition guarantees reproducibility across environments |
| **Google Colab + Google Drive** | Compute + persistent storage | Free GPU access for the learned matchers, with Drive as a durable filesystem across sessions for the multi-stage, disk-chained pipeline design |

---

## 🔬 Pipeline Stages in Detail

### ① `01_ingest_pds4.ipynb` — PDS4 Ingestion
Recursively scans the raw archive for PDS4 `.xml` labels, skipping browse/geometry folders. Reads pixel data using a **three-tier fallback strategy**: GDAL's native PDS4 driver → GDAL's ENVI driver (for `.qub`+`.hdr` IIRS cubes) → a hand-written PDS4 label parser as a last resort. IIRS's 256-band cube is streamed **band-by-band** directly to GeoTIFF to avoid loading the full cube into memory. Produces a `manifest.csv` tracking every converted product.

### ② `02_radiometric_preprocess.ipynb` — Illumination Normalization
Computes robust percentile stretch bounds from a decimated read, then applies **blockwise CLAHE** (tiled, overlap-padded) so contrast enhancement stays consistent across tile boundaries without ever holding a full-resolution array. IIRS is first reduced from 256 bands to a single representative channel via a **streaming PCA** (computed from randomly sampled row-blocks, never the full cube).

### ③ `03_geometric_preprocess.ipynb` — Real Ground-Control Georeferencing
Parses the per-product `geometry/*.csv` GCP files (Longitude/Latitude/Pixel/Scan) that ship alongside OHRC and TMC-2, fits an affine transform with `rasterio.transform.from_gcps`, and attaches a proper CRS on a spherical Moon datum. Also **derives the true ground-sampling distance** from great-circle distances between adjacent GCPs, rather than trusting nominal mission specs. IIRS, lacking any GCP file, is intentionally left unregistered at this stage — its geolocation is inherited transitively later via feature matches to TMC-2.

### ③b `03b_common_footprint_crop.ipynb` — Common-Footprint Cropping
Computes the true geographic bounding-box intersection between sensor pairs and crops both images to just that shared area — using an **orientation-agnostic inverse-transform window computation** (avoiding a sign-convention bug in `rasterio.windows.from_bounds` on GCP-derived transforms) and a **fresh minimal GTiff profile** (avoiding stale tiling metadata conflicts). For IIRS, attempts to recover an approximate 4-corner footprint from PDS4 label bounding coordinates; falls back safely to full-frame passthrough if unavailable. This step means every downstream matcher operates only on pixels that could possibly correspond — dramatically improving match yield and runtime.

### ④ `04_scale_space.ipynb` — Scale-Space Construction
Builds a Gaussian pyramid (`cv2.pyrDown`) on the finer image, selecting the pyramid level whose effective GSD best matches the coarser sensor's GSD — using **measured** (GCP-derived) GSD where available, nominal mission values only as fallback. Preserves and correctly rescales georeferencing transforms through each pyramid level.

### ⑤ `05_correspondence_matching.ipynb` — Multi-Method Feature Matching
Runs **four** correspondence methods per pair for genuine comparison: classical **SIFT**, learned sparse **SuperPoint+LightGlue**, dense robust **RoMa**, and a **MINIMA** scaffold for modality-invariant matching. All matching happens on **GDAL-decimated reads** (capped at a configurable pixel budget) to stay memory-safe, with resulting keypoints **rescaled back** to each image's own full-resolution pixel grid immediately after detection — so downstream stages need no awareness of the decimation.

### ⑥ `06_transform_and_warp.ipynb` — Outlier Rejection, Refinement & Warping
For each pair, evaluates all matcher outputs, fits a robust homography via **MAGSAC (falling back to RANSAC)**, and **keeps whichever method yields the most inliers**. Enforces uniform spatial coverage of surviving matches via an 8×8 grid cap, then refines each kept match to **sub-pixel accuracy** using windowed phase correlation. A 20% held-out test split is set aside *before* fitting, purely for an unbiased RMSE later. Final resampling is delegated to **`gdalwarp`**, which handles arbitrarily large rasters block-by-block without a full in-memory decode.

### ⑦ `07_evaluate.ipynb` — Evaluation & Final Report
Computes held-out RMSE on the test split, reports inlier count/ratio and spatial coverage score from stage ⑥, generates checkerboard-blend and side-by-side visual QA, and compiles a per-pair `evaluation_report.json` plus an aggregate `final_report.csv` — the final submission deliverable.

---

## 📊 Evaluation Metrics

| Metric | Meaning | Computed In |
|---|---|---|
| **RMSE (px)** | Residual error on a held-out 20% test split, never seen during transform fitting | `07_evaluate.ipynb` |
| **Inlier count / ratio** | Geometric consistency of matches under the fitted homography | `06_transform_and_warp.ipynb` |
| **Coverage score** | Fraction of an 8×8 spatial grid containing at least one inlier match — penalizes matches clustered in one corner | `06_transform_and_warp.ipynb` |
| **Checkerboard / side-by-side QA** | Visual alignment sanity check between reference and warped-registered product | `07_evaluate.ipynb` |

---

## 📁 Data & Output Layout

```
26166_DATA/
├── raw/{ohrc,tmc2,iirs}/{browse,data,geometry,miscellaneous}/{raw|calibrated}/<date>/...
└── processed/
    ├── geotiff/{ohrc,tmc2,iirs}/<product>.tif           # ① ingest
    ├── manifest.csv
    ├── radiometric/{ohrc,tmc2,iirs}/<product>.tif        # ② radiometric
    ├── manifest_radiometric.csv
    ├── geocoded/{ohrc,tmc2}/<product>.tif                # ③ geometric
    ├── manifest_geocoded.csv
    ├── common_crop/{ohrc_to_tmc2,tmc2_to_iirs}/*.tif     # ③b common-footprint crop
    ├── manifest_common_crop.csv
    ├── scalespace/<pair>/*.tif                            # ④ scale-space
    ├── manifest_scalespace.csv
    ├── matches/<pair>/{sift,lightglue,roma}_raw_matches.csv  # ⑤ matching
    ├── registered/<pair>/
    │   ├── registered.tif            ⟵ final registered product
    │   ├── match_points.csv          ⟵ final match points
    │   ├── transform_meta.json
    │   └── evaluation_report.json    ⟵ per-pair metrics
    └── final_report.csv              ⟵ submission-level summary
```

---

## 🚀 Getting Started

1. **Mount your data** in Google Drive following the `raw/` layout shown above.
2. **Open notebooks in order** in Google Colab, `01` → `07`, running each top-to-bottom.
3. **Enable a GPU runtime** (`Runtime → Change runtime type → GPU`) before `05_correspondence_matching.ipynb` — required for LightGlue/RoMa.
4. Each notebook reads the manifest produced by its predecessor — re-running any single notebook is safe as long as upstream manifests still exist.
5. Inspect `final_report.csv` for the submission-ready summary of every sensor pair.

### Dependencies
```bash
pip install rasterio pyproj tqdm opencv-python torch numpy matplotlib
pip install git+https://github.com/cvg/LightGlue.git
pip install git+https://github.com/Parskatt/RoMa.git
apt-get install -y gdal-bin python3-gdal
```

---

## ⚠️ Known Limitations

- **IIRS has no independent ground truth** in this dataset (no `geometry/` GCP file) — its absolute geolocation accuracy is bounded by the quality of its matched-pixel chain through TMC-2, not by any direct measurement.
- The OHRC↔TMC-2 approximate footprint check should be verified visually (Notebook 03, §4) before investing in matching, since the sample acquisitions are several years apart and could in principle cover different ground.
- Homography order defaults to 1 (affine-like); order 2 is worth trying when ≥10 well-spread GCPs are available and RMSE is higher than expected.

---

## 📜 License

MIT — see `LICENSE` for details.

==============================================================================================================================

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
