# LunarXReg — Chandrayaan-2 Multi-Modal Image Registration

**A geometry-guided, confidence-aware registration framework for Chandrayaan-2 OHRC, TMC-2 and IIRS imagery.**

> **Diagnose before matching. Match adaptively. Register with confidence.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![GDAL](https://img.shields.io/badge/GDAL%20%2F%20Rasterio-Geospatial-2E7D32?logo=gdal&logoColor=white)](https://rasterio.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Optional%20GPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Runtime-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

---

## Overview

**LunarXReg** is an adaptive image-correspondence and registration pipeline for Chandrayaan-2 optical imagery. It is designed for the practical difficulty of aligning images acquired by different sensors, at different times, with different illumination, viewing geometry and spatial resolution.

The system supports three Chandrayaan-2 payload data types:

| Instrument | Role in LunarXReg | Approx. spatial resolution | Data character |
|---|---|---:|---|
| **OHRC** — Orbiter High Resolution Camera | Fine morphological detail | ~0.25–0.32 m/pixel | High-resolution panchromatic image |
| **TMC-2** — Terrain Mapping Camera-2 | Intermediate-resolution bridge | ~5 m/pixel | Panchromatic stereo-capable imagery |
| **IIRS** — Imaging Infrared Spectrometer | Spectral/mineralogical context | ~80 m/pixel | Hyperspectral cube with ~256 bands |

Rather than applying one matcher blindly to every image pair, LunarXReg follows a **diagnose → prepare → route → verify → certify** approach:

1. Diagnose scale gap, texture, illumination difference, modality and possible overlap.
2. Create a structural, illumination-reduced view for matching.
3. Use the hierarchical bridge:

   \[
   \text{OHRC} \leftrightarrow \text{TMC-2} \leftrightarrow \text{IIRS}
   \]

4. Select the most suitable matching route for the image pair.
5. Verify geometry robustly, balance match distribution, refine locally and quantify confidence.
6. Return **Accept**, **Review** or **Reject** instead of forcing an unreliable registration.

The pipeline produces:

- Registered image-space output and GeoTIFF **only when valid georeferencing is available**.
- Uniformly distributed match points.
- Transformation matrix and metadata.
- Before/after overlays and checkerboard visual QA.
- RMSE, inlier count, inlier ratio, spatial coverage and transform-stability metrics.
- A **Registration Confidence Score (RCS)** and final reliability decision.

---

## Problem Context

The problem requires multi-modal, Sun-angle and scale-invariant correspondence across Chandrayaan-2 OHRC, TMC-2 and IIRS imagery.

### Core challenges

| Challenge | Why it is difficult | LunarXReg response |
|---|---|---|
| **Illumination variation** | Different Sun azimuth/elevation changes shadows and surface appearance | Structural matching using CLAHE, homomorphic normalization, gradient/edge representations and shadow-confidence masks |
| **Viewpoint variation** | Different acquisition geometry creates shifts, rotation, scale and local distortion | Available geometry/GCP inspection, robust MAGSAC/USAC verification and local phase-correlation refinement |
| **Scale variation** | OHRC, TMC-2 and IIRS differ by large resolution ratios | Gaussian pyramids, compatible working resolution and hierarchical OHRC ↔ TMC-2 ↔ IIRS bridging |
| **Modality variation** | IIRS is hyperspectral while OHRC/TMC-2 are mainly panchromatic | IIRS PCA/selected-band structural view; original hyperspectral cube remains unchanged |
| **Uncertain overlap** | Different dates may observe different lunar footprints | Overlap diagnostics, explicit confidence thresholds and Reject/Review output rather than forced alignment |
| **Large data volume** | High-resolution strips and hyperspectral cubes can exceed Colab memory | Windowed reads, decimated working views, tiling, selected-band reads and streaming PCA |

---

## Key Innovation

### 1. Registration Quality Controller

LunarXReg diagnoses every source–reference pair before registration using:

- Approximate scale ratio.
- Image texture availability.
- Illumination/shadow reliability.
- Sensor or modality pair type.
- Available metadata and geometry.
- Preliminary overlap confidence.

This information routes each pair through the lightest reliable matching path instead of running every deep model for every image.

### 2. Hierarchical Resolution Bridge

Direct OHRC-to-IIRS matching is an extreme cross-scale task. LunarXReg reduces risk by using TMC-2 as a bridge:

\[
\text{OHRC} \leftrightarrow \text{TMC-2} \leftrightarrow \text{IIRS}
\]

The system uses compatible working resolutions and coarse-to-fine Gaussian pyramids. It does **not** claim that simple resampling creates missing spatial detail.

### 3. Sun-Neutral Structural Matching

Matching is performed on derived structural views rather than only raw brightness:

- Percentile normalization.
- CLAHE local-contrast enhancement.
- Homomorphic illumination normalization.
- Gradient magnitude and edge structure.
- Shadow/low-confidence masking.
- PCA-1 or selected-band structural representations for IIRS.

The original IIRS cube is preserved for science use; only a derived view is used for registration.

### 4. Grid-Balanced Tie Point Selection

LunarXReg prevents a transform from being fitted mainly from one texture-rich or repetitive region.

- The valid overlap is divided into grid cells.
- Candidate inliers are ranked using reprojection error, matcher confidence and local validity.
- A limited number of high-quality points is retained per cell.
- The final transform is refitted using spatially balanced evidence.

### 5. Confidence-Certified Registration

Every registration is assessed using a transparent **Registration Confidence Score (RCS)**.

\[
\text{RCS} = 100 \times
\left(
0.30R +
0.25I +
0.20C +
0.15S +
0.10L
\right)
\]

where:

| Symbol | Component | Meaning |
|---|---|---|
| \(R\) | Residual score | Low reprojection RMSE yields higher confidence |
| \(I\) | Inlier reliability | Uses both inlier count and inlier ratio |
| \(C\) | Spatial coverage | Measures grid-wide distribution of reliable matches |
| \(S\) | Transform stability | Measures sensitivity of the fitted transform under resampling |
| \(L\) | Illumination validity | Rewards matches outside shadow/no-data/low-confidence regions |

### Decision policy

| RCS | Decision | Interpretation |
|---:|---|---|
| **≥ 75** | **Accept** | Strong geometric support, adequate coverage and stable transform |
| **50–74** | **Review** | Partially reliable result; local uncertainty or limited coverage should be inspected |
| **< 50** | **Reject / Escalate** | Insufficient evidence; try a stronger matcher, a better overlap ROI or manual review |

Hard safety gates can prevent acceptance even when average RCS is high:

- Too few inliers.
- Low inlier ratio.
- Low spatial coverage.
- Invalid or degenerate transform.
- Non-finite or excessive reprojection error.

> RCS is an evidence-based reliability estimate. It does not guarantee correct registration.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             MULTI-SENSOR INPUT                              │
│    OHRC (.img + .xml + geometry CSV) -  TMC-2 (.img + .xml + geometry CSV)  │
│                    IIRS (.qub + .hdr + .xml) -  Browse PNG                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. DATA & METADATA INGESTION                                                │
│  PDS4/XML inspection -  ENVI header parsing -  geometry CSV inspection        │
│  GDAL/Rasterio first -  XML-assisted binary-reader fallback                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. SCENE DIAGNOSIS — Registration Quality Controller                       │
│  Scale gap -  texture -  illumination/shadow -  modality -  possible overlap    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. SENSOR-AWARE STRUCTURAL PREPARATION                                     │
│  CLAHE -  homomorphic normalization -  gradient/edge structure -  shadow mask  │
│  IIRS PCA/selected-band matching view; original cube preserved              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. HIERARCHICAL SCALE BRIDGE                                               │
│         OHRC ↔ TMC-2 ↔ IIRS -  compatible working GSD -  pyramids             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. ADAPTIVE MATCHER CASCADE                                                │
│  Fast baseline: SIFT / ORB / AKAZE                                          │
│  Optional learned routes: SuperPoint + LightGlue -  RoMa -  MINIMA            │
│  Selection is based on verified RCS, not raw match count                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. ROBUST GEOMETRY VERIFICATION                                            │
│  USAC_MAGSAC when available → RANSAC fallback                               │
│  Homography / affine model selection -  outlier rejection -  plausibility     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  7. GRID-BALANCED SUB-PIXEL REFINEMENT                                      │
│  Uniform tie-point selection -  local phase correlation -  quality checks     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  8. CONFIDENCE-CERTIFIED OUTPUT                                             │
│  Overlay -  image-space registration / GeoTIFF when valid georef exists      │
│  Match points -  RCS -  metrics -  HTML/JSON report -  Accept/Review/Reject     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Layout

Expected Google Drive root:

```text
MyDrive/
└── 26166_DATA/
    ├── raw/
    │   ├── iirs/
    │   │   ├── browse/raw/<date>/
    │   │   │   ├── *_b_brw_*.png
    │   │   │   └── *_b_brw_*.xml
    │   │   ├── data/raw/<date>/
    │   │   │   ├── *_d_img_*.hdr
    │   │   │   ├── *_d_img_*.qub
    │   │   │   └── *_d_img_*.xml
    │   │   └── miscellaneous/raw/<date>/
    │   │
    │   ├── ohrc/
    │   │   ├── browse/calibrated/<date>/
    │   │   ├── data/calibrated/<date>/
    │   │   │   ├── *_d_img_*.img
    │   │   │   └── *_d_img_*.xml
    │   │   └── geometry/calibrated/<date>/
    │   │       ├── *_g_grd_*.csv
    │   │       └── *_g_grd_*.xml
    │   │
    │   └── tmc2/
    │       ├── browse/calibrated/<date>/
    │       ├── data/calibrated/<date>/
    │       │   ├── *_d_img_*.img
    │       │   └── *_d_img_*.xml
    │       └── geometry/calibrated/<date>/
    │           ├── *_g_grd_*.csv
    │           └── *_g_grd_*.xml
    │
    └── results/
        ├── previews/
        ├── metadata/
        ├── structural_views/
        ├── matches/
        ├── registered/
        ├── reports/
        ├── synthetic_validation/
        └── logs/
```

The notebook creates the `results/` tree automatically.

---

## Implementation Details

### Data ingestion

LunarXReg uses a layered reader strategy:

1. Attempt GDAL/Rasterio data access.
2. Parse PDS4 XML labels for dimensions, data type, byte order, offsets, scaling and missing values.
3. Use NumPy binary-reading fallback for `.img` data when generic readers fail.
4. Use Spectral Python/ENVI header parsing for IIRS `.hdr + .qub` products.
5. Use browse PNG files only as a clearly labelled demo fallback if raw-product parsing fails.

The notebook never silently replaces raw data with a browse image. It logs the reason and marks the run accordingly.

### IIRS handling

IIRS is handled as a hyperspectral data product:

- Open only selected bands, row windows or spatial subsets where possible.
- Estimate memory before attempting full-cube reads.
- Use PCA-1 or selected spectral bands to create a structural matching view.
- Preserve the original cube; no registration preprocessing overwrites spectral science data.

### Geometric metadata

For OHRC and TMC-2, geometry CSV files are inspected for possible line/sample and latitude/longitude information.

Geometry metadata may help with:

- Scene inspection.
- Approximate overlap diagnosis.
- Optional mapping of matched image points to latitude/longitude.
- Validation consistency checks.

The MVP does **not** invent a CRS or claim full orthorectification when required metadata are unavailable.

---

## Matching Strategy

### Default robust CPU pipeline

The default MVP uses:

- **SIFT** with ratio-test and mutual-nearest-neighbour filtering.
- **ORB** fallback for fast binary features.
- **AKAZE** fallback for nonlinear scale-space features.
- **USAC_MAGSAC** geometric verification when available.
- **RANSAC** fallback when USAC_MAGSAC is unavailable.
- **Affine or homography** selection based on residual quality and transform plausibility.
- **Local phase correlation** for conservative sub-pixel refinement.

### Optional learned matching

The pipeline exposes optional integration points for:

| Method | Role |
|---|---|
| **SuperPoint + LightGlue** | Fast learned sparse correspondence |
| **RoMa** | Dense correspondence with confidence/certainty estimation |
| **MINIMA** | Modality-invariant matching for difficult cross-modal cases |

These are optional by default. The core notebook remains functional without GPU access or network-dependent installations.

---

## Output Products

Each run creates a dedicated timestamped result folder:

```text
results/
└── <source>_to_<reference>_<timestamp>/
    ├── previews/
    ├── metadata/
    ├── structural_views/
    ├── matches/
    │   ├── raw_matches.png
    │   ├── verified_inliers.png
    │   ├── grid_balanced_matches.png
    │   └── grid_coverage_heatmap.png
    ├── registered/
    │   ├── registered_image_space.png
    │   ├── overlay_alpha_blend.png
    │   ├── overlay_checkerboard.png
    │   ├── absolute_difference.png
    │   ├── transform.json
    │   └── match_points.csv
    ├── reports/
    │   ├── metrics.json
    │   ├── confidence_report.json
    │   ├── registration_report.html
    │   └── experiment_summary.csv
    └── logs/
        └── lunarxreg.log
```

### Match-point CSV fields

```text
source_x_working
source_y_working
reference_x_working
reference_y_working
source_x_original
source_y_original
reference_x_original
reference_y_original
reprojection_error_px
phase_response
shadow_valid_source
shadow_valid_reference
selected_after_grid_balance
```

### GeoTIFF policy

A GeoTIFF is generated **only** when trustworthy georeferencing/CRS information is available from the input or supplied explicitly.

If valid geographic metadata are unavailable, LunarXReg saves an **image-space registered TIFF/PNG** and explicitly labels it as such. It never invents a lunar CRS, projection or map coordinates.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Tentative matches** | Number of matches before robust geometric verification |
| **Inlier count** | Number of geometrically consistent matches after verification |
| **Inlier ratio** | Inliers divided by tentative matches |
| **RMSE** | Root-mean-square reprojection error in image pixels |
| **Median / P95 residual** | Robust residual statistics that reveal long-tail geometric error |
| **Coverage score** | Fraction of valid grid cells with retained tie points |
| **Transform stability** | Variation in warp prediction under bootstrap/resampling tests |
| **Illumination validity** | Fraction of selected matches outside shadow/no-data regions |
| **RCS** | Combined 0–100 evidence-based registration confidence score |
| **Decision** | Accept, Review or Reject |

### Evaluation principle

Where a held-out set of reliable points exists, it is excluded from transform fitting and used for independent residual evaluation.

For synthetic validation, the known synthetic transform provides controlled ground truth. Synthetic validation demonstrates pipeline correctness; it is **not** a substitute for scientific inter-sensor validation on physically overlapping lunar scenes.

---

## Synthetic Validation Mode

The supplied Colab workflow includes a controlled validation mode because images from different acquisition dates may not share a footprint.

Synthetic mode:

1. Loads a real OHRC/TMC-2/IIRS structural view.
2. Applies known rotation, scale, translation, optional perspective perturbation and illumination change.
3. Runs the exact same pipeline used in a real pair.
4. Compares the estimated transform with known ground truth.
5. Reports corner reprojection error, center displacement and RCS.

Example configuration:

```python
CONFIG["mode"] = "synthetic_validation"
CONFIG["synthetic_source_sensor"] = "tmc2"
```

> Synthetic Transformation Validation confirms technical pipeline behavior. It must be presented as a controlled algorithmic demonstration, not as final Chandrayaan-2 cross-sensor geolocation accuracy.

---

## Real-Pair Mode

For real image-pair experiments:

```python
CONFIG["mode"] = "real_pair"
CONFIG["source_sensor"] = "tmc2"
CONFIG["reference_sensor"] = "ohrc"
CONFIG["use_browse_preview_first"] = True
```

Other supported configurations include:

```python
# High-to-medium resolution attempt
CONFIG["source_sensor"] = "ohrc"
CONFIG["reference_sensor"] = "tmc2"

# Medium-to-hyperspectral attempt
CONFIG["source_sensor"] = "tmc2"
CONFIG["reference_sensor"] = "iirs"

# Reverse direction
CONFIG["source_sensor"] = "iirs"
CONFIG["reference_sensor"] = "tmc2"
```

A real pair may return **Reject** because acquisition dates, footprints, Sun geometry or available structural overlap may not support reliable correspondence. This is expected safe behavior.

---

## Getting Started

### 1. Upload the data

Arrange data in the `MyDrive/26166_DATA/raw/` structure shown in the **Data Layout** section.

### 2. Open the Colab notebook

Open the LunarXReg Colab notebook and run cells from top to bottom.

### 3. Mount Google Drive

The notebook mounts:

```python
/content/drive/MyDrive/26166_DATA/raw
```

and writes all outputs to:

```python
/content/drive/MyDrive/26166_DATA/results
```

### 4. Start with synthetic validation

```python
CONFIG["mode"] = "synthetic_validation"
CONFIG["synthetic_source_sensor"] = "tmc2"
```

This provides the most controlled first demo.

### 5. Run a real pair

After confirming ingestion and synthetic validation:

```python
CONFIG["mode"] = "real_pair"
CONFIG["source_sensor"] = "tmc2"
CONFIG["reference_sensor"] = "ohrc"
```

### 6. Review the final dashboard

The notebook displays:

- Source and reference structural views.
- Raw and verified matches.
- Grid-balanced correspondences.
- Registration overlay and checkerboard.
- RMSE, coverage, inlier statistics and RCS.
- Final **Accept / Review / Reject** result.
- Google Drive output paths.

---

## Dependencies

The CPU-compatible MVP uses:

```bash
pip install opencv-contrib-python-headless
pip install spectral rasterio tifffile pandas scikit-image
pip install matplotlib seaborn lxml tqdm
```

Core libraries:

```text
Python 3.10+
NumPy
OpenCV
Pandas
Matplotlib
Scikit-image
Spectral Python
Rasterio / GDAL
lxml
PyTorch (optional GPU-aware runtime detection)
```

Optional learned-matcher integrations should be enabled only after verifying compatible package versions and GPU availability.

---
## Prototype Scope and Validation Status

LunarXReg is a research-oriented MVP designed to demonstrate a complete, reproducible and confidence-aware registration workflow for Chandrayaan-2 multi-sensor imagery.

The publicly available sample products currently used in this prototype were acquired on different dates and may not have confirmed geographic overlap. Therefore, the repository supports two complementary validation modes:

1. **Synthetic Transformation Validation**  
   A known geometric and illumination transformation is applied to a real lunar input image. The pipeline then estimates the transformation and compares it with known ground truth. This validates feature matching, robust geometry estimation, grid balancing, local refinement and Registration Confidence Score computation.

2. **Real-Pair Feasibility Evaluation**  
   OHRC–TMC-2 and TMC-2–IIRS pairs are ingested, structurally prepared and evaluated without forcing a result. The pipeline outputs **Accept**, **Review** or **Reject** according to geometric evidence, spatial coverage and confidence thresholds.

A successful real cross-sensor registration requires verified physical overlap, adequate shared terrain structure, compatible acquisition geometry and sufficient illumination support. When an input pair lacks this evidence, LunarXReg reports the limitation rather than generating a misleading alignment.

The architecture is designed to ingest the official SIH-provided overlapping data products when available.

## Known Limitations

- **Physical overlap is required.** Separate Chandrayaan-2 acquisitions may observe different lunar footprints; the pipeline must not force a result.
- **Browse images are not equivalent to calibrated science products.** They are used only as clearly labelled preview/demo fallbacks when raw-product ingestion fails.
- **IIRS direct absolute control may be limited.** Its alignment can be supported through TMC-2, but propagated uncertainty must be reported.
- **Pretrained learned matchers may exhibit lunar domain gap.** Natural-image training may not generalize perfectly to repeated crater textures, extreme shadows or spectral representations.
- **Sub-pixel refinement is local.** Phase correlation refines local image-space offsets; it does not by itself prove sub-pixel geolocation accuracy.
- **GeoTIFF output depends on valid input georeferencing.** The pipeline will not fabricate a lunar projection or CRS.
- **A global homography is not always physically sufficient.** Local terrain, pushbroom acquisition geometry and residual sensor distortion may require tiled/local models in future upgrades.
- **The default notebook prioritizes a reliable CPU-compatible MVP.** Learned models are optional enhancements, not prerequisites.

---

## Future Extensions

- Camera-model and SPICE-kernel integration for improved geometry initialization.
- DEM-informed map projection and slope-aware transform modelling.
- Lunar-domain fine-tuning or synthetic relighting augmentation.
- RoMa/LightGlue/MINIMA production integration with controlled version pinning.
- Tile-wise local affine / thin-plate-spline correction after stable global initialization.
- DEM/LOLA/LRO-based independent validation where available.
- GeoJSON/CSV match exports with reliable latitude/longitude interpolation.
- Automated overlap prediction from metadata and geometry grids.
- Interactive analyst review interface for **Review** cases.

---

## Research Foundation

1. Indian Space Science Data Centre (ISSDC), “Chandrayaan-II: ISRO Science Data Archive (ISDA),” PRADAN.  
   https://pradan.issdc.gov.in/ch2/

2. Indian Space Science Data Centre (ISSDC), “Chandrayaan Data Explorer,” CHMAP Browse.  
   https://chmapbrowse.issdc.gov.in/

3. P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, “LightGlue: Local Feature Matching at Light Speed,” *Proc. IEEE/CVF ICCV*, 2023, pp. 17627–17638.

4. J. Ren, X. Jiang, Z. Li, D. Liang, X. Zhou, and X. Bai, “MINIMA: Modality Invariant Image Matching,” *Proc. IEEE/CVF CVPR*, 2025, pp. 23059–23068.

5. J. Edstedt, Q. Sun, G. Bökman, M. Wadenbäck, and M. Felsberg, “RoMa: Robust Dense Feature Matching,” *Proc. IEEE/CVF CVPR*, 2024, pp. 19790–19800.

6. D. Baráth, J. Noskova, M. Ivashechkin, and J. Matas, “MAGSAC++, a Fast, Reliable and Accurate Robust Estimator,” *Proc. IEEE/CVF CVPR*, 2020, pp. 1301–1309.

7. D. DeTone, T. Malisiewicz, and A. Rabinovich, “SuperPoint: Self-Supervised Interest Point Detection and Description,” *Proc. IEEE/CVF CVPR Workshops*, 2018, pp. 224–236.

8. NASA Ames Stereo Pipeline, “Chandrayaan-2 Lunar Orbiter Processing Documentation.”  
   https://stereopipeline.readthedocs.io/en/latest/examples/chandrayaan2.html

9. Mapbox, “Rasterio Documentation.”  
   https://rasterio.readthedocs.io/

10. Indian Space Research Organisation (ISRO), “Chandrayaan-2 Begins Spectroscopic Studies of Lunar Surface.”  
    https://www.isro.gov.in/Chandrayaan2begins.html

---

## License

MIT — see `LICENSE` for details.
