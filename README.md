<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="Pose Heatmaps: Where is every joint, and how sure is the model? Measured against Kinovea.">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-E6C35C?style=for-the-badge" alt="Python"> <img src="https://img.shields.io/badge/MMPose-E6C35C?style=for-the-badge" alt="MMPose"> <img src="https://img.shields.io/badge/HRNet--W32-E6C35C?style=for-the-badge" alt="HRNet-W32"> <img src="https://img.shields.io/badge/OpenCV-E6C35C?style=for-the-badge" alt="OpenCV"> <img src="https://img.shields.io/badge/Matplotlib-E6C35C?style=for-the-badge" alt="Matplotlib"> <img src="https://img.shields.io/badge/Kinovea-E6C35C?style=for-the-badge" alt="Kinovea">
</p>

<img src="./assets/readme/title-features.svg" width="100%" alt="01 Features: What it does.">
<img src="./assets/readme/features.svg" width="100%" alt="Pose Heatmaps features">

<img src="./assets/readme/title-flow.svg" width="100%" alt="02 Pipeline: From video to verdict.">
<img src="./assets/readme/flow.svg" width="100%" alt="Detect → Estimate → Analyse → Validate">

<img src="./assets/readme/title-table.svg" width="100%" alt="03 Outputs: What each step produces.">
<img src="./assets/readme/table.svg" width="100%" alt="Outputs">

<img src="./assets/readme/title-stack.svg" width="100%" alt="04 Tech stack: The tools behind it.">
<img src="./assets/readme/stack.svg" width="100%" alt="Pose Heatmaps tech stack">

<img src="./assets/readme/title-run.svg" width="100%" alt="05 Run it: Try it yourself.">

## Quick start (Google Colab)

Open `notebooks/Human_Pose_Estimation_MMPose.ipynb` in Colab with a **GPU runtime** and run the cells in order.
The notebook doesn't need anything else from this repo: it writes all the code itself.

The first install compiles MMCV from source, which can take 30–60 min on Colab. The compiled wheel is
cached in Google Drive, so later sessions install it in seconds.


## Local install (Linux, Python 3.10–3.12)

```bash
pip install torch torchvision            # pick the CUDA build for your machine
pip install "setuptools<70" ninja "mmengine==0.10.7" json_tricks munkres opencv-python matplotlib pandas scipy openpyxl
pip install --no-cache-dir --no-build-isolation xtcocotools       # must be built against your NumPy
MMCV_WITH_OPS=1 pip install --no-build-isolation "mmcv==2.2.0"   # compiles; needs torch already installed
pip install --no-deps "mmpose==1.3.2" "mmdet==3.3.0"              # --no-deps skips chumpy (breaks on py3.12)
# mmpose imports mmdet utils; mmdet asserts mmcv<2.2.0 although 2.2.0 works -> relax the check:
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" \
    "$(python -c 'import importlib.util; print(importlib.util.find_spec("mmdet").origin)')"
```

Why it's done this way:

* **MMCV** is built from source because OpenMMLab has no prebuilt wheels for recent PyTorch.
* **setuptools<70** is needed because MMCV's `setup.py` still imports `pkg_resources`.
* **mmdet** is installed only because MMPose imports a few of its utilities. Its MMCV version check is relaxed, and people are detected with torchvision's Faster R-CNN, not mmdet.
* **xtcocotools** has to be compiled against the NumPy you have installed. Otherwise you get `numpy.dtype size changed`.


## Usage

```bash
# 1. pose estimation (video, image, or folder of frames)
python scripts/run_pose.py --input data/trial.mp4 --out results/trial
#    --no-detector   whole frame = person box (subject fills the frame)
#    --model resnet50 | hrnet-w48,  --stride 2,  --max-frames 300,  --device cpu

# 2. heatmap analysis (+ overlay video)
python scripts/analyse_heatmaps.py --results results/trial --video
python scripts/analyse_heatmaps.py --results results/trial --video --joint left_knee

# 3. Kinovea validation
python scripts/compare_kinovea.py --kinovea data/trial_kinovea.csv --inspect     # see the columns
cp data/kinovea_mapping_example.json data/kinovea_mapping.json                   # edit column names
python scripts/compare_kinovea.py --results results/trial --kinovea data/trial_kinovea.csv \
       --mapping data/kinovea_mapping.json --auto-offset 10
```

### Kinovea data
Track the joints in Kinovea on the **same video**, using the trajectory tool and/or angle tools,
and export to CSV or XLSX. Because the export layout differs between Kinovea versions and
languages, a small JSON mapping tells the loader which column is which:

```json
{
  "time_column": "Time (ms)", "time_unit": "ms",
  "points": {"left_knee": {"x": "Knee/X", "y": "Knee/Y"}},
  "angles": {"left_knee": "Knee angle"},
  "coords": {"px_per_unit": 1.0, "origin_px": [0, 0], "y_up": false}
}
```

* Joint names must be COCO names (`left_knee`, `right_ankle`, …).
* Files that use `;` as the separator and `,` as the decimal mark are handled automatically.
* `coords` converts Kinovea's coordinate system into image pixels. If Kinovea was calibrated
  in metres, compare **joint angles**: they don't depend on units.
* `--auto-offset N` fixes a start-time mismatch by searching ±N frames.
* `--auto-align` fits scale and offset automatically. Use it only when the coordinate system is
  unknown, because it also hides any global bias of the model.


## Outputs

```
results/<name>/
  keypoints.csv  heatmaps.npz  meta.json
  heatmap_analysis/   heatmap_metrics.csv, joint_summary.csv, confidence_timeline.png,
                      joint_summary.png, confidence_vs_spread.png, grid_frame_*.png,
                      overlay_frame_*.jpg, heatmap_overlay.mp4
  kinovea_validation/ keypoint_summary.csv, angle_agreement.csv, confidence_vs_error.csv,
                      error_per_joint.png, trajectory_<joint>.png, angles.png, report.json
```


## Repo layout

```
posestudy/   skeleton.py (COCO joints, angles) · estimator.py (MMPose) · heatmaps.py · kinovea.py
scripts/     run_pose.py · analyse_heatmaps.py · compare_kinovea.py
notebooks/   Colab notebook (generated by tools/build_notebook.py; re-run it after editing code)
docs/assets/ README banner (generated by tools/make_preview.py)
tests/       test_synthetic_pipeline.py: checks steps 2–3 on synthetic data, no MMPose needed
```


## About the author

<p align="center">
  <img src="docs/assets/author-banner.png" alt="Hemashree" width="100%">
</p>

Built during my research internship at VIT Chennai (May–Jul 2025) under Prof. Pavithra,
then rebuilt and extended as this open-source project. GitHub: [@HemashreeVI](https://github.com/HemashreeVI)


## License

[MIT](LICENSE)

<p align="center">
  <img src="./assets/readme/footer.svg" width="100%" alt="Research internship · VIT Chennai · May–Jul 2025 Built by Hemashree.">
</p>
