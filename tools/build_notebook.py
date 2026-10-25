"""Generate notebooks/Human_Pose_Estimation_MMPose.ipynb from the source files.

The notebook is self-contained for Google Colab: it writes the `posestudy`
package and scripts with %%writefile cells, so it runs without cloning this
repo. Re-run this after editing any .py file:

    python tools/build_notebook.py
"""

import json
import os.path as osp

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
OUT = osp.join(ROOT, "notebooks", "Human_Pose_Estimation_MMPose.ipynb")
PROJECT_FILES = [
    "posestudy/__init__.py", "posestudy/skeleton.py", "posestudy/estimator.py",
    "posestudy/heatmaps.py", "posestudy/kinovea.py",
    "scripts/run_pose.py", "scripts/analyse_heatmaps.py", "scripts/compare_kinovea.py",
    "data/kinovea_mapping_example.json",
]

cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text, hidden=False):
    meta = {"cellView": "form"} if hidden else {}
    cells.append(dict(cell_type="code", metadata=meta, execution_count=None,
                      outputs=[], source=text.strip("\n")))


md("""
# Human Pose Estimation using Heatmaps (MMPose)

Research-internship project: **heatmap-based 2-D human pose estimation** with
[MMPose](https://github.com/open-mmlab/mmpose), analysis of the
**joint-confidence heatmaps** with OpenCV + Matplotlib, and **validation against
Kinovea** motion-analysis measurements.

| Step | What happens |
|---|---|
| 1 | Install MMPose (MMEngine + MMCV) on Colab |
| 2 | Write the project code (`posestudy/` package) |
| 3 | Run HRNet-W32 (top-down heatmap model, COCO-17 joints) on an image / video, keeping the raw heatmaps |
| 4 | Visualise heatmaps, measure how well each joint is localised, flag unreliable predictions |
| 5 | Compare keypoints and joint angles with Kinovea (error px, PCK, angle MAE, Bland-Altman) |

**Runtime → Change runtime type → GPU (T4)** before starting.
""")

md("## 1. Setup")
code("""
!nvidia-smi -L || echo "No GPU - inference will run on CPU (slow but works)"
import torch, sys
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| python", sys.version.split()[0])
""")

code("""
#@title Optional: mount Google Drive (keeps results + the compiled MMCV wheel between sessions)
USE_DRIVE = True  #@param {type:"boolean"}
DRIVE_DIR = "/content/drive/MyDrive/pose-heatmap-study"  #@param {type:"string"}
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    import os; os.makedirs(DRIVE_DIR, exist_ok=True)
""")

md("""
### Install MMPose

MMPose 1.x needs **MMCV with compiled CUDA ops**. OpenMMLab has no prebuilt MMCV
wheels for the PyTorch version Colab ships today, so MMCV is compiled from source
the first time (**roughly 30-60 min on Colab** — a CPU-only build took 23 min on 4 cores; Colab has 2 cores and also compiles CUDA kernels). The wheel is then saved to Drive, and
later sessions install it in seconds (as long as Colab's PyTorch version stays
the same).

Other notes:
* `mmpose` is installed with `--no-deps` to skip `chumpy` (only needed for SMPL, fails to build on Python 3.12).
* MMPose imports a few `mmdet` utilities, so `mmdet` is installed (`--no-deps`) and its MMCV version check is relaxed. People are detected with torchvision's Faster R-CNN, not mmdet.
* `setuptools<70` is needed because MMCV's `setup.py` still imports `pkg_resources`.
""")
code("""
import glob, os, subprocess, sys, torch

def mmcv_ok():
    r = subprocess.run([sys.executable, "-c", "from mmcv.ops import nms; import mmcv; print(mmcv.__version__)"],
                       capture_output=True, text=True)
    return r.returncode == 0

!pip install -q "setuptools<70" ninja "mmengine==0.10.7" json_tricks munkres
# --no-cache-dir: compile xtcocotools against the NumPy that is installed now
# (a cached wheel built for another NumPy fails with "numpy.dtype size changed")
!pip install -q --no-cache-dir --no-build-isolation xtcocotools

tag = f"torch{torch.__version__.split('+')[0]}_cu{(torch.version.cuda or 'cpu').replace('.', '')}"
wheel_dir = f"{DRIVE_DIR}/wheels/{tag}" if USE_DRIVE else f"/content/wheels/{tag}"
os.makedirs(wheel_dir, exist_ok=True)

if not mmcv_ok():
    cached = glob.glob(f"{wheel_dir}/mmcv-2.2.0*.whl")
    if cached:
        print("Installing cached MMCV wheel:", cached[0])
        !pip install -q {cached[0]}
    else:
        print(f"Compiling MMCV 2.2.0 for {tag} - this takes a while (only once) ...")
        os.environ.update(MMCV_WITH_OPS="1", FORCE_CUDA="1" if torch.cuda.is_available() else "0",
                          MAX_JOBS=str(os.cpu_count()))
        !pip wheel --no-build-isolation --no-deps "mmcv==2.2.0" -w {wheel_dir}
        built = glob.glob(wheel_dir + "/mmcv-2.2.0*.whl")[0]
        !pip install -q {built}

!pip install -q --no-deps "mmpose==1.3.2" "mmdet==3.3.0"
# mmpose imports a few mmdet utilities; mmdet asserts mmcv < 2.2.0 but works fine with 2.2.0
import importlib.util, re
init = importlib.util.find_spec("mmdet").origin
src = open(init).read()
open(init, "w").write(re.sub(r"mmcv_maximum_version = '[0-9.]+'", "mmcv_maximum_version = '2.3.0'", src))
assert mmcv_ok(), "MMCV ops failed to import - see the build log above"
!python -c "import mmcv, mmengine, mmpose; print('mmcv', mmcv.__version__, '| mmengine', mmengine.__version__, '| mmpose', mmpose.__version__)"
""")

md("""
## 2. Project code

These cells write the `posestudy` package and command-line scripts to
`/content/pose-heatmap-study`. Same code as the GitHub repo.
""")
code("""
import os
os.makedirs("/content/pose-heatmap-study/posestudy", exist_ok=True)
os.makedirs("/content/pose-heatmap-study/scripts", exist_ok=True)
os.makedirs("/content/pose-heatmap-study/data", exist_ok=True)
%cd /content/pose-heatmap-study
""")
for rel in PROJECT_FILES:
    with open(osp.join(ROOT, rel)) as f:
        src = f.read()
    code(f"%%writefile {rel}\n{src}")

md("## 3. Run pose estimation")
code("""
import sys, os
sys.path.insert(0, "/content/pose-heatmap-study")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, Video, display
from posestudy import heatmaps as hm, kinovea as kv
from posestudy.estimator import load_pose_model, PersonDetector, run_on_video

pose_model = load_pose_model("hrnet-w32")   # or "hrnet-w48", "resnet50"
detector = PersonDetector()                 # set to None if the subject fills the frame
""")

code("""
#@title Input: sample image, or upload your own video
SOURCE = "sample_image"  #@param ["sample_image", "upload"]
os.makedirs("data", exist_ok=True)
if SOURCE == "sample_image":
    !wget -q -O data/sample.jpg https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg
    INPUT = "data/sample.jpg"
else:
    from google.colab import files
    up = files.upload()
    INPUT = "data/" + list(up)[0]
    os.replace(list(up)[0], INPUT)
print("input:", INPUT)
""")

code("""
RESULTS = "results/" + os.path.splitext(os.path.basename(INPUT))[0]
STRIDE = 1          # process every n-th frame (use 2-3 for long videos)
MAX_FRAMES = None   # e.g. 300 for a quick test
df = run_on_video(pose_model, INPUT, RESULTS, detector=detector, stride=STRIDE, max_frames=MAX_FRAMES)
df.head(17)
""")

md("""
## 4. Joint-confidence heatmaps

For every joint the model outputs a 64×48 heatmap; its peak position is the
keypoint and the peak height is the confidence. The heatmap shape is used to
judge localisation quality:

* **peak** – confidence; low → joint probably occluded / out of frame
* **spread (px)** – width of the blob in image pixels; wide → uncertain position
* **second peak** – strongest other local maximum ÷ main peak; high → the model is torn between two places (typical for left/right swaps or limbs crossing)

A joint is flagged **unreliable** when peak < 0.3, spread > 8 % of the person box, or second peak > 0.5.
""")
code("""
res = hm.load_results(RESULTS)
metrics = hm.metrics_table(res)        # conf_thr=0.3, spread_frac=0.08, second_thr=0.5
summary = hm.joint_summary(metrics)
summary.round(3)
""")
code("""
t = int(np.argmin(res["scores"][:, 5:].mean(1)))   # frame with the lowest body-joint confidence
frame = hm.read_frame(res["meta"]["source"], int(res["frames"][t]))
hm.plot_joint_heatmap_grid(res, t, frame); plt.show()

m = hm.heatmap_metrics(res["heatmaps"], res["input_scale"])
flags = hm.flag_unreliable(m, hm.bbox_sizes(res["bboxes"]))
overlay = hm.render_frame(res, t, frame, flags=flags)   # red ring = flagged joint
plt.figure(figsize=(10, 7)); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()
""")
code("""
if len(res["frames"]) > 1:
    hm.plot_confidence_timeline(metrics); plt.show()
hm.plot_joint_summary(summary); plt.show()
hm.plot_metric_scatter(metrics); plt.show()
""")
code("""
#@title Heatmap overlay video + save all analysis files
VIDEO_FLAG = "--video" if len(res["frames"]) > 1 else ""
!python scripts/analyse_heatmaps.py --results {RESULTS} {VIDEO_FLAG}
if VIDEO_FLAG:
    !ffmpeg -y -loglevel error -i {RESULTS}/heatmap_analysis/heatmap_overlay.mp4 -vcodec libx264 {RESULTS}/heatmap_analysis/heatmap_overlay_h264.mp4
    display(Video(RESULTS + "/heatmap_analysis/heatmap_overlay_h264.mp4", embed=True, width=720))
""")

md("""
## 5. Validation against Kinovea

1. Open the **same video** in Kinovea, track the joints you care about
   (trajectory tool on hip / knee / ankle …) and/or add angle tools.
2. Export the data to a spreadsheet / CSV (the menu name differs by Kinovea version).
3. Upload it here, look at its columns, and fill the mapping in the next cell:
   which column is time, which columns are X/Y of which COCO joint, which are angles.

**Coordinates.** If Kinovea was *not* calibrated the values are pixels; set
`origin_px`/`y_up` to match its coordinate system (e.g. origin bottom-left and
y pointing up → `"origin_px": [0, IMAGE_HEIGHT], "y_up": true`). If it *was*
calibrated (metres), fill in `px_per_unit` or rely on **joint angles**, which
are unit-free. `AUTO_ALIGN` can fit the scale/offset automatically, but it also
hides any global bias of the model, so report that you used it.
""")
code("""
from google.colab import files
up = files.upload()
KINOVEA = "data/" + list(up)[0]; os.replace(list(up)[0], KINOVEA)
kin = kv.inspect(KINOVEA)
""")
code("""
H = res["meta"]["height"]
mapping = {
    "time_column": "Time (ms)",   # or "frame_column": "Frame"
    "time_unit": "ms",
    "points": {                   # COCO joint name -> Kinovea X/Y columns
        "left_hip":   {"x": "Hip/X",   "y": "Hip/Y"},
        "left_knee":  {"x": "Knee/X",  "y": "Knee/Y"},
        "left_ankle": {"x": "Ankle/X", "y": "Ankle/Y"},
    },
    "angles": {"left_knee": "Knee angle"},
    "coords": {"px_per_unit": 1.0, "origin_px": [0, 0], "y_up": False},
}
import json; json.dump(mapping, open("data/kinovea_mapping.json", "w"), indent=2)
AUTO_OFFSET = 10    # search +-10 frames for the best time alignment (0 = off)
AUTO_ALIGN = False
""")
code("""
ALIGN_FLAG = "--auto-align" if AUTO_ALIGN else ""
!python scripts/compare_kinovea.py --results {RESULTS} --kinovea {KINOVEA} --mapping data/kinovea_mapping.json --auto-offset {AUTO_OFFSET} {ALIGN_FLAG}
import glob
for png in sorted(glob.glob(RESULTS + "/kinovea_validation/*.png")):
    print(png); display(Image(png, width=900))
""")

md("## 6. Save results")
code("""
from google.colab import files
!zip -qr results.zip results
if USE_DRIVE:
    !cp -r results "{DRIVE_DIR}/"
    print("copied to", DRIVE_DIR)
files.download("results.zip")
""")

nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
    accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
    kernelspec=dict(display_name="Python 3", name="python3"),
    language_info=dict(name="python")))
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT)
