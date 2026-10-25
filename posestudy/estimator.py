"""MMPose top-down heatmap inference that keeps the raw per-joint heatmaps.

Pipeline per frame:
    1. find the person box (torchvision Faster R-CNN, or the whole frame)
    2. MMPose top-down heatmap model (HRNet-W32 by default) on that crop
    3. keep keypoints (image px), keypoint scores (= heatmap peak value)
       and the raw 17 x 64 x 48 heatmaps + the crop geometry, so the
       heatmaps can be projected back onto the image later.
"""

import json
import os
import os.path as osp
from glob import glob

import cv2
import numpy as np

from .skeleton import COCO_KEYPOINTS

# name -> (config path inside the mmpose package, checkpoint URL)
_CFG_DIR = "configs/body_2d_keypoint/topdown_heatmap/coco"
_CKPT = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/"
MODEL_ZOO = {
    "hrnet-w32": (f"{_CFG_DIR}/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
                  _CKPT + "td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth"),
    "hrnet-w48": (f"{_CFG_DIR}/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
                  _CKPT + "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"),
    "resnet50": (f"{_CFG_DIR}/td-hm_res50_8xb64-210e_coco-256x192.py",
                 _CKPT + "td-hm_res50_8xb64-210e_coco-256x192-04af38ce_20220923.pth"),
}


def _default_device():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def mmpose_config_path(rel_path):
    """Resolve a config shipped inside the installed mmpose package."""
    import mmpose
    root = osp.dirname(mmpose.__file__)
    for base in (osp.join(root, ".mim"), osp.dirname(root)):
        path = osp.join(base, rel_path)
        if osp.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {rel_path} inside the mmpose install")


def load_pose_model(name="hrnet-w32", device=None, config=None, checkpoint=None):
    """Build an MMPose top-down heatmap model that also returns its heatmaps.

    `name` picks an entry from MODEL_ZOO; pass `config`/`checkpoint` to use
    any other MMPose top-down heatmap model instead.
    """
    # PyTorch >= 2.6 defaults torch.load(weights_only=True), which rejects the
    # metadata stored in OpenMMLab checkpoints. They are official files, so allow it.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from mmpose.apis import init_model

    if config is None:
        rel_cfg, url = MODEL_ZOO[name]
        config = mmpose_config_path(rel_cfg)
        checkpoint = url if checkpoint is None else checkpoint
    device = device or _default_device()
    # output_heatmaps=True makes MMPose put the heatmaps in pred_fields.
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    return init_model(config, checkpoint, device=device, cfg_options=cfg_options)


class PersonDetector:
    """Picks the most confident person box with torchvision Faster R-CNN.

    Keeps the project free of mmdet (whose mmcv version pin clashes with
    current Colab). Good enough for single-subject lab recordings.
    """

    def __init__(self, device=None, score_thr=0.5):
        import torch
        import torchvision
        self.torch = torch
        self.device = device or _default_device()
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights).eval().to(self.device)
        self.score_thr = score_thr

    def __call__(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
        with self.torch.no_grad():
            out = self.model([tensor.to(self.device)])[0]
        keep = (out["labels"] == 1) & (out["scores"] >= self.score_thr)  # 1 = person
        if not keep.any():
            return None
        best = out["scores"][keep].argmax()
        return out["boxes"][keep][best].cpu().numpy()


def full_frame_bbox(frame):
    h, w = frame.shape[:2]
    return np.array([0, 0, w, h], dtype=np.float32)


def estimate_frame(model, frame_bgr, bbox):
    """Run the pose model on one person box (x1, y1, x2, y2).

    Returns keypoints (17, 2) in image px, scores (17,), heatmaps (17, h, w)
    in crop space, and the crop centre/scale needed to map them back.
    """
    from mmpose.apis import inference_topdown

    sample = inference_topdown(model, frame_bgr, np.asarray(bbox)[None], bbox_format="xyxy")[0]
    heatmaps = sample.pred_fields.heatmaps
    if hasattr(heatmaps, "cpu"):
        heatmaps = heatmaps.cpu().numpy()
    return dict(
        keypoints=np.asarray(sample.pred_instances.keypoints[0], dtype=np.float32),
        scores=np.asarray(sample.pred_instances.keypoint_scores[0], dtype=np.float32),
        heatmaps=np.asarray(heatmaps, dtype=np.float32),
        input_center=np.asarray(sample.metainfo["input_center"], dtype=np.float32).reshape(2),
        input_scale=np.asarray(sample.metainfo["input_scale"], dtype=np.float32).reshape(2),
    )


def iter_frames(source, stride=1, max_frames=None):
    """Yield (frame_idx, frame_bgr) from a video file, an image, or an image folder."""
    if osp.isdir(source):
        files = sorted(f for ext in ("jpg", "jpeg", "png", "bmp")
                       for f in glob(osp.join(source, f"*.{ext}")))
        frames = ((i, cv2.imread(f)) for i, f in enumerate(files))
    elif source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        frames = iter([(0, cv2.imread(source))])
    else:
        def _video():
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise IOError(f"Cannot open video {source}")
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
                idx += 1
            cap.release()
        frames = _video()

    n = 0
    for idx, frame in frames:
        if idx % stride:
            continue
        yield idx, frame
        n += 1
        if max_frames and n >= max_frames:
            break


def video_fps(source, default=30.0):
    if osp.isdir(source) or source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        return default
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default


def run_on_video(model, source, out_dir, detector=None, stride=1, max_frames=None,
                 fps=None, track_bbox=True, model_name="hrnet-w32", verbose=True):
    """Run pose estimation over a whole video and save everything to `out_dir`.

    Saves:
        keypoints.csv   tidy table: frame, time_s, joint, x, y, score
        heatmaps.npz    raw heatmaps (T, 17, h, w) float16 + crop geometry + boxes
        meta.json       fps, frame size, model info
    """
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    fps = fps or video_fps(source)
    rows, frames_idx, kpts, scores, hms, centers, scales, boxes = [], [], [], [], [], [], [], []
    last_bbox, frame_shape = None, None

    for idx, frame in iter_frames(source, stride, max_frames):
        frame_shape = frame.shape[:2]
        bbox = detector(frame) if detector is not None else None
        if bbox is None:
            # No detection: reuse the previous box, else fall back to the full frame.
            bbox = last_bbox if (track_bbox and last_bbox is not None) else full_frame_bbox(frame)
        last_bbox = bbox

        res = estimate_frame(model, frame, bbox)
        frames_idx.append(idx)
        kpts.append(res["keypoints"])
        scores.append(res["scores"])
        hms.append(res["heatmaps"].astype(np.float16))
        centers.append(res["input_center"])
        scales.append(res["input_scale"])
        boxes.append(np.asarray(bbox, dtype=np.float32))
        for j, name in enumerate(COCO_KEYPOINTS):
            x, y = res["keypoints"][j]
            rows.append((idx, idx / fps, name, float(x), float(y), float(res["scores"][j])))
        if verbose and len(frames_idx) % 25 == 0:
            print(f"  processed {len(frames_idx)} frames (frame {idx})")

    if not frames_idx:
        raise RuntimeError(f"No frames read from {source}")

    df = pd.DataFrame(rows, columns=["frame", "time_s", "joint", "x", "y", "score"])
    df.to_csv(osp.join(out_dir, "keypoints.csv"), index=False)
    np.savez_compressed(
        osp.join(out_dir, "heatmaps.npz"),
        frames=np.array(frames_idx), keypoints=np.stack(kpts), scores=np.stack(scores),
        heatmaps=np.stack(hms), input_center=np.stack(centers),
        input_scale=np.stack(scales), bboxes=np.stack(boxes))
    meta = dict(source=osp.abspath(source), fps=fps, height=frame_shape[0],
                width=frame_shape[1], stride=stride, n_frames=len(frames_idx),
                model=model_name, keypoints=COCO_KEYPOINTS)
    with open(osp.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"Saved {len(frames_idx)} frames of predictions to {out_dir}")
    return df
