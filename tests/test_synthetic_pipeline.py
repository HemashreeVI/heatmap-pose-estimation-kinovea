"""End-to-end check of the analysis + Kinovea code on SYNTHETIC data.

No MMPose needed: it fakes what run_pose.py writes (Gaussian heatmaps around
known keypoints, some made blurry or double-peaked on purpose) plus a
Kinovea-style export (';' separator, ',' decimals, y-up coordinates), then
runs scripts/analyse_heatmaps.py and scripts/compare_kinovea.py.
The numbers it prints are NOT real results - only a test of the code.

    python tests/test_synthetic_pipeline.py
"""

import json
import os
import os.path as osp
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
from posestudy.skeleton import COCO_KEYPOINTS, SKELETON_IDX, compute_joint_angles  # noqa: E402

H, W, T, FPS = 480, 640, 60, 30.0
HM_H, HM_W = 64, 48


def make_pose(t):
    """A stick figure that squats: knee angle changes over time."""
    phase = np.sin(2 * np.pi * t / T)
    kp = np.zeros((17, 2))
    cx = 320
    hip_y = 260 + 40 * phase
    kp[0] = (cx, hip_y - 170)                              # nose
    kp[1:5] = [(cx - 8, hip_y - 178), (cx + 8, hip_y - 178), (cx - 18, hip_y - 172), (cx + 18, hip_y - 172)]
    kp[5], kp[6] = (cx - 40, hip_y - 130), (cx + 40, hip_y - 130)
    kp[7], kp[8] = (cx - 60, hip_y - 70), (cx + 60, hip_y - 70)
    kp[9], kp[10] = (cx - 70, hip_y - 15), (cx + 70, hip_y - 15)
    kp[11], kp[12] = (cx - 25, hip_y), (cx + 25, hip_y)
    knee_dx = 30 + 30 * (phase + 1)
    kp[13], kp[14] = (cx - 25 - knee_dx, hip_y + 90 - 20 * phase), (cx + 25 + knee_dx, hip_y + 90 - 20 * phase)
    kp[15], kp[16] = (cx - 30, 440), (cx + 30, 440)
    return kp


def gaussian(cx, cy, sigma=2.0, amp=1.0):
    ys, xs = np.mgrid[0:HM_H, 0:HM_W]
    return amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))


def main():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix="posestudy_test_")
    video = osp.join(tmp, "synthetic.mp4")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    true_kp = np.stack([make_pose(t) for t in range(T)])
    bbox = np.array([200, 40, 440, 470], dtype=np.float32)
    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    # MMPose pads the box to the 3:4 aspect ratio and enlarges it by 1.25.
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bh = max(bh, bw * 4 / 3)
    scale = np.array([bh * 3 / 4, bh], dtype=np.float32) * 1.25

    pred_kp, scores, hms = [], [], []
    for t in range(T):
        frame = np.full((H, W, 3), 40, np.uint8)
        for a, b in SKELETON_IDX:
            cv2.line(frame, tuple(true_kp[t, a].astype(int)), tuple(true_kp[t, b].astype(int)), (220, 220, 220), 6)
        writer.write(frame)

        noise = rng.normal(0, 2.0, (17, 2))
        noise[15] += rng.normal(0, 12.0, 2) if 20 <= t < 30 else 0     # ankle goes wrong for a while
        kp = true_kp[t] + noise
        hm = np.zeros((17, HM_H, HM_W), np.float32)
        sc = np.zeros(17, np.float32)
        for j in range(17):
            u = (kp[j, 0] - center[0] + scale[0] / 2) / scale[0] * HM_W
            v = (kp[j, 1] - center[1] + scale[1] / 2) / scale[1] * HM_H
            amp, sigma = 0.9, 2.0
            if j == 15 and 20 <= t < 30:
                amp, sigma = 0.25, 5.0                                   # low + diffuse
            hm[j] = gaussian(u, v, sigma, amp)
            if j == 9 and 40 <= t < 45:
                hm[j] += gaussian(u + 10, v, 2.0, 0.8 * amp)             # double peak
            sc[j] = hm[j].max()
        pred_kp.append(kp)
        scores.append(sc)
        hms.append(hm.astype(np.float16))
    writer.release()

    res_dir = osp.join(tmp, "results")
    os.makedirs(res_dir)
    np.savez_compressed(osp.join(res_dir, "heatmaps.npz"), frames=np.arange(T),
                        keypoints=np.stack(pred_kp).astype(np.float32), scores=np.stack(scores),
                        heatmaps=np.stack(hms), input_center=np.tile(center, (T, 1)),
                        input_scale=np.tile(scale, (T, 1)), bboxes=np.tile(bbox, (T, 1)))
    with open(osp.join(res_dir, "meta.json"), "w") as f:
        json.dump(dict(source=video, fps=FPS, height=H, width=W, stride=1, n_frames=T,
                       model="synthetic", keypoints=COCO_KEYPOINTS), f)

    # Kinovea-like export: y-up coords with origin at the bottom-left, 2-frame delay,
    # European formatting.
    rows = []
    knee = compute_joint_angles(true_kp)["left_knee"]
    for t in range(T):
        tk = t + 2
        if tk >= T:
            break
        r = {"Time (ms)": tk / FPS * 1000}
        for joint, name in (("left_hip", "Hip"), ("left_knee", "Knee"), ("left_ankle", "Ankle")):
            x, y = true_kp[t, COCO_KEYPOINTS.index(joint)] + rng.normal(0, 1.0, 2)
            r[f"{name}/X"], r[f"{name}/Y"] = x, H - y
        r["Knee angle"] = knee[t] + rng.normal(0, 1.0)
        rows.append(r)
    kin_csv = osp.join(tmp, "kinovea.csv")
    pd.DataFrame(rows).to_csv(kin_csv, sep=";", decimal=",", index=False)
    with open(osp.join(ROOT, "data", "kinovea_mapping_example.json")) as f:
        mapping = json.load(f)
    mapping["coords"] = {"px_per_unit": 1.0, "origin_px": [0, H], "y_up": True}
    map_path = osp.join(tmp, "mapping.json")
    with open(map_path, "w") as f:
        json.dump(mapping, f)

    py = sys.executable
    subprocess.run([py, osp.join(ROOT, "scripts", "analyse_heatmaps.py"), "--results", res_dir, "--video"], check=True)
    subprocess.run([py, osp.join(ROOT, "scripts", "compare_kinovea.py"), "--kinovea", kin_csv, "--inspect"], check=True)
    subprocess.run([py, osp.join(ROOT, "scripts", "compare_kinovea.py"), "--results", res_dir,
                    "--kinovea", kin_csv, "--mapping", map_path, "--auto-offset", "5"], check=True)

    # Checks: the planted problems must be detected.
    m = pd.read_csv(osp.join(res_dir, "heatmap_analysis", "heatmap_metrics.csv"))
    ankle = m[(m.joint == "left_ankle")]
    assert ankle[(ankle.frame >= 20) & (ankle.frame < 30)].unreliable.all(), "diffuse ankle not flagged"
    wrist = m[(m.joint == "left_wrist")]
    assert wrist[(wrist.frame >= 40) & (wrist.frame < 45)].ambiguous.all(), "double peak not flagged"
    assert not m[m.joint == "left_hip"].unreliable.any(), "clean joint wrongly flagged"
    rep = json.load(open(osp.join(res_dir, "kinovea_validation", "report.json")))
    assert rep["auto_offset"] == -2, rep
    summ = pd.read_csv(osp.join(res_dir, "kinovea_validation", "keypoint_summary.csv"), index_col=0)
    assert summ.loc["left_hip", "median_err_px"] < 5, summ
    assert summ.loc["left_ankle", "mean_err_px"] > summ.loc["left_hip", "mean_err_px"]
    ang = pd.read_csv(osp.join(res_dir, "kinovea_validation", "angle_agreement.csv"), index_col=0)
    assert ang.loc["left_knee", "MAE_deg"] < 5, ang
    print("\nALL CHECKS PASSED - outputs in", tmp)


if __name__ == "__main__":
    main()
