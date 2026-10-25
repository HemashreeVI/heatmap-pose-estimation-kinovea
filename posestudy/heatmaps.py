"""Joint-confidence heatmap analysis and visualisation (OpenCV + Matplotlib).

Works on the files written by `estimator.run_on_video`; MMPose is not needed.

For a heatmap model each joint gets a 2-D "probability" map. Its shape tells
us how well the joint was localised:
    peak          max value; MMPose reports it as the keypoint score
    spread_px     spatial std-dev of the map around its centre, in image px
                  (wide blob = uncertain location)
    second_peak   height of the strongest competing peak / main peak
                  (close to 1 = model torn between two places, e.g. left/right swap)
"""

import json
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .skeleton import COCO_KEYPOINTS, SKELETON_IDX, kpt_color


# ----------------------------------------------------------------------------- io
def load_results(out_dir):
    data = dict(np.load(osp.join(out_dir, "heatmaps.npz")))
    data["heatmaps"] = data["heatmaps"].astype(np.float32)
    with open(osp.join(out_dir, "meta.json")) as f:
        data["meta"] = json.load(f)
    return data


def read_frame(source, frame_idx):
    if osp.isdir(source):
        files = sorted(f for f in os.listdir(source)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        return cv2.imread(osp.join(source, files[frame_idx]))
    if source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        return cv2.imread(source)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Could not read frame {frame_idx} from {source}")
    return frame


# ------------------------------------------------------------------- geometry
def heatmap_to_image(heatmaps, center, scale, img_shape):
    """Project (K, h, w) crop-space heatmaps onto the (H, W) image.

    Uses the same mapping MMPose applies to keypoints:
        x_img = u * scale_w / w + center_x - scale_w / 2
    """
    hm = np.asarray(heatmaps, dtype=np.float32)
    k, h, w = hm.shape
    sx, sy = scale[0] / w, scale[1] / h
    mat = np.array([[sx, 0, center[0] - scale[0] / 2],
                    [0, sy, center[1] - scale[1] / 2]], dtype=np.float32)
    H, W = img_shape[:2]
    # warpAffine handles at most 4 channels reliably, so warp joint by joint.
    return np.stack([cv2.warpAffine(hm[j], mat, (W, H), flags=cv2.INTER_LINEAR,
                                    borderValue=0) for j in range(k)])


# -------------------------------------------------------------------- metrics
def heatmap_metrics(heatmaps, input_scale, suppress_radius=2, rel_floor=0.1):
    """Per-frame, per-joint quality metrics.

    heatmaps: (T, K, h, w); input_scale: (T, 2) crop size in image px.
    Returns dict of (T, K) arrays: peak, spread_px, second_peak.
    """
    hm = np.asarray(heatmaps, dtype=np.float32)
    T, K, h, w = hm.shape
    flat = hm.reshape(T, K, -1)
    peak = flat.max(-1)
    arg = flat.argmax(-1)
    py, px = np.unravel_index(arg, (h, w))

    ys, xs = np.mgrid[0:h, 0:w]
    # Spread: weighted std of the map after dropping the low background floor.
    wts = np.clip(hm - rel_floor * peak[..., None, None], 0, None)
    tot = wts.sum((-1, -2)) + 1e-12
    mx = (wts * xs).sum((-1, -2)) / tot
    my = (wts * ys).sum((-1, -2)) / tot
    var_x = (wts * (xs - mx[..., None, None]) ** 2).sum((-1, -2)) / tot
    var_y = (wts * (ys - my[..., None, None]) ** 2).sum((-1, -2)) / tot
    px_per_hm = (np.asarray(input_scale)[:, 0] / w + np.asarray(input_scale)[:, 1] / h) / 2
    spread_px = np.sqrt(var_x + var_y) * px_per_hm[:, None]

    # Second peak: highest *local maximum* away from the main peak (a plain
    # "max outside a disc" would just pick up the flank of the main blob).
    padded = np.pad(hm, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    neigh = np.max([padded[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx], axis=0)
    is_peak = hm >= neigh
    dist2 = (xs[None, None] - px[..., None, None]) ** 2 + (ys[None, None] - py[..., None, None]) ** 2
    masked = np.where(is_peak & (dist2 > suppress_radius ** 2), hm, 0.0)
    second = masked.reshape(T, K, -1).max(-1)
    second_peak = np.clip(second, 0, None) / np.maximum(peak, 1e-6)

    return dict(peak=peak, spread_px=spread_px, second_peak=second_peak)


def flag_unreliable(metrics, bbox_size, conf_thr=0.3, spread_frac=0.08, second_thr=0.5):
    """Boolean (T, K) mask of predictions we should not trust, plus the reason.

    A joint is unreliable when its heatmap peak is low, the blob is wide
    relative to the person size, or a competing peak is nearly as strong.
    """
    bbox_size = np.asarray(bbox_size, dtype=float)[:, None]
    low_conf = metrics["peak"] < conf_thr
    diffuse = metrics["spread_px"] > spread_frac * bbox_size
    ambiguous = metrics["second_peak"] > second_thr
    return dict(unreliable=low_conf | diffuse | ambiguous,
                low_conf=low_conf, diffuse=diffuse, ambiguous=ambiguous)


def bbox_sizes(bboxes):
    b = np.asarray(bboxes, dtype=float)
    return np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])


def metrics_table(results, **flag_kwargs):
    """Tidy DataFrame (frame, joint, score, peak, spread_px, second_peak, flags...)."""
    m = heatmap_metrics(results["heatmaps"], results["input_scale"])
    flags = flag_unreliable(m, bbox_sizes(results["bboxes"]), **flag_kwargs)
    T, K = m["peak"].shape
    fps = results["meta"]["fps"]
    frames = np.repeat(results["frames"], K)
    df = pd.DataFrame(dict(
        frame=frames, time_s=frames / fps, joint=np.tile(COCO_KEYPOINTS, T),
        x=results["keypoints"][..., 0].ravel(), y=results["keypoints"][..., 1].ravel(),
        score=results["scores"].ravel(),
        **{k: v.ravel() for k, v in m.items()},
        **{k: v.ravel() for k, v in flags.items()},
    ))
    return df


def joint_summary(metrics_df):
    """Per-joint averages and % of frames flagged unreliable."""
    g = metrics_df.groupby("joint", sort=False)
    out = pd.DataFrame(dict(
        mean_score=g["score"].mean(), min_score=g["score"].min(),
        mean_spread_px=g["spread_px"].mean(), mean_second_peak=g["second_peak"].mean(),
        pct_unreliable=100 * g["unreliable"].mean(), pct_low_conf=100 * g["low_conf"].mean(),
        pct_diffuse=100 * g["diffuse"].mean(), pct_ambiguous=100 * g["ambiguous"].mean(),
    ))
    return out.reindex(COCO_KEYPOINTS)


# -------------------------------------------------------------- OpenCV drawing
def colorize(hm_img, vmax=1.0):
    """Single (H, W) map -> BGR JET image."""
    norm = np.clip(hm_img / vmax, 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def overlay_heatmap(frame, hm_img, alpha=0.5, vmax=1.0):
    color = colorize(hm_img, vmax)
    # Blend only where there is heat so the background stays visible.
    weight = (np.clip(hm_img / vmax, 0, 1) ** 0.5 * alpha)[..., None]
    return (frame * (1 - weight) + color * weight).astype(np.uint8)


def draw_pose(frame, keypoints, scores, thr=0.3, unreliable=None, radius=4, thickness=2):
    """Skeleton on top of the frame. Low-score joints are skipped; joints
    flagged unreliable are drawn with a red ring."""
    img = frame.copy()
    for a, b in SKELETON_IDX:
        if scores[a] >= thr and scores[b] >= thr:
            pa, pb = tuple(np.round(keypoints[a]).astype(int)), tuple(np.round(keypoints[b]).astype(int))
            cv2.line(img, pa, pb, kpt_color(COCO_KEYPOINTS[b]), thickness, cv2.LINE_AA)
    for j, (x, y) in enumerate(keypoints):
        if scores[j] < thr:
            continue
        c = (int(round(x)), int(round(y)))
        cv2.circle(img, c, radius, kpt_color(COCO_KEYPOINTS[j]), -1, cv2.LINE_AA)
        if unreliable is not None and unreliable[j]:
            cv2.circle(img, c, radius + 4, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def render_frame(results, t, frame, joint=None, alpha=0.55, flags=None):
    """Overlay heatmap (all joints = max over joints, or a single joint) + skeleton."""
    hm_img = heatmap_to_image(results["heatmaps"][t], results["input_center"][t],
                              results["input_scale"][t], frame.shape)
    hm = hm_img[COCO_KEYPOINTS.index(joint)] if joint else hm_img.max(0)
    img = overlay_heatmap(frame, hm, alpha)
    unrel = flags["unreliable"][t] if flags is not None else None
    img = draw_pose(img, results["keypoints"][t], results["scores"][t], unreliable=unrel)
    label = f"frame {int(results['frames'][t])}" + (f" | {joint}" if joint else "")
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def render_heatmap_video(results, out_path, joint=None, alpha=0.55, flags=None, max_width=960):
    """Write an mp4 with the heatmap overlay + skeleton for every processed frame."""
    source = results["meta"]["source"]
    fps = results["meta"]["fps"] / results["meta"].get("stride", 1)
    writer = None
    for t, f_idx in enumerate(results["frames"]):
        frame = read_frame(source, int(f_idx))
        img = render_frame(results, t, frame, joint, alpha, flags)
        if img.shape[1] > max_width:
            s = max_width / img.shape[1]
            img = cv2.resize(img, (max_width, int(img.shape[0] * s)))
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer is not None:
        writer.release()
    return out_path


# --------------------------------------------------------- Matplotlib figures
def plot_joint_heatmap_grid(results, t, frame, out_path=None, crop=True):
    """17 panels: each joint's heatmap over the frame, titled with its metrics."""
    hm_img = heatmap_to_image(results["heatmaps"][t], results["input_center"][t],
                              results["input_scale"][t], frame.shape)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    m = heatmap_metrics(results["heatmaps"][t:t + 1], results["input_scale"][t:t + 1])
    if crop:
        x1, y1, x2, y2 = results["bboxes"][t]
        pad = 0.1 * max(x2 - x1, y2 - y1)
        H, W = frame.shape[:2]
        x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        x2, y2 = int(min(W, x2 + pad)), int(min(H, y2 + pad))
    else:
        x1, y1, (y2, x2) = 0, 0, frame.shape[:2]

    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    for j, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if j == len(COCO_KEYPOINTS):
            ax.imshow(rgb[y1:y2, x1:x2])
            ax.imshow(hm_img.max(0)[y1:y2, x1:x2], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax.set_title("all joints (max)", fontsize=10)
            continue
        if j > len(COCO_KEYPOINTS):
            continue
        ax.imshow(rgb[y1:y2, x1:x2])
        ax.imshow(hm_img[j][y1:y2, x1:x2], cmap="jet", alpha=0.55, vmin=0, vmax=1)
        kx, ky = results["keypoints"][t, j]
        ax.plot(kx - x1, ky - y1, "w+", ms=10, mew=2)
        ax.set_title(f"{COCO_KEYPOINTS[j]}\nconf {m['peak'][0, j]:.2f} | "
                     f"spread {m['spread_px'][0, j]:.0f}px | 2nd {m['second_peak'][0, j]:.2f}",
                     fontsize=8)
    fig.suptitle(f"Joint-confidence heatmaps, frame {int(results['frames'][t])}", fontsize=14)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_confidence_timeline(metrics_df, joints=None, conf_thr=0.3, out_path=None):
    """Heatmap peak (confidence) over time per joint, unreliable frames marked."""
    joints = joints or COCO_KEYPOINTS[5:]  # body joints; face points are rarely of interest
    fig, axes = plt.subplots(len(joints), 1, figsize=(12, 1.3 * len(joints)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, joint in zip(axes, joints):
        d = metrics_df[metrics_df.joint == joint]
        ax.plot(d.time_s, d.score, lw=1.2, color="#2b6cb0")
        bad = d[d.unreliable]
        ax.scatter(bad.time_s, bad.score, s=10, color="#e53e3e", zorder=3)
        ax.axhline(conf_thr, color="gray", ls="--", lw=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(joint.replace("_", "\n"), rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Heatmap confidence per joint (red = flagged unreliable)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_joint_summary(summary, out_path=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    y = np.arange(len(summary))
    a1.barh(y, summary.mean_score, color="#2b6cb0")
    a1.set_yticks(y, summary.index, fontsize=8)
    a1.invert_yaxis()
    a1.set_xlim(0, 1)
    a1.set_xlabel("mean heatmap confidence")
    left = 0
    for col, c in (("pct_low_conf", "#e53e3e"), ("pct_diffuse", "#dd6b20"), ("pct_ambiguous", "#805ad5")):
        a2.barh(y, summary[col], left=left, color=c, label=col.replace("pct_", ""))
        left = left + summary[col]
    a2.set_yticks(y, summary.index, fontsize=8)
    a2.invert_yaxis()
    a2.set_xlabel("% of frames (reasons can overlap)")
    a2.legend(fontsize=8)
    a2.set_title("Why predictions were flagged unreliable")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_metric_scatter(metrics_df, out_path=None):
    """Confidence vs spread: wide, low-peak heatmaps cluster bottom-right."""
    fig, ax = plt.subplots(figsize=(7, 5))
    body = metrics_df[~metrics_df.joint.isin(COCO_KEYPOINTS[:5])]
    sc = ax.scatter(body.spread_px, body.score, c=body.second_peak, cmap="viridis",
                    s=6, alpha=0.6, vmin=0, vmax=1)
    fig.colorbar(sc, label="second-peak ratio")
    ax.set_xlabel("heatmap spread (px)")
    ax.set_ylabel("heatmap peak (confidence)")
    ax.set_title("Heatmap shape vs confidence (body joints)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig
