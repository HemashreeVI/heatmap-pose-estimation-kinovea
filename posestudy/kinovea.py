"""Validate predicted keypoints / joint angles against Kinovea measurements.

Kinovea (https://www.kinovea.org) is used to manually track markers / joints
and measure angles frame by frame. Its spreadsheet export differs between
versions and languages, so loading is driven by a small JSON mapping that
says which column holds what. Example `mapping.json`:

    {
      "time_column": "Time (ms)",          # or "frame_column": "Frame"
      "time_unit": "ms",                   # ms | s
      "points": {
        "left_knee":  {"x": "Knee/X",  "y": "Knee/Y"},
        "left_ankle": {"x": "Ankle/X", "y": "Ankle/Y"}
      },
      "angles": {"left_knee": "Knee angle"},
      "coords": {"px_per_unit": 1.0, "origin_px": [0, 0], "y_up": false}
    }

`coords` converts Kinovea coordinates to image pixels:
    x_px = origin_x + x * px_per_unit
    y_px = origin_y - y * px_per_unit  (if y_up)   else  origin_y + y * px_per_unit
Uncalibrated Kinovea data in pixels with a top-left origin needs the
defaults. If the coordinate system was calibrated (metres/cm, moved origin),
either fill these in or validate with joint angles, which do not depend on
the coordinate system.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .skeleton import COCO_KEYPOINTS, compute_joint_angles


# --------------------------------------------------------------------- loading
def read_table(path):
    """Read a Kinovea CSV/TSV/XLSX, coping with ';' separators and ',' decimals."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        head = f.read(4096)
    header = head.splitlines()[0] if head else ""
    # Decide from the header row: data rows may use ',' as the decimal mark.
    sep = max((";", "\t", ","), key=header.count) if header else ","
    decimal = "," if sep in (";", "\t") and _has_comma_decimals(head, sep) else "."
    df = pd.read_csv(path, sep=sep, decimal=decimal, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _has_comma_decimals(text, sep):
    cells = [c for line in text.splitlines()[1:] for c in line.split(sep)]
    return any(c.strip().replace(",", "", 1).lstrip("-").isdigit() and "," in c for c in cells)


def inspect(path, n=5):
    """Print the columns and first rows, to help write mapping.json."""
    df = read_table(path)
    print(f"{path}: {len(df)} rows")
    print("columns:", list(df.columns))
    print(df.head(n).to_string())
    return df


def load_kinovea(path, mapping, fps, frame_offset=0):
    """Return (points_df, angles_df) in tidy form, keyed by video frame.

    points_df: frame, joint, x, y   (image pixels)
    angles_df: frame, angle, value  (degrees)
    """
    if isinstance(mapping, str):
        with open(mapping) as f:
            mapping = json.load(f)
    df = read_table(path)
    frame = _frame_index(df, mapping, fps) + frame_offset

    coords = mapping.get("coords", {})
    ppu = float(coords.get("px_per_unit", 1.0))
    ox, oy = coords.get("origin_px", [0, 0])
    y_up = bool(coords.get("y_up", False))

    pts = []
    for joint, cols in mapping.get("points", {}).items():
        if joint not in COCO_KEYPOINTS:
            raise ValueError(f"'{joint}' is not a COCO keypoint name: {COCO_KEYPOINTS}")
        x = pd.to_numeric(df[cols["x"]], errors="coerce").to_numpy() * ppu + ox
        y = pd.to_numeric(df[cols["y"]], errors="coerce").to_numpy() * ppu
        y = oy - y if y_up else oy + y
        pts.append(pd.DataFrame(dict(frame=frame, joint=joint, x=x, y=y)))
    ang = []
    for name, col in mapping.get("angles", {}).items():
        v = pd.to_numeric(df[col], errors="coerce").to_numpy()
        ang.append(pd.DataFrame(dict(frame=frame, angle=name, value=v)))

    points = pd.concat(pts).dropna() if pts else pd.DataFrame(columns=["frame", "joint", "x", "y"])
    angles = pd.concat(ang).dropna() if ang else pd.DataFrame(columns=["frame", "angle", "value"])
    points["frame"] = points.frame.astype(int)
    angles["frame"] = angles.frame.astype(int)
    # Several Kinovea rows can round to the same frame; average them.
    points = points.groupby(["frame", "joint"], as_index=False)[["x", "y"]].mean()
    angles = angles.groupby(["frame", "angle"], as_index=False)["value"].mean()
    return points, angles


def _frame_index(df, mapping, fps):
    if "frame_column" in mapping:
        return pd.to_numeric(df[mapping["frame_column"]], errors="coerce").round().to_numpy(dtype=float)
    col = mapping.get("time_column") or df.columns[0]
    t = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    t_s = t / 1000.0 if mapping.get("time_unit", "ms") == "ms" else t
    return np.round(t_s * fps)  # float so missing times stay NaN


# ------------------------------------------------------------------ alignment
def fit_similarity(src, dst):
    """Least-squares scale + translation (+ optional y flip) mapping src -> dst.

    Only use this when Kinovea's coordinate system is unknown: it removes any
    global offset/scale between the two, so a global bias in the model would
    be hidden too. Returns (params, transform_fn).
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    best = None
    for flip in (1.0, -1.0):
        s2 = src * [1.0, flip]
        sc, dc = s2.mean(0), dst.mean(0)
        a, b = s2 - sc, dst - dc
        scale = (a * b).sum() / max((a * a).sum(), 1e-12)
        resid = np.linalg.norm(b - scale * a, axis=1).mean()
        if best is None or resid < best[0]:
            best = (resid, flip, scale, dc - scale * sc)
    _, flip, scale, t = best

    def transform(p):
        return np.asarray(p, float) * [1.0, flip] * scale + t
    return dict(scale=float(scale), y_flip=flip < 0, translation=t.tolist()), transform


# ------------------------------------------------------------------ comparison
def predictions_table(results, metrics_df=None):
    """Tidy predicted keypoints (frame, joint, x_pred, y_pred, score, bbox_size[, unreliable])."""
    b = results["bboxes"]
    size = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    T, K = results["scores"].shape
    df = pd.DataFrame(dict(
        frame=np.repeat(results["frames"], K), joint=np.tile(COCO_KEYPOINTS, T),
        x_pred=results["keypoints"][..., 0].ravel(), y_pred=results["keypoints"][..., 1].ravel(),
        score=results["scores"].ravel(), bbox_size=np.repeat(size, K)))
    if metrics_df is not None:
        df = df.merge(metrics_df[["frame", "joint", "spread_px", "second_peak", "unreliable"]],
                      on=["frame", "joint"], how="left")
    return df


def predicted_angles(results):
    ang = compute_joint_angles(results["keypoints"])
    return pd.DataFrame([dict(frame=int(f), angle=name, pred=float(v[i]))
                         for name, v in ang.items() for i, f in enumerate(results["frames"])])


def match_keypoints(pred_df, gt_points):
    m = pred_df.merge(gt_points.rename(columns={"x": "x_gt", "y": "y_gt"}), on=["frame", "joint"])
    m["error_px"] = np.hypot(m.x_pred - m.x_gt, m.y_pred - m.y_gt)
    m["error_norm"] = m.error_px / m.bbox_size
    return m


def best_frame_offset(pred_df, gt_points, max_offset=10):
    """Shift Kinovea frames by -max..+max and keep the shift with the lowest
    median error; fixes a start-time mismatch between the two."""
    scores = {}
    for off in range(-max_offset, max_offset + 1):
        g = gt_points.assign(frame=gt_points.frame + off)
        m = match_keypoints(pred_df, g)
        if len(m):
            scores[off] = m.error_px.median()
    return min(scores, key=scores.get), scores


def keypoint_summary(matched, pck_alphas=(0.05, 0.1)):
    """Per-joint error statistics. PCK@a = % of frames with error < a * person box size."""
    def stats(d):
        out = dict(n=len(d), mean_err_px=d.error_px.mean(), median_err_px=d.error_px.median(),
                   rmse_px=np.sqrt((d.error_px ** 2).mean()), max_err_px=d.error_px.max(),
                   mean_score=d.score.mean())
        for a in pck_alphas:
            out[f"PCK@{a}"] = 100 * (d.error_norm < a).mean()
        return pd.Series(out)
    per_joint = matched.groupby("joint", sort=False).apply(stats, include_groups=False)
    overall = stats(matched).rename("ALL")
    return pd.concat([per_joint, overall.to_frame().T])


def confidence_vs_error(matched, conf_bins=(0, 0.3, 0.5, 0.7, 0.9, 1.01)):
    """Does a low heatmap confidence actually mean a larger error?"""
    m = matched.copy()
    m["conf_bin"] = pd.cut(m.score, conf_bins, right=False)
    t = m.groupby("conf_bin", observed=True).error_px.agg(["count", "mean", "median"])
    rho = m[["score", "error_px"]].corr(method="spearman").iloc[0, 1]
    if "unreliable" in m:
        flagged = m.groupby("unreliable").error_px.agg(["count", "mean", "median"])
    else:
        flagged = None
    return t, rho, flagged


def angle_agreement(pred_ang, gt_angles):
    """MAE, RMSE, bias, Bland-Altman 95% limits and Pearson r per angle."""
    m = pred_ang.merge(gt_angles.rename(columns={"value": "kinovea"}), on=["frame", "angle"]).dropna()
    rows = {}
    for name, d in m.groupby("angle"):
        diff = d.pred - d.kinovea
        sd = diff.std(ddof=1) if len(d) > 1 else np.nan
        rows[name] = dict(n=len(d), MAE_deg=diff.abs().mean(), RMSE_deg=np.sqrt((diff ** 2).mean()),
                          bias_deg=diff.mean(), LoA_low=diff.mean() - 1.96 * sd,
                          LoA_high=diff.mean() + 1.96 * sd,
                          pearson_r=d.pred.corr(d.kinovea) if len(d) > 2 else np.nan)
    return m, pd.DataFrame(rows).T


# ----------------------------------------------------------------------- plots
def plot_error_per_joint(summary, out_path=None):
    s = summary.drop(index="ALL", errors="ignore")
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(s) + 1.5))
    y = np.arange(len(s))
    ax.barh(y, s.mean_err_px, color="#2b6cb0", label="mean")
    ax.scatter(s.median_err_px, y, color="k", marker="|", s=200, label="median", zorder=3)
    ax.set_yticks(y, s.index)
    ax.invert_yaxis()
    ax.set_xlabel("error vs Kinovea (px)")
    ax.legend()
    ax.set_title("Keypoint error per joint")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_trajectories(matched, joint, fps, out_path=None):
    d = matched[matched.joint == joint].sort_values("frame")
    t = d.frame / fps
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, c in zip(axes[:2], ("x", "y")):
        ax.plot(t, d[f"{c}_gt"], "k-", lw=1.5, label="Kinovea")
        ax.plot(t, d[f"{c}_pred"], "-", color="#dd6b20", lw=1.2, label="MMPose")
        ax.set_ylabel(f"{c} (px)")
        ax.legend(fontsize=8)
    axes[2].plot(t, d.error_px, color="#e53e3e", lw=1)
    if "unreliable" in d:
        bad = d[d.unreliable.fillna(False).astype(bool)]
        axes[2].scatter(bad.frame / fps, bad.error_px, s=12, color="#805ad5", label="flagged unreliable", zorder=3)
        axes[2].legend(fontsize=8)
    axes[2].set_ylabel("error (px)")
    axes[2].set_xlabel("time (s)")
    fig.suptitle(f"{joint}: MMPose vs Kinovea")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_confidence_vs_error(matched, out_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(matched.score, matched.error_px, s=8, alpha=0.5, color="#2b6cb0")
    ax.set_xlabel("heatmap confidence")
    ax.set_ylabel("error vs Kinovea (px)")
    ax.set_title("Is low heatmap confidence a warning sign?")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_angles(angle_matched, fps, out_path=None):
    names = sorted(angle_matched.angle.unique())
    fig, axes = plt.subplots(len(names), 2, figsize=(13, 3.2 * len(names)), squeeze=False)
    for (a1, a2), name in zip(axes, names):
        d = angle_matched[angle_matched.angle == name].sort_values("frame")
        a1.plot(d.frame / fps, d.kinovea, "k-", label="Kinovea")
        a1.plot(d.frame / fps, d.pred, "-", color="#dd6b20", label="MMPose")
        a1.set_title(f"{name} angle")
        a1.set_xlabel("time (s)")
        a1.set_ylabel("deg")
        a1.legend(fontsize=8)
        # Bland-Altman
        mean, diff = (d.pred + d.kinovea) / 2, d.pred - d.kinovea
        bias, sd = diff.mean(), diff.std(ddof=1)
        a2.scatter(mean, diff, s=10, alpha=0.6)
        for v, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
            a2.axhline(v, color="gray", ls=ls)
        a2.set_title(f"Bland-Altman: bias {bias:.1f}°, LoA ±{1.96 * sd:.1f}°")
        a2.set_xlabel("mean of methods (deg)")
        a2.set_ylabel("MMPose - Kinovea (deg)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig
