"""Step 2 - visualise and analyse the joint-confidence heatmaps.

    python scripts/analyse_heatmaps.py --results results/trial
    python scripts/analyse_heatmaps.py --results results/trial --video --joint left_knee

Writes into <results>/heatmap_analysis/:
    heatmap_metrics.csv       per frame x joint: score, spread, second peak, flags
    joint_summary.csv         per joint averages + % unreliable frames
    confidence_timeline.png   confidence over time, unreliable frames in red
    joint_summary.png         mean confidence and reasons for flags per joint
    confidence_vs_spread.png  heatmap shape vs confidence
    grid_frame_XXXX.png       17-panel per-joint heatmaps (best/worst/middle frame)
    overlay_frame_XXXX.jpg    OpenCV overlay of all heatmaps + skeleton
    heatmap_overlay.mp4       (with --video) overlay for every processed frame
"""

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from posestudy import heatmaps as hm  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="folder written by run_pose.py")
    p.add_argument("--conf-thr", type=float, default=0.3, help="flag joints with peak below this")
    p.add_argument("--spread-frac", type=float, default=0.08,
                   help="flag joints whose heatmap spread > this fraction of the person box")
    p.add_argument("--second-thr", type=float, default=0.5,
                   help="flag joints whose second peak is > this fraction of the main peak")
    p.add_argument("--frames", type=int, nargs="*", help="frame indices to plot (default: auto)")
    p.add_argument("--video", action="store_true", help="also render an overlay video")
    p.add_argument("--joint", default=None, help="overlay video shows only this joint")
    args = p.parse_args()

    res = hm.load_results(args.results)
    out = osp.join(args.results, "heatmap_analysis")
    os.makedirs(out, exist_ok=True)
    flag_kw = dict(conf_thr=args.conf_thr, spread_frac=args.spread_frac, second_thr=args.second_thr)

    metrics = hm.metrics_table(res, **flag_kw)
    metrics.to_csv(osp.join(out, "heatmap_metrics.csv"), index=False)
    summary = hm.joint_summary(metrics)
    summary.to_csv(osp.join(out, "joint_summary.csv"))
    print(summary.round(3).to_string())

    hm.plot_confidence_timeline(metrics, conf_thr=args.conf_thr,
                                out_path=osp.join(out, "confidence_timeline.png"))
    hm.plot_joint_summary(summary, out_path=osp.join(out, "joint_summary.png"))
    hm.plot_metric_scatter(metrics, out_path=osp.join(out, "confidence_vs_spread.png"))
    plt.close("all")

    # Frames to inspect: best, median and worst mean body-joint confidence.
    body_conf = res["scores"][:, 5:].mean(1)
    if args.frames:
        ts = [int(np.where(res["frames"] == f)[0][0]) for f in args.frames]
    else:
        order = np.argsort(body_conf)
        ts = sorted({int(order[-1]), int(order[len(order) // 2]), int(order[0])})
    m = hm.heatmap_metrics(res["heatmaps"], res["input_scale"])
    flags = hm.flag_unreliable(m, hm.bbox_sizes(res["bboxes"]), **flag_kw)
    for t in ts:
        f_idx = int(res["frames"][t])
        frame = hm.read_frame(res["meta"]["source"], f_idx)
        hm.plot_joint_heatmap_grid(res, t, frame, out_path=osp.join(out, f"grid_frame_{f_idx:05d}.png"))
        cv2.imwrite(osp.join(out, f"overlay_frame_{f_idx:05d}.jpg"), hm.render_frame(res, t, frame, flags=flags))
        plt.close("all")
        print(f"frame {f_idx}: mean body confidence {body_conf[t]:.3f}")

    if args.video:
        path = hm.render_heatmap_video(res, osp.join(out, "heatmap_overlay.mp4"), joint=args.joint, flags=flags)
        print("video:", path)
    print("Saved analysis to", out)


if __name__ == "__main__":
    main()
