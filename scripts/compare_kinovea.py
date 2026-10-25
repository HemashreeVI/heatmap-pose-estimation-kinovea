"""Step 3 - validate predicted keypoints / joint angles against Kinovea.

    # look at the export first, then write a mapping.json (see data/kinovea_mapping_example.json)
    python scripts/compare_kinovea.py --kinovea data/trial_kinovea.csv --inspect

    python scripts/compare_kinovea.py --results results/trial \
        --kinovea data/trial_kinovea.csv --mapping data/kinovea_mapping.json --auto-offset 10

Writes into <results>/kinovea_validation/:
    matched_keypoints.csv, keypoint_summary.csv (error px, RMSE, PCK per joint)
    confidence_vs_error.csv, angle_agreement.csv (MAE, RMSE, bias, Bland-Altman LoA, r)
    error_per_joint.png, trajectory_<joint>.png, confidence_vs_error.png, angles.png
"""

import argparse
import json
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from posestudy import heatmaps as hm  # noqa: E402
from posestudy import kinovea as kv  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kinovea", required=True, help="Kinovea export (.csv / .txt / .xlsx)")
    p.add_argument("--inspect", action="store_true", help="only print the export's columns")
    p.add_argument("--results", help="folder written by run_pose.py")
    p.add_argument("--mapping", help="JSON column mapping (see posestudy/kinovea.py)")
    p.add_argument("--frame-offset", type=int, default=0, help="add to Kinovea frame numbers")
    p.add_argument("--auto-offset", type=int, default=0,
                   help="search +-N frames for the best time alignment")
    p.add_argument("--auto-align", action="store_true",
                   help="fit scale/translation/y-flip from Kinovea coords to pixels "
                        "(only if the Kinovea coordinate system is unknown)")
    args = p.parse_args()

    if args.inspect:
        kv.inspect(args.kinovea)
        return
    if not (args.results and args.mapping):
        p.error("--results and --mapping are required (or use --inspect)")

    res = hm.load_results(args.results)
    fps = res["meta"]["fps"]
    out = osp.join(args.results, "kinovea_validation")
    os.makedirs(out, exist_ok=True)
    gt_pts, gt_ang = kv.load_kinovea(args.kinovea, args.mapping, fps, args.frame_offset)
    print(f"Kinovea: {len(gt_pts)} point samples ({gt_pts.joint.nunique()} joints), "
          f"{len(gt_ang)} angle samples")

    metrics = hm.metrics_table(res)
    pred = kv.predictions_table(res, metrics)
    report = dict(frame_offset=args.frame_offset)

    if len(gt_pts):
        if args.auto_offset:
            off, _ = kv.best_frame_offset(pred, gt_pts, args.auto_offset)
            print(f"best extra frame offset: {off:+d}")
            gt_pts = gt_pts.assign(frame=gt_pts.frame + off)
            gt_ang = gt_ang.assign(frame=gt_ang.frame + off)
            report["auto_offset"] = off
        if args.auto_align:
            m0 = kv.match_keypoints(pred, gt_pts)
            params, f = kv.fit_similarity(m0[["x_gt", "y_gt"]].to_numpy(), m0[["x_pred", "y_pred"]].to_numpy())
            gt_pts[["x", "y"]] = f(gt_pts[["x", "y"]].to_numpy())
            print("auto-align:", params)
            report["auto_align"] = params

        matched = kv.match_keypoints(pred, gt_pts)
        if matched.empty:
            sys.exit("No overlapping frames/joints - check the time column, fps and frame offset.")
        matched.to_csv(osp.join(out, "matched_keypoints.csv"), index=False)
        summary = kv.keypoint_summary(matched)
        summary.to_csv(osp.join(out, "keypoint_summary.csv"))
        print("\nKeypoint error vs Kinovea:\n", summary.round(2).to_string())

        bins, rho, flagged = kv.confidence_vs_error(matched)
        bins.to_csv(osp.join(out, "confidence_vs_error.csv"))
        print(f"\nError by heatmap confidence (Spearman rho score~error = {rho:.2f}):\n{bins.round(2)}")
        if flagged is not None:
            print("\nError for frames flagged unreliable from heatmaps:\n", flagged.round(2))
        report["spearman_conf_error"] = rho

        kv.plot_error_per_joint(summary, osp.join(out, "error_per_joint.png"))
        kv.plot_confidence_vs_error(matched, osp.join(out, "confidence_vs_error.png"))
        for joint in matched.joint.unique():
            kv.plot_trajectories(matched, joint, fps, osp.join(out, f"trajectory_{joint}.png"))
        plt.close("all")

    if len(gt_ang):
        ang_m, ang_s = kv.angle_agreement(kv.predicted_angles(res), gt_ang)
        if ang_m.empty:
            print("No overlapping angle frames.")
        else:
            ang_m.to_csv(osp.join(out, "matched_angles.csv"), index=False)
            ang_s.to_csv(osp.join(out, "angle_agreement.csv"))
            print("\nJoint-angle agreement vs Kinovea:\n", ang_s.round(2).to_string())
            kv.plot_angles(ang_m, fps, osp.join(out, "angles.png"))
            plt.close("all")

    with open(osp.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("\nSaved validation to", out)


if __name__ == "__main__":
    main()
