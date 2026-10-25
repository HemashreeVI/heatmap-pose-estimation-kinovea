"""Step 1 - run MMPose heatmap pose estimation on a video / image / image folder.

    python scripts/run_pose.py --input data/trial.mp4 --out results/trial
    python scripts/run_pose.py --input data/trial.mp4 --out results/trial --model resnet50 --no-detector
"""

import argparse
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from posestudy.estimator import MODEL_ZOO, PersonDetector, load_pose_model, run_on_video  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="video file, image, or folder of images")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--model", default="hrnet-w32", choices=sorted(MODEL_ZOO))
    p.add_argument("--config", help="custom MMPose top-down heatmap config (overrides --model)")
    p.add_argument("--checkpoint", help="custom checkpoint path/URL")
    p.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    p.add_argument("--no-detector", action="store_true",
                   help="use the whole frame as the person box (subject fills the frame)")
    p.add_argument("--stride", type=int, default=1, help="process every n-th frame")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=None, help="override video fps")
    args = p.parse_args()

    model = load_pose_model(args.model, args.device, args.config, args.checkpoint)
    detector = None if args.no_detector else PersonDetector(args.device)
    run_on_video(model, args.input, args.out, detector=detector, stride=args.stride,
                 max_frames=args.max_frames, fps=args.fps,
                 model_name=args.config or args.model)


if __name__ == "__main__":
    main()
