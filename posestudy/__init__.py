"""Human pose estimation with heatmap-based models (MMPose).

Modules
-------
skeleton   : COCO keypoint names, skeleton edges, joint-angle helpers (no MMPose needed)
estimator  : MMPose model loading + per-frame inference that keeps the raw heatmaps
heatmaps   : heatmap quality metrics and OpenCV / Matplotlib visualisations
kinovea    : loading Kinovea exports and validating predictions against them
"""

__version__ = "1.0.0"
