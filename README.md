# Referring 3D Object Detection on LiDAR

Language-guided (open-vocabulary) 3D object detection on nuScenes LiDAR point clouds.
A frozen **PointPillars** detector provides Bird's-Eye-View features and oriented box
proposals; a Deformable-DETR–style decoder fuses them with **RoBERTa** language
embeddings to ground a free-form natural-language query to 3D boxes, while learning to
correctly reject **null-target** queries (objects that are not present).

This is the **complete, self-contained release package** for the thesis: models,
training and evaluation entry points, the referring dataset loader, the parametrised
run scripts, and every analysis/experiment reported in the thesis. It ships as a fork
of **[mmdetection3d v1.4.0](README.mmdet3d.md)** and is built to run **from a single
Docker container** — pull the repo, build the image once, and you are ready to run the
whole extract → train → evaluate pipeline. Nothing has to be compiled by hand.

## Quick start (Docker)

Requirements on the host: an NVIDIA GPU + recent driver, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone https://github.com/OrestisZach/thesis_repo_models.git
cd thesis_repo_models/refer_lidar

# 1. build the image once (PyTorch 2.4/CUDA 12.4 + mmdet3d + MSDeformAttn CUDA op,
#    RoBERTa, nuScenes devkit — all baked in). ~15-20 min.
REFER_DATA_ROOT=/path/to/nuscenes docker compose build refer-lidar

# 2. drop into the container (working dir = /workspace/refer_lidar)
REFER_DATA_ROOT=/path/to/nuscenes docker compose run --rm refer-lidar
```

`REFER_DATA_ROOT` is your host directory holding nuScenes, the prepared referring
queries, and the feature caches; it is mounted at `/data`. `NVIDIA_VISIBLE_DEVICES`
selects the GPUs (default: all). Once inside the container, follow
[`refer_lidar/README.md`](refer_lidar/README.md) for the extract → train → eval flow.

The MSDeformAttn CUDA op and mmdet3d are already compiled inside the image, so there is
**no manual build step** — the container-first workflow is the supported path.

## Where the code is

All of the project's contribution lives in **[`refer_lidar/`](refer_lidar/README.md)** —
the architectures (a 3-model cumulative ablation chain + a tight-NMS final model), the extract → train → eval
pipeline, the parametrised run scripts, and the thesis analyses (proposal recall/noise,
per-type & per-class mAP, class-confusion matrices, the yaw-flip diagnosis, and the
center-distance NMS de-duplication study).

## Relationship to mmdetection3d

This repository is a fork of **[mmdetection3d v1.4.0](README.mmdet3d.md)**. The
`mmdet3d/` library and all upstream configs/tools are **unmodified** and used purely as a
dependency (frozen PointPillars detector, 3D box structures, BEV NMS). The referring
model, dataset, training, evaluation, and analysis code are entirely contained in
`refer_lidar/`. Upstream mmdetection3d is licensed under Apache-2.0 (see `LICENSE`).

## Data

nuScenes (`v1.0-trainval`) and a PointPillars checkpoint are the detector-side inputs.
The referring-query dataset (the `ablation_fixed/sampled_*` directories) is produced by a
**separate generation repository**
([thesis_repo_dataset](https://github.com/OrestisZach/thesis_repo_dataset)) and consumed
here as prepared files. See [`refer_lidar/README.md`](refer_lidar/README.md) for the full
data layout and pipeline.
