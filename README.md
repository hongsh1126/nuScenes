# nuScenes mini: validation of a camera-first range-gated perception pipeline

This repository checks, end to end, a camera-first range-gating pipeline on the **nuScenes mini** split before any full-scale study. It verifies the data loading, coordinate transforms, ground-truth eligibility rules, the camera arm, a LiDAR arm, and the routing/fusion logic.

> **Scope.** The detectors are not trained for this setting: the camera arm is a COCO-pretrained YOLO11n and the LiDAR arm is either a KITTI-trained PointPillars stand-in or a nuScenes-trained PointPillars checkpoint run through MMDetection3D. The recall values produced here are **mechanics checks, not scientific results.** No claims about detector or fusion performance should be drawn from them.

## What the pipeline checks
- Ground-truth geometry: points-in-box counts are compared with the annotated `num_lidar_pts`, and frame axes are checked for consistency.
- Eligibility rules and distance bins (0-20, 20-40, 40-60, 60-80 m).
- A camera arm, a LiDAR arm, and the range-gated routing and fusion arms.
- Stage 2 re-runs the verification checks and the routing/fusion arms on the saved frames.

## Contents
| Path | Purpose |
|---|---|
| `nus_pipeline_check.py` | Stage 1: loads nuScenes mini, runs the camera and LiDAR arms, saves per-frame results |
| `nus_pipeline_eval.py` | Stage 2: verification checks and routing/fusion arms on the saved frames |
| `patch_json_lidar.py` | Adds the optional `NUS_LIDAR_JSON` input to stage 1 (LiDAR arm from an MMDetection3D `results_nusc.json`) |
| `colab/run_mmdet3d_nuscenes.ipynb` | Colab notebook that runs the public nuScenes PointPillars checkpoint with MMDetection3D 1.4.0 (torch 2.1.0+cu118, mmcv 2.1.0, mmdet 3.2.0) on a T4 GPU |
| `colab/make_notebook.py` | Generates the notebook |
| `results/` | Outputs of the pipeline check (`pipeline_check*.json`, `stage1.log`) |

## How to run
Set the paths with environment variables (defaults are relative to the working directory):

| Variable | Meaning | Default |
|---|---|---|
| `NUS_ROOT` | nuScenes mini data root (`v1.0-mini`) | `data/v1.0-mini` |
| `NUS_WORK` | output directory | `work` |
| `PP_ROOT` | PointPillars implementation with a KITTI checkpoint (stand-in LiDAR arm), e.g. a clone of [zhulf0804/PointPillars](https://github.com/zhulf0804/PointPillars) | `PointPillars` |
| `NUS_LIDAR_JSON` | optional `results_nusc.json` from the Colab notebook (after running `patch_json_lidar.py` once) | none |
| `NUS_LIMIT` | optional number of samples | all |

```bash
pip install nuscenes-devkit numpy torch ultralytics
python nus_pipeline_check.py   # stage 1
python nus_pipeline_eval.py    # stage 2
```

## Data
The nuScenes data are **not** included. Download them from [nuscenes.org](https://www.nuscenes.org) under its [terms of use](https://www.nuscenes.org/terms-of-use) (CC BY-NC-SA 4.0).

## License
MIT for the code in this repository (see `LICENSE`). nuScenes, MMDetection3D, PointPillars and the pretrained checkpoints have their own licenses.
