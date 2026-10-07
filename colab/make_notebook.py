"""Writes run_mmdet3d_nuscenes.ipynb (Colab notebook: nuScenes LiDAR detections from the public MMDetection3D PointPillars checkpoint)."""
import json
from pathlib import Path

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)})


md("""
# nuScenes LiDAR detections with the public MMDetection3D PointPillars checkpoint (Colab)

Output: `results_nusc.json` in the nuScenes detection-submission format (global-frame boxes), which
`nus_pipeline_check.py` reads through `NUS_LIDAR_JSON`.

* Runtime: **GPU** (Runtime > Change runtime type). The T4 is enough for the mini split.
* This notebook was written without access to a Colab runtime. Package versions are pinned to the combination that
  MMDetection3D 1.4 supports (mmcv < 2.2, mmdet < 3.4); if a cell fails, the notes under it list the usual fixes.
* nuScenes data are CC BY-NC-SA 4.0; use is subject to the nuScenes terms of use (https://www.nuscenes.org/terms-of-use).
""")

md("## 1. GPU and Python version")
code("""
import subprocess, sys
print(sys.version)
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:600])
""")

md("""
## 2. Isolated Python 3.10 environment
Colab's default Python and PyTorch are newer than what mmcv 2.1 (the newest release allowed by MMDetection3D) has
prebuilt wheels for. An isolated environment with Python 3.10 and PyTorch 2.1 / CUDA 11.8 avoids compiling mmcv.
""")
code("""
!pip -q install uv
!uv venv /content/venv --python 3.10
PY = "/content/venv/bin/python"
!uv pip install --python {PY} "torch==2.1.0" "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu118
!uv pip install --python {PY} "numpy==1.23.5" "setuptools<70" wheel openmim "mmengine>=0.8,<1" nuscenes-devkit numba scipy
!uv pip install --python {PY} "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
!uv pip install --python {PY} "mmdet==3.2.0" "mmdet3d==1.4.0"
!{PY} -c "import torch, mmcv, mmdet, mmdet3d; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__, mmdet.__version__, mmdet3d.__version__)"
""")
md("""
Usual fixes: if `mmcv` cannot be imported with *No module named mmcv._ext*, the wheel and torch/CUDA do not match
(check the index URL). If `numpy` errors appear, keep numpy at 1.23.5 or lower than 2.0.
""")

md("## 3. Configs and tools of MMDetection3D (same version) and the PointPillars checkpoint")
code("""
!git clone -q --depth 1 -b v1.4.0 https://github.com/open-mmlab/mmdetection3d.git /content/mmdetection3d
!mkdir -p /content/ckpt
!wget -q -O /content/ckpt/pp_fpn_nus.pth https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth
!ls -la /content/ckpt /content/mmdetection3d/configs/pointpillars | head -n 30
""")
md("""
Config: `configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py` (nuScenes, 10 classes, 10 LiDAR sweeps;
reported mAP 39.7 / NDS 53.2). A CenterPoint checkpoint is stronger but needs spconv; start with PointPillars, which
is also the architecture used in the KITTI study.
""")

md("""
## 4. nuScenes mini data
Option A (default): download the 4.2 GB archive directly. Option B: copy `v1.0-mini.tgz` to Google Drive and mount it.
""")
code("""
!mkdir -p /content/data/nuscenes
!wget -q -O /content/v1.0-mini.tgz https://www.nuscenes.org/data/v1.0-mini.tgz
!tar -xzf /content/v1.0-mini.tgz -C /content/data/nuscenes
!ls /content/data/nuscenes
""")
code("""
# Option B (use instead of the cell above):
# from google.colab import drive; drive.mount('/content/drive')
# !tar -xzf "/content/drive/MyDrive/nuscenes/v1.0-mini.tgz" -C /content/data/nuscenes
""")

md("## 5. Create the MMDetection3D info files (mini)")
code("""
!rm -rf /content/mmdetection3d/data/data
!mkdir -p /content/mmdetection3d/data
!ln -sfn /content/data/nuscenes /content/mmdetection3d/data/nuscenes
!cd /content/mmdetection3d && PYTHONPATH=/content/mmdetection3d /content/venv/bin/python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini 2>&1 | tail -n 12
!ls /content/data/nuscenes
""")
md("""
Expected: `nuscenes_infos_train.pkl` (323 keyframes) and `nuscenes_infos_val.pkl` (81 keyframes) of the mini split.
`create_data.py` needs `PYTHONPATH` (it imports `tools.*`) and the relative path `./data/nuscenes` (its info-update step uses it),
therefore the symlink inside the repository. The mini split has 10 scenes; the main run needs the full validation scenes.
""")

md("## 6. Run the detector on the mini keyframes (val and train infos separately) and write the submission-format json")
code("""
!cd /content/mmdetection3d && for s in val train; do PYTHONPATH=/content/mmdetection3d /content/venv/bin/python tools/test.py configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py /content/ckpt/pp_fpn_nus.pth --cfg-options test_dataloader.dataset.ann_file=nuscenes_infos_$s.pkl test_evaluator.ann_file=data/nuscenes/nuscenes_infos_$s.pkl test_evaluator.format_only=True test_evaluator.jsonfile_prefix=/content/out/$s test_dataloader.num_workers=2 2>&1 | tail -n 4; done; find /content/out -name results_nusc.json
""")
md("""
Notes. The two runs take about 4 minutes on a T4 (81 + 323 keyframes) and write
`/content/out/val/pred_instances_3d/results_nusc.json` (about 10 MB) and `/content/out/train/pred_instances_3d/results_nusc.json`
(about 26 MB). Do not merge the info pickles in the system Python: it has NumPy 2, which writes `numpy._core` references that
the NumPy 1.23 environment cannot read.
""")

md("## 7. Sanity check and download")
code("""
import json
res = {}
for sp in ("val", "train"):
    res.update(json.load(open(f"/content/out/{sp}/pred_instances_3d/results_nusc.json"))["results"])
n_boxes = sum(len(v) for v in res.values())
print("samples:", len(res), "boxes:", n_boxes)
k = next(iter(res)); print(k, res[k][:1])
import collections
print(collections.Counter(b["detection_name"] for v in res.values() for b in v).most_common())
""")
code("""
from google.colab import files
files.download("/content/out/val/pred_instances_3d/results_nusc.json")
files.download("/content/out/train/pred_instances_3d/results_nusc.json")
""")

md("""
## 8. Use the result in the local pipeline
Copy `results_nusc.json` to the PC and run (Windows):

    set NUS_LIDAR_JSON=F:\\nuscenes_work\\val_results_nusc.json;F:\\nuscenes_work\\train_results_nusc.json
    python nus_pipeline_check.py     # stage 1: LiDAR arm now comes from this file
    python nus_pipeline_eval.py      # stage 2: routing, matching and recall

## Notes for the full study
* The official nuScenes download is split in blobs of several tens of GB, and a Colab VM has about 100 GB of disk.
  For the validation scenes either use a machine with more disk, mount a cloud bucket, or process one blob at a time
  (verify the keyframe-only archives on the nuScenes download page).
* The detector predicts 10 classes. `bicycle` and `motorcycle` detections do not know whether a rider is present, which
  matters for the *two-wheeler with rider* class of the study design.
* The checkpoint was trained on the nuScenes training scenes, so the validation scenes are unseen, as the study design requires.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).parent / "run_mmdet3d_nuscenes.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
