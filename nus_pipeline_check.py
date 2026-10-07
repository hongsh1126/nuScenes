"""Pipeline check of the camera-first range-gating comparator on nuScenes mini.

Purpose: verify data loading, coordinate transforms, eligibility rules, the camera arm (COCO YOLO11n), a LiDAR
stand-in (KITTI-trained PointPillars on converted nuScenes points) and the routing/fusion code end to end.
The detectors are NOT trained for nuScenes, so the recall values are mechanics checks, not scientific results.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes

DATAROOT = os.environ.get("NUS_ROOT", r"F:\Research\T-IV Dataset\v1.0-mini")
WORK = Path(os.environ.get("NUS_WORK", r"F:\nuscenes_work"))
WORK.mkdir(parents=True, exist_ok=True)
PP_ROOT = r"G:\내 드라이브\Research\Journal\2026\In progress\Rejected\27. TR-C\experiments\PointPillars"
YOLO_W = r"G:\내 드라이브\Research\Journal\2026\In progress\TRC\yolo11n.pt"
BINS = ((0, 20, "0-20"), (20, 40, "20-40"), (40, 60, "40-60"), (60, 80, "60-80"))
CLS = {"car": 0, "pedestrian": 1, "twowheeler": 2}
COCO = {0: "pedestrian", 1: "twowheeler", 3: "twowheeler", 2: "car"}
GATE, CAM_T, LID_T, FB_T, ROI_T, ASSOC_IOU, MATCH_M = 40.0, 0.25, 0.50, 0.75, 0.35, 0.30, 2.0
PRIOR_WLH = {"car": (1.95, 4.6, 1.7), "pedestrian": (0.7, 0.7, 1.75), "twowheeler": (0.8, 1.9, 1.6)}

nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)


def T(rot, trans):
    M = np.eye(4)
    M[:3, :3] = Quaternion(rot).rotation_matrix
    M[:3, 3] = trans
    return M


def sd_info(sd_token):
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    K = np.array(cs["camera_intrinsic"]) if cs["camera_intrinsic"] else None
    return dict(S2E=T(cs["rotation"], cs["translation"]), E2G=T(ep["rotation"], ep["translation"]), K=K,
                path=os.path.join(nusc.dataroot, sd["filename"]))


def cls_of(ann):
    name = ann["category_name"]
    attrs = [nusc.get("attribute", t)["name"] for t in ann["attribute_tokens"]]
    if name == "vehicle.car":
        return "car"
    if name.startswith("human.pedestrian"):
        return "pedestrian"
    if name in ("vehicle.bicycle", "vehicle.motorcycle") and "cycle.with_rider" in attrs:
        return "twowheeler"
    return None


def condition(desc):
    d = desc.lower()
    return "night" if "night" in d else "rain" if "rain" in d else "day-dry"


def box_corners(center, wlh, yaw_rot):
    w, l, h = wlh
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    return (yaw_rot @ np.vstack([x, y, z])).T + center


def project(pts_cam, K):
    uvw = (K @ pts_cam.T).T
    return uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)


def iou2d(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0); inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


# ------------------------------------------------------------------ stage 1: data, GT, camera, LiDAR stand-in
def stage1():
    from ultralytics import YOLO
    import torch
    lidar_json = os.environ.get("NUS_LIDAR_JSON", "")
    json_res = None
    if lidar_json:
        json_res = {}
        for part in lidar_json.split(";"):
            json_res.update(json.load(open(part.strip()))["results"])
    sys.path.insert(0, PP_ROOT)
    from pointpillars.model import PointPillars
    torch.set_num_threads(12)
    yolo = YOLO(YOLO_W)
    pp = None
    if json_res is None:
        pp = PointPillars(nclasses=3)
        pp.load_state_dict(torch.load(os.path.join(PP_ROOT, "pretrained", "epoch_160.pth"), map_location="cpu"))
        pp.eval()
    splits = create_splits_scenes()
    scene_split = {s: ("val" if s in splits["mini_val"] else "train") for s in splits["mini_train"] + splits["mini_val"]}
    frames = []
    t0 = time.time()
    limit = int(os.environ.get("NUS_LIMIT", "0")) or len(nusc.sample)
    for si, sample in enumerate(nusc.sample[:limit]):
        scene = nusc.get("scene", sample["scene_token"])
        lid = sd_info(sample["data"]["LIDAR_TOP"])
        cam = sd_info(sample["data"]["CAM_FRONT"])
        G2Eref = np.linalg.inv(lid["E2G"])            # global -> ego frame at the LiDAR time (x forward, y left)
        G2C = np.linalg.inv(cam["E2G"] @ cam["S2E"])   # global -> camera frame
        Eref2C = G2C @ lid["E2G"]
        L2Eref = lid["S2E"]
        L2C = Eref2C @ L2Eref
        img = Image.open(cam["path"]).convert("RGB")
        W, H = img.size
        # ---------------- ground truth
        gts = []
        for tok in sample["anns"]:
            ann = nusc.get("sample_annotation", tok)
            ann = dict(ann, category_name=nusc.get("category", nusc.get("instance", ann["instance_token"])["category_token"])["name"])
            c = cls_of(ann)
            if c is None:
                continue
            vis = int(nusc.get("visibility", ann["visibility_token"])["token"])
            ctr_g = np.array(ann["translation"])
            ctr_e = (G2Eref @ np.append(ctr_g, 1))[:3]
            ctr_c = (G2C @ np.append(ctr_g, 1))[:3]
            rot_g = Quaternion(ann["rotation"]).rotation_matrix
            corners_g = box_corners(ctr_g, ann["size"], rot_g)
            corners_c = (G2C @ np.vstack([corners_g.T, np.ones(8)]))[:3].T
            in_front = ctr_c[2] > 0.1
            uv = project(ctr_c[None], cam["K"])[0] if in_front else np.array([-1, -1])
            in_img = in_front and 0 <= uv[0] < W and 0 <= uv[1] < H
            box2d = None
            if (corners_c[:, 2] > 0.1).all():
                p = project(corners_c, cam["K"])
                box2d = np.array([max(p[:, 0].min(), 0), max(p[:, 1].min(), 0), min(p[:, 0].max(), W), min(p[:, 1].max(), H)])
            gts.append(dict(cls=c, vis=vis, ctr_e=ctr_e, ctr_c=ctr_c, size=ann["size"], rot_g=rot_g,
                            n_lidar=ann["num_lidar_pts"], n_radar=ann["num_radar_pts"], in_img=bool(in_img), box2d=box2d,
                            dist=float(np.hypot(ctr_e[0], ctr_e[1])), inst=ann["instance_token"]))
        # ---------------- camera detections (COCO YOLO11n)
        r = yolo.predict(source=np.array(img)[:, :, ::-1], imgsz=1280, conf=0.10, device="cpu", verbose=False)[0]
        cam_dets = []
        for b, s, k in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy().astype(int)):
            if k in COCO:
                cam_dets.append(dict(cls=COCO[k], box=b.astype(float), score=float(s)))
        # ---------------- LiDAR points and stand-in detector
        pc = LidarPointCloud.from_file(lid["path"])
        pts_n = pc.points[:4].T.copy()                  # x right, y forward, z up, intensity 0..255 (lidar frame)
        pts_k = np.stack([pts_n[:, 1], -pts_n[:, 0], pts_n[:, 2], pts_n[:, 3] / 255.0], 1).astype(np.float32)
        keep = (pts_k[:, 0] > 0) & (pts_k[:, 0] < 69.12) & (np.abs(pts_k[:, 1]) < 39.68) & (pts_k[:, 2] > -3) & (pts_k[:, 2] < 1)
        lid_dets = []
        if json_res is not None:
            NAME = {"car": "car", "pedestrian": "pedestrian", "bicycle": "twowheeler", "motorcycle": "twowheeler"}
            for b in json_res.get(sample["token"], []):
                if b["detection_name"] not in NAME:
                    continue
                ctr_g = np.array(b["translation"])
                rot_g = Quaternion(b["rotation"]).rotation_matrix
                cor_g = box_corners(ctr_g, b["size"], rot_g)
                cor_c = (G2C @ np.vstack([cor_g.T, np.ones(8)]))[:3].T
                box2d = None
                if (cor_c[:, 2] > 0.1).all():
                    p = project(cor_c, cam["K"])
                    box2d = np.array([max(p[:, 0].min(), 0), max(p[:, 1].min(), 0), min(p[:, 0].max(), W), min(p[:, 1].max(), H)])
                lid_dets.append(dict(cls=NAME[b["detection_name"]], score=float(b["detection_score"]),
                                     ctr_e=(G2Eref @ np.append(ctr_g, 1))[:3], ctr_c=(G2C @ np.append(ctr_g, 1))[:3], box2d=box2d))
        else:
            with torch.inference_mode():
                res = pp([torch.from_numpy(pts_k[keep])], mode="test")[0]
        for bx, s, k in ([] if json_res is not None else zip(np.asarray(res["lidar_bboxes"]), np.asarray(res["scores"]), np.asarray(res["labels"]))):
            # KITTI-lidar (x fwd, y left) -> nuScenes lidar (x right, y fwd)
            # z of the detector output is the bottom centre; the box size order of the vendored model is not relied on,
            # class-prior sizes (w, l, h) with the predicted heading are used for the image-plane projection
            name = {0: "pedestrian", 1: "twowheeler", 2: "car"}[int(k)]
            pw, pl, ph = PRIOR_WLH[name]
            c_n = np.array([-bx[1], bx[0], bx[2] + ph / 2])
            rot = Quaternion(axis=[0, 0, 1], angle=bx[6] + np.pi / 2).rotation_matrix
            cor_n = box_corners(c_n, (pw, pl, ph), rot)
            c_e = (L2Eref @ np.append(c_n, 1))[:3]
            cor_c = (L2C @ np.vstack([cor_n.T, np.ones(8)]))[:3].T
            box2d = None
            if (cor_c[:, 2] > 0.1).all():
                p = project(cor_c, cam["K"])
                box2d = np.array([max(p[:, 0].min(), 0), max(p[:, 1].min(), 0), min(p[:, 0].max(), W), min(p[:, 1].max(), H)])
            c_c = (L2C @ np.append(c_n, 1))[:3]
            lid_dets.append(dict(cls={0: "pedestrian", 1: "twowheeler", 2: "car"}[int(k)], score=float(s), ctr_e=c_e, ctr_c=c_c, box2d=box2d))
        # points in the camera frame for the ROI proxy (keep those in front)
        pts_c = (L2C @ np.vstack([pts_n[:, :3].T, np.ones(len(pts_n))]))[:3].T
        front = pts_c[:, 2] > 0.1
        uv_all = project(pts_c[front], cam["K"])
        frames.append(dict(token=sample["token"], scene=scene["name"], split=scene_split[scene["name"]], cond=condition(scene["description"]),
                           size=(W, H), K=cam["K"], gts=gts, cam=cam_dets, lid=lid_dets, roi_uv=uv_all, roi_xyz=pts_c[front],
                           L2Eref=L2Eref, Eref2C=Eref2C, G2Eref=G2Eref, pts_n=pts_n[:, :3].astype(np.float32)))
        if (si + 1) % 20 == 0:
            print(f"stage1 {si + 1}/{len(nusc.sample)} ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump(frames, open(WORK / "frames.pkl", "wb"))
    print("stage1 saved", len(frames), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    stage1()
