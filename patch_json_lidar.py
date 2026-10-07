"""Adds the NUS_LIDAR_JSON path to nus_pipeline_check.py: LiDAR arm from an MMDetection3D results_nusc.json."""
from pathlib import Path

P = Path(__file__).parent / "nus_pipeline_check.py"
s = P.read_text(encoding="utf-8")
if "NUS_LIDAR_JSON" in s:
    raise SystemExit("already patched")

s = s.replace('''    sys.path.insert(0, PP_ROOT)
    from pointpillars.model import PointPillars
''', '''    lidar_json = os.environ.get("NUS_LIDAR_JSON", "")
    json_res = json.load(open(lidar_json))["results"] if lidar_json else None
    sys.path.insert(0, PP_ROOT)
    from pointpillars.model import PointPillars
''', 1)
s = s.replace('''    pp = PointPillars(nclasses=3)
    pp.load_state_dict(torch.load(os.path.join(PP_ROOT, "pretrained", "epoch_160.pth"), map_location="cpu"))
    pp.eval()
''', '''    pp = None
    if json_res is None:
        pp = PointPillars(nclasses=3)
        pp.load_state_dict(torch.load(os.path.join(PP_ROOT, "pretrained", "epoch_160.pth"), map_location="cpu"))
        pp.eval()
''', 1)

old_start = '''        with torch.inference_mode():
            res = pp([torch.from_numpy(pts_k[keep])], mode="test")[0]
        lid_dets = []
        for bx, s, k in zip('''
new_start = '''        lid_dets = []
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
        for bx, s, k in zip('''
assert old_start in s
s = s.replace(old_start, new_start, 1)
s = s.replace('''        for bx, s, k in zip(np.asarray(res["lidar_bboxes"]), np.asarray(res["scores"]), np.asarray(res["labels"])):''',
              '''        for bx, s, k in ([] if json_res is not None else zip(np.asarray(res["lidar_bboxes"]), np.asarray(res["scores"]), np.asarray(res["labels"]))):''', 1)
P.write_text(s, encoding="utf-8")
print("patched")
