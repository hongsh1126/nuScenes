"""Stage 2: verification checks and routing/fusion arms on the saved nuScenes mini frames (see nus_pipeline_check.py)."""
import json
import os
import pickle
from pathlib import Path

import numpy as np

WORK = Path(os.environ.get("NUS_WORK", r"F:\nuscenes_work"))
frames = pickle.load(open(WORK / "frames.pkl", "rb"))
BINS = ((0, 20, "0-20"), (20, 40, "20-40"), (40, 60, "40-60"), (60, 80, "60-80"))
GATE, CAM_T, LID_T, FB_T, ROI_T, ASSOC_IOU, MATCH_M = 40.0, 0.25, 0.50, 0.75, 0.35, 0.30, 2.0
LEN_PRIOR = {"car": 4.6, "pedestrian": 0.7, "twowheeler": 1.9}
OUT = {}


def iou2d(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0); inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def eligible(g, std=False):
    ok = g["in_img"] and g["vis"] >= 2 and g["dist"] <= 80.0
    return ok and (g["n_lidar"] + g["n_radar"] > 0 if std else ok)


def bin_of(d):
    return next((n for lo, hi, n in BINS if lo <= d < hi or (n == "60-80" and d == 80)), None)


# class height priors from the mini training scenes (no fine-tuning data in mini, so this is a stand-in for detector-train)
hs = {}
for f in frames:
    if f["split"] == "train":
        for g in f["gts"]:
            hs.setdefault(g["cls"], []).append(g["size"][2])
H_PRIOR = {k: float(np.median(v)) for k, v in hs.items()}
OUT["height_prior_m"] = H_PRIOR
print("height priors", H_PRIOR)

# ============================================================ verification checks
# V1 counts
cnt = {}
for f in frames:
    for g in f["gts"]:
        if eligible(g):
            b = bin_of(g["dist"])
            cnt.setdefault((g["cls"], b), [0, 0, 0])
            cnt[(g["cls"], b)][0] += 1
            cnt[(g["cls"], b)][1] += int(g["n_lidar"] + g["n_radar"] > 0)
conds = {}
for f in frames:
    conds[f["cond"]] = conds.get(f["cond"], 0) + 1
OUT["frames"] = len(frames)
OUT["scenes"] = len({f["scene"] for f in frames})
OUT["conditions_frames"] = conds
OUT["eligible_counts"] = {f"{k[0]}|{k[1]}": {"P_vis": v[0], "P_std": v[1]} for k, v in sorted(cnt.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))}
print("frames", len(frames), "scenes", OUT["scenes"], "conditions", conds)
print("eligible counts (class|bin: P_vis, P_std):")
for k, v in OUT["eligible_counts"].items():
    print(" ", k, v)

# V2 frame consistency: forward/lateral axes of the ego frame vs the camera frame
dx = [abs(g["ctr_e"][0] - g["ctr_c"][2]) for f in frames for g in f["gts"] if g["in_img"]]
dy = [abs(g["ctr_e"][1] + g["ctr_c"][0]) for f in frames for g in f["gts"] if g["in_img"]]
OUT["frame_consistency"] = dict(median_abs_forward_diff_m=float(np.median(dx)), median_abs_lateral_diff_m=float(np.median(dy)), n=len(dx))
print("ego vs camera axes: median |x_e - z_c| = %.2f m, median |y_e + x_c| = %.2f m (n=%d)" % (np.median(dx), np.median(dy), len(dx)))

# V3 LiDAR points inside GT boxes vs annotated num_lidar_pts
pairs = []
for f in frames:
    Rr = f["G2Eref"][:3, :3]
    Rl = f["L2Eref"][:3, :3]
    for g in f["gts"]:
        if not g["in_img"]:
            continue
    # points in the lidar frame
    P = f["pts_n"]
    for g in f["gts"]:
        if not (g["in_img"] and g["dist"] <= 80):
            continue
        c_l = np.linalg.inv(f["L2Eref"]) @ np.append(g["ctr_e"], 1)
        R_box = np.linalg.inv(Rl) @ Rr @ g["rot_g"]       # box local axes in the lidar frame
        local = (P - c_l[:3]) @ R_box                       # coordinates in the box frame (x = length, y = width)
        w, l, h = g["size"]
        inside = (np.abs(local[:, 0]) <= l / 2) & (np.abs(local[:, 1]) <= w / 2) & (np.abs(local[:, 2]) <= h / 2)
        pairs.append((int(inside.sum()), g["n_lidar"]))
pairs = np.array(pairs)
exact = float(np.mean(pairs[:, 0] == pairs[:, 1]))
corr = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
OUT["points_in_box_check"] = dict(n=int(len(pairs)), exact_match_share=exact, correlation=corr,
                                  median_abs_diff=float(np.median(np.abs(pairs[:, 0] - pairs[:, 1]))))
print("LiDAR points inside GT boxes vs annotated count: exact %.2f, corr %.4f, median |diff| %.1f (n=%d)" % (exact, corr, np.median(np.abs(pairs[:, 0] - pairs[:, 1])), len(pairs)))


# ============================================================ arms
def mono_pos(f, d):
    """Camera detection -> monocular position in the camera frame and BEV position in the ego frame."""
    K = f["K"]
    x1, y1, x2, y2 = d["box"]
    h = max(y2 - y1, 2.0)
    z = K[1, 1] * H_PRIOR[d["cls"]] / h
    u, v = (x1 + x2) / 2, y2
    xc, yc = (u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1]
    p_c = np.array([xc, yc, z])
    p_e = (np.linalg.inv(f["Eref2C"]) @ np.append(p_c, 1))[:3]
    return p_c, p_e


def roi_proxy(f, d):
    x1, y1, x2, y2 = d["box"]
    w, h = x2 - x1, y2 - y1
    roi = (x1 + 0.15 * w, y1 + 0.20 * h, x2 - 0.15 * w, y1 + 0.95 * h)
    uv, xyz = f["roi_uv"], f["roi_xyz"]
    k = (uv[:, 0] >= roi[0]) & (uv[:, 0] <= roi[2]) & (uv[:, 1] >= roi[1]) & (uv[:, 1] <= roi[3])
    pts = xyz[k]
    need = 5 if d["cls"] == "car" else 2
    if len(pts) < need:
        return None
    bins = np.floor(pts[:, 2]).astype(int)
    vals, c = np.unique(bins, return_counts=True)
    peak = vals[np.argmax(c)]
    cl = pts[np.abs(pts[:, 2] - (peak + 0.5)) <= 1.0]
    if len(cl) < need or np.std(cl[:, 2]) > 1.25:
        return None
    loc = np.median(cl, axis=0)
    loc[2] += LEN_PRIOR[d["cls"]] / 2
    p_e = (np.linalg.inv(f["Eref2C"]) @ np.append(loc, 1))[:3]
    return loc, p_e


def associate(cams, lids):
    cand = []
    for i, c in enumerate(cams):
        for j, l in enumerate(lids):
            if c["cls"] == l["cls"] and l["box2d"] is not None:
                iou = iou2d(c["box"][None], l["box2d"][None])[0, 0]
                if iou >= ASSOC_IOU:
                    cand.append((iou, i, j))
    pairs, uc, ul = {}, set(), set()
    for iou, i, j in sorted(cand, reverse=True):
        if i not in uc and j not in ul:
            pairs[i] = j; uc.add(i); ul.add(j)
    return pairs


def arms(f, gate):
    cams = [dict(d, **dict(zip(("p_c", "p_e"), mono_pos(f, d)))) for d in f["cam"] if d["score"] >= CAM_T]
    cams = [c for c in cams if 0 < c["p_c"][2] < 80]
    lids = [l for l in f["lid"] if l["score"] >= LID_T and 0 < l["ctr_c"][2] < 80]
    pr = associate(cams, lids)
    out = {"camera": [(c["cls"], c["p_e"][:2], c["score"]) for c in cams],
           "lidar": [(l["cls"], l["ctr_e"][:2], l["score"]) for l in lids],
           "agreement": [(lids[j]["cls"], lids[j]["ctr_e"][:2], (cams[i]["score"] + lids[j]["score"]) / 2) for i, j in pr.items()]}
    gated, used = [], set(pr.values())
    for i, c in enumerate(cams):
        if c["p_c"][2] > gate:
            gated.append((c["cls"], c["p_e"][:2], c["score"], "far_cam"))
        elif i in pr:
            gated.append((c["cls"], lids[pr[i]]["ctr_e"][:2], max(c["score"], lids[pr[i]]["score"]), "lidar_match"))
        elif c["score"] >= ROI_T:
            rp = roi_proxy(f, c)
            if rp is not None:
                gated.append((c["cls"], rp[1][:2], c["score"], "roi"))
    for j, l in enumerate(lids):
        if j not in used and l["ctr_c"][2] <= gate and l["score"] >= FB_T:
            gated.append((l["cls"], l["ctr_e"][:2], l["score"], "fallback"))
    out["gate"] = [(a, b, c) for a, b, c, _ in gated]
    out["gate_src"] = [s for *_, s in gated]
    return out, cams


def match(preds, gts):
    """class-wise greedy matching by BEV centre distance <= 2 m; returns dict gt_index -> error, list of fp indices"""
    cand = []
    for i, (c, p, _) in enumerate(preds):
        for j, g in enumerate(gts):
            if g["cls"] == c:
                d = float(np.hypot(*(p - g["ctr_e"][:2])))
                if d <= MATCH_M:
                    cand.append((d, i, j))
    mp, ug, up = {}, set(), set()
    for d, i, j in sorted(cand):
        if i not in up and j not in ug:
            mp[j] = d; up.add(i); ug.add(j)
    return mp, [i for i in range(len(preds)) if i not in up]


names = ["camera", "lidar", "agreement", "gate"]
res = {n: {"vis": {b[2]: [0, 0] for b in BINS}, "std": {b[2]: [0, 0] for b in BINS}, "fp": 0, "err": [], "cls": {}} for n in names + ["gate_all"]}
rng_err = []
for f in frames:
    for gate_name, gate in (("gate", GATE), ("gate_all", 80.0)):
        o, cams = arms(f, gate)
        sets = [("camera", o["camera"]), ("lidar", o["lidar"]), ("agreement", o["agreement"]), (gate_name, o["gate"])] if gate_name == "gate" else [(gate_name, o["gate"])]
        elig = [g for g in f["gts"] if eligible(g)]
        non_elig = [g for g in f["gts"] if not eligible(g)]
        for n, preds in sets:
            mp, fps = match(preds, elig)
            res[n]["fp"] += sum(1 for i in fps if not any(g["cls"] == preds[i][0] and np.hypot(*(preds[i][1] - g["ctr_e"][:2])) <= MATCH_M for g in non_elig))
            for j, g in enumerate(elig):
                b = bin_of(g["dist"])
                for key, std in (("vis", False), ("std", True)):
                    if eligible(g, std):
                        res[n][key][b][1] += 1
                        res[n][key][b][0] += int(j in mp)
                if j in mp and g["dist"] < 40:
                    res[n]["err"].append(mp[j])
                k = (g["cls"], b)
                res[n]["cls"].setdefault(k, [0, 0]); res[n]["cls"][k][1] += 1; res[n]["cls"][k][0] += int(j in mp)
        if gate_name == "gate":
            # monocular range error of camera detections matched to GT (camera-frame forward distance)
            mpc, _ = match(o["camera"], elig)
            for j, g in enumerate(elig):
                if j in mpc:
                    # find the matched camera detection forward z for this GT
                    best = min((np.hypot(*(c["p_e"][:2] - g["ctr_e"][:2])), c) for c in cams if c["cls"] == g["cls"])
                    rng_err.append((g["dist"], (best[1]["p_c"][2] - g["ctr_c"][2]) / g["ctr_c"][2]))
    print("frame", f["token"][:6], end="\r")
nf = len(frames)
table = {}
print("\n\nRecall by range bin (P_vis = visibility>=2 in CAM_FRONT; P_std = also >=1 LiDAR/radar point). Pipeline check only.")
print(f"{'arm':10s} " + " ".join(f"{b[2]:>14s}" for b in BINS) + "   FP/frame  err_mean(m)  err_med(m)")
for n in names + ["gate_all"]:
    row = []
    for b in BINS:
        k, m = res[n]["vis"][b[2]]
        row.append(f"{k}/{m} ({k / m:.2f})" if m else "-")
    e = np.array(res[n]["err"])
    print(f"{n:10s} " + " ".join(f"{x:>14s}" for x in row) + f"   {res[n]['fp'] / nf:7.2f}   {e.mean() if len(e) else float('nan'):9.2f}   {np.median(e) if len(e) else float('nan'):9.2f}")
    table[n] = dict(vis={b: v for b, v in res[n]["vis"].items()}, std={b: v for b, v in res[n]["std"].items()}, fp_per_frame=res[n]["fp"] / nf,
                    bev_err_mean=float(e.mean()) if len(e) else None, bev_err_median=float(np.median(e)) if len(e) else None)
OUT["arms"] = table
re_ = np.array(rng_err)
OUT["monocular_range"] = {b[2]: dict(n=int(((re_[:, 0] >= b[0]) & (re_[:, 0] < b[1])).sum()),
                                     median_rel_err=float(np.median(re_[(re_[:, 0] >= b[0]) & (re_[:, 0] < b[1]), 1])) if ((re_[:, 0] >= b[0]) & (re_[:, 0] < b[1])).any() else None) for b in BINS}
print("monocular forward-range relative error of matched camera detections:", OUT["monocular_range"])
src = {}
for f in frames:
    o, _ = arms(f, GATE)
    for s in o["gate_src"]:
        src[s] = src.get(s, 0) + 1
OUT["gate_output_sources"] = src
OUT["detections_total"] = dict(camera_raw=sum(len(f["cam"]) for f in frames), lidar_raw=sum(len(f["lid"]) for f in frames))
print("gate output sources:", src, " raw detections:", OUT["detections_total"])
json.dump(OUT, open(WORK / "pipeline_check.json", "w"), indent=1, default=float)
print("saved", WORK / "pipeline_check.json")
