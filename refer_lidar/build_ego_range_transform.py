"""Build ego_range_transform.json: per LIDAR_TOP keyframe, the exact transform
that reproduces the devkit's ego_dist (loaders.py add_center_dist:195-197 +
DetectionBox.ego_dist = xy-norm) from SENSOR-frame coordinates.

Devkit: ego_translation = box.translation_global - ego_pose.translation
      = R_ge @ (R_es @ x_s + t_es)          (global rotation of the ego-frame vec)
      ego_dist = || (ego_translation)_xy ||
So with M = R_ge @ R_es and b = R_ge @ t_es (rows 0,1 only needed):
      ego_dist(x_s) = hypot(M0 . x_s + b0, M1 . x_s + b1)
Exact including vehicle roll/pitch. Keyed by basename(lidar filename).
Value: [m00,m01,m02, m10,m11,m12, b0,b1].
"""
import argparse
import json
import numpy as np
import os.path as osp

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument('--table-dir', default='/data/nuscenes/v1.0-trainval')
_ap.add_argument('--out', default='/data/nuscenes/ego_range_transform.json')
_a = _ap.parse_args()
TABLE_DIR = _a.table_dir
OUT = _a.out


def quat_to_rot(w, x, y, z):
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


print('[map] loading calibrated_sensor.json / ego_pose.json ...')
cs = {r['token']: r for r in json.load(open(osp.join(TABLE_DIR, 'calibrated_sensor.json')))}
ep = {r['token']: r for r in json.load(open(osp.join(TABLE_DIR, 'ego_pose.json')))}
print(f'[map] {len(cs):,} calibrated_sensor, {len(ep):,} ego_pose records')

print('[map] loading sample_data.json (large) ...')
sd = json.load(open(osp.join(TABLE_DIR, 'sample_data.json')))
print(f'[map] {len(sd):,} sample_data records')

out = {}
off_norms, tilts = [], []
for r in sd:
    fn = r.get('filename', '')
    if not fn.startswith('samples/LIDAR_TOP/'):
        continue
    c = cs[r['calibrated_sensor_token']]
    p = ep[r['ego_pose_token']]
    R_es = quat_to_rot(*c['rotation'])          # sensor -> ego
    t_es = np.asarray(c['translation'], dtype=np.float64)
    R_ge = quat_to_rot(*p['rotation'])          # ego -> global (rotation only)
    M = R_ge @ R_es
    b = R_ge @ t_es
    out[osp.basename(fn)] = [round(float(v), 8) for v in
                             [M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2], b[0], b[1]]]
    off_norms.append(float(np.hypot(b[0], b[1])))
    # vehicle tilt = angle of ego z-axis vs global z
    tilts.append(float(np.degrees(np.arccos(np.clip(R_ge[2, 2], -1, 1)))))

off_norms = np.asarray(off_norms)
tilts = np.asarray(tilts)
print(f'[map] {len(out):,} LIDAR_TOP keyframes')
print(f'[map] |b| (ego-origin offset in global xy): median {np.median(off_norms):.4f} m, '
      f'max {off_norms.max():.4f} m')
print(f'[map] vehicle tilt: median {np.median(tilts):.3f} deg, p99 {np.percentile(tilts, 99):.3f}, '
      f'max {tilts.max():.3f} deg')
json.dump(out, open(OUT, 'w'))
print(f'[map] wrote {OUT}')
