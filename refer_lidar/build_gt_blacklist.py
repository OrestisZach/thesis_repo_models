#!/usr/bin/env python3
"""Build the GT-token blacklist = GT boxes the official nuScenes eval never scores.

Two devkit rules (nuscenes/eval/common/loaders.py::filter_eval_boxes), applied
to annotation tokens so ONE list can be used identically at training-target
build time (--gt-blacklist in nuscenes_lidar_simple) and at eval GT filtering:

  1. GHOSTS: sample_annotations with num_lidar_pts + num_radar_pts == 0
     ("we cannot guarantee they are visible"). Matches the devkit criterion
     exactly (keyframe counts; predictions are never point-filtered).
     Note: mmdet3d's NuScenesDataset default is the stricter num_lidar_pts > 0;
     we use the devkit/eval criterion so the training-target universe equals
     the scored GT universe (radar-only GTs stay).
  2. BIKE-RACK: bicycle/motorcycle annotations whose CENTER lies inside any
     static_object.bicycle_rack box of the same sample (devkit points_in_box
     on the translation point, global frame).

Usage (host or container, raw tables only, no devkit dependency):
  python build_gt_blacklist.py \
      --table-dir /data/nuscenes/v1.0-trainval \
      --out /data/nuscenes/gt_blacklist_trainval.json
"""
import argparse
import json
import os.path as osp
import sys
from collections import defaultdict

import numpy as np

BIKE_CATS = {'vehicle.bicycle', 'vehicle.motorcycle'}
RACK_CAT = 'static_object.bicycle_rack'
# For reporting only (blacklist itself is token-based).
DET_CLASS_OF = {
    'movable_object.barrier': 'barrier', 'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car', 'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle', 'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone', 'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
}


def _quat_to_rot(q):
    """nuScenes stores quaternions as [w, x, y, z]. Returns 3x3 R."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def _center_in_box(point, box_translation, box_size_wlh, box_rotation):
    """Devkit points_in_box semantics for a single point.

    nuScenes Box local frame: x-extent = l/2, y-extent = w/2, z-extent = h/2,
    with size stored as [w, l, h].
    """
    w, l, h = box_size_wlh
    R = _quat_to_rot(box_rotation)
    local = R.T @ (np.asarray(point, dtype=np.float64) - np.asarray(box_translation, dtype=np.float64))
    return (abs(local[0]) <= l / 2.0 and abs(local[1]) <= w / 2.0
            and abs(local[2]) <= h / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table-dir', default='/data/nuscenes/v1.0-trainval')
    ap.add_argument('--out', default='gt_blacklist_trainval.json')
    args = ap.parse_args()

    def load(name):
        fp = osp.join(args.table_dir, name)
        if not osp.isfile(fp):
            sys.exit(f'[fatal] missing table: {fp}')
        print(f'[load] {name} ...', flush=True)
        with open(fp, 'r') as f:
            return json.load(f)

    category = {c['token']: c['name'] for c in load('category.json')}
    instance_cat = {i['token']: category.get(i['category_token'], '')
                    for i in load('instance.json')}
    anns = load('sample_annotation.json')
    print(f'[load] {len(anns):,} sample_annotations', flush=True)

    # Pass 1: ghosts + collect racks / bikes per sample.
    ghost_tokens = []
    racks_by_sample = defaultdict(list)
    bikes_by_sample = defaultdict(list)
    ghost_per_class = defaultdict(int)

    for a in anns:
        cat = instance_cat.get(a['instance_token'], '')
        if int(a.get('num_lidar_pts', 0)) + int(a.get('num_radar_pts', 0)) == 0:
            ghost_tokens.append(a['token'])
            ghost_per_class[DET_CLASS_OF.get(cat, 'void/ignore')] += 1
        if cat == RACK_CAT:
            racks_by_sample[a['sample_token']].append(
                (a['translation'], a['size'], a['rotation']))
        elif cat in BIKE_CATS:
            bikes_by_sample[a['sample_token']].append(
                (a['token'], a['translation']))

    # Pass 2: bike/moto centers inside a rack box (same sample, global frame).
    bikerack_tokens = []
    for sample_token, racks in racks_by_sample.items():
        for tok, center in bikes_by_sample.get(sample_token, []):
            for (t, s, r) in racks:
                if _center_in_box(center, t, s, r):
                    bikerack_tokens.append(tok)
                    break

    ghost_set = set(ghost_tokens)
    rack_only = [t for t in bikerack_tokens if t not in ghost_set]
    all_tokens = sorted(ghost_set.union(bikerack_tokens))

    out = {
        'criterion': 'devkit filter_eval_boxes: (num_lidar_pts+num_radar_pts==0) OR '
                     '(bicycle/motorcycle center inside static_object.bicycle_rack box)',
        'table_dir': args.table_dir,
        'num_ghost': len(ghost_set),
        'num_bikerack': len(set(bikerack_tokens)),
        'num_bikerack_not_ghost': len(rack_only),
        'num_total': len(all_tokens),
        'ghost_per_detection_class': dict(sorted(ghost_per_class.items())),
        'all': all_tokens,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f)

    print('\n================ GT BLACKLIST ================')
    print(f"ghosts (lidar+radar==0)        : {out['num_ghost']:,}")
    print(f"bike/moto in racks             : {out['num_bikerack']:,} "
          f"({out['num_bikerack_not_ghost']:,} not already ghosts)")
    print(f"TOTAL blacklisted ann tokens   : {out['num_total']:,}")
    print('ghosts per detection class:')
    for k, v in out['ghost_per_detection_class'].items():
        print(f'  {k:<22} {v:,}')
    print(f'[write] {args.out}')
    print('==============================================')


if __name__ == '__main__':
    main()
