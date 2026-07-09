# ---------------------------------------------------------------------------
# nuscenes_lidar_simple.py — Single-Frame LiDAR Referring Detection Dataset
# ---------------------------------------------------------------------------
# Simplified variant of nuscenes_lidar.py for the refer_model pipeline:
#   - train split uses type-stratified query sampling (one query per item)
#   - val/eval splits iterate every query exactly once (one query per item)
#   - point clouds are still loaded per frame and reused for the selected query
#   - supports batch_size > 1 via simple_collate_fn
#
# Layout and JSON schema is identical to nuscenes_lidar.py.
# ---------------------------------------------------------------------------

import math
import os
import os.path as osp
import json
import random
import hashlib
from collections import defaultdict
from glob import glob

import numpy as np
import torch
import torch.utils.data
from mmengine.dataset import Compose
import mmengine

from models.structures import Instances
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules


register_all_modules(init_default_scope=True)


# Local helpers keep this dataset independent from temporal variants.
SHORT_CLASS_TO_ID = {
    'car': 0,
    'truck': 1,
    'construction_vehicle': 2,
    'bus': 3,
    'trailer': 4,
    'barrier': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'pedestrian': 8,
    'traffic_cone': 9,
}

DETECTION_RANGE = {
    0: 50.0,
    1: 50.0,
    2: 50.0,
    3: 50.0,
    4: 50.0,
    5: 30.0,
    6: 40.0,
    7: 40.0,
    8: 40.0,
    9: 30.0,
}

DEFAULT_NUSCENES_ROOT = '/data/nuscenes'
DATA_ROOT_TRAIN = osp.join(DEFAULT_NUSCENES_ROOT, 'refer_detection_with_negatives', 'train')
DATA_ROOT_VAL_SAMPLED = osp.join(DEFAULT_NUSCENES_ROOT, 'refer_detection_with_negatives', 'val')
DATA_ROOT_VAL_EXHAUSTIVE = osp.join(DEFAULT_NUSCENES_ROOT, 'complete_validation')


def _scene_from_lidar_path(lidar_path: str) -> str:
    fname = os.path.basename(lidar_path)
    return fname.split('__')[0]


def metres_to_norm(x, y, pc_range):
    cx = (x - pc_range[0]) / (pc_range[3] - pc_range[0])
    cy = (y - pc_range[1]) / (pc_range[4] - pc_range[1])
    return cx, cy


def size_to_norm(w, l, pc_range):
    rx = pc_range[3] - pc_range[0]
    ry = pc_range[4] - pc_range[1]
    return w / rx, l / ry


def _has_json_shards(root: str) -> bool:
    return osp.isdir(root) and len(glob(osp.join(root, 'refer_detection_*.json'))) > 0


def _is_negative_query(query: dict) -> bool:
    targets = query.get('targets') or []
    return len(targets) == 0 or bool(query.get('is_synthesized_negative', False))


def _normalize_legacy_entries(entries):
    for entry in entries:
        if 'captions' in entry and 'refer_queries' not in entry:
            entry['refer_queries'] = []
            for cap in entry['captions']:
                mapped_query = cap.copy()
                mapped_query['query'] = cap.get('question', '')
                mapped_query['query_type'] = cap.get('question_type', '')
                mapped_query['targets'] = cap.get('bounding_boxes', [])
                entry['refer_queries'].append(mapped_query)


def _is_explicit_split_root(root: str, split: str) -> bool:
    norm = osp.normpath(root).replace('\\', '/').rstrip('/')
    if split == 'train':
        return norm.endswith('/train')
    if split == 'val':
        return norm.endswith('/val')
    if split == 'eval':
        return norm.endswith('/complete_validation')
    return False


def _candidate_split_roots(refer_data_dir: str, split: str):
    base = osp.normpath(refer_data_dir)
    candidates = []

    if split == 'train':
        candidates.extend([
            osp.join(base, 'train'),
            osp.join(base, 'refer_with_negatives', 'train'),
            osp.join(base, 'refer_detection_with_negatives', 'train'),
            osp.join(base, 'refer_detection_with_negatives', 'refer_with_negatives', 'train'),
            osp.join(base, 'refer_detection_with_negatives'),
            DATA_ROOT_TRAIN,
        ])
    elif split == 'val':
        candidates.extend([
            osp.join(base, 'val'),
            osp.join(base, 'refer_with_negatives', 'val'),
            osp.join(base, 'refer_detection_with_negatives', 'val'),
            osp.join(base, 'refer_detection_with_negatives', 'refer_with_negatives', 'val'),
            osp.join(base, 'refer_detection_with_negatives'),
            DATA_ROOT_VAL_SAMPLED,
        ])
    elif split == 'eval':
        candidates.extend([
            osp.join(base, 'complete_validation'),
            osp.join(base, 'final_mAP_val'),
            osp.join(base, 'complete_validation', 'final_mAP_val'),
            DATA_ROOT_VAL_EXHAUSTIVE,
        ])

    candidates.append(base)

    out = []
    seen = set()
    for cand in candidates:
        norm = osp.normpath(cand)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _resolve_split_root(refer_data_dir: str, split: str) -> str:
    for cand in _candidate_split_roots(refer_data_dir, split):
        if _has_json_shards(cand):
            return cand
    raise FileNotFoundError(
        f'No refer_detection_*.json shards found for split={split!r}. '
        f'Tried: {_candidate_split_roots(refer_data_dir, split)}'
    )


def _load_shards(refer_data_dir: str, split: str):
    split_root = _resolve_split_root(refer_data_dir, split)
    pattern = osp.join(split_root, 'refer_detection_*.json')
    shard_paths = sorted(glob(pattern))

    if len(shard_paths) == 0:
        raise FileNotFoundError(f'No json files found in {split_root}')

    all_entries = []
    has_entry_split = False
    for path in shard_paths:
        print(f"  Scanning shard: {osp.basename(path)}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data and isinstance(data[0], dict) and 'split' in data[0]:
            has_entry_split = True
        all_entries.extend(data)

    _normalize_legacy_entries(all_entries)

    if split == 'eval' or _is_explicit_split_root(split_root, split):
        print(f"  Using explicit split root: {split_root}")
        return all_entries

    if has_entry_split:
        entries = [e for e in all_entries if e.get('split') == split]
        print(f"  Using per-entry split field: split='{split}', kept {len(entries)} entries")
        return entries

    split_map = {}
    for i, path in enumerate(shard_paths):
        split_map[path] = 'train' if i < 8 else ('val' if i == 8 else 'test')

    entries = []
    for path in shard_paths:
        if split_map[path] != split:
            continue
        print(f"  Loading shard: {osp.basename(path)}  ({split})")
        with open(path, 'r', encoding='utf-8') as f:
            entries.extend(json.load(f))
    return entries


# ── Collate function ─────────────────────────────────────────────────────

def simple_collate_fn(batch):
    """Collate single-frame dict samples into a batched dict.

    Always returns lists (even batch_size=1) so the model can rely on
    a consistent interface.
    """
    result = {}
    for key in batch[0].keys():
        result[key] = [b[key] for b in batch]
    return result


# ── Dataset ──────────────────────────────────────────────────────────────

class LiDARReferDetectionSimple(torch.utils.data.Dataset):
    """Single-frame LiDAR referring detection dataset.

    Train and validation iterate frames, selecting K queries within-frame.
    Final eval iterates every query exactly once.

    Designed for the ``refer_model`` architecture (no temporal tracking).
    """

    def __init__(self, args, split: str = 'train', pillar_encoder=None):
        self.args = args
        self.split = split
        self.K = max(1, int(getattr(args, 'queries_per_frame', 3)))
        self.seed = int(getattr(args, 'seed', 42))
        self._current_epoch = 0
        self._logged_train_examples = False
        self._flat_queries = []
        self._frame_query_indices_by_type = []
        self._by_type = defaultdict(list)
        self._types = []
        self._w = []
        self._type_to_weight = {}
        self._epoch_size = 0
        self._train_item_count = 0
        self._val_epoch_size = 0
        self._example_by_type = {}
        self._stats_by_type = defaultdict(lambda: {'pos': 0, 'neg': 0, 'total': 0})

        self.point_cloud_range = getattr(
            args, 'point_cloud_range',
            [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        self.voxel_size = getattr(args, 'voxel_size', [0.08, 0.08, 4.0])

        # Optional root remapping
        self.lidar_root = getattr(args, 'lidar_root', None)

        # Build LiDAR filename → full-path index
        self.nuscenes_dataroot = getattr(args, 'nuscenes_dataroot', None)
        self.sweeps_num = int(getattr(args, 'sweeps_num', 10))
        self.num_point_features = int(getattr(args, 'num_point_features', 4))
        self.feature_cache_dir = getattr(args, 'feature_cache_dir', None)
        self.feature_cache_strict = bool(getattr(args, 'feature_cache_strict', False))
        self._feature_cache_index = {}
        self._feature_cache_detector = 'centerpoint'
        self._lidar_index = {}
        self._nus_dataset = None
        self._sample_token_to_idx = {}
        self._legacy_info_by_token = {}
        self._sweep_pipeline = None
        if self.nuscenes_dataroot:
            self._build_nuscenes_loader(split)
            self._build_lidar_index()
        self._load_feature_cache_index(split)

        # ── GT blacklist (devkit filter_eval_boxes parity) ──
        # Annotation tokens the official nuScenes eval never scores: GTs with
        # zero lidar+radar keyframe points ("ghosts") and bicycles/motorcycles
        # inside bike racks. Built by build_gt_blacklist.py. Applied at the
        # frame-object build so training supervision, val loss, and eval GT
        # all share the identical GT universe.
        self.gt_blacklist = set()
        _bl_path = getattr(args, 'gt_blacklist', None)
        if _bl_path:
            with open(_bl_path, 'r', encoding='utf-8') as f:
                _bl = json.load(f)
            self.gt_blacklist = set(_bl['all'] if isinstance(_bl, dict) else _bl)
            print(f'[Dataset] GT blacklist: {len(self.gt_blacklist):,} tokens '
                  f'({_bl_path})', flush=True)
        else:
            print('[Dataset] GT blacklist: not set. Fine for datasets whose '
                  'generation already applies the devkit eligibility at source '
                  '(ghosts/bike-racks never emitted); REQUIRED for legacy sets '
                  'generated before that, or their GT keeps devkit-unscored ghosts.',
                  flush=True)

        # ── Ego-range transform (devkit ego_dist parity, EXACT) ──
        # filter_eval_boxes keeps boxes with ego_dist < class_range, where
        # ego_dist is the GLOBAL-frame xy-norm of (box - ego_pose) at the
        # LIDAR_TOP timestamp (devkit loaders.py:189-197, data_classes.py:54-56).
        # Our boxes are sensor-frame; with M = R_ge @ R_es, b = R_ge @ t_es:
        #   ego_dist(x_s) = hypot(M0.x_s + b0, M1.x_s + b1)
        # exact incl. vehicle roll/pitch (tilt p99 ~4.9 deg would otherwise
        # leave up to ~0.5 m residual at 50 m). Map value per keyframe:
        # [m00,m01,m02, m10,m11,m12, b0,b1]. Built by build_ego_range_transform.
        self.ego_range_transform = {}
        _eo_path = getattr(args, 'ego_range_transform', '') or ''
        if not _eo_path and self.nuscenes_dataroot:
            _cand = osp.join(self.nuscenes_dataroot, 'ego_range_transform.json')
            _eo_path = _cand if osp.isfile(_cand) else ''
        if _eo_path:
            with open(_eo_path, 'r', encoding='utf-8') as f:
                self.ego_range_transform = {
                    k: tuple(float(x) for x in v) for k, v in json.load(f).items()}
            print(f'[Dataset] Ego-range transform: {len(self.ego_range_transform):,} '
                  f'frames ({_eo_path}) — devkit-exact ego_dist for class ranges.',
                  flush=True)
        else:
            print('[Dataset] Ego-range transform: NOT FOUND — class range measured '
                  'from the LIDAR origin (~0.9-1.2 m off ego; deviates from devkit).',
                  flush=True)

        # Pre-compute BEV (when pillar encoder is frozen)
        self.precompute_bev = getattr(args, 'precompute_bev', False)
        if self.precompute_bev and pillar_encoder is not None:
            self.pillar_encoder = pillar_encoder.eval()
            for p in self.pillar_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.pillar_encoder = None

        # ── Question-type filter ──
        qt = None
        qt_json = getattr(args, 'question_types_json', None)
        if qt_json is not None:
            with open(qt_json, 'r', encoding='utf-8') as f:
                qt = json.load(f)
            assert isinstance(qt, list)
        else:
            qt = getattr(args, 'question_types', None)
        self.question_types = set(qt) if qt else None

        # ── Load shards ──
        refer_data_dir = getattr(args, 'refer_data_dir', None)
        assert refer_data_dir is not None, \
            "Must specify --refer_data_dir"

        print(f"[SimpleDS] Loading shards from {refer_data_dir} ...")
        entries = _load_shards(refer_data_dir, split)
        print(f"  {len(entries)} raw entries for split '{split}'")

        # ── Build per-frame object catalog ──
        # De-duplicate objects across all queries in a frame.
        # Build scene-level unique object IDs for tracking continuity
        # (not strictly needed for single-frame, but kept for
        #  compatibility and potential eval).
        scene_entries = defaultdict(list)
        for e in entries:
            scene = _scene_from_lidar_path(e['lidar_path'])
            scene_entries[scene].append(e)

        self.scene_to_idx = {}
        self.obj_token_to_id = {}
        for sidx, scene in enumerate(sorted(scene_entries.keys())):
            self.scene_to_idx[scene] = sidx
            scene_tokens = set()
            for e in scene_entries[scene]:
                for rq in e.get('refer_queries', []):
                    for t in rq.get('targets', []):
                        scene_tokens.add(t['token'])
            for local_id, tok in enumerate(sorted(scene_tokens)):
                self.obj_token_to_id[(scene, tok)] = sidx * 1_000_000 + local_id

        # ── Flatten into per-frame structures ──
        self.frames = []
        self.frame_scene = []
        self.frame_objects = []
        self.frame_refer_queries = []
        self.frame_ego_transform = []   # per-frame devkit ego_dist transform (or None)

        # Protocol GT filter bookkeeping (blacklist + ego class-range + BEV square)
        self._dropped_queries_by_type = defaultdict(int)
        self._dropped_targets_by_type = defaultdict(int)
        _ego_tf_misses = 0

        for entry in entries:
            scene = _scene_from_lidar_path(entry['lidar_path'])
            # lidar_path may be a Windows path from the generation repo —
            # normalise separators before taking the basename.
            ego_tf = self.ego_range_transform.get(
                osp.basename((entry.get('lidar_path', '') or '').replace('\\', '/')))
            if ego_tf is None and self.ego_range_transform:
                _ego_tf_misses += 1

            # De-duplicate objects
            seen_tokens = set()
            objs = []
            for rq in entry.get('refer_queries', []):
                for t in rq.get('targets', []):
                    # Safely get token with a fallback
                    tok = t.get('token', 'unknown_token')
                    if tok in seen_tokens:
                        continue
                    if tok in self.gt_blacklist:
                        # devkit-unscored GT (ghost / bike-rack): excluded from
                        # supervision and from eval GT alike.
                        continue
                    seen_tokens.add(tok)

                    # Try common keys for class names in newer formats
                    cname = t.get('class') or t.get('category') or t.get('label') or t.get('class_name', 'unknown')
                    cls_id = SHORT_CLASS_TO_ID.get(cname, -1)

                    # Safely get all other properties
                    objs.append({
                        'token': tok,
                        'class': cname,
                        'class_id': cls_id,
                        'center_sensor': t.get('center_sensor', [0.0, 0.0, 0.0]),
                        'extents_lwh': t.get('extents_lwh', [0.1, 0.1, 0.1]),
                        'yaw_deg': t.get('yaw_deg', 0.0),
                        'obj_id': self.obj_token_to_id.get((scene, tok), -1),
                    })
            # Filter refer queries by question_type, then apply the protocol GT
            # filter to each query's target list (single choke point: the SAME
            # filtered targets feed training is_ref, val loss and eval GT).
            queries = []
            for rq in entry.get('refer_queries', []):
                if self.question_types and rq.get('query_type') not in self.question_types:
                    continue
                self._validate_query(rq, entry)
                orig_targets = rq.get('targets') or []
                if orig_targets:
                    kept = [t for t in orig_targets
                            if t.get('token', 'unknown_token') not in self.gt_blacklist
                            and self._gt_scoreable(t, ego_tf)]
                    if not kept:
                        # Generation-time positive whose whole answer set is
                        # devkit-unscorable (ghost / out of class range): the
                        # premise is broken (e.g. "closest car" -> a ghost), so
                        # the query is excluded outright — never re-designated
                        # to another object, never scored as a fake negative.
                        self._dropped_queries_by_type[
                            rq.get('query_type', 'unknown')] += 1
                        continue
                    if len(kept) != len(orig_targets):
                        self._dropped_targets_by_type[
                            rq.get('query_type', 'unknown')] += len(orig_targets) - len(kept)
                        rq = dict(rq)          # shallow copy; entry stays pristine
                        rq['targets'] = kept
                queries.append(rq)

            self.frames.append(entry)
            self.frame_scene.append(scene)
            self.frame_objects.append(objs)
            self.frame_refer_queries.append(queries)
            self.frame_ego_transform.append(ego_tf)

        if self._dropped_queries_by_type or self._dropped_targets_by_type:
            _dq = dict(sorted(self._dropped_queries_by_type.items()))
            _dt = dict(sorted(self._dropped_targets_by_type.items()))
            print(f'[Dataset] Protocol GT filter: dropped queries (all targets '
                  f'unscorable) by type: {_dq}; dropped targets by type: {_dt}',
                  flush=True)
        if _ego_tf_misses:
            print(f'[Dataset] WARNING: {_ego_tf_misses} frames missing from the '
                  f'ego-range transform map (lidar-origin fallback).', flush=True)

        # ── Valid indices: frames with ≥1 filtered refer_query ──
        self.valid_indices = [
            i for i in range(len(self.frames))
            if len(self.frame_refer_queries[i]) > 0
        ]

        # ── CBGS: scene-level class-balanced resampling (train only) ──
        # Composes with the per-scene type-stratified query sampling: CBGS picks
        # WHICH frame (oversampling rare-class scenes), the stratified sampler
        # picks WHICH K queries inside it. Implemented as a constant-epoch-size
        # weighted draw (epoch length unchanged => compute-matched), so it is a
        # clean one-variable change.
        self.cbgs_enabled = bool(getattr(args, 'cbgs', False)) and self.split == 'train'
        self._cbgs_cum = None
        if self.cbgs_enabled:
            self._build_cbgs_weights()

        n_all = sum(len(e.get('refer_queries', [])) for e in entries)
        n_filt = sum(len(rqs) for rqs in self.frame_refer_queries)
        self._build_query_index()

        print(f"  {n_filt}/{n_all} queries after filtering")
        print(f"  {len(self.valid_indices)}/{len(self.frames)} frames have ≥1 query")
        if self.split in {'train', 'val'}:
            frame_item_count = len(self)
            query_count = self._count_frame_queries_for_items(frame_item_count)
            print(
                f"  __len__ = {frame_item_count} frame-items for split '{self.split}' "
                f"(queries_per_frame={self.K}, queries_per_pass={query_count})"
            )
        else:
            print(f"  __len__ = {len(self)} queries for split '{self.split}'")
        self._print_query_distribution()

    @staticmethod
    def _validate_query(query: dict, entry: dict) -> None:
        sample_token = entry.get('sample_token', '<unknown>')
        qa_token = query.get('qa_token', '<missing>')
        assert query.get('query'), (
            f'Malformed refer query: missing query text for sample_token={sample_token} qa_token={qa_token}'
        )
        assert query.get('query_type'), (
            f'Malformed refer query: missing query_type for sample_token={sample_token} qa_token={qa_token}'
        )

    def _build_query_index(self) -> None:
        for frame_idx, queries in enumerate(self.frame_refer_queries):
            frame_by_type = defaultdict(list)
            for query_idx, query in enumerate(queries):
                query_type = str(query['query_type'])
                pair = (frame_idx, query_idx)
                self._flat_queries.append(pair)
                self._by_type[query_type].append(pair)
                frame_by_type[query_type].append(query_idx)

                stats = self._stats_by_type[query_type]
                stats['total'] += 1
                if _is_negative_query(query):
                    stats['neg'] += 1
                else:
                    stats['pos'] += 1

                if query_type not in self._example_by_type:
                    self._example_by_type[query_type] = {
                        'query': query.get('query', ''),
                        'target_count': len(query.get('targets') or []),
                    }
            self._frame_query_indices_by_type.append(dict(frame_by_type))

        if not self._flat_queries:
            raise ValueError(f'No refer queries available for split={self.split!r}')

        self._types = sorted(self._by_type.keys())
        counts = [len(self._by_type[t]) for t in self._types]
        if self.split == 'train':
            type_weighting = getattr(self.args, 'type_weighting', 'uniform')
            if type_weighting == 'uniform':
                self._w = [1.0 / len(self._types)] * len(self._types)
            elif type_weighting == 'inverse_sqrt':
                raw = [1.0 / (count ** 0.5) for count in counts]
                denom = sum(raw)
                self._w = [value / denom for value in raw]
            else:
                raise ValueError(f'Unsupported type_weighting={type_weighting!r}')
            self._type_to_weight = dict(zip(self._types, self._w))
            requested_epoch_size = int(getattr(self.args, 'epoch_size', 0) or 0)
            self._epoch_size = requested_epoch_size
            self._train_item_count = (
                max(1, math.ceil(requested_epoch_size / max(self.K, 1)))
                if requested_epoch_size > 0 else len(self.valid_indices)
            )
        else:
            self._w = []
            self._type_to_weight = {}
            self._epoch_size = 0
            self._train_item_count = 0

        if self.split == 'val':
            requested_val_epoch_size = int(getattr(self.args, 'val_epoch_size', 0) or 0)
            self._val_epoch_size = requested_val_epoch_size
        else:
            self._val_epoch_size = 0

    def _count_frame_queries_for_items(self, item_count: int) -> int:
        if item_count <= 0 or not self.valid_indices:
            return 0

        frame_count = len(self.valid_indices)
        total = 0
        for item_idx in range(item_count):
            flat_idx = self.valid_indices[item_idx % frame_count]
            total += min(self.K, len(self.frame_refer_queries[flat_idx]))
        return total

    def _print_query_distribution(self) -> None:
        total_pos = 0
        total_neg = 0

        if self.split == 'train':
            n_items = len(self)
            print(f"[ReferDataset] {len(self.frames)} frames, {len(self._flat_queries)} queries")
            for query_type, weight in zip(self._types, self._w):
                count = len(self._by_type[query_type])
                print(f"  {query_type:<48s} count={count:>8d}  sampling weight={weight:.4f}")
            print(
                f"  queries_per_frame={self.K}  frame_items_per_epoch={n_items}  "
                f"queries_per_epoch~={self._count_frame_queries_for_items(n_items)}"
            )
            return

        label = 'Exhaustive eval' if self.split == 'eval' else 'Validation'
        print(f"[ReferDataset] {label} distribution")
        for query_type in self._types:
            stats = self._stats_by_type[query_type]
            total_pos += stats['pos']
            total_neg += stats['neg']
            print(
                f"  {query_type:<48s} pos={stats['pos']:>6d}  neg={stats['neg']:>6d}  total={stats['total']:>6d}"
            )
        denom = max(total_neg, 1)
        print(f"  global pos={total_pos} neg={total_neg} pos:neg={total_pos / denom:.4f}")
        if self.split == 'val':
            n_items = len(self)
            print(
                f"  val: frame_items_per_pass={n_items}  queries_per_frame={self.K}  "
                f"queries_per_pass={self._count_frame_queries_for_items(n_items)}"
            )

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)

    def _frame_class_ids(self, flat_idx: int) -> set:
        """Class ids present in a frame, under the same filter as _build_targets
        (valid class, devkit ego_dist class range, inside the normalized BEV).
        """
        out = set()
        ego_tf = (self.frame_ego_transform[flat_idx]
                  if flat_idx < len(self.frame_ego_transform) else None)
        for obj in self.frame_objects[flat_idx]:
            if obj['class_id'] < 0:
                continue
            if not self._gt_scoreable(obj, ego_tf):
                continue
            out.add(int(obj['class_id']))
        return out

    def _build_cbgs_weights(self) -> None:
        """Per-valid-frame CBGS sampling weights (class-balanced grouping/sampling).

        Each frame's weight is the sum over the classes it contains of the inverse
        of that class's frame-frequency -- the expected-replication distribution of
        standard CBGS. Rare-class frames get high weight. Stored as a normalized
        cumulative array for O(log n) weighted draws at sampling time.
        """
        cls_frame_count = defaultdict(int)
        frame_classes = []
        for flat_idx in self.valid_indices:
            cset = self._frame_class_ids(flat_idx)
            frame_classes.append(cset)
            for c in cset:
                cls_frame_count[c] += 1

        weights = np.zeros(len(self.valid_indices), dtype=np.float64)
        for i, cset in enumerate(frame_classes):
            if cset:
                weights[i] = sum(1.0 / cls_frame_count[c] for c in cset)
        total = weights.sum()
        if total <= 0:
            self.cbgs_enabled = False
            self._cbgs_cum = None
            return
        cum = np.cumsum(weights)
        cum /= cum[-1]
        self._cbgs_cum = cum

        # Log the rebalancing effect on rank 0.
        is_rank0 = not (torch.distributed.is_available() and torch.distributed.is_initialized()) \
            or torch.distributed.get_rank() == 0
        if is_rank0:
            probs = weights / total
            uniform_share = {c: cls_frame_count[c] / len(self.valid_indices) for c in cls_frame_count}
            cbgs_share = defaultdict(float)
            for i, cset in enumerate(frame_classes):
                for c in cset:
                    cbgs_share[c] += probs[i]
            print(f"[CBGS] enabled: {len(cls_frame_count)} classes over {len(self.valid_indices)} frames "
                  f"(epoch size unchanged)")
            for c in sorted(cls_frame_count, key=lambda k: cls_frame_count[k]):
                print(f"[CBGS]   class_id={c:2d}  frames={cls_frame_count[c]:6d}  "
                      f"uniform_share={uniform_share[c]:.3f} -> cbgs_share={cbgs_share[c]:.3f}")

    def _sample_train_queries(self, idx: int):
        """Frame-iterated training with within-frame inverse_sqrt query sampling."""
        frame_count = len(self.valid_indices)
        rng = random.Random(self.seed + self._current_epoch * max(frame_count, 1) + int(idx))

        if self.cbgs_enabled and self._cbgs_cum is not None:
            # Scene-level class-balanced draw (seeded per epoch+item), then the
            # within-frame stratified query selection proceeds on this frame.
            pos = int(np.searchsorted(self._cbgs_cum, rng.random(), side='left'))
            flat_idx = self.valid_indices[min(pos, frame_count - 1)]
        else:
            flat_idx = self.valid_indices[idx % frame_count]
        frame_queries = self.frame_refer_queries[flat_idx]
        if len(frame_queries) <= self.K:
            return flat_idx, list(range(len(frame_queries)))

        frame_by_type = self._frame_query_indices_by_type[flat_idx]
        selected = []
        selected_set = set()

        while len(selected) < self.K:
            candidate_types = []
            candidate_weights = []
            for query_type, query_indices in frame_by_type.items():
                available = [query_index for query_index in query_indices if query_index not in selected_set]
                if not available:
                    continue
                candidate_types.append(query_type)
                candidate_weights.append(self._type_to_weight.get(query_type, 0.0))

            if not candidate_types:
                break

            chosen_type = rng.choices(candidate_types, weights=candidate_weights, k=1)[0]
            chosen_candidates = [
                query_index
                for query_index in frame_by_type[chosen_type]
                if query_index not in selected_set
            ]
            chosen_query_idx = rng.choice(chosen_candidates)
            selected.append(chosen_query_idx)
            selected_set.add(chosen_query_idx)

        return flat_idx, selected

    def _select_val_queries(self, idx: int):
        """Frame-iterated val with epoch-dependent uniform within-frame sampling."""
        frame_count = len(self.valid_indices)
        flat_idx = self.valid_indices[idx % frame_count]
        frame_queries = self.frame_refer_queries[flat_idx]
        if len(frame_queries) <= self.K:
            return flat_idx, list(range(len(frame_queries)))

        rng = random.Random(self.seed + self._current_epoch * max(frame_count, 1) + int(idx))
        chosen = rng.sample(range(len(frame_queries)), self.K)
        return flat_idx, chosen

    def _maybe_log_train_examples(self) -> None:
        if self.split != 'train' or self._logged_train_examples:
            return
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.id != 0:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        print('[ReferDataset] First-train-batch examples by type:')
        for query_type in self._types:
            example = self._example_by_type[query_type]
            print(
                f"  {query_type:<48s} target_count={example['target_count']:>3d}  query={example['query']}"
            )
        self._logged_train_examples = True

    @staticmethod
    def _cache_key_for_entry(entry: dict) -> str:
        sample_token = entry.get('sample_token', '')
        lidar_path = str(entry.get('lidar_path', ''))
        raw = f'{sample_token}|{lidar_path}'
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()

    @staticmethod
    def _cache_alts_for_entry(entry: dict) -> list:
        sample_token = entry.get('sample_token', '')
        lidar_path = str(entry.get('lidar_path', ''))
        lidar_name = osp.basename(lidar_path) if lidar_path else ''
        return [sample_token, lidar_path, lidar_name]

    def _load_feature_cache_index(self, split: str) -> None:
        self._feature_cache_index = {}
        if not self.feature_cache_dir:
            return

        cache_dir = self.feature_cache_dir
        split_names = [split]
        if split == 'eval':
            split_names.append('val')

        candidates = []
        for split_name in split_names:
            candidates.extend([
                osp.join(cache_dir, f'index_{split_name}.json'),
                osp.join(cache_dir, split_name, f'index_{split_name}.json'),
                osp.join(cache_dir, split_name, 'index.json'),
            ])
        candidates.extend([
            osp.join(cache_dir, 'index_all.json'),
            osp.join(cache_dir, 'index.json'),
        ])
        index_path = None
        for p in candidates:
            if osp.isfile(p):
                index_path = p
                break
        if index_path is None:
            msg = f"[SimpleDS] feature cache index not found in {cache_dir}"
            if self.feature_cache_strict:
                raise FileNotFoundError(msg)
            print(msg)
            return

        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Store the detector type declared at the top level of the index.
        if isinstance(data, dict):
            self._feature_cache_detector = data.get('detector', 'centerpoint')
        else:
            self._feature_cache_detector = 'centerpoint'

        rows = data.get('samples') if isinstance(data, dict) else data
        if not isinstance(rows, list):
            msg = f"[SimpleDS] Invalid feature cache index format: {index_path}"
            if self.feature_cache_strict:
                raise RuntimeError(msg)
            print(msg)
            return

        for row in rows:
            if not isinstance(row, dict):
                continue
            row_split = row.get('split')
            if row_split is not None and str(row_split) not in {str(name) for name in split_names}:
                continue
            rel = row.get('feature_file')
            if not rel:
                continue
            abs_path = rel if osp.isabs(rel) else osp.join(cache_dir, rel)
            if not osp.isfile(abs_path):
                continue
            row = dict(row)
            row['feature_file'] = abs_path

            keys = set()
            for k in ('cache_key', 'sample_token', 'lidar_path', 'lidar_basename'):
                v = row.get(k)
                if v:
                    keys.add(str(v))
            for k in keys:
                self._feature_cache_index[k] = row

        print(
            f"[SimpleDS] Feature cache enabled: {len(self._feature_cache_index)} keys from {index_path}",
            flush=True,
        )

    def _resolve_feature_cache_record(self, entry: dict):
        keys = [self._cache_key_for_entry(entry)] + self._cache_alts_for_entry(entry)
        for k in keys:
            rec = self._feature_cache_index.get(k)
            if rec is not None:
                return rec
        return None

    def _load_cached_features(self, entry: dict):
        rec = self._resolve_feature_cache_record(entry)
        if rec is None:
            if self.feature_cache_strict:
                raise KeyError(
                    f"Feature cache miss for sample_token={entry.get('sample_token', '<none>')} "
                    f"lidar_path={entry.get('lidar_path', '<none>')}"
                )
            return None

        payload = torch.load(rec['feature_file'], map_location='cpu')
        srcs = payload.get('feature_maps')
        props = payload.get('proposal_boxes_8d')
        scores = payload.get('proposal_scores')
        if srcs is None or props is None or scores is None:
            raise RuntimeError(f"Invalid cached feature payload: {rec['feature_file']}")
        if not isinstance(srcs, list):
            raise RuntimeError(f"feature_maps must be a list in cached payload: {rec['feature_file']}")

        # Per-proposal heading (radians), used by the +angle model; None for a
        # cache built without it.
        yaw = payload.get('proposal_yaw')

        return {
            'srcs': [s.float().contiguous() for s in srcs],
            'props': props.float().contiguous(),
            'scores': scores.float().contiguous(),
            'yaw': None if yaw is None else yaw.float().contiguous(),
            'record': rec,
        }

    def _default_ann_file(self, split: str) -> str:
        # train/val use trainval infos; test can use test infos.
        if split == 'eval':
            fname = 'nuscenes_infos_val.pkl'
        elif split == 'test':
            fname = 'nuscenes_infos_test.pkl'
        elif split == 'val':
            fname = 'nuscenes_infos_val.pkl'
        else:
            fname = 'nuscenes_infos_train.pkl'
        return osp.join(self.nuscenes_dataroot, fname)

    def _build_nuscenes_loader(self, split: str):
        ann_file = getattr(self.args, 'nuscenes_ann_file', None)
        if not ann_file:
            ann_file = self._default_ann_file(split)
        backend_args = getattr(self.args, 'backend_args', None)

        dataset_cfg = dict(
            type='NuScenesDataset',
            data_root=self.nuscenes_dataroot,
            ann_file=ann_file,
            pipeline=[],
            modality=dict(use_lidar=True, use_camera=False),
            data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP', img=''),
            test_mode=True,
            box_type_3d='LiDAR',
        )
        try:
            self._nus_dataset = DATASETS.build(dataset_cfg)
            for i in range(len(self._nus_dataset)):
                info = self._nus_dataset.get_data_info(i)
                tok = info.get('token')
                if tok is not None:
                    self._sample_token_to_idx[tok] = i
        except Exception as exc:
            # Backward-compatible path for legacy nuscenes_infos_*.pkl
            print(f"  [SimpleDS] Official NuScenesDataset build failed: {exc}")
            print("  [SimpleDS] Falling back to manual legacy .pkl parsing...")
            
            loaded = mmengine.load(ann_file)
            self._legacy_info_by_token = {}
            
            if isinstance(loaded, dict):
                if 'infos' in loaded:
                    infos = loaded['infos']
                    self._legacy_info_by_token = {i['token']: i for i in infos if 'token' in i}
                elif 'data_list' in loaded:
                    infos = loaded['data_list']
                    self._legacy_info_by_token = {i['token']: i for i in infos if 'token' in i}
                else:
                    # THIS IS YOUR FORMAT: {token: info_dict}
                    for k, v in loaded.items():
                        if isinstance(v, dict):
                            # Ensure the token is stored inside the info dict for the pipeline
                            v['token'] = v.get('token', k)
                            self._legacy_info_by_token[k] = v
            elif isinstance(loaded, list):
                self._legacy_info_by_token = {i['token']: i for i in loaded if 'token' in i}
                
            if not self._legacy_info_by_token:
                raise RuntimeError(f"Failed to parse .pkl file {ann_file}. No valid tokens found.")

            self._nus_dataset = None
            self._sample_token_to_idx = {}
            print(
                f"  NuScenes legacy infos fallback enabled: {len(self._legacy_info_by_token)} tokens loaded successfully!",
                flush=True,
            )

        self._sweep_pipeline = Compose([
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                backend_args=backend_args,
            ),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=self.sweeps_num,
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4] if self.num_point_features >= 5 else [0, 1, 2, 4],
                pad_empty_sweeps=True,
                test_mode=True,
                backend_args=backend_args,
            ),
            dict(type='Pack3DDetInputs', keys=['points']),
        ])
        print(f"  NuScenes info loader enabled: ann_file={ann_file}, sweeps_num={self.sweeps_num}")

    @staticmethod
    def _legacy_lidar2sensor_from_sensor2lidar(sweep: dict):
        rot_s2l = np.asarray(sweep['sensor2lidar_rotation'], dtype=np.float32)
        trans_s2l = np.asarray(sweep['sensor2lidar_translation'], dtype=np.float32)

        # Legacy sweep applies: p_target = p_src @ R + T.
        # New transform applies: p_target = p_src @ lidar2sensor_rot - lidar2sensor_trans.
        # So set lidar2sensor_rot = R, lidar2sensor_trans = -T.
        rot = rot_s2l.T
        trans = -trans_s2l

        lidar2sensor = np.eye(4, dtype=np.float32)
        lidar2sensor[:3, :3] = rot
        lidar2sensor[:3, 3] = trans
        return lidar2sensor.tolist()

    def _legacy_info_to_pipeline_input(self, info: dict) -> dict:
        lidar_path = self._resolve_lidar_path(info['lidar_path'])
        sweeps = info.get('sweeps', info.get('lidar_sweeps', []))
        token = info.get('token', info.get('sample_token', '<unknown>'))
        lidar_sweeps = []
        for sw in sweeps:
            if 'data_path' in sw:
                sweep_path = self._resolve_lidar_path(sw['data_path'])
            else:
                sweep_path = self._resolve_lidar_path(sw['lidar_points']['lidar_path'])

            if 'sensor2lidar_rotation' in sw and 'sensor2lidar_translation' in sw:
                lidar2sensor = self._legacy_lidar2sensor_from_sensor2lidar(sw)
            elif 'lidar_points' in sw and 'lidar2sensor' in sw['lidar_points']:
                lidar2sensor = sw['lidar_points']['lidar2sensor']
            else:
                # Identity fallback for custom infos that do not store sweep transforms.
                lidar2sensor = np.eye(4, dtype=np.float32).tolist()

            sweep_ts = float(sw.get('timestamp', 0.0))
            if sweep_ts > 1e12:
                sweep_ts /= 1e6

            lidar_sweeps.append({
                'lidar_points': {
                    'lidar_path': sweep_path,
                    'lidar2sensor': lidar2sensor,
                },
                'timestamp': sweep_ts,
            })

        ts = float(info.get('timestamp', 0.0))
        if ts > 1e12:
            ts /= 1e6

        out = {
            'token': token,
            'timestamp': ts,
            'lidar_points': {
                'lidar_path': lidar_path,
            },
            'lidar_sweeps': lidar_sweeps,
        }
        return out

    # ── LiDAR file index ──

    def _build_lidar_index(self):
        root = self.nuscenes_dataroot
        shard_dirs = sorted(glob(osp.join(root, 'v1.0-trainval*_keyframes')))
        if not shard_dirs:
            lidar_dir = osp.join(root, 'samples', 'LIDAR_TOP')
            if osp.isdir(lidar_dir):
                shard_dirs = [root]
        for sd in shard_dirs:
            lidar_dir = osp.join(sd, 'samples', 'LIDAR_TOP')
            if not osp.isdir(lidar_dir):
                continue
            for fname in os.listdir(lidar_dir):
                if fname.endswith('.pcd.bin'):
                    self._lidar_index[fname] = osp.join(lidar_dir, fname)
        print(f"  LiDAR file index: {len(self._lidar_index)} files")

    def _resolve_lidar_path(self, lidar_path: str) -> str:
        lidar_path = lidar_path.replace('\\', '/')
        if self._lidar_index:
            fname = lidar_path.rsplit('/', 1)[-1]
            if fname in self._lidar_index:
                return self._lidar_index[fname]
        if self.lidar_root is not None:
            idx = lidar_path.find('samples/')
            if idx >= 0:
                return osp.join(self.lidar_root, lidar_path[idx:])
            return osp.join(self.lidar_root, osp.basename(lidar_path))
        return lidar_path

    def _load_point_cloud(self, path: str) -> torch.Tensor:
        """Single-keyframe fallback used when the sweep pipeline can't resolve
        a sample. Returns a tensor that matches the layout the sweep path
        normally emits, so downstream detectors see a consistent channel
        ordering.

        Raw nuScenes ``.pcd.bin`` is 5-D ``[x, y, z, intensity, ring]``. The
        sweep-accumulated path returns either:
          * 4-D ``[x, y, z, dt]`` when ``num_point_features=4`` (drops intensity,
            puts sweep time-lag at col 3).
          * 5-D ``[x, y, z, intensity, dt]`` when ``num_point_features=5``
            (col 4 is the sweep time-lag, replacing the raw ring index).

        With only the keyframe, dt is identically zero. We honour that and zero
        the dt channel rather than leaving the raw ``ring`` value at col 4 (or
        the intensity value at col 3 for 4-D), which would be out-of-distribution
        vs the stock pretraining and silently corrupt downstream features.
        """
        num_features = int(getattr(self.args, 'num_point_features', 4))
        raw = np.fromfile(path, dtype=np.float32).reshape(-1, 5).copy()
        if num_features == 4:
            # [x, y, z, dt=0]; intensity (raw col 3) and ring (raw col 4) dropped
            out = np.zeros((raw.shape[0], 4), dtype=np.float32)
            out[:, :3] = raw[:, :3]
            return torch.from_numpy(out)
        if num_features == 5:
            # [x, y, z, intensity, dt=0]; overwrite ring (raw col 4) with dt=0
            out = raw[:, :5].copy()
            out[:, 4] = 0.0
            return torch.from_numpy(out)
        raise ValueError(
            f"Unsupported num_point_features={num_features}; expected 4 or 5."
        )

    def _load_points_with_sweeps(self, entry: dict) -> torch.Tensor:
        token = entry.get('sample_token')

        # MMDetection3D version compatibility:
        # - older versions return LiDARPoints (has .tensor)
        # - newer versions may return torch.Tensor directly
        def _extract_tensor(packed_data):
            pts = packed_data['inputs']['points']
            return pts.tensor if hasattr(pts, 'tensor') else pts

        if self._nus_dataset is not None and token in self._sample_token_to_idx:
            data_idx = self._sample_token_to_idx[token]
            raw_info = self._nus_dataset.get_data_info(data_idx)
            packed = self._sweep_pipeline(raw_info)
            return _extract_tensor(packed)

        if token in self._legacy_info_by_token:
            raw_info = self._legacy_info_to_pipeline_input(self._legacy_info_by_token[token])
            packed = self._sweep_pipeline(raw_info)
            return _extract_tensor(packed)

        # Fallback to single keyframe path for rare unmatched tokens.
        lidar_path = self._resolve_lidar_path(entry['lidar_path'])
        return self._load_point_cloud(lidar_path)

    # ── Scoreable-GT predicate (single source of truth) ──

    def _gt_scoreable(self, t: dict, ego_tf) -> bool:
        """The devkit-faithful scoreable-GT predicate, shared by training
        targets (_build_targets), CBGS (_frame_class_ids) and eval GT (query
        target filtering): known detection class, devkit ego_dist STRICTLY
        inside the per-class range (filter_eval_boxes: `ego_dist < max_dist`),
        and inside the BEV square (model domain). Blacklist is checked by the
        callers (it needs the annotation token).

        ego_tf: per-frame (m00,m01,m02,m10,m11,m12,b0,b1) reproducing the
        devkit's global-frame ego_dist from sensor coords; None -> lidar-origin
        fallback (legacy behaviour, ~1 m off)."""
        cname = (t.get('class') or t.get('category') or t.get('label')
                 or t.get('class_name') or 'unknown')
        cls_id = SHORT_CLASS_TO_ID.get(cname, -1)
        if cls_id < 0:
            return False
        c = t.get('center_sensor') or [None, None, None]
        if c[0] is None or c[1] is None:
            return False
        x, y = float(c[0]), float(c[1])
        z = float(c[2]) if len(c) > 2 and c[2] is not None else 0.0
        if ego_tf is not None:
            m00, m01, m02, m10, m11, m12, b0, b1 = ego_tf
            ego_dist = math.hypot(m00 * x + m01 * y + m02 * z + b0,
                                  m10 * x + m11 * y + m12 * z + b1)
        else:
            ego_dist = math.hypot(x, y)
        if not ego_dist < DETECTION_RANGE.get(cls_id, 50.0):
            return False
        cx_norm, cy_norm = metres_to_norm(x, y, self.point_cloud_range)
        return 0.0 <= cx_norm <= 1.0 and 0.0 <= cy_norm <= 1.0

    # ── Build targets for a single frame ──

    def _build_targets(self, flat_idx: int, ref_tokens: set):
        """Build detection targets dict for one frame."""
        objects = self.frame_objects[flat_idx]
        pc_range = self.point_cloud_range
        ego_tf = (self.frame_ego_transform[flat_idx]
                  if flat_idx < len(self.frame_ego_transform) else None)

        targets = {
            'boxes': [],
            'area': [],
            'labels': [],
            'cat_labels': [],
            'obj_ids': [],
            'is_ref': [],
            'attrs_3d': [],
        }

        for obj in objects:
            cls_id = obj['class_id']
            if cls_id < 0:
                continue

            # Devkit-faithful gate (ego_dist class range + BEV square), shared
            # with the eval-GT query filter so supervision == scoring universe.
            if not self._gt_scoreable(obj, ego_tf):
                continue

            x, y, z = obj['center_sensor']
            l, w, h = obj['extents_lwh']
            yaw_rad = np.deg2rad(obj['yaw_deg'])

            cx_norm, cy_norm = metres_to_norm(x, y, pc_range)
            w_norm, l_norm = size_to_norm(w, l, pc_range)

            targets['boxes'].append([cx_norm, cy_norm, w_norm, l_norm])
            targets['area'].append(w_norm * l_norm)
            targets['labels'].append(0)  # binary objectness (foreground)
            # Ground-truth nuScenes class index (SHORT_CLASS_TO_ID order), carried
            # on the GT boxes for per-class evaluation metrics. The models train
            # only binary objectness, not class.
            targets['cat_labels'].append(int(cls_id))
            targets['obj_ids'].append(float(obj['obj_id']))
            targets['is_ref'].append(
                1.0 if obj['token'] in ref_tokens else 0.0)
            targets['attrs_3d'].append([
                z, h,
                np.sin(yaw_rad), np.cos(yaw_rad),
                0.0, 0.0,
            ])

        targets['boxes'] = torch.as_tensor(
            targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['area'] = torch.as_tensor(
            targets['area'], dtype=torch.float32)
        targets['labels'] = torch.as_tensor(
            targets['labels'], dtype=torch.int64)
        targets['cat_labels'] = torch.as_tensor(
            targets['cat_labels'], dtype=torch.int64)
        targets['obj_ids'] = torch.as_tensor(
            targets['obj_ids'], dtype=torch.float32)
        targets['is_ref'] = torch.as_tensor(
            targets['is_ref'], dtype=torch.float32)
        targets['attrs_3d'] = torch.as_tensor(
            targets['attrs_3d'], dtype=torch.float32).reshape(-1, 6)

        return targets

    @staticmethod
    def _targets_to_instances(targets: dict, bev_shape) -> Instances:
        gt = Instances(tuple(bev_shape))
        gt.boxes = targets['boxes']
        gt.labels = targets['labels']
        if 'cat_labels' in targets:
            gt.cat_labels = targets['cat_labels']
        gt.obj_ids = targets['obj_ids']
        gt.area = targets['area']
        gt.is_ref = targets['is_ref']
        if 'attrs_3d' in targets:
            gt.attrs_3d = targets['attrs_3d']
        return gt

    # ── Main entry points ──

    def __len__(self):
        if self.split == 'train':
            if self._epoch_size and self._epoch_size > 0:
                return max(1, math.ceil(self._epoch_size / max(self.K, 1)))
            return len(self.valid_indices)
        if self.split == 'val':
            if self._val_epoch_size and self._val_epoch_size > 0:
                return max(1, math.ceil(self._val_epoch_size / max(self.K, 1)))
            return len(self.valid_indices)
        return len(self._flat_queries)

    def __getitem__(self, idx):
        if self.split == 'train':
            self._maybe_log_train_examples()
            flat_idx, query_indices = self._sample_train_queries(idx)
            queries = [self.frame_refer_queries[flat_idx][query_idx] for query_idx in query_indices]
        elif self.split == 'val':
            flat_idx, query_indices = self._select_val_queries(idx)
            queries = [self.frame_refer_queries[flat_idx][query_idx] for query_idx in query_indices]
        else:
            flat_idx, query_idx = self._flat_queries[idx]
            queries = [self.frame_refer_queries[flat_idx][query_idx]]

        # Load point cloud (once per frame)
        entry = self.frames[flat_idx]
        cached = self._load_cached_features(entry)
        points = None if cached is not None else self._load_points_with_sweeps(entry)

        # Optional BEV precompute
        bev_tensor = None
        if self.pillar_encoder is not None and points is not None:
            with torch.no_grad():
                bev_tensor = self.pillar_encoder([points])[0]  # (C, H, W)

        # Determine BEV shape
        if cached is not None and len(cached['srcs']) > 0:
            shape = tuple(cached['srcs'][0].shape[-2:])
        elif bev_tensor is not None:
            shape = bev_tensor.shape[-2:]
        else:
            pc = np.array(self.point_cloud_range)
            vs = np.array(self.voxel_size)
            g = ((pc[3:] - pc[:3]) / vs).astype(int)
            shape = (g[1], g[0])

        # Build targets for each query (boxes/labels shared, only is_ref differs)
        all_sentences = []
        all_gt_instances = []
        all_qa_tokens = []
        for chosen in queries:
            sentence = chosen['query']
            all_qa_tokens.append(chosen.get('qa_token'))
            if _is_negative_query(chosen):
                ref_tokens = set()
            else:
                ref_tokens = {t['token'] for t in chosen.get('targets', []) if 'token' in t}
            targets = self._build_targets(flat_idx, ref_tokens)
            gt = self._targets_to_instances(targets, shape)
            all_sentences.append(sentence)
            all_gt_instances.append(gt)

        # Return dict (collated by simple_collate_fn into nested lists)
        data = {
            'sentences': all_sentences,          # list of K strings
            'gt_instances': all_gt_instances,    # list of K Instances
            'dataset_name': 'LIDAR',
            'qa_tokens': all_qa_tokens,
            'sample_token': entry.get('sample_token'),
            'lidar_path': entry.get('lidar_path'),
            # Keep keys stable across cache hits/misses so mixed batches collate safely.
            'points': points,
            'centerpoint_srcs': None,
            'centerpoint_props': None,
            'centerpoint_scores': None,
            'pointpillars_srcs': None,
            'pointpillars_props': None,
            'pointpillars_scores': None,
            'pointpillars_yaw': None,
        }
        if cached is not None:
            detector = getattr(self, '_feature_cache_detector', 'centerpoint')
            if detector == 'pointpillars':
                data['pointpillars_srcs'] = cached['srcs']
                data['pointpillars_props'] = cached['props']
                data['pointpillars_scores'] = cached['scores']
                data['pointpillars_yaw'] = cached.get('yaw')
            else:
                data['centerpoint_srcs'] = cached['srcs']
                data['centerpoint_props'] = cached['props']
                data['centerpoint_scores'] = cached['scores']
        elif bev_tensor is not None:
            data['imgs'] = bev_tensor            # (C, H, W)

        return data


# ── Builder ──────────────────────────────────────────────────────────────

def build(image_set, args):
    """Build single-frame LiDAR referring detection dataset."""
    pillar_encoder = None
    if getattr(args, 'precompute_bev', False):
        try:
            from models.pillar_encoder import PillarEncoderWrapper
        except ImportError as exc:
            raise ImportError(
                'precompute_bev=True requires models.pillar_encoder.PillarEncoderWrapper, '
                'but this module is not present in this repository. '
                'Disable precompute_bev or add the pillar encoder module.'
            ) from exc
        pillar_encoder = PillarEncoderWrapper(
            voxel_size=getattr(args, 'voxel_size', [0.08, 0.08, 4.0]),
            point_cloud_range=getattr(
                args, 'point_cloud_range',
                [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            in_channel=getattr(args, 'pillar_in_channel', 9),
            out_channel=getattr(args, 'pillar_out_channel', 64),
            max_num_points=getattr(args, 'max_points_per_voxel', 32),
            max_voxels=(getattr(args, 'max_voxels_train', 30000),
                        getattr(args, 'max_voxels_eval', 60000)),
        )
        ckpt = getattr(args, 'pillar_encoder_ckpt', None)
        if ckpt:
            pillar_encoder.load_pretrained(ckpt)

    dataset = LiDARReferDetectionSimple(
        args, split=image_set, pillar_encoder=pillar_encoder)
    return dataset
