import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import mmengine
import numpy as np
import torch
from mmengine.dataset import Compose

try:
    from mmcv.ops import box_iou_rotated as _mmcv_box_iou_rotated
except Exception:
    _mmcv_box_iou_rotated = None


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

# Prioritize local refer_lidar modules (models/*)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdet3d.registry import DATASETS  # noqa: E402
from models import build_model  # noqa: E402
from nuscenes_lidar_simple import build as build_refer_dataset  # noqa: E402


NUSCENES_DET_CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]


_USE_MMCV_BEV_IOU = _mmcv_box_iou_rotated is not None
_MMCV_BEV_IOU_WARNED = False
_MMCV_BEV_IOU_INFO_PRINTED = False


def _build_model_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    num_point_features = 5 if cli_args.meta_arch in {'refer_model_second', 'refer_model_second_v2', 'refer_model_second_v3'} else 4
    return SimpleNamespace(
        meta_arch=cli_args.meta_arch,
        hidden_dim=cli_args.hidden_dim,
        nheads=cli_args.nheads,
        enc_layers=cli_args.enc_layers,
        dec_layers=cli_args.dec_layers,
        dim_feedforward=cli_args.dim_feedforward,
        dropout=cli_args.dropout,
        num_feature_levels=cli_args.num_feature_levels,
        dec_n_points=cli_args.dec_n_points,
        enc_n_points=cli_args.enc_n_points,
        two_stage=False,
        num_queries=cli_args.proposal_queries,
        decoder_cross_self=False,
        sigmoid_attn=False,
        extra_track_attn=False,
        pointpillars_config=cli_args.pointpillars_config,
        pointpillars_ckpt=cli_args.pointpillars_ckpt,
        centerpoint_config=cli_args.centerpoint_config,
        centerpoint_ckpt=cli_args.centerpoint_ckpt,
        detector_backbone=cli_args.detector_backbone,
        centerpoint_feature_mode=cli_args.centerpoint_feature_mode,
        utonia_ckpt=cli_args.utonia_ckpt,
        pointcept_root=cli_args.pointcept_root,
        utonia_bev_size=cli_args.utonia_bev_size,
        utonia_bev_hidden_dim=cli_args.utonia_bev_hidden_dim,
        utonia_compute_normals=cli_args.utonia_compute_normals,
        utonia_normals_k=cli_args.utonia_normals_k,
        proposal_queries=cli_args.proposal_queries,
        proposal_w_from=cli_args.proposal_w_from,
        freeze_pointpillars=True,
        freeze_centerpoint=True,
        freeze_utonia_detector=cli_args.freeze_utonia_detector,
        freeze_utonia_encoder=cli_args.freeze_utonia_encoder,
        point_cloud_range=cli_args.point_cloud_range,
        # Matcher/loss compatibility for SEED-like checkpoints.
        aux_loss=cli_args.aux_loss,
        set_cost_class=cli_args.set_cost_class,
        set_cost_bbox=cli_args.set_cost_bbox,
        set_cost_center=cli_args.set_cost_center,
        set_cost_refer=cli_args.set_cost_refer,
        set_cost_refer_beta=cli_args.set_cost_refer_beta,
        cls_loss_coef=cli_args.cls_loss_coef,
        bbox_loss_coef=cli_args.bbox_loss_coef,
        giou_loss_coef=cli_args.giou_loss_coef,
        refer_loss_coef=cli_args.refer_loss_coef,
        loss_3d_coef=cli_args.loss_3d_coef,
        loss_dir_coef=getattr(cli_args, 'loss_dir_coef', 0.2),
        quality_loss_coef=cli_args.quality_loss_coef,
        focal_alpha=cli_args.focal_alpha,
        # SEED-like architecture toggles.
        use_dga=cli_args.use_dga,
        dga_grid_size=cli_args.dga_grid_size,
        dqs_topk=cli_args.dqs_topk,
        dqs_beta=cli_args.dqs_beta,
        no_decoder_lang_attn=getattr(cli_args, 'no_decoder_lang_attn', False),
        num_point_features=num_point_features,
    )


def _build_nus_dataset(args: argparse.Namespace) -> Tuple[Optional[object], Optional[List[dict]]]:
    dataset_cfg = dict(
        type='NuScenesDataset',
        data_root=args.data_root,
        ann_file=args.ann_file,
        pipeline=[],
        modality=dict(use_lidar=True, use_camera=False),
        data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP', img=''),
        test_mode=True,
        box_type_3d='LiDAR',
    )
    try:
        return DATASETS.build(dataset_cfg), None
    except Exception as exc:
        print(f'[Dataset] NuScenesDataset build failed: {exc}')
        print('[Dataset] Falling back to legacy ann-file parser...')
        return None, _load_legacy_infos(args.ann_file)


def _load_legacy_infos(ann_file: str) -> List[dict]:
    loaded = mmengine.load(ann_file)

    if isinstance(loaded, dict):
        if 'data_list' in loaded and isinstance(loaded['data_list'], list):
            return loaded['data_list']
        if 'infos' in loaded and isinstance(loaded['infos'], list):
            return loaded['infos']

        out = []
        for token, info in loaded.items():
            if isinstance(info, dict):
                info = dict(info)
                info['token'] = info.get('token', token)
                out.append(info)
        if out:
            return out

    if isinstance(loaded, list):
        return loaded

    raise RuntimeError(f'Unsupported annotation format in {ann_file}')


def _resolve_lidar_path(data_root: str, lidar_path: str) -> str:
    lidar_path = str(lidar_path).replace('\\', '/')
    if os.path.isabs(lidar_path):
        return lidar_path
    if lidar_path.startswith('samples/') or lidar_path.startswith('sweeps/'):
        return os.path.join(data_root, lidar_path)
    return os.path.join(data_root, os.path.basename(lidar_path))


def _legacy_lidar2sensor_from_sensor2lidar(sweep: dict) -> List[List[float]]:
    rot_s2l = np.asarray(sweep['sensor2lidar_rotation'], dtype=np.float32)
    trans_s2l = np.asarray(sweep['sensor2lidar_translation'], dtype=np.float32)

    rot = rot_s2l.T
    trans = -trans_s2l

    lidar2sensor = np.eye(4, dtype=np.float32)
    lidar2sensor[:3, :3] = rot
    lidar2sensor[:3, 3] = trans
    return lidar2sensor.tolist()


def _legacy_info_to_pipeline_input(info: dict, data_root: str) -> dict:
    lidar_path = info.get('lidar_path')
    if not lidar_path:
        lidar_path = info.get('lidar_points', {}).get('lidar_path', '')
    lidar_path = _resolve_lidar_path(data_root, lidar_path)

    sweeps = info.get('sweeps')
    if sweeps is None:
        sweeps = info.get('lidar_sweeps')
    if sweeps is None:
        sweeps = info.get('lidar_points', {}).get('lidar_sweeps')
    if sweeps is None:
        sweeps = info.get('lidar_points', {}).get('sweeps')
    if sweeps is None:
        sweeps = []

    if isinstance(sweeps, dict):
        sweeps = list(sweeps.values())

    token = info.get('token', info.get('sample_token', '<unknown>'))

    lidar_sweeps = []
    for sw in sweeps:
        if not isinstance(sw, dict):
            continue

        if 'data_path' in sw:
            sweep_path = _resolve_lidar_path(data_root, sw['data_path'])
        elif 'lidar_path' in sw:
            sweep_path = _resolve_lidar_path(data_root, sw['lidar_path'])
        else:
            sweep_path = _resolve_lidar_path(
                data_root, sw.get('lidar_points', {}).get('lidar_path', '')
            )

        if 'sensor2lidar_rotation' in sw and 'sensor2lidar_translation' in sw:
            lidar2sensor = _legacy_lidar2sensor_from_sensor2lidar(sw)
        elif 'lidar2sensor' in sw:
            lidar2sensor = sw['lidar2sensor']
        elif 'lidar_points' in sw and 'lidar2sensor' in sw['lidar_points']:
            lidar2sensor = sw['lidar_points']['lidar2sensor']
        else:
            lidar2sensor = np.eye(4, dtype=np.float32).tolist()

        sweep_ts = float(sw.get('timestamp', sw.get('ts', 0.0)))
        if sweep_ts > 1e12:
            sweep_ts /= 1e6

        if not sweep_path:
            continue

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

    return {
        'token': token,
        'timestamp': ts,
        'lidar_points': {
            'lidar_path': lidar_path,
        },
        # Keep both keys for cross-version compatibility.
        'sweeps': lidar_sweeps,
        'lidar_sweeps': lidar_sweeps,
    }


def _find_sample_index(
    dataset,
    legacy_infos: Optional[List[dict]],
    token: Optional[str],
    lidar_suffix: Optional[str],
) -> int:
    if dataset is None and legacy_infos is None:
        raise RuntimeError('No dataset source available')

    def _num_sweeps(info: dict) -> int:
        if 'lidar_sweeps' in info and isinstance(info['lidar_sweeps'], list):
            return len(info['lidar_sweeps'])
        if 'sweeps' in info and isinstance(info['sweeps'], list):
            return len(info['sweeps'])
        if isinstance(info.get('lidar_points', None), dict):
            lp = info['lidar_points']
            if isinstance(lp.get('lidar_sweeps', None), list):
                return len(lp['lidar_sweeps'])
            if isinstance(lp.get('sweeps', None), list):
                return len(lp['sweeps'])
        return 0

    if token is None and lidar_suffix is None:
        n = len(dataset) if dataset is not None else len(legacy_infos)
        for i in range(n):
            info = dataset.get_data_info(i) if dataset is not None else legacy_infos[i]
            if _num_sweeps(info) > 0:
                return i
        return 0

    n = len(dataset) if dataset is not None else len(legacy_infos)
    for i in range(n):
        info = dataset.get_data_info(i) if dataset is not None else legacy_infos[i]
        lidar_path = info.get('lidar_path') or info.get('lidar_points', {}).get('lidar_path', '')
        if token is not None and info.get('token') == token:
            return i
        if lidar_suffix is not None and str(lidar_path).endswith(lidar_suffix):
            return i

    raise ValueError(
        'Could not locate sample. Provide a valid --sample-token or --lidar-suffix.'
    )


def _build_10_sweep_pipeline(sweeps_num: int, num_point_features: int = 4) -> Compose:
    return Compose([
        dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=sweeps_num,
            load_dim=5,
            use_dim=[0, 1, 2, 3, 4] if num_point_features >= 5 else [0, 1, 2, 4],
            pad_empty_sweeps=True,
            test_mode=True,
        ),
        dict(type='Pack3DDetInputs', keys=['points']),
    ])


def _decode_boxes_to_meters(norm_boxes: torch.Tensor, point_cloud_range: List[float]) -> torch.Tensor:
    x_min, y_min, _z_min, x_max, y_max, _z_max = point_cloud_range
    x_span = x_max - x_min
    y_span = y_max - y_min

    out = norm_boxes.clone()
    out[..., 0] = norm_boxes[..., 0] * x_span + x_min
    out[..., 1] = norm_boxes[..., 1] * y_span + y_min
    out[..., 2] = norm_boxes[..., 2] * y_span
    out[..., 3] = norm_boxes[..., 3] * x_span
    return out


def _extract_pred_attrs_3d(outputs: Dict[str, torch.Tensor], query_index: int,
                           point_cloud_range: Optional[List[float]] = None) -> Optional[np.ndarray]:
    """Extract per-query 3D attrs in [z, h, sin(yaw), cos(yaw)].

    V2 keeps 3D attrs in pred_boxes[..., 4:7] as *normalized* values
    (z_norm, h_norm, yaw_half_norm) with a separate pred_dirs (direction logits).
    yaw_half_norm maps [0,1] → [-π/2, π/2].  The full yaw is reconstructed
    from yaw_half + dir * π where dir = sigmoid(logit) > 0.5.

    When ``point_cloud_range`` is provided, z and h are converted back to metres
    and yaw is converted to sin/cos for downstream IoU computations.
    """
    pred_boxes = outputs.get('pred_boxes', None)
    if pred_boxes is not None and pred_boxes.shape[-1] >= 7:
        raw = pred_boxes[query_index, :, 4:7].detach().cpu().numpy()  # (N, 3)

        # Direction reconstruction
        pred_dirs = outputs.get('pred_dirs', None)
        if pred_dirs is not None:
            dir_logits = pred_dirs[query_index, :, 0].detach().cpu().numpy()  # (N,)
            dir_class = (1.0 / (1.0 + np.exp(-dir_logits)) > 0.5).astype(np.float32)
        else:
            dir_class = np.zeros(raw.shape[0], dtype=np.float32)

        if point_cloud_range is not None:
            z_min = float(point_cloud_range[2])
            z_max = float(point_cloud_range[5])
            z_span = max(z_max - z_min, 1e-6)
            z = raw[:, 0] * z_span + z_min        # z_norm → z metres
            h = raw[:, 1] * 10.0                   # h_norm → h metres
            # yaw_half_norm → yaw_half → full yaw via dir
            yaw_half = raw[:, 2] * np.pi - (np.pi / 2.0)
            yaw = yaw_half + dir_class * np.pi
            out = np.stack([z, h, np.sin(yaw), np.cos(yaw)], axis=-1)
        else:
            yaw_half = raw[:, 2] * np.pi - (np.pi / 2.0)
            yaw = yaw_half + dir_class * np.pi
            out = np.stack([raw[:, 0], raw[:, 1], np.sin(yaw), np.cos(yaw)], axis=-1)
        return out

    if 'pred_3d' in outputs:
        base = outputs['pred_3d'][query_index].detach().cpu().numpy()
        # refer_model_angle: heading comes from the iteratively-refined pred_yaw
        # (radians), not head_3d's sin/cos. Keep z, h from head_3d (dims 0, 1)
        # and overwrite the sin/cos slots (dims 2, 3) from pred_yaw.
        pred_yaw = outputs.get('pred_yaw', None)
        if pred_yaw is not None and base.shape[-1] >= 4:
            yaw = pred_yaw[query_index].detach().cpu().numpy()  # (N,)
            base = base.copy()
            base[:, 2] = np.sin(yaw)
            base[:, 3] = np.cos(yaw)
        return base

    return None


def _box_iou_2d(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) boxes in [cx, cy, w, l] format."""
    if box_a.shape[0] == 0 or box_b.shape[0] == 0:
        return np.zeros((box_a.shape[0], box_b.shape[0]), dtype=np.float32)

    a = np.stack([
        box_a[:, 0] - box_a[:, 3] / 2,
        box_a[:, 1] - box_a[:, 2] / 2,
        box_a[:, 0] + box_a[:, 3] / 2,
        box_a[:, 1] + box_a[:, 2] / 2,
    ], axis=1)
    b = np.stack([
        box_b[:, 0] - box_b[:, 3] / 2,
        box_b[:, 1] - box_b[:, 2] / 2,
        box_b[:, 0] + box_b[:, 3] / 2,
        box_b[:, 1] + box_b[:, 2] / 2,
    ], axis=1)

    iou = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    for i in range(a.shape[0]):
        xx1 = np.maximum(a[i, 0], b[:, 0])
        yy1 = np.maximum(a[i, 1], b[:, 1])
        xx2 = np.minimum(a[i, 2], b[:, 2])
        yy2 = np.minimum(a[i, 3], b[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_a = (a[i, 2] - a[i, 0]) * (a[i, 3] - a[i, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou[i] = inter / (area_a + area_b - inter + 1e-8)
    return iou


def _center_distance(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    if box_a.shape[0] == 0 or box_b.shape[0] == 0:
        return np.zeros((box_a.shape[0], box_b.shape[0]), dtype=np.float32)
    dist = np.zeros((box_a.shape[0], box_b.shape[0]), dtype=np.float32)
    for i in range(box_a.shape[0]):
        dist[i] = np.sqrt((box_a[i, 0] - box_b[:, 0]) ** 2 + (box_a[i, 1] - box_b[:, 1]) ** 2)
    return dist

def _calc_scale_error(pred_w, pred_l, pred_h, gt_w, gt_l, gt_h) -> float:
    """Calculate 1 - 3D IoU after aligning centers and orientation (Scale Error)."""
    # 3D intersection of aligned boxes
    inter_w = min(pred_w, gt_w)
    inter_l = min(pred_l, gt_l)
    inter_h = min(pred_h, gt_h)
    
    if inter_w <= 0 or inter_l <= 0 or inter_h <= 0:
        return 1.0
        
    inter_vol = inter_w * inter_l * inter_h
    pred_vol = pred_w * pred_l * pred_h
    gt_vol = gt_w * gt_l * gt_h
    union_vol = pred_vol + gt_vol - inter_vol
    
    iou = inter_vol / max(union_vol, 1e-8)
    return 1.0 - float(iou)

def _calc_yaw_error(pred_sin, pred_cos, gt_sin, gt_cos, class_name) -> float:
    """Calculate absolute yaw difference, accounting for symmetry."""
    pred_yaw = np.arctan2(pred_sin, pred_cos)
    gt_yaw = np.arctan2(gt_sin, gt_cos)
    
    # nuScenes ignores orientation for traffic cones entirely
    if class_name == 'traffic_cone':
        return np.nan
        
    diff = abs(pred_yaw - gt_yaw)
    
    # Barriers look the same if rotated 180 degrees (pi)
    period = np.pi if class_name == 'barrier' else 2 * np.pi
    
    diff = diff % period
    if diff > period / 2.0:
        diff = period - diff
        
    return float(diff)


def _rotated_rect_corners(cx: float, cy: float, w: float, l: float, yaw: float) -> np.ndarray:
    hw, hl = 0.5 * float(w), 0.5 * float(l)
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = local @ rot.T
    pts[:, 0] += float(cx)
    pts[:, 1] += float(cy)
    return pts


def _polygon_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 1e-7


def _line_intersection(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-8:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], dtype=np.float32)


def _convex_intersection(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    out = subject.copy()
    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        inp = out
        if len(inp) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        out_pts = []
        s = inp[-1]
        for e in inp:
            if _inside(e, a, b):
                if not _inside(s, a, b):
                    out_pts.append(_line_intersection(s, e, a, b))
                out_pts.append(e)
            elif _inside(s, a, b):
                out_pts.append(_line_intersection(s, e, a, b))
            s = e
        if len(out_pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        out = np.asarray(out_pts, dtype=np.float32)
    return out


def _oriented_bev_intersection_area(box_a: np.ndarray, yaw_a: float, box_b: np.ndarray, yaw_b: float) -> float:
    pa = _rotated_rect_corners(box_a[0], box_a[1], box_a[2], box_a[3], yaw_a)
    pb = _rotated_rect_corners(box_b[0], box_b[1], box_b[2], box_b[3], yaw_b)
    inter_poly = _convex_intersection(pa, pb)
    return _polygon_area(inter_poly)


def _box_iou_bev_oriented_mmcv(
    pred_boxes: np.ndarray,
    pred_attrs_3d: np.ndarray,
    gt_boxes: np.ndarray,
    gt_attrs_3d: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute oriented BEV IoU via mmcv op (CUDA when available).

    Project box format is [cx, cy, w(y-span), l(x-span)], while mmcv expects
    [cx, cy, w(x-span), h(y-span), yaw], so width/height are swapped here.
    """
    global _USE_MMCV_BEV_IOU
    global _MMCV_BEV_IOU_WARNED
    global _MMCV_BEV_IOU_INFO_PRINTED

    if not _USE_MMCV_BEV_IOU:
        return None

    try:
        pred_yaw = np.arctan2(pred_attrs_3d[:, 2], pred_attrs_3d[:, 3]).astype(np.float32)
        gt_yaw = np.arctan2(gt_attrs_3d[:, 2], gt_attrs_3d[:, 3]).astype(np.float32)

        pred_xywhr = np.stack(
            [pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 3], pred_boxes[:, 2], pred_yaw],
            axis=1,
        ).astype(np.float32)
        gt_xywhr = np.stack(
            [gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 3], gt_boxes[:, 2], gt_yaw],
            axis=1,
        ).astype(np.float32)

        pred_xywhr[:, 2:4] = np.maximum(pred_xywhr[:, 2:4], 1e-4)
        gt_xywhr[:, 2:4] = np.maximum(gt_xywhr[:, 2:4], 1e-4)

        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        pred_t = torch.as_tensor(pred_xywhr, dtype=torch.float32, device=device)
        gt_t = torch.as_tensor(gt_xywhr, dtype=torch.float32, device=device)

        iou_t = _mmcv_box_iou_rotated(pred_t, gt_t)
        if not _MMCV_BEV_IOU_INFO_PRINTED:
            print(f'[Eval][IoU] Using mmcv box_iou_rotated on device={device}.', flush=True)
            _MMCV_BEV_IOU_INFO_PRINTED = True
        return iou_t.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as exc:
        if not _MMCV_BEV_IOU_WARNED:
            print(f'[Eval][IoU] mmcv box_iou_rotated failed once: {exc}. Falling back to Python IoU.', flush=True)
            _MMCV_BEV_IOU_WARNED = True
        _USE_MMCV_BEV_IOU = False
        return None


def _box_iou_bev_oriented(
    pred_boxes: np.ndarray,
    pred_attrs_3d: Optional[np.ndarray],
    gt_boxes: np.ndarray,
    gt_attrs_3d: Optional[np.ndarray],
) -> np.ndarray:
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)
    if pred_attrs_3d is None or gt_attrs_3d is None:
        return _box_iou_2d(pred_boxes, gt_boxes)

    iou_mmcv = _box_iou_bev_oriented_mmcv(pred_boxes, pred_attrs_3d, gt_boxes, gt_attrs_3d)
    if iou_mmcv is not None:
        return iou_mmcv

    pred_yaw = np.arctan2(pred_attrs_3d[:, 2], pred_attrs_3d[:, 3])
    gt_yaw = np.arctan2(gt_attrs_3d[:, 2], gt_attrs_3d[:, 3])
    area_p = pred_boxes[:, 2] * pred_boxes[:, 3]
    area_g = gt_boxes[:, 2] * gt_boxes[:, 3]
    out = np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)
    for i in range(pred_boxes.shape[0]):
        for j in range(gt_boxes.shape[0]):
            inter = _oriented_bev_intersection_area(pred_boxes[i], pred_yaw[i], gt_boxes[j], gt_yaw[j])
            if inter <= 0.0:
                continue
            union = area_p[i] + area_g[j] - inter + 1e-8
            out[i, j] = inter / union
    return out


def _iou_3d_oriented(
    pred_boxes_2d: np.ndarray,
    pred_attrs_3d: Optional[np.ndarray],
    gt_boxes_2d: np.ndarray,
    gt_attrs_3d: Optional[np.ndarray],
) -> np.ndarray:
    if pred_boxes_2d.shape[0] == 0 or gt_boxes_2d.shape[0] == 0:
        return np.zeros((pred_boxes_2d.shape[0], gt_boxes_2d.shape[0]), dtype=np.float32)
    if pred_attrs_3d is None or gt_attrs_3d is None:
        return _box_iou_2d(pred_boxes_2d, gt_boxes_2d)

    n, m = pred_boxes_2d.shape[0], gt_boxes_2d.shape[0]
    iou3d = np.zeros((n, m), dtype=np.float32)
    pred_yaw = np.arctan2(pred_attrs_3d[:, 2], pred_attrs_3d[:, 3])
    gt_yaw = np.arctan2(gt_attrs_3d[:, 2], gt_attrs_3d[:, 3])

    pz1 = pred_attrs_3d[:, 0] - pred_attrs_3d[:, 1] / 2.0
    pz2 = pred_attrs_3d[:, 0] + pred_attrs_3d[:, 1] / 2.0
    gz1 = gt_attrs_3d[:, 0] - gt_attrs_3d[:, 1] / 2.0
    gz2 = gt_attrs_3d[:, 0] + gt_attrs_3d[:, 1] / 2.0

    area_p = pred_boxes_2d[:, 2] * pred_boxes_2d[:, 3]
    area_g = gt_boxes_2d[:, 2] * gt_boxes_2d[:, 3]
    height_p = np.maximum(0.0, pz2 - pz1)
    height_g = np.maximum(0.0, gz2 - gz1)

    for i in range(n):
        for j in range(m):
            inter_bev = _oriented_bev_intersection_area(pred_boxes_2d[i], pred_yaw[i], gt_boxes_2d[j], gt_yaw[j])
            if inter_bev <= 0.0:
                continue
            iz1 = max(pz1[i], gz1[j])
            iz2 = min(pz2[i], gz2[j])
            iz = max(0.0, iz2 - iz1)
            if iz <= 0.0:
                continue
            inter_vol = inter_bev * iz
            vol_p = area_p[i] * height_p[i]
            vol_g = area_g[j] * height_g[j]
            union = vol_p + vol_g - inter_vol + 1e-8
            iou3d[i, j] = inter_vol / union
    return iou3d


def _compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def _compute_ap_nuscenes_style(recalls: np.ndarray, precisions: np.ndarray, min_recall: float = 0.1, min_precision: float = 0.0) -> float:
    """Calculates AP using the exact 101-point interpolation from nuScenes algo.py"""
    if len(recalls) == 0:
        return 0.0
    rec_interp = np.linspace(0, 1, 101) 
    prec_interp = np.interp(rec_interp, recalls, precisions, right=0.0)
    first_ind = round(100 * min_recall) + 1 
    prec_interp = prec_interp[first_ind:]
    prec_interp -= min_precision
    prec_interp[prec_interp < 0] = 0
    return float(np.mean(prec_interp)) / (1.0 - min_precision)

def _compute_tp_errors_nuscenes(
    query_records: List[dict], 
    class_name: str, 
    dist_th_tp: float = 2.0, 
    min_recall: float = 0.1
) -> Dict[str, float]:
    """
    Computes nuScenes True Positive metrics (mATE, mASE, mAOE) using the official 
    Recall-Normalized Cumulative Mean method. Matches are STRICTLY capped at 2.0 meters.
    """
    gt_total = int(sum(r['gt_boxes'].shape[0] for r in query_records))
    if gt_total == 0:
        return {'mATE': 1.0, 'mASE': 1.0, 'mAOE': 1.0}

    # 1. Flatten and sort all predictions globally by confidence
    detections = []
    for qi, rec in enumerate(query_records):
        boxes = rec['pred_boxes']
        scores = rec['pred_scores']
        attrs = rec.get('pred_attrs_3d', None)
        for pi in range(boxes.shape[0]):
            detections.append({
                'score': float(scores[pi]), 'qi': qi, 'pi': pi,
                'box': boxes[pi], 'attr': attrs[pi] if attrs is not None else None
            })
            
    detections.sort(key=lambda x: x['score'], reverse=True)

    tp, fp, conf = [], [], []
    match_data = {'trans_err': [], 'scale_err': [], 'orient_err': [], 'conf': []}
    gt_matched = [np.zeros(r['gt_boxes'].shape[0], dtype=bool) for r in query_records]

    # 2. Strict 2.0m Matching Loop
    for det in detections:
        qi, pi = det['qi'], det['pi']
        pred_box, pred_attr = det['box'], det['attr']
        gt_boxes = query_records[qi]['gt_boxes']
        gt_attrs = query_records[qi].get('gt_attrs_3d', None)
        gt_cnames = query_records[qi].get('gt_class_names', None)

        if gt_boxes.shape[0] == 0:
            tp.append(0); fp.append(1); conf.append(det['score'])
            continue

        dists = np.sqrt((pred_box[0] - gt_boxes[:, 0])**2 + (pred_box[1] - gt_boxes[:, 1])**2)
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, dist in enumerate(dists):
            if not gt_matched[qi][gt_idx] and dist < min_dist:
                min_dist = dist
                match_gt_idx = gt_idx

        # TP Errors are evaluated strictly at 2.0m regardless of AP settings
        if match_gt_idx is not None and min_dist < dist_th_tp:
            gt_matched[qi][match_gt_idx] = True
            tp.append(1); fp.append(0); conf.append(det['score'])

            match_data['trans_err'].append(min_dist)
            
            gt_box = gt_boxes[match_gt_idx]
            if pred_attr is not None and gt_attrs is not None and gt_attrs.shape[0] > match_gt_idx:
                gt_attr = gt_attrs[match_gt_idx]
                # nuScenes orientation rules are PER-OBJECT-CLASS (cone excluded,
                # barrier mod-pi). Prefer the per-object GT class; fall back to
                # the caller's class_name only if the record predates the
                # gt_class_names field.
                cn = (gt_cnames[match_gt_idx]
                      if (gt_cnames is not None and len(gt_cnames) > match_gt_idx)
                      else class_name)
                se = _calc_scale_error(pred_box[2], pred_box[3], pred_attr[1], gt_box[2], gt_box[3], gt_attr[1])
                oe = _calc_yaw_error(pred_attr[2], pred_attr[3], gt_attr[2], gt_attr[3], cn)
            else:
                se, oe = 1.0, 1.0

            match_data['scale_err'].append(se)
            match_data['orient_err'].append(oe)
            match_data['conf'].append(det['score'])
        else:
            tp.append(0); fp.append(1); conf.append(det['score'])

    if len(match_data['trans_err']) == 0:
        return {'mATE': 1.0, 'mASE': 1.0, 'mAOE': 1.0}

    # 3. Calculate 101-point Interpolation mapping
    tp_cum = np.cumsum(tp).astype(float)
    rec = tp_cum / gt_total
    rec_interp = np.linspace(0, 1, 101)
    conf_interp = np.interp(rec_interp, rec, np.array(conf), right=0)

    # 4. Apply nuScenes Recall-Normalized Cumulative Mean — devkit-exact.
    def _devkit_cummean(x: np.ndarray) -> np.ndarray:
        """nuscenes.eval.common.utils.cummean — NaN-aware cumulative mean.
        NaN slots carry the running mean; all-NaN input returns ones."""
        if np.all(np.isnan(x)):
            return np.ones(len(x))
        sum_vals = np.nancumsum(x.astype(float))
        count_vals = np.cumsum(~np.isnan(x))
        return np.divide(sum_vals, count_vals,
                         out=np.zeros_like(sum_vals), where=count_vals != 0)

    # devkit max_recall_ind (DetectionMetricData.max_recall_ind): the LAST
    # index with confidence > 0; 0 when there are no matches at all. (NOT the
    # first zero index — that is one bin too far.)
    _nz = np.nonzero(conf_interp)[0]
    last_ind = int(_nz[-1]) if len(_nz) > 0 else 0
    first_ind = round(100 * min_recall) + 1

    out_metrics = {}
    for key in ['trans_err', 'scale_err', 'orient_err']:
        raw_errs = np.array(match_data[key], dtype=float)
        match_confs = np.array(match_data['conf'], dtype=float)

        # devkit-exact: NaN-aware cummean over ALL matches (e.g. traffic-cone
        # orientation NaNs carry the running mean), then interpolate over the
        # full match-confidence grid (algo.py::accumulate resampling).
        tmp = _devkit_cummean(raw_errs)
        interp_errs = np.interp(conf_interp[::-1], match_confs[::-1], tmp[::-1])[::-1]

        if last_ind < first_ind:
            out_metrics[key] = 1.0  # devkit calc_tp fallback
        else:
            out_metrics[key] = float(np.mean(interp_errs[first_ind:last_ind + 1]))

    return {
        'mATE': out_metrics['trans_err'],
        'mASE': out_metrics['scale_err'],
        'mAOE': out_metrics['orient_err']
    }

def _compute_detection_pr_nuscenes(
    query_records: List[dict],
    dist_threshold: float,
    max_curve_points: int = 0,
) -> Dict[str, Any]:
    gt_total = int(sum(r['gt_boxes'].shape[0] for r in query_records))

    detections = []
    global_idx = 0  # Used to match nuScenes exact tie-breaking
    for qi, rec in enumerate(query_records):
        boxes = rec['pred_boxes']
        scores = rec['pred_scores']
        for pi in range(boxes.shape[0]):
            detections.append((float(scores[pi]), qi, pi, boxes[pi], global_idx))
            global_idx += 1
            
    # 1. Sort strictly by confidence score (Descending)
    # nuScenes Tie-breaker: If scores are equal, the higher original index goes first.
    # numpy lexsort replaces the Python lambda-key sort: it produces the IDENTICAL
    # order (global_idx is unique -> total order, no residual ties) at C speed.
    # lexsort uses its LAST key as primary and sorts ascending, so we negate both
    # score and global_idx to get (score desc, then global_idx desc).
    if detections:
        _scores = np.fromiter((d[0] for d in detections), dtype=np.float64, count=len(detections))
        _gidx = np.fromiter((d[4] for d in detections), dtype=np.int64, count=len(detections))
        _order = np.lexsort((-_gidx, -_scores))
        detections = [detections[i] for i in _order]

    if gt_total == 0 or len(detections) == 0:
        return {
            'ap': 0.0, 'num_gt': gt_total, 'num_detections': len(detections),
            'curve': {'recall': [], 'precision': [], 'score': []}
        }

    tp = np.zeros(len(detections), dtype=np.float32)
    fp = np.zeros(len(detections), dtype=np.float32)
    gt_matched = [np.zeros(r['gt_boxes'].shape[0], dtype=bool) for r in query_records]

    # Match predictions to GT based on 2D Center Distance
    for di, (score, qi, pi, pred_box, _) in enumerate(detections):

        if di % 1000000 == 0 and len(detections) > 1000000:
            print(f"    [PR Math] Processed {di}/{len(detections)} predictions...")

        gt_boxes = query_records[qi]['gt_boxes']
        if gt_boxes.shape[0] == 0:
            fp[di] = 1.0
            continue
            
        # READ FROM CACHE INSTEAD OF CALCULATING
        dists = query_records[qi]['dist_matrix'][pi] 
        
        min_dist = np.inf
        match_gt_idx = None
        
        for gt_idx, dist in enumerate(dists):
            if not gt_matched[qi][gt_idx] and dist < min_dist:
                min_dist = dist
                match_gt_idx = gt_idx
                
        if match_gt_idx is not None and min_dist < dist_threshold:
            tp[di] = 1.0
            gt_matched[qi][match_gt_idx] = True
        else:
            fp[di] = 1.0

    # 3. Calculate Precision and Recall
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(gt_total, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
    
    # 4. Calculate AP using nuScenes 101-point interpolation.
    # min_precision=0.1 is REQUIRED for devkit parity: official calc_ap clips
    # precision below 0.1 and renormalizes by 0.9 (algo.py::calc_ap). Without
    # it, AP is systematically inflated relative to the nuScenes protocol.
    ap = _compute_ap_nuscenes_style(recalls, precisions, min_recall=0.1, min_precision=0.1)

    curve = {'recall': [], 'precision': [], 'score': []}
    if max_curve_points > 0:
        det_scores = np.asarray([d[0] for d in detections], dtype=np.float32)
        curve = _downsample_curve_points(recalls, precisions, det_scores, max_curve_points)

    return {'ap': float(ap), 'num_gt': int(gt_total), 'num_detections': int(len(detections)), 'curve': curve}


def _downsample_curve_points(
    recalls: np.ndarray,
    precisions: np.ndarray,
    scores: np.ndarray,
    max_points: int,
) -> Dict[str, List[float]]:
    if recalls.size == 0:
        return {'recall': [], 'precision': [], 'score': []}

    if max_points <= 0 or recalls.size <= max_points:
        idx = np.arange(recalls.size, dtype=np.int64)
    else:
        idx = np.linspace(0, recalls.size - 1, num=max_points, dtype=np.int64)
        idx = np.unique(idx)

    return {
        'recall': [float(x) for x in recalls[idx].tolist()],
        'precision': [float(x) for x in precisions[idx].tolist()],
        'score': [float(x) for x in scores[idx].tolist()],
    }




def _filter_records_by_confidence(query_records: List[dict], min_conf: float) -> List[dict]:
    out = []
    thr = float(min_conf)
    for rec in query_records:
        scores = np.asarray(rec.get('pred_scores', np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        pred_boxes = rec.get('pred_boxes', np.zeros((0, 4), dtype=np.float32))
        if pred_boxes.shape[0] != scores.shape[0]:
            n = min(pred_boxes.shape[0], scores.shape[0])
            pred_boxes = pred_boxes[:n]
            scores = scores[:n]
        keep = scores >= thr

        pred_attrs = rec.get('pred_attrs_3d', None)
        if pred_attrs is not None:
            # Align pred_attrs to n (minimum of pred_boxes and scores)
            if pred_boxes.shape[0] != scores.shape[0]:
                n = min(pred_boxes.shape[0], scores.shape[0])
                pred_attrs = pred_attrs[:n]
            else:
                pred_attrs = pred_attrs[:scores.shape[0]]
            pred_attrs = pred_attrs[keep]

        new_rec = {
            'pred_boxes': pred_boxes[keep],
            'pred_scores': scores[keep],
            'gt_boxes': rec.get('gt_boxes', np.zeros((0, 4), dtype=np.float32)),
            'pred_attrs_3d': pred_attrs,
            'gt_attrs_3d': rec.get('gt_attrs_3d', None),
            'gt_class_names': rec.get('gt_class_names', None),  # per-object class for nuScenes TP rules
            'cname': rec.get('cname', 'unknown'),
            'qtype': rec.get('qtype', 'unknown')
        }

        # Propagate cached matrices (slice rows for kept predictions).
        for key in rec:
            if key.startswith('_iou_cache_') or key == 'dist_matrix': # <--- UPDATE THIS IF CONDITION
                mat = rec[key]
                if mat is not None:
                    new_rec[key] = mat[keep]
                else:
                    new_rec[key] = None

        out.append(new_rec)
    return out


def _nms_records_by_center(query_records: List[dict], radius: float) -> List[dict]:
    """Greedy center-distance NMS *within each referring record*: keep boxes by
    descending refer score, drop any whose BEV center lies within `radius` m of an
    already-kept box. DIAGNOSTIC only -- the reported eval is NMS-free (radius<=0
    is a no-op). Per-box arrays are subset exactly as in
    _filter_records_by_confidence so cached matrices stay row-consistent."""
    if radius is None or float(radius) <= 0.0:
        return query_records
    r2 = float(radius) ** 2
    out = []
    for rec in query_records:
        scores = np.asarray(rec.get('pred_scores', np.zeros((0,), np.float32)), np.float32)
        pred_boxes = np.asarray(rec.get('pred_boxes', np.zeros((0, 4), np.float32)), np.float32)
        n = min(pred_boxes.shape[0], scores.shape[0])
        if n <= 1:
            out.append(rec)
            continue
        pred_boxes = pred_boxes[:n]
        scores = scores[:n]
        centers = pred_boxes[:, :2]
        # Vectorised greedy: precompute pairwise squared distances once, then
        # walk boxes best-first suppressing all neighbours within radius in bulk.
        diff = centers[:, None, :] - centers[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)   # (n, n)
        suppressed = np.zeros(n, dtype=bool)
        keep = np.zeros(n, dtype=bool)
        for i in np.argsort(-scores, kind='stable'):
            if suppressed[i]:
                continue
            keep[i] = True
            nb = d2[i] < r2
            nb[i] = False
            suppressed |= nb
        new_rec = dict(rec)
        new_rec['pred_boxes'] = pred_boxes[keep]
        new_rec['pred_scores'] = scores[keep]
        pred_attrs = rec.get('pred_attrs_3d', None)
        new_rec['pred_attrs_3d'] = pred_attrs[:n][keep] if pred_attrs is not None else None
        for key in rec:
            if key.startswith('_iou_cache_') or key == 'dist_matrix':
                mat = rec[key]
                new_rec[key] = mat[:n][keep] if mat is not None else None
        out.append(new_rec)
    return out


# Official detection_cvpr_2019 class_range (metres, ego distance).
DEVKIT_CLASS_RANGE = {
    'car': 50.0, 'truck': 50.0, 'bus': 50.0, 'trailer': 50.0,
    'construction_vehicle': 50.0, 'pedestrian': 40.0, 'motorcycle': 40.0,
    'bicycle': 40.0, 'traffic_cone': 30.0, 'barrier': 30.0,
}


def _ego_dists(boxes: np.ndarray, attrs, ego_tf) -> np.ndarray:
    """Devkit ego_dist for (N,4) BEV boxes [cx,cy,w,l]. With a per-record
    ego-range transform (8 floats: m00,m01,m02,m10,m11,m12,b0,b1) this
    reproduces the devkit's GLOBAL-frame xy distance from the ego pose exactly
    (incl. vehicle roll/pitch, using z from attrs[:, 0] when available).
    Legacy records (ego_tf None): lidar-origin xy norm (~1 m off ego)."""
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    x = boxes[:, 0].astype(np.float64)
    y = boxes[:, 1].astype(np.float64)
    if ego_tf is None:
        return np.sqrt(x * x + y * y)
    z = (np.asarray(attrs)[:, 0].astype(np.float64)
         if attrs is not None and len(attrs) == boxes.shape[0]
         else np.zeros_like(x))
    m00, m01, m02, m10, m11, m12, b0, b1 = ego_tf
    return np.hypot(m00 * x + m01 * y + m02 * z + b0,
                    m10 * x + m11 * y + m12 * z + b1)


def _filter_records_by_class_range(query_records: List[dict]) -> Tuple[List[dict], int, int]:
    """Devkit filter_eval_boxes parity: remove GT *and predictions* beyond the
    per-class ego-distance range BEFORE matching (evaluate.py:113/116). Dropped
    predictions are neither TP nor FP. Note the BEV square reaches ~71 m in the
    corners, so this is material even for 50 m classes. With a per-record
    'ego_range_transform' the distance is the devkit's exact global-frame
    ego_dist and the comparison is strict (`ego_dist < range`, loaders.py:227);
    legacy records fall back to the lidar-origin xy norm with `<=` (documented
    ~1 m deviation). Records with unknown class use 50 m.
    Returns (filtered_records, n_preds_dropped, n_gts_dropped)."""
    out = []
    pred_dropped = gt_dropped = 0
    for rec in query_records:
        cname = _canonicalize_class_name(rec.get('cname')) or 'unknown'
        rng = DEVKIT_CLASS_RANGE.get(cname, 50.0)
        pb = rec['pred_boxes']
        gb = rec['gt_boxes']
        ego_tf = rec.get('ego_range_transform')
        pdist = _ego_dists(pb, rec.get('pred_attrs_3d'), ego_tf)
        gdist = _ego_dists(gb, rec.get('gt_attrs_3d'), ego_tf)
        if ego_tf is not None:
            # Intersection protocol: devkit rule (ego_dist < range, strict,
            # loaders.py:227) AND the generation-time rule (lidar_dist <= range
            # — the refer GT was generated with sensor-origin ranges, so the
            # ego-side annulus has no GT by construction; dropping predictions
            # there keeps the pred and GT universes identical instead of
            # scoring guaranteed FPs against structurally-absent GT).
            plid = np.sqrt(pb[:, 0].astype(np.float64) ** 2 + pb[:, 1].astype(np.float64) ** 2) \
                if pb.shape[0] else np.zeros(0, dtype=np.float64)
            glid = np.sqrt(gb[:, 0].astype(np.float64) ** 2 + gb[:, 1].astype(np.float64) ** 2) \
                if gb.shape[0] else np.zeros(0, dtype=np.float64)
            pkeep = (pdist < rng) & (plid <= rng)
            gkeep = (gdist < rng) & (glid <= rng)
        else:
            pkeep, gkeep = pdist <= rng, gdist <= rng   # legacy behaviour
        pred_dropped += int((~pkeep).sum())
        gt_dropped += int((~gkeep).sum())
        if pkeep.all() and gkeep.all():
            out.append(rec)
            continue
        new_rec = dict(rec)
        new_rec['pred_boxes'] = pb[pkeep]
        new_rec['pred_scores'] = rec['pred_scores'][pkeep]
        if rec.get('pred_attrs_3d') is not None:
            new_rec['pred_attrs_3d'] = rec['pred_attrs_3d'][pkeep]
        new_rec['gt_boxes'] = gb[gkeep]
        if rec.get('gt_attrs_3d') is not None:
            new_rec['gt_attrs_3d'] = rec['gt_attrs_3d'][gkeep]
        gcn = rec.get('gt_class_names', None)
        if gcn is not None:
            new_rec['gt_class_names'] = [c for c, k in zip(gcn, gkeep) if k]
        for key in rec:
            if key.startswith('_iou_cache_') or key == 'dist_matrix':
                mat = rec[key]
                new_rec[key] = mat[pkeep][:, gkeep] if mat is not None else None
        out.append(new_rec)
    return out, pred_dropped, gt_dropped


def _aggregate_tp_fp_fn(
    query_records: List[dict],
    dist_threshold: float,
    iou_mode: str,
) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    gt_total = 0
    det_total = 0

    for rec in query_records:
        pred_boxes = rec.get('pred_boxes', np.zeros((0, 4), dtype=np.float32))
        gt_boxes = rec.get('gt_boxes', np.zeros((0, 4), dtype=np.float32))

        gt_total += int(gt_boxes.shape[0])
        det_total += int(pred_boxes.shape[0])

        if pred_boxes.shape[0] == 0 and gt_boxes.shape[0] == 0:
            continue
        if pred_boxes.shape[0] == 0:
            fn += int(gt_boxes.shape[0])
            continue
        if gt_boxes.shape[0] == 0:
            fp += int(pred_boxes.shape[0])
            continue

        # 1. Calculate center distances
        dist_matrix = _center_distance(pred_boxes, gt_boxes)
        
        # 2. Sort predictions by confidence
        scores = rec.get('pred_scores', np.zeros(pred_boxes.shape[0]))
        pred_indices = np.arange(pred_boxes.shape[0])
        sorted_pred_indices = sorted(pred_indices.tolist(), key=lambda i: (scores[i], i), reverse=True)
        
        used_g = np.zeros(gt_boxes.shape[0], dtype=bool)
        tp_local = 0
        
        # 3. Match highest confidence to closest GT
        for pi in sorted_pred_indices:
            min_dist = np.inf
            match_gi = None
            
            for gi in range(gt_boxes.shape[0]):
                if not used_g[gi] and dist_matrix[pi, gi] < min_dist:
                    min_dist = dist_matrix[pi, gi]
                    match_gi = gi
            
            # dist_threshold is the center distance threshold (meters)
            if match_gi is not None and min_dist < dist_threshold:
                used_g[match_gi] = True
                tp_local += 1

        tp += int(tp_local)
        fp += int(pred_boxes.shape[0] - tp_local)
        fn += int(gt_boxes.shape[0] - tp_local)

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    return {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'precision': precision,
        'recall': recall,
        'num_gt': int(gt_total),
        'num_detections': int(det_total),
    }


def _compute_threshold_sweep(
    query_records: List[dict],
    thresholds: List[float],
    iou_mode: str,  # Kept for signature compatibility, but unused
    match_iou_threshold: float, # Acts as dist_threshold
) -> Dict[str, dict]:
    out = {}
    for thr in thresholds:
        thr_f = float(thr)
        recs_thr = _filter_records_by_confidence(query_records, thr_f)
        
        # Swapped to use nuScenes distance PR instead of IoU PR!
        pr05 = _compute_detection_pr_nuscenes(recs_thr, dist_threshold=0.5, max_curve_points=0)
        pr10 = _compute_detection_pr_nuscenes(recs_thr, dist_threshold=1.0, max_curve_points=0)
        pr20 = _compute_detection_pr_nuscenes(recs_thr, dist_threshold=2.0, max_curve_points=0)
        pr40 = _compute_detection_pr_nuscenes(recs_thr, dist_threshold=4.0, max_curve_points=0)
        
        det_stats = _aggregate_tp_fp_fn(recs_thr, dist_threshold=match_iou_threshold, iou_mode=iou_mode)
        
        out[f'{thr_f:.2f}'] = {
            'AP@0.5m': float(pr05['ap']), 
            'AP@1.0m': float(pr10['ap']),
            'AP@2.0m': float(pr20['ap']),
            'AP@4.0m': float(pr40['ap']),
            'mAP': float((pr10['ap'] + pr20['ap']) / 2.0),
            'tp': int(det_stats['tp']),
            'fp': int(det_stats['fp']),
            'fn': int(det_stats['fn']),
            'precision': float(det_stats['precision']),
            'recall': float(det_stats['recall']),
            'num_gt': int(det_stats['num_gt']),
            'num_detections': int(det_stats['num_detections']),
        }
    return out


def _safe_slug(text: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(text)).strip('._')
    return slug if slug else 'unknown'


def _save_pr_curve_plot(
    out_path: Path,
    title: str,
    curves: Dict[str, Dict[str, Any]],  # e.g., {'Dist 0.5m': pr_05, 'Dist 2.0m': pr_20}
    threshold_metrics: Dict[str, dict],
) -> bool:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'[PR] matplotlib unavailable, skip plot {out_path.name}: {exc}')
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # Dynamically plot whichever curves are passed in
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (label, curve_data) in enumerate(curves.items()):
        c = curve_data.get('curve', {}) if isinstance(curve_data, dict) else {}
        r = c.get('recall', [])
        p = c.get('precision', [])
        if r and p:
            ap = float(curve_data.get("ap", 0.0))
            color = colors[idx % len(colors)]
            ax.plot(r, p, lw=2.0, color=color, label=f'{label} (AP={ap:.3f})')

    for thr_key, row in sorted(threshold_metrics.items(), key=lambda x: x[0]):
        p = row.get('precision', None)
        r = row.get('recall', None)
        if p is None or r is None:
            continue
        ax.scatter([float(r)], [float(p)], s=28, marker='o', label=f'Score Thr={thr_key}')

    ax.set_title(title)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _build_lookup_from_dataset_sources(
    dataset,
    legacy_infos: Optional[List[dict]],
    data_root: str,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Create token/basename lookup to reuse 10-sweep pipeline for refer entries."""
    by_token: Dict[str, dict] = {}
    by_basename: Dict[str, dict] = {}

    if dataset is not None:
        for i in range(len(dataset)):
            info = dataset.get_data_info(i)
            tok = info.get('token')
            lidar_path = info.get('lidar_path') or info.get('lidar_points', {}).get('lidar_path', '')
            base = os.path.basename(str(lidar_path))
            if tok:
                by_token[str(tok)] = info
            if base:
                by_basename[base] = info
        return by_token, by_basename

    for info in legacy_infos or []:
        converted = _legacy_info_to_pipeline_input(info, data_root)
        tok = converted.get('token')
        lidar_path = converted.get('lidar_points', {}).get('lidar_path', '')
        base = os.path.basename(str(lidar_path))
        if tok:
            by_token[str(tok)] = converted
        if base:
            by_basename[base] = converted

    return by_token, by_basename


def _gt_boxes_from_targets(targets: List[dict]) -> np.ndarray:
    boxes = []
    for t in targets:
        center = t.get('center_sensor', [None, None, None])
        extents = t.get('extents_lwh', [None, None, None])
        if center[0] is None or center[1] is None or extents[0] is None or extents[1] is None:
            continue
        # Keep [cx, cy, w, l] to match model decoding.
        boxes.append([float(center[0]), float(center[1]), float(extents[1]), float(extents[0])])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def _gt_attrs_3d_from_targets(targets: List[dict]) -> np.ndarray:
    attrs = []
    for t in targets:
        center = t.get('center_sensor', [None, None, None])
        extents = t.get('extents_lwh', [None, None, None])
        yaw_deg = t.get('yaw_deg', None)
        if center[2] is None or extents[2] is None or yaw_deg is None:
            continue
        yaw = np.deg2rad(float(yaw_deg))
        attrs.append([float(center[2]), float(extents[2]), float(np.sin(yaw)), float(np.cos(yaw))])
    if not attrs:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(attrs, dtype=np.float32)


def _gt_class_names_from_targets(targets: List[dict]) -> List[str]:
    """Per-object nuScenes class, aligned 1:1 with ``_gt_boxes_from_targets``.

    Uses the IDENTICAL filter as the box extractor (BEV center + extents) so
    ``gt_class_names[i]`` corresponds to ``gt_boxes[i]``. Needed for the
    nuScenes TP-metric class rules in ``_calc_yaw_error`` (cone orientation is
    excluded, barrier orientation is only defined mod pi). Unmappable labels
    fall back to 'unknown' (2*pi period, never excluded).
    """
    names: List[str] = []
    for t in targets:
        center = t.get('center_sensor', [None, None, None])
        extents = t.get('extents_lwh', [None, None, None])
        if center[0] is None or center[1] is None or extents[0] is None or extents[1] is None:
            continue
        names.append(_canonicalize_class_name(t.get('class')) or 'unknown')
    return names


def _build_gt_class_map(args) -> Dict[bytes, List[str]]:
    """Reload the refer eval split and map gt_boxes-content -> per-object class.

    Used by the augment-merge path to retro-fit ``gt_class_names`` onto shard
    records saved before that field existed. No model / features / points are
    touched — only ``targets`` are read. The key is the exact bytes of the
    ``_gt_boxes_from_targets`` output, which is deterministic, so it matches the
    ``gt_boxes`` already stored in each record bit-for-bit. (Two queries with
    identical gt_boxes necessarily reference the same physical objects, hence
    the same classes, so a key collision is harmless.)
    """
    eval_ds_args = SimpleNamespace(
        refer_data_dir=args.refer_data_dir, nuscenes_dataroot=args.data_root,
        nuscenes_ann_file=args.ann_file, sweeps_num=args.sweeps_num,
        point_cloud_range=args.point_cloud_range, question_types_json=None,
        question_types=None, queries_per_frame=1, precompute_bev=False,
        num_point_features=4, voxel_size=[0.08, 0.08, 4.0], lidar_root=None,
        backend_args=None, feature_cache_dir=None, feature_cache_strict=False,
        gt_blacklist=getattr(args, 'gt_blacklist', None),
        ego_range_transform=getattr(args, 'ego_range_transform', ''),
    )
    ds = build_refer_dataset(args.eval_split, eval_ds_args)
    idxs = getattr(ds, 'valid_indices', None)
    if idxs is None:
        idxs = range(len(ds.frames))
    gtmap: Dict[bytes, List[str]] = {}
    n_q = 0
    for flat_idx in idxs:
        for rq in ds.frame_refer_queries[flat_idx]:
            tgts = rq.get('targets', [])
            boxes = _gt_boxes_from_targets(tgts)
            names = _gt_class_names_from_targets(tgts)
            if boxes.shape[0] != len(names):
                continue
            gtmap[boxes.astype(np.float32).tobytes()] = names
            n_q += 1
    print(f"[Augment] gt-class map: {len(list(idxs))} frames, {n_q} queries, "
          f"{len(gtmap)} unique gt-box sets")
    return gtmap


def _augment_records_with_gt_class(per_type_records, per_class_records, gtmap) -> None:
    """Inject per-object ``gt_class_names`` into loaded shard records in place."""
    hit = miss = already = 0
    seen = set()
    for bucket in (per_type_records, per_class_records):
        for recs in bucket.values():
            for r in recs:
                rid = id(r)
                if rid in seen:
                    continue
                seen.add(rid)
                if r.get('gt_class_names') is not None:
                    already += 1
                    continue
                gb = np.asarray(r['gt_boxes'], dtype=np.float32)
                names = gtmap.get(gb.tobytes())
                if names is not None and len(names) == gb.shape[0]:
                    r['gt_class_names'] = names
                    hit += 1
                else:
                    miss += 1
    print(f"[Augment] injected per-object class into records: "
          f"matched={hit} already_present={already} unmatched={miss}")
    if miss:
        print(f"[Augment] WARNING: {miss} records unmatched -> those fall back to "
              f"class 'unknown' (no cone/barrier handling). Check refer-data-dir/eval-split.")


def _load_question_type_filter(question_types_json: Optional[str]) -> Optional[set]:
    if not question_types_json:
        return None

    cand = Path(question_types_json)
    if not cand.is_absolute():
        local = THIS_DIR / cand
        repo = REPO_ROOT / cand
        if local.exists():
            cand = local
        elif repo.exists():
            cand = repo

    if not cand.exists():
        print(f'[Eval] question_types_json not found: {cand}. Using all question types.')
        return None

    with open(cand, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return set(str(x) for x in data)
    if isinstance(data, dict):
        if isinstance(data.get('question_types', None), list):
            return set(str(x) for x in data['question_types'])
        # Fallback: treat keys as types.
        return set(str(k) for k in data.keys())

    print(f'[Eval] Unsupported question_types_json format in {cand}. Using all question types.')
    return None


def _load_checkpoint_if_any(model: torch.nn.Module, ckpt_path: Optional[str]) -> None:
    if not ckpt_path:
        return
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[Model CKPT] Loaded: {ckpt_path}')
    if missing:
        # Older checkpoints (baseline/intermediate) may not include quality head.
        expected_prefixes = ('quality_embed.',)
        expected_missing = [k for k in missing if k.startswith(expected_prefixes)]
        other_missing = [k for k in missing if not k.startswith(expected_prefixes)]

        if expected_missing:
            print(
                f'[Model CKPT] Missing keys (expected, older ckpt): '
                f'{len(expected_missing)} quality-head params'
            )
        if other_missing:
            print(f'[Model CKPT] Missing keys (unexpected): {len(other_missing)}')
            print(f'[Model CKPT] Unexpected-missing sample: {other_missing[:8]}')
    if unexpected:
        print(f'[Model CKPT] Unexpected keys: {len(unexpected)}')


def _canonicalize_class_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip().lower()
    if not s:
        return None

    det_classes = set(NUSCENES_DET_CLASSES)

    alias = {
        'bus': 'bus',
        'cars': 'car',
        'trucks': 'truck',
        'construction vehicle': 'construction_vehicle',
        'construction vehicles': 'construction_vehicle',
        'construction_vehicle': 'construction_vehicle',
        'buses': 'bus',
        'trailers': 'trailer',
        'barriers': 'barrier',
        'motorcycles': 'motorcycle',
        'bicycles': 'bicycle',
        'pedestrians': 'pedestrian',
        'traffic cone': 'traffic_cone',
        'traffic cones': 'traffic_cone',
        'traffic_cone': 'traffic_cone',
        # Full nuScenes names (if any leak into refer JSON).
        'vehicle.car': 'car',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.trailer': 'trailer',
        'movable_object.barrier': 'barrier',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
    }

    if s in alias:
        return alias[s]

    s = s.replace('-', '_').replace(' ', '_')

    # Keep exact class labels before any plural/suffix normalization.
    if s in det_classes:
        return s

    # Conservative singularization to avoid corrupting labels like "bus".
    if s.endswith('s') and s not in {'bus'}:
        s = s[:-1]
    if s == 'trafficcone':
        return 'traffic_cone'
    if s == 'constructionvehicle':
        return 'construction_vehicle'
    if s in det_classes:
        return s
    return None


def _infer_class_id_from_query_record(rq: dict, class_to_id: Dict[str, int]) -> Optional[int]:
    # 1) Direct class-like fields on query record.
    for key in ('class', 'category', 'target_class', 'object_class', 'label'):
        cname = _canonicalize_class_name(rq.get(key))
        if cname in class_to_id:
            return class_to_id[cname]

    # 2) Infer from query targets; this is the most reliable path for this dataset.
    target_classes = set()
    for t in rq.get('targets', []):
        cname = _canonicalize_class_name(t.get('class'))
        if cname in class_to_id:
            target_classes.add(cname)
    if len(target_classes) == 1:
        cname = next(iter(target_classes))
        return class_to_id[cname]

    # 3) Fallback to query text parsing.
    q = str(rq.get('query', '')).lower().replace('-', ' ').replace('_', ' ')
    if q:
        for cname in NUSCENES_DET_CLASSES:
            terms = {cname, cname.replace('_', ' ')}
            if cname.endswith('y'):
                terms.add(cname[:-1] + 'ies')
            else:
                terms.add(cname + 's')
                terms.add(cname.replace('_', ' ') + 's')
            if any(term in q for term in terms):
                return class_to_id[cname]

    return None


def _targets_to_classed_arrays(targets: List[dict], class_to_id: Dict[str, int]):
    boxes, attrs, cls_ids, tokens = [], [], [], []
    for t in targets:
        center = t.get('center_sensor', [None, None, None])
        extents = t.get('extents_lwh', [None, None, None])
        yaw_deg = t.get('yaw_deg', None)
        cname = _canonicalize_class_name(t.get('class'))
        if cname not in class_to_id:
            continue
        if (
            center[0] is None or center[1] is None or center[2] is None
            or extents[0] is None or extents[1] is None or extents[2] is None
            or yaw_deg is None
        ):
            continue

        boxes.append([float(center[0]), float(center[1]), float(extents[1]), float(extents[0])])
        yaw = np.deg2rad(float(yaw_deg))
        attrs.append([float(center[2]), float(extents[2]), float(np.sin(yaw)), float(np.cos(yaw))])
        cls_ids.append(int(class_to_id[cname]))
        tok = t.get('token')
        if tok is None or tok == '':
            # Stable fallback key if token is missing in source JSON.
            tok = (
                f'no_token_{cname}_'
                f'{float(center[0]):.3f}_{float(center[1]):.3f}_{float(center[2]):.3f}_'
                f'{float(extents[0]):.3f}_{float(extents[1]):.3f}_{float(extents[2]):.3f}'
            )
        tokens.append(str(tok))

    if not boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )
    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(attrs, dtype=np.float32),
        np.asarray(cls_ids, dtype=np.int64),
        tokens,
    )


def _write_confusion_matrix_csv(path: Path, matrix: np.ndarray, labels: List[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['gt\\pred'] + labels)
        for i, row_name in enumerate(labels):
            writer.writerow([row_name] + [int(x) for x in matrix[i].tolist()])


def _build_eval_frame_model_inputs(eval_dataset, entry: dict, device: torch.device) -> Tuple[dict, bool]:
    """Return model inputs for one frame, mirroring dataset cache key layout."""
    data = {
        'points': None,
        'centerpoint_srcs': None,
        'centerpoint_props': None,
        'centerpoint_scores': None,
        'pointpillars_srcs': None,
        'pointpillars_props': None,
        'pointpillars_scores': None,
        'pointpillars_yaw': None,
    }

    cached = None
    if getattr(eval_dataset, 'feature_cache_dir', None):
        cached = eval_dataset._load_cached_features(entry)

    if cached is not None:
        detector = getattr(eval_dataset, '_feature_cache_detector', 'centerpoint')
        srcs = [lvl.to(device, non_blocking=True) for lvl in cached['srcs']]
        props = cached['props'].to(device, non_blocking=True)
        scores = cached['scores'].to(device, non_blocking=True)
        yaw = cached.get('yaw')

        if detector == 'pointpillars':
            data['pointpillars_srcs'] = [srcs]
            data['pointpillars_props'] = [props]
            data['pointpillars_scores'] = [scores]
            if yaw is not None:
                data['pointpillars_yaw'] = [yaw.to(device, non_blocking=True)]
        else:
            data['centerpoint_srcs'] = [srcs]
            data['centerpoint_props'] = [props]
            data['centerpoint_scores'] = [scores]
        return data, True

    points = eval_dataset._load_points_with_sweeps(entry).cpu().to(device)
    data['points'] = [points]
    return data, False


def run(args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    device = torch.device(args.device)
    if device.type == 'cuda' and device.index not in (None, 0):
        # MSDeformAttn's CUDA op assumes the CURRENT device holds its
        # tensors; without this, --device cuda:1 hits an illegal memory
        # access inside the deformable-attention kernel.
        torch.cuda.set_device(device)

    # MMDetection3D compatibility:
    # - older versions return LiDARPoints (has .tensor)
    # - newer versions may return torch.Tensor directly
    def _extract_points_tensor(packed_data):
        pts = packed_data['inputs']['points']
        return pts.tensor if hasattr(pts, 'tensor') else pts

    model_args = _build_model_args(args)
    model, _, _ = build_model(model_args)
    _load_checkpoint_if_any(model, args.model_ckpt)
    model.to(device)
    model.eval()

    dataset, legacy_infos = _build_nus_dataset(args)
    index = _find_sample_index(dataset, legacy_infos, args.sample_token, args.lidar_suffix)
    if dataset is not None:
        raw_info = dataset.get_data_info(index)
    else:
        raw_info = _legacy_info_to_pipeline_input(legacy_infos[index], args.data_root)

    sweep_count = len(raw_info.get('lidar_sweeps', raw_info.get('sweeps', [])))
    print(f'[Sweep Check] raw_sweeps_available={sweep_count}')

    num_point_features = 5 if args.meta_arch in {'refer_model_second', 'refer_model_second_v2', 'refer_model_second_v3'} else 4
    pipeline = _build_10_sweep_pipeline(args.sweeps_num, num_point_features=num_point_features)
    packed = pipeline(raw_info)
    points = _extract_points_tensor(packed).to(device)

    lag = points[:, 3].detach().cpu()
    nonzero_ratio = float((lag != 0).float().mean().item())
    print(f'[Sweep Check] sweeps_num={args.sweeps_num}')
    print(f'[Sweep Check] points_total={points.shape[0]}')
    print(f'[Sweep Check] nonzero_time_lag_ratio={nonzero_ratio:.4f}')

    with torch.no_grad():
        outputs = model(
            {
                'points': [points],
                'sentences': [args.text],
            }
        )

    pred_boxes = outputs['pred_boxes'][0]
    pred_ref = outputs['pred_refers'][0, :, 0]
    pred_cls = outputs['pred_logits'][0, :, 0]

    topk = min(args.topk, pred_boxes.shape[0])
    order = torch.argsort(pred_ref, descending=True)[:topk]

    top_norm = pred_boxes[order].detach().cpu()
    top_meter = _decode_boxes_to_meters(top_norm, args.point_cloud_range)
    top_ref = pred_ref[order].detach().cpu()
    top_cls = pred_cls[order].detach().cpu()

    print('\nTop predictions (sorted by refer score):')
    for i in range(topk):
        cx, cy, w, l = top_meter[i].tolist()
        print(
            f'  #{i+1:02d} '
            f'ref={top_ref[i]:+.4f} cls={top_cls[i]:+.4f} '
            f'cx={cx:+.2f}m cy={cy:+.2f}m w={w:.2f}m l={l:.2f}m'
        )

    return outputs


def evaluate(args: argparse.Namespace) -> None:
    import pickle
    import math
    device = torch.device(args.device)
    if device.type == 'cuda' and device.index not in (None, 0):
        # MSDeformAttn's CUDA op assumes the CURRENT device holds its
        # tensors; without this, --device cuda:1 hits an illegal memory
        # access inside the deformable-attention kernel.
        torch.cuda.set_device(device)

    # Standard setup
    eval_iou_mode = 'distance'
    dist_thr = float(args.dist_threshold)
    conf_mode = args.map_conf_mode
    min_conf = float(args.map_min_conf)
    use_all_proposals = bool(getattr(args, 'eval_use_all_proposals', True))
    pr_thresholds = [float(x) for x in getattr(args, 'pr_thresholds', [0.3, 0.5, 0.7])]
    if not pr_thresholds: pr_thresholds = [0.3, 0.5, 0.7]
    pr_curve_max_points = max(1, int(getattr(args, 'pr_curve_max_points', 400)))
    save_pr_plots = bool(getattr(args, 'save_pr_plots', True))

    class_names = list(NUSCENES_DET_CLASSES)
    class_to_id = {name: i for i, name in enumerate(class_names)}

    per_type_records: Dict[str, List[dict]] = defaultdict(list)
    per_class_records: Dict[str, List[dict]] = defaultdict(list) # NEW: Track by class
    per_type_tp_fp_fn: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            'tp': 0, 'fp': 0, 'fn': 0, 'queries': 0,
            'center_hits_0_5m': 0, 'center_hits_1_0m': 0, 'center_hits_2_0m': 0,
            'eval_gt_count': 0, 'center_dist_sum': 0.0, 'center_dist_cnt': 0,
        }
    )
    processed_queries = skipped = cache_hit_frames = cache_miss_frames = total_queries = 0

    # ---------------- MERGE MODE ----------------
    allowed_question_types = None 
    num_entries = 0
    if args.merge_predictions:
        print(f"Merging {len(args.merge_predictions)} prediction shards...")
        # The merge is a short-lived batch job that retains the full ~60GB shard
        # heap (hundreds of millions of live objects). Python's cyclic GC then
        # re-scans that heap on every per-(type,threshold,class) PR pass, which
        # made later passes 4-5x slower than identical earlier ones. Disabling it
        # is numerically inert (non-cyclic tuples/ndarrays are freed by refcount).
        import gc
        gc.disable()
        for pkl_file in args.merge_predictions:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            for k, v in data['per_type_records'].items(): per_type_records[k].extend(v)
            for k, v in data['per_class_records'].items(): per_class_records[k].extend(v)
            for k, v in data['per_type_tp_fp_fn'].items():
                for mk, mv in v.items(): per_type_tp_fp_fn[k][mk] += mv
            total_queries += data['total_queries']
            skipped += data['skipped']
            cache_hit_frames += data['cache_hit_frames']
            cache_miss_frames += data['cache_miss_frames']
            num_entries = data.get('num_entries', 0)

        if getattr(args, 'augment_gt_class', False):
            print("[Augment] retro-fitting per-object GT class onto shard records "
                  "(no inference; reloading refer targets)...")
            gtmap = _build_gt_class_map(args)
            _augment_records_with_gt_class(per_type_records, per_class_records, gtmap)

        # Devkit parity: NO confidence floor by default. The devkit keeps every
        # submitted box (<=500/sample; we have 150/query), and an 0.01 floor was
        # measured to move AP by up to ~0.009 (barrier) — NOT negligible.
        # --merge-min-conf > 0 remains available for quick previews only.
        _merge_floor = float(getattr(args, 'merge_min_conf', 0.0) or 0.0)
        if _merge_floor > 0.0:
            print(f"[Merge] WARNING: confidence floor {_merge_floor} (preview mode; "
                  f"deviates from devkit — final numbers must use 0)")
            for k in per_type_records.keys():
                per_type_records[k] = _filter_records_by_confidence(per_type_records[k], _merge_floor)
            for k in per_class_records.keys():
                per_class_records[k] = _filter_records_by_confidence(per_class_records[k], _merge_floor)

        _nms_radius = float(getattr(args, 'merge_nms_radius', 0.0) or 0.0)
        if _nms_radius > 0.0:
            print(f"[Merge] DIAGNOSTIC center-distance NMS radius={_nms_radius}m "
                  f"(reported eval is NMS-free; this is a what-if pass, not the protocol)")
            for k in list(per_type_records.keys()):
                per_type_records[k] = _nms_records_by_center(per_type_records[k], _nms_radius)
            for k in list(per_class_records.keys()):
                per_class_records[k] = _nms_records_by_center(per_class_records[k], _nms_radius)
    else:
        # ---------------- INFERENCE MODE ----------------
        model_args = _build_model_args(args)
        model, _, _ = build_model(model_args)
        _load_checkpoint_if_any(model, args.model_ckpt)
        model.to(device)
        model.eval()

        eval_ds_args = SimpleNamespace(
            refer_data_dir=args.refer_data_dir, nuscenes_dataroot=args.data_root, nuscenes_ann_file=args.ann_file,
            sweeps_num=args.sweeps_num, point_cloud_range=args.point_cloud_range, question_types_json=None,
            question_types=None, queries_per_frame=1, precompute_bev=False,
            num_point_features=5 if args.meta_arch in {'refer_model_second', 'refer_model_second_v2', 'refer_model_second_v3'} else 4,
            voxel_size=[0.08, 0.08, 4.0], lidar_root=None, backend_args=None,
            feature_cache_dir=args.feature_cache_dir, feature_cache_strict=args.feature_cache_strict,
            gt_blacklist=getattr(args, 'gt_blacklist', None),
            ego_range_transform=getattr(args, 'ego_range_transform', ''),
        )
        eval_dataset = build_refer_dataset(args.eval_split, eval_ds_args)
        num_entries = len(eval_dataset.frames)
        allowed_question_types = _load_question_type_filter(args.question_types_json)

        # Apply Sharding
        if args.num_shards > 1:
            chunk_size = math.ceil(len(eval_dataset.valid_indices) / args.num_shards)
            start_idx = args.shard_id * chunk_size
            end_idx = min((args.shard_id + 1) * chunk_size, len(eval_dataset.valid_indices))
            eval_dataset.valid_indices = eval_dataset.valid_indices[start_idx:end_idx]
            print(f"[Shard {args.shard_id+1}/{args.num_shards}] Evaluating {len(eval_dataset.valid_indices)} frames.")

        for flat_idx in eval_dataset.valid_indices:
            for rq in eval_dataset.frame_refer_queries[flat_idx]:
                qtype = rq.get('query_type', 'unknown')
                if allowed_question_types is not None and qtype not in allowed_question_types: continue
                total_queries += 1

        t0 = time.time()
        chunk_size = int(getattr(args, 'eval_chunk_size', 32))

        for flat_idx in eval_dataset.valid_indices:
            entry = eval_dataset.frames[flat_idx]
            frame_queries = eval_dataset.frame_refer_queries[flat_idx]
            if not frame_queries: continue

            if allowed_question_types is not None:
                frame_queries = [rq for rq in frame_queries if rq.get('query_type', 'unknown') in allowed_question_types]
            if not frame_queries: continue

            frame_model_inputs_base, cache_hit = _build_eval_frame_model_inputs(eval_dataset, entry, device)
            if cache_hit: cache_hit_frames += 1
            else: cache_miss_frames += 1

            for i in range(0, len(frame_queries), chunk_size):
                chunk_rqs = frame_queries[i:i + chunk_size]
                sentences = [rq.get('query', '') for rq in chunk_rqs]

                with torch.no_grad():
                    model_inputs = {'sentences': [sentences]}
                    model_inputs.update(frame_model_inputs_base)
                    if getattr(args, 'no_proposal_yaw', False):
                        # GUARD: legacy angle-blind checkpoints (v1 / +lang
                        # trained BEFORE the classyaw cache) must not see
                        # proposal yaw at eval — otherwise the shared decoder
                        # silently switches on rotated sampling (ref_angles),
                        # a train/eval mismatch. Strip it so eval stays blind.
                        model_inputs['pointpillars_yaw'] = None
                    outputs = model(model_inputs)

                for j, rq in enumerate(chunk_rqs):
                    processed_queries += 1
                    sentence = sentences[j]
                    if not sentence:
                        skipped += 1
                        continue

                    pred_boxes = outputs['pred_boxes'][j].detach().cpu().numpy()
                    pred_scores = torch.sigmoid(outputs['pred_logits'][j, :, 0]).detach().cpu().numpy()
                    pred_refers = torch.sigmoid(outputs['pred_refers'][j, :, 0]).detach().cpu().numpy()
                    pred_attrs_3d_all = _extract_pred_attrs_3d(outputs, j, args.point_cloud_range)

                    if conf_mode == 'score': conf = pred_scores
                    elif conf_mode == 'refer': conf = pred_refers
                    else: conf = pred_scores * pred_refers

                    if use_all_proposals: conf_mask = np.ones(conf.shape[0], dtype=bool)
                    else: conf_mask = conf >= min_conf
                    
                    pred_boxes = pred_boxes[conf_mask]
                    conf = conf[conf_mask]
                    pred_attrs_3d = pred_attrs_3d_all[conf_mask] if pred_attrs_3d_all is not None else None

                    if pred_boxes.shape[0] > 0:
                        pred_boxes_m = _decode_boxes_to_meters(torch.from_numpy(pred_boxes), args.point_cloud_range).numpy()
                    else:
                        pred_boxes_m = np.zeros((0, 4), dtype=np.float32)

                    gt_boxes_m = _gt_boxes_from_targets(rq.get('targets', []))
                    gt_attrs_3d = _gt_attrs_3d_from_targets(rq.get('targets', []))
                    gt_class_names = _gt_class_names_from_targets(rq.get('targets', []))
                    qtype = rq.get('query_type', 'unknown')
                    
                    # Extract Class Name for side-by-side metrics
                    cid = _infer_class_id_from_query_record(rq, class_to_id)
                    cname = class_names[cid] if cid is not None else 'unknown_class'

                    dist_mat = _center_distance(pred_boxes_m, gt_boxes_m)

                    record = {
                        'pred_boxes': pred_boxes_m,
                        'pred_scores': conf.astype(np.float32),
                        'gt_boxes': gt_boxes_m,
                        'pred_attrs_3d': pred_attrs_3d,
                        'gt_attrs_3d': gt_attrs_3d,
                        'gt_class_names': gt_class_names,  # per-object nuScenes class (aligned w/ gt_boxes)
                        'dist_matrix': dist_mat,
                        'qtype': qtype,
                        'cname': cname,
                        # devkit ego_dist transform for exact class-range
                        # filtering at merge time (None on legacy caches).
                        'ego_range_transform': eval_dataset.frame_ego_transform[flat_idx]
                        if hasattr(eval_dataset, 'frame_ego_transform') else None,
                    }

                    
                    per_type_records[qtype].append(record)
                    per_class_records[cname].append(record) # NEW: Save to class tracker

                    # --- Per-query TP/FP/FN tracking (correctly indented inside the j-loop) ---
                    per_type_tp_fp_fn[qtype]['queries'] += 1
                    per_type_tp_fp_fn[qtype]['eval_gt_count'] += int(gt_boxes_m.shape[0])

                    if pred_boxes_m.shape[0] == 0 and gt_boxes_m.shape[0] == 0:
                        pass
                    elif pred_boxes_m.shape[0] == 0:
                        per_type_tp_fp_fn[qtype]['fn'] += int(gt_boxes_m.shape[0])
                    elif gt_boxes_m.shape[0] == 0:
                        per_type_tp_fp_fn[qtype]['fp'] += int(pred_boxes_m.shape[0])
                    else:
                        # Pure nuScenes distance matching tracking
                        dist_matrix = _center_distance(pred_boxes_m, gt_boxes_m)

                        # Sort predictions by confidence score to mimic nuScenes
                        pred_indices = np.arange(pred_boxes_m.shape[0])
                        sorted_pred_indices = sorted(pred_indices.tolist(), key=lambda i: (conf[i], i), reverse=True)

                        # Track hits at 0.5m, 1.0m, and 2.0m distances (Just for secondary tracking)
                        for thr, key in [(0.5, 'center_hits_0_5m'), (1.0, 'center_hits_1_0m'), (2.0, 'center_hits_2_0m')]:
                            used_g = np.zeros(gt_boxes_m.shape[0], dtype=bool)
                            hits = 0
                            for pi in sorted_pred_indices:
                                min_dist = np.inf
                                match_gi = None
                                for gi in range(gt_boxes_m.shape[0]):
                                    if not used_g[gi] and dist_matrix[pi, gi] < min_dist:
                                        min_dist = dist_matrix[pi, gi]
                                        match_gi = gi
                                if match_gi is not None and min_dist < thr:
                                    used_g[match_gi] = True
                                    hits += 1
                            per_type_tp_fp_fn[qtype][key] += hits

                        # For the main Precision/Recall calculation (Using your dist_thr)
                        used_g = np.zeros(gt_boxes_m.shape[0], dtype=bool)
                        tp_local = 0
                        matched_pairs = []

                        for pi in sorted_pred_indices:
                            min_dist = np.inf
                            match_gi = None
                            for gi in range(gt_boxes_m.shape[0]):
                                if not used_g[gi] and dist_matrix[pi, gi] < min_dist:
                                    min_dist = dist_matrix[pi, gi]
                                    match_gi = gi

                            if match_gi is not None and min_dist < dist_thr:
                                used_g[match_gi] = True
                                tp_local += 1
                                matched_pairs.append((pi, match_gi))


                        per_type_tp_fp_fn[qtype]['tp'] += tp_local
                        per_type_tp_fp_fn[qtype]['fp'] += int(pred_boxes_m.shape[0] - tp_local)
                        per_type_tp_fp_fn[qtype]['fn'] += int(gt_boxes_m.shape[0] - tp_local)

                        if tp_local > 0:
                            for pi, gi in matched_pairs:
                                per_type_tp_fp_fn[qtype]['center_dist_sum'] += float(dist_matrix[pi, gi])
                                per_type_tp_fp_fn[qtype]['center_dist_cnt'] += 1

                if processed_queries % 200 == 0 or processed_queries == total_queries:
                    elapsed = time.time() - t0
                    qps = processed_queries / max(elapsed, 1e-6)
                    print(f'[Eval] {processed_queries}/{total_queries} queries, {qps:.1f} q/s')

        # IF SAVING SHARD, DUMP AND EXIT
        if args.save_predictions:
            dump_data = {
                'per_type_records': dict(per_type_records),
                'per_class_records': dict(per_class_records),
                'per_type_tp_fp_fn': dict(per_type_tp_fp_fn),
                'total_queries': total_queries,
                'skipped': skipped,
                'cache_hit_frames': cache_hit_frames,
                'cache_miss_frames': cache_miss_frames,
                'num_entries': len(eval_dataset.frames)
            }
            with open(args.save_predictions, 'wb') as f:
                pickle.dump(dump_data, f)
            print(f"Saved predictions to {args.save_predictions}")
            return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pr_plot_dir = output_dir / f'pr_curves_eval_{args.eval_split}'

    per_type_results = {}
    overall_records = []
    total_tp = total_fp = total_fn = 0



    # Qtypes where the headline mAP should be the unweighted mean of per-class APs
    # (mirrors the nuScenes detection mAP convention).
    # ── Devkit class-range filter (evaluate.py:113/116 parity) ──
    # Remove GT and predictions beyond the per-class ego range BEFORE matching.
    # Applied at metric time so saved shards stay raw. Default ON.
    class_range_filter = not bool(getattr(args, 'no_class_range_filter', False))
    if class_range_filter:
        _tot_p = _tot_g = 0
        for k in list(per_type_records.keys()):
            per_type_records[k], _p, _g = _filter_records_by_class_range(per_type_records[k])
            _tot_p += _p; _tot_g += _g
        for k in list(per_class_records.keys()):
            per_class_records[k], _, _ = _filter_records_by_class_range(per_class_records[k])
        print(f'[Eval] class-range filter (devkit parity): dropped {_tot_p:,} preds, '
              f'{_tot_g:,} GTs beyond per-class range (per-type buckets)', flush=True)

    CLASS_AVG_QTYPES = {
        'object_detection',
        'object_detection_all_category',
        'object_detection_closest',
        'object_detection_closest_in_sector',
    }

    for qtype in sorted(per_type_records.keys()):
        recs = per_type_records[qtype]
        stats = per_type_tp_fp_fn[qtype]

        # Group this qtype's records by inferred class (cname stored on each record).
        class_grouped = defaultdict(list)
        for rec in recs:
            class_grouped[rec.get('cname', 'unknown_class')].append(rec)

        # Per-class AP breakdown (populated only for CLASS_AVG_QTYPES); persisted below.
        per_class_ap = {}

        if qtype in CLASS_AVG_QTYPES and len([k for k in class_grouped.keys() if k != 'unknown_class']) > 1:
            # Per-class AP, then unweighted mean across classes (skip unknown_class).
            per_thr_aps = {0.5: [], 1.0: [], 2.0: [], 4.0: []}
            for cname, crecs in class_grouped.items():
                if cname == 'unknown_class':
                    continue
                pr_c_05 = _compute_detection_pr_nuscenes(crecs, dist_threshold=0.5, max_curve_points=pr_curve_max_points)
                pr_c_10 = _compute_detection_pr_nuscenes(crecs, dist_threshold=1.0, max_curve_points=pr_curve_max_points)
                pr_c_20 = _compute_detection_pr_nuscenes(crecs, dist_threshold=2.0, max_curve_points=pr_curve_max_points)
                pr_c_40 = _compute_detection_pr_nuscenes(crecs, dist_threshold=4.0, max_curve_points=pr_curve_max_points)
                per_thr_aps[0.5].append(float(pr_c_05['ap']))
                per_thr_aps[1.0].append(float(pr_c_10['ap']))
                per_thr_aps[2.0].append(float(pr_c_20['ap']))
                per_thr_aps[4.0].append(float(pr_c_40['ap']))
                per_class_ap[cname] = {
                    'AP@0.5m': float(pr_c_05['ap']),
                    'AP@1.0m': float(pr_c_10['ap']),
                    'AP@2.0m': float(pr_c_20['ap']),
                    'AP@4.0m': float(pr_c_40['ap']),
                    'mAP': float((pr_c_05['ap'] + pr_c_10['ap'] + pr_c_20['ap'] + pr_c_40['ap']) / 4.0),
                    'num_gt': int(pr_c_40['num_gt']),
                    'num_detections': int(pr_c_40['num_detections']),
                }
            ap_05 = float(np.mean(per_thr_aps[0.5])) if per_thr_aps[0.5] else 0.0
            ap_10 = float(np.mean(per_thr_aps[1.0])) if per_thr_aps[1.0] else 0.0
            ap_20 = float(np.mean(per_thr_aps[2.0])) if per_thr_aps[2.0] else 0.0
            ap_40 = float(np.mean(per_thr_aps[4.0])) if per_thr_aps[4.0] else 0.0
            # Lumped PR dicts only for plotting / curve export.
            pr_05 = _compute_detection_pr_nuscenes(recs, dist_threshold=0.5, max_curve_points=pr_curve_max_points)
            pr_10 = _compute_detection_pr_nuscenes(recs, dist_threshold=1.0, max_curve_points=pr_curve_max_points)
            pr_20 = _compute_detection_pr_nuscenes(recs, dist_threshold=2.0, max_curve_points=pr_curve_max_points)
            pr_40 = _compute_detection_pr_nuscenes(recs, dist_threshold=4.0, max_curve_points=pr_curve_max_points)
        else:
            # Single-class or non-class-average qtype: lump all records.
            pr_05 = _compute_detection_pr_nuscenes(recs, dist_threshold=0.5, max_curve_points=pr_curve_max_points)
            pr_10 = _compute_detection_pr_nuscenes(recs, dist_threshold=1.0, max_curve_points=pr_curve_max_points)
            pr_20 = _compute_detection_pr_nuscenes(recs, dist_threshold=2.0, max_curve_points=pr_curve_max_points)
            pr_40 = _compute_detection_pr_nuscenes(recs, dist_threshold=4.0, max_curve_points=pr_curve_max_points)
            ap_05 = float(pr_05['ap'])
            ap_10 = float(pr_10['ap'])
            ap_20 = float(pr_20['ap'])
            ap_40 = float(pr_40['ap'])

        map_val = float((ap_05 + ap_10 + ap_20 + ap_40) / 4.0)

        threshold_metrics = _compute_threshold_sweep(recs, pr_thresholds, eval_iou_mode, dist_thr) # Keep this for legacy metrics if needed

        precision = stats['tp'] / max(stats['tp'] + stats['fp'], 1)
        recall = stats['tp'] / max(stats['tp'] + stats['fn'], 1)

        pr_plot_rel = None
        if save_pr_plots:
            pr_plot_path = pr_plot_dir / f'{_safe_slug(qtype)}.png'
            if _save_pr_curve_plot(
                pr_plot_path,
                title=f'eval split={args.eval_split} type={qtype}',
                curves={'Dist 0.5m': pr_05, 'Dist 1.0m': pr_10, 'Dist 2.0m': pr_20, 'Dist 4.0m': pr_40},
                threshold_metrics=threshold_metrics,
            ):
                pr_plot_rel = str(pr_plot_path.relative_to(output_dir))


        inferred_class = _canonicalize_class_name(qtype) or 'unknown'
        tp_errors = _compute_tp_errors_nuscenes(recs, class_name=inferred_class)

        per_type_results[qtype] = {
            'queries': stats['queries'],
            'tp': stats['tp'],
            'fp': stats['fp'],
            'fn': stats['fn'],
            'precision': precision,
            'recall': recall,
            'AP@0.5m': ap_05,
            'AP@1.0m': ap_10,
            'AP@2.0m': ap_20,
            'AP@4.0m': ap_40,
            'mAP': map_val,
            'mATE': tp_errors['mATE'],
            'mASE': tp_errors['mASE'],
            'mAOE': tp_errors['mAOE'],
            'Acc@0.5m': stats['center_hits_0_5m'] / max(stats['eval_gt_count'], 1),
            'Acc@1.0m': stats['center_hits_1_0m'] / max(stats['eval_gt_count'], 1),
            'Acc@2.0m': stats['center_hits_2_0m'] / max(stats['eval_gt_count'], 1),
            'mean_center_dist_m': stats['center_dist_sum'] / max(stats['center_dist_cnt'], 1),
            'pr_plot': pr_plot_rel,
            'per_class': per_class_ap,
        }

        total_tp += stats['tp']
        total_fp += stats['fp']
        total_fn += stats['fn']
        overall_records.extend(recs)

    overall_pr_05 = _compute_detection_pr_nuscenes(overall_records, dist_threshold=0.5, max_curve_points=pr_curve_max_points)
    overall_pr_10 = _compute_detection_pr_nuscenes(overall_records, dist_threshold=1.0, max_curve_points=pr_curve_max_points)
    overall_pr_20 = _compute_detection_pr_nuscenes(overall_records, dist_threshold=2.0, max_curve_points=pr_curve_max_points)
    overall_pr_40 = _compute_detection_pr_nuscenes(overall_records, dist_threshold=4.0, max_curve_points=pr_curve_max_points)
    
    overall_ap_05 = float(overall_pr_05['ap'])
    overall_ap_10 = float(overall_pr_10['ap'])
    overall_ap_20 = float(overall_pr_20['ap'])
    overall_ap_40 = float(overall_pr_40['ap'])
    overall_map = float((overall_ap_05 + overall_ap_10 + overall_ap_20 + overall_ap_40) / 4.0)

    valid_mATE = [res['mATE'] for res in per_type_results.values() if res['mATE'] < 1.0]
    valid_mASE = [res['mASE'] for res in per_type_results.values() if res['mASE'] < 1.0]
    valid_mAOE = [res['mAOE'] for res in per_type_results.values() if res['mAOE'] < 1.0]

    overall_threshold_metrics = _compute_threshold_sweep(overall_records, pr_thresholds, eval_iou_mode, dist_thr)
    overall_pr_plot_rel = None
    if save_pr_plots:
        overall_plot_path = pr_plot_dir / 'overall.png'
        if _save_pr_curve_plot(
            overall_plot_path,
            title=f'eval split={args.eval_split} overall',
            curves={'Dist 0.5m': overall_pr_05, 'Dist 1.0m': overall_pr_10, 'Dist 2.0m': overall_pr_20, 'Dist 4.0m': overall_pr_40},
            threshold_metrics=overall_threshold_metrics,
        ):
            overall_pr_plot_rel = str(overall_plot_path.relative_to(output_dir))

    overall_precision = total_tp / max(total_tp + total_fp, 1)
    overall_recall = total_tp / max(total_tp + total_fn, 1)

    chunk_size = int(getattr(args, 'eval_chunk_size', 32))

    results = {
        'split': args.eval_split,
        'num_entries': num_entries,
        'num_queries': total_queries,
        'num_skipped': skipped,
        'question_types_filter': sorted(list(allowed_question_types)) if allowed_question_types is not None else 'all',
        'map_conf_mode': conf_mode,
        'map_min_conf': min_conf,
        'feature_cache_dir': args.feature_cache_dir,
        'feature_cache_strict': bool(args.feature_cache_strict),
        'cache_hit_frames': int(cache_hit_frames),
        'cache_miss_frames': int(cache_miss_frames),
        'eval_use_all_proposals': use_all_proposals,
        'no_proposal_yaw': bool(getattr(args, 'no_proposal_yaw', False)),
        'class_range_filter': class_range_filter,
        'pr_thresholds': pr_thresholds,
        'pr_curve_max_points': pr_curve_max_points,
        'eval_chunk_size': chunk_size,
        'eval_iou_mode': eval_iou_mode,
        'iou_threshold_matching': dist_thr,
        'overall': {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'precision': overall_precision,
            'recall': overall_recall,
            'AP@0.5m': overall_ap_05,
            'AP@1.0m': overall_ap_10,
            'AP@2.0m': overall_ap_20,
            'AP@4.0m': overall_ap_40,
            'mAP': overall_map,
            'mATE': float(np.mean([res['mATE'] for res in per_type_results.values() if res['mATE'] < 1.0] or [1.0])),
            'mASE': float(np.mean([res['mASE'] for res in per_type_results.values() if res['mASE'] < 1.0] or [1.0])),
            'mAOE': float(np.mean([res['mAOE'] for res in per_type_results.values() if res['mAOE'] < 1.0] or [1.0])),
            'pr': {
                'dist_0.5m': overall_pr_05,
                'dist_1.0m': overall_pr_10,
                'dist_2.0m': overall_pr_20,
                'dist_4.0m': overall_pr_40,
            },
            'threshold_metrics': overall_threshold_metrics,
            'pr_plot': overall_pr_plot_rel,
        },
        'per_question_type': per_type_results,
        'pr_plot_dir': str(pr_plot_dir.relative_to(output_dir)) if save_pr_plots else None,
    }

    # ── nuScenes-comparable PER-CLASS metrics (devkit convention) ──
    # Official mAP/mATE/mASE/mAOE are macro-averages over the 10 detection
    # classes (evaluate.py::DetectionMetrics), NOT over question types. Classes
    # absent from GT contribute AP=0 / TP=1.0 (devkit no_predictions rule);
    # traffic_cone orientation is excluded at CLASS level (NaN, dropped by
    # nanmean) — matching the official PointPillars log convention.
    # SCOPE: only object_detection_all_category queries ("find all X") — the
    # referring analogue of full-scene per-class detection. Pooling the other
    # query types would mix single-target tasks (closest/sector: 1 GT vs 150
    # preds) into the class buckets and is NOT comparable to detector reports.
    per_class_nusc = {}
    for cname in NUSCENES_DET_CLASSES:
        recs = [r for r in per_class_records.get(cname, [])
                if r.get('qtype') == 'object_detection_all_category']
        n_gt = int(sum(r['gt_boxes'].shape[0] for r in recs))
        if n_gt == 0:
            per_class_nusc[cname] = {
                'num_queries': len(recs), 'num_gt': 0, 'mAP': 0.0,
                'AP@0.5m': 0.0, 'AP@1.0m': 0.0, 'AP@2.0m': 0.0, 'AP@4.0m': 0.0,
                'mATE': 1.0, 'mASE': 1.0,
                'mAOE': float('nan') if cname == 'traffic_cone' else 1.0,
            }
            continue
        aps = {th: float(_compute_detection_pr_nuscenes(recs, dist_threshold=th)['ap'])
               for th in (0.5, 1.0, 2.0, 4.0)}
        tp_err = _compute_tp_errors_nuscenes(recs, class_name=cname)
        per_class_nusc[cname] = {
            'num_queries': len(recs), 'num_gt': n_gt,
            'mAP': float(np.mean(list(aps.values()))),
            'AP@0.5m': aps[0.5], 'AP@1.0m': aps[1.0],
            'AP@2.0m': aps[2.0], 'AP@4.0m': aps[4.0],
            'mATE': tp_err['mATE'], 'mASE': tp_err['mASE'],
            'mAOE': float('nan') if cname == 'traffic_cone' else tp_err['mAOE'],
        }
    _pc = list(per_class_nusc.values())
    results['per_class_nuscenes'] = per_class_nusc
    results['nuscenes_macro'] = {
        'mAP': float(np.mean([c['mAP'] for c in _pc])),
        'mATE': float(np.mean([c['mATE'] for c in _pc])),
        'mASE': float(np.mean([c['mASE'] for c in _pc])),
        'mAOE': float(np.nanmean([c['mAOE'] for c in _pc])),
        'note': 'macro over 10 nuScenes classes, object_detection_all_category '
                'queries only; cone AOE excluded (nanmean); comparable to '
                'official detector reports (e.g. PointPillars mAOE=0.529). '
                'per_question_type numbers are task metrics, pooled across '
                'classes, and are NOT comparable to these.',
    }

    out_path = output_dir / f'eval_{args.eval_split}_all_types.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('\n[Eval Summary]')
    print(f"  split={args.eval_split} queries={total_queries} skipped={skipped}")
    print(f"  matching_mode=nuscenes_2d_distance")
    print(f"  use_all_proposals={use_all_proposals} map_min_conf={min_conf:.3f}")
    print(f"  overall: P={overall_precision:.4f} R={overall_recall:.4f} mAP={overall_map:.4f} "
          f"(AP@0.5m={overall_ap_05:.4f}, AP@1.0m={overall_ap_10:.4f}, AP@2.0m={overall_ap_20:.4f}, AP@4.0m={overall_ap_40:.4f})")
    print('  per question type mAP:')
    for qtype in sorted(per_type_results.keys()):
        row = per_type_results[qtype]
        print(f"    {qtype}: mAP={row['mAP']:.4f} (AP@0.5m={row['AP@0.5m']:.4f}, AP@1.0m={row['AP@1.0m']:.4f}, "
              f"AP@2.0m={row['AP@2.0m']:.4f}, AP@4.0m={row['AP@4.0m']:.4f}) "
              f"queries={row['queries']}")
    print('\n  per class (nuScenes devkit convention, all_category queries only):')
    for cname in NUSCENES_DET_CLASSES:
        c = per_class_nusc[cname]
        aoe_s = '  excl' if np.isnan(c['mAOE']) else f"{c['mAOE']:.4f}"
        print(f"    {cname:<25}: mAP={c['mAP']:.4f} mATE={c['mATE']:.4f} "
              f"mASE={c['mASE']:.4f} mAOE={aoe_s} "
              f"(AP@0.5m={c['AP@0.5m']:.4f}, AP@2.0m={c['AP@2.0m']:.4f}, n_gt={c['num_gt']})")
    nm = results['nuscenes_macro']
    print(f"  NUSCENES MACRO (10-class, cone-AOE excluded): mAP={nm['mAP']:.4f} "
          f"mATE={nm['mATE']:.4f} mASE={nm['mASE']:.4f} mAOE={nm['mAOE']:.4f}")
    print(f'  saved: {out_path}')

    if not getattr(args, 'skip_class_breakdown', False):
    # --- PRINTING PER QUERY TYPE AND CLASS BREAKDOWN ---
        print('\n[Class Breakdown per Query Type]')
        for qt in sorted(per_type_records.keys()):
            print(f"\n======================================")
            print(f"Query Type: {qt}")
            print(f"======================================")

            # Group records for this specific query type by their class
            class_grouped_recs = defaultdict(list)
            for rec in per_type_records[qt]:
                cname = rec.get('cname', 'unknown')
                class_grouped_recs[cname].append(rec)

            for cname in sorted(class_grouped_recs.keys()):
                if cname == 'unknown': continue

                recs = class_grouped_recs[cname]
                if not recs: continue

                # The filter guarantees no math bottlenecks here!
                filtered_recs = _filter_records_by_confidence(recs, 0.001) 

                pr_05 = _compute_detection_pr_nuscenes(filtered_recs, dist_threshold=0.5)
                pr_10 = _compute_detection_pr_nuscenes(filtered_recs, dist_threshold=1.0)
                pr_20 = _compute_detection_pr_nuscenes(filtered_recs, dist_threshold=2.0)
                pr_40 = _compute_detection_pr_nuscenes(filtered_recs, dist_threshold=4.0)

                ap05 = float(pr_05['ap'])
                ap10 = float(pr_10['ap'])
                ap20 = float(pr_20['ap'])
                ap40 = float(pr_40['ap'])
                cmap = (ap05 + ap10 + ap20 + ap40) / 4.0

                print(f"  {cname:<25}: mAP={cmap:.4f} (AP@0.5m={ap05:.4f}, AP@1.0m={ap10:.4f}, AP@2.0m={ap20:.4f}, AP@4.0m={ap40:.4f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ReferModel inference with explicit 10-sweep nuScenes loading via MMDetection3D Compose.'
    )
    parser.add_argument(
        '--meta-arch',
        default='refer_model',
        choices=['refer_model', 'refer_model_lang_dec', 'refer_model_angle'],
    )
    parser.add_argument(
        '--merge-min-conf', type=float, default=0.0,
        help='Confidence floor applied to merged records before metrics. '
             'Default 0 = devkit-exact (no floor). >0 is preview-only.')
    parser.add_argument(
        '--no-class-range-filter', action='store_true',
        help='Disable devkit-parity per-class ego-range filtering of GT and '
             'predictions before matching (filter is ON by default).')
    parser.add_argument(
        '--no-proposal-yaw', action='store_true',
        help='Strip proposal yaw from cached eval inputs so the decoder runs '
             'angle-BLIND (fixed axis-aligned sampling). REQUIRED when '
             'evaluating legacy checkpoints trained before the classyaw cache '
             '(v1 / +lang) on the classyaw cache, else rotated sampling '
             'silently activates at eval time (train/eval mismatch).',
    )
    parser.add_argument('--data-root', default='data/nuscenes', help='nuScenes root directory')
    parser.add_argument('--ann-file', default='nuscenes_infos_val.pkl', help='Annotation pkl relative to data_root')
    parser.add_argument('--sample-token', default=None, help='nuScenes sample token to run')
    parser.add_argument('--lidar-suffix', default=None, help='Suffix match for lidar path, e.g. n015-...pcd.bin')
    parser.add_argument('--text', default=None, help='Referring expression (single-sample mode)')
    parser.add_argument('--mode', choices=['single', 'eval'], default='single')
    parser.add_argument('--refer-data-dir', default=None,
                        help='Refer dataset root, refer_detection_with_negatives root, or explicit split directory for eval mode')
    parser.add_argument('--feature-cache-dir', type=str, default=None,
                        help='Optional CenterPoint cache directory (index + feature tensors) for eval.')
    parser.add_argument('--feature-cache-strict', action='store_true',
                        help='Fail on cache miss when --feature-cache-dir is set.')
    parser.add_argument('--gt-blacklist', type=str, default=None,
                        help='JSON from build_gt_blacklist.py (devkit ghost + '
                             'bike-rack tokens). Excluded from eval GT at the '
                             'dataset source — must match the training flag.')
    parser.add_argument('--ego-range-transform', type=str, default='',
                        help='JSON mapping LIDAR_TOP keyframe -> devkit ego_dist '
                             'transform (8 floats) for exact class-range filtering '
                             'from the EGO pose. Auto-discovered at '
                             '<nuscenes-dataroot>/ego_range_transform.json.')
    parser.add_argument('--eval-split', choices=['train', 'val', 'eval', 'test'], default='val')
    parser.add_argument('--question-types-json', default='configs/question_types_det.json', help='Optional json list to filter question types')
    parser.add_argument('--output-dir', default='outputs/inference_eval', help='Output directory for eval json')
    parser.add_argument('--dist-threshold', type=float, default=2.0, help='Center distance threshold in meters for TP/FP/FN matching')
    parser.add_argument('--map-conf-mode', choices=['score', 'refer', 'product'], default='product')
    parser.add_argument('--map-min-conf', type=float, default=0.05)
    parser.add_argument(
        '--eval-use-all-proposals',
        dest='eval_use_all_proposals',
        action='store_true',
        help='Use all proposal queries for eval metrics and PR curves (no hard confidence cutoff).',
    )
    parser.add_argument(
        '--no-eval-use-all-proposals',
        dest='eval_use_all_proposals',
        action='store_false',
        help='Apply --map-min-conf before eval metric computation.',
    )
    parser.set_defaults(eval_use_all_proposals=True)
    parser.add_argument(
        '--pr-thresholds',
        type=float,
        nargs='+',
        default=[0.3, 0.5, 0.7],
        help='Score thresholds for numeric summaries (e.g., AP/mAP, precision/recall) in eval outputs.',
    )
    parser.add_argument(
        '--pr-curve-max-points',
        type=int,
        default=400,
        help='Maximum number of points saved per PR curve (downsampled).',
    )
    parser.add_argument(
        '--save-pr-plots',
        dest='save_pr_plots',
        action='store_true',
        help='Save PR curve PNG files in addition to JSON curve data.',
    )
    parser.add_argument(
        '--no-save-pr-plots',
        dest='save_pr_plots',
        action='store_false',
        help='Do not save PR curve PNG files.',
    )
    parser.set_defaults(save_pr_plots=True)
    parser.add_argument('--eval-chunk-size', type=int, default=32, help='Queries per frame-chunk for batched eval forward')

    parser.add_argument('--skip_class_breakdown', action='store_true', help='Skip the per-class breakdown section in the printed summary.')

    parser.add_argument(
        '--pointpillars-config',
        default='configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py',
    )
    parser.add_argument('--pointpillars-ckpt', default=None, help='Pretrained PointPillars checkpoint')
    parser.add_argument(
        '--centerpoint-config',
        default='configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py',
    )
    parser.add_argument('--centerpoint-ckpt', default=None, help='Pretrained CenterPoint checkpoint')
    parser.add_argument(
        '--detector-backbone',
        choices=['centerpoint', 'utonia'],
        default='centerpoint',
        help='Detector backend for refer_model_second. centerpoint uses pretrained MMDet3D CenterPoint; utonia uses Utonia -> SECOND -> SECONDFPN -> CenterHead.',
    )
    parser.add_argument(
        '--centerpoint-feature-mode',
        choices=['fused', 'two_levels'],
        default='fused',
        help='CenterPoint features for RMOT: fused SECONDFPN map or two separate backbone levels',
    )
    parser.add_argument('--utonia-ckpt', default=None, help='Utonia checkpoint for --detector-backbone utonia')
    parser.add_argument('--pointcept-root', default=None, help='Optional Pointcept root; defaults to /opt/Pointcept when available')
    parser.add_argument('--utonia-bev-size', type=int, default=256)
    parser.add_argument('--utonia-bev-hidden-dim', type=int, default=256)
    parser.add_argument('--utonia-normals-k', type=int, default=16)
    parser.add_argument('--no-utonia-compute-normals', dest='utonia_compute_normals', action='store_false')
    parser.add_argument('--train-utonia-detector', dest='freeze_utonia_detector', action='store_false')
    parser.add_argument('--train-utonia-encoder', dest='freeze_utonia_encoder', action='store_false')
    parser.set_defaults(
        utonia_compute_normals=True,
        freeze_utonia_detector=True,
        freeze_utonia_encoder=True,
    )
    parser.add_argument('--model-ckpt', default=None, help='Optional ReferModel checkpoint')

    parser.add_argument('--proposal-queries', type=int, default=150)
    parser.add_argument('--proposal-w-from', choices=['dx', 'dy'], default='dy')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--enc-layers', type=int, default=6)
    parser.add_argument('--dec-layers', type=int, default=6)
    parser.add_argument('--dim-feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num-feature-levels', type=int, default=3)
    parser.add_argument('--dec-n-points', type=int, default=4)
    parser.add_argument('--enc-n-points', type=int, default=4)
    parser.add_argument('--aux-loss', dest='aux_loss', action='store_true')
    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.set_defaults(aux_loss=True)

    # SEED-like flags for current training checkpoints.
    parser.add_argument('--use-dga', action='store_true', default=False)
    parser.add_argument('--dga-grid-size', type=int, default=5)
    parser.add_argument('--dqs-topk', type=int, default=50)
    parser.add_argument('--dqs-beta', type=float, default=0.35)
    parser.add_argument('--no-decoder-lang-attn', action='store_true',
                        help='V3.3: disable per-layer language cross-attention in decoder')

    # Cost/loss compatibility flags.
    parser.add_argument('--set-cost-class', type=float, default=2.0)
    parser.add_argument('--set-cost-bbox', type=float, default=5.0)
    parser.add_argument('--set-cost-center', type=float, default=5.0)
    parser.add_argument('--set-cost-refer', type=float, default=2.0)
    parser.add_argument('--set-cost-refer-beta', type=float, default=0.35)
    parser.add_argument('--cls-loss-coef', type=float, default=2.0)
    parser.add_argument('--bbox-loss-coef', type=float, default=5.0)
    parser.add_argument('--giou-loss-coef', type=float, default=2.0)
    parser.add_argument('--refer-loss-coef', type=float, default=2.0)
    parser.add_argument('--loss-3d-coef', type=float, default=2.0)
    parser.add_argument('--quality-loss-coef', type=float, default=1.0)
    parser.add_argument('--focal-alpha', type=float, default=0.25)

    parser.add_argument('--sweeps-num', type=int, default=10)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--point-cloud-range',
        type=float,
        nargs=6,
        default=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
    )

    parser.add_argument('--num-shards', type=int, default=1, help='Number of parallel shards')
    parser.add_argument('--shard-id', type=int, default=0, help='Shard index (0 to num-shards - 1)')
    parser.add_argument('--save-predictions', type=str, default=None, help='Save raw predictions to a pickle file')
    parser.add_argument('--merge-predictions', type=str, nargs='+', default=None, help='Merge multiple pickle files')
    parser.add_argument('--merge-nms-radius', type=float, default=0.0,
                        help='DIAGNOSTIC: greedy center-distance NMS radius (m) applied to merged '
                             'records before AP. 0 = off (default, NMS-free reported protocol).')
    parser.add_argument('--augment-gt-class', action='store_true',
                        help='During --merge-predictions, retro-fit per-object GT class onto '
                             'records saved before the gt_class_names field existed (reloads '
                             'refer targets, no model inference). Required to get correct '
                             'nuScenes mAOE (cone-exclude / barrier mod-pi) from legacy shards.')

    args = parser.parse_args()
    args.ann_file = os.path.join(args.data_root, args.ann_file)
    if args.mode == 'single' and not args.text:
        parser.error('--text is required in single mode')
    if args.mode == 'eval' and not args.refer_data_dir:
        parser.error('--refer-data-dir is required in eval mode')
    if args.meta_arch in {'refer_model', 'refer_model_lang_dec', 'refer_model_angle'} and not args.pointpillars_ckpt:
        parser.error(f'--pointpillars-ckpt is required for --meta-arch {args.meta_arch}')
    if args.meta_arch == 'refer_model_second' and args.detector_backbone == 'centerpoint' and not args.centerpoint_ckpt:
        parser.error('--centerpoint-ckpt is required for --meta-arch refer_model_second when --detector-backbone centerpoint')
    if args.meta_arch == 'refer_model_second' and args.detector_backbone == 'utonia' and not args.utonia_ckpt:
        parser.error('--utonia-ckpt is required for --meta-arch refer_model_second when --detector-backbone utonia')
    if args.meta_arch == 'refer_model_second_v2' and not args.centerpoint_ckpt:
        parser.error('--centerpoint-ckpt is required for --meta-arch refer_model_second_v2')
    if args.meta_arch == 'refer_model_second_v3' and not args.centerpoint_ckpt:
        parser.error('--centerpoint-ckpt is required for --meta-arch refer_model_second_v3')
    if args.meta_arch != 'refer_model_second' and args.detector_backbone != 'centerpoint':
        parser.error('--detector-backbone utonia is currently supported only for --meta-arch refer_model_second')
    if args.feature_cache_dir and args.detector_backbone != 'centerpoint':
        parser.error('--feature-cache-dir is only supported with --detector-backbone centerpoint')
    return args


if __name__ == '__main__':
    parsed = parse_args()
    if parsed.mode == 'eval':
        evaluate(parsed)
    else:
        run(parsed)