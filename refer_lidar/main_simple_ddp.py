import argparse
import csv
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import build_model  # noqa: E402


try:
    from nuscenes_lidar_simple import build as build_dataset  # noqa: E402
    from nuscenes_lidar_simple import simple_collate_fn  # noqa: E402
except Exception as exc:
    raise ImportError(
        'Failed to import refer_lidar dataset builder. '\
        'Check refer_lidar/nuscenes_lidar_simple.py dependencies and PYTHONPATH.'
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DDP trainer for refer_lidar refer_model (2+ GPUs).')
    parser.add_argument(
        '--meta-arch',
        default='refer_model',
        choices=['refer_model', 'refer_model_lang_dec', 'refer_model_angle'],
    )

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr-drop', type=int, default=40)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-grad-norm', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='outputs/refer_model')
    parser.add_argument('--resume', type=str, default='')

    parser.add_argument('--split', default='train')
    parser.add_argument('--refer-data-dir', required=True)
    parser.add_argument('--nuscenes-dataroot', required=True)
    parser.add_argument('--queries-per-frame', type=int, default=3)
    parser.add_argument('--type-weighting', choices=['uniform', 'inverse_sqrt'], default='uniform')
    parser.add_argument('--cbgs', action='store_true',
                        help='E1: scene-level class-balanced (CBGS) frame resampling, '
                             'composed with the per-scene query stratification.')
    parser.add_argument('--epoch-size', type=int, default=0,
                        help='Optional query budget cap for frame-iterated training; 0 visits all train frames once per epoch.')
    parser.add_argument('--val-epoch-size', type=int, default=0,
                        help='Optional query budget cap for frame-iterated val; 0 visits all val frames once per val pass.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Run training-time validation every N epochs. Final epoch is always validated.')
    parser.add_argument('--question-types-json', type=str, default='configs/question_types_det.json')
    parser.add_argument('--question-types', type=str, nargs='*', default=None)
    parser.add_argument('--nuscenes-ann-file', type=str, default=None)
    parser.add_argument('--sweeps-num', type=int, default=10)
    parser.add_argument('--feature-cache-dir', type=str, default=None)
    parser.add_argument('--feature-cache-strict', action='store_true')

    parser.add_argument(
        '--pointpillars-config',
        default='configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py',
    )
    parser.add_argument('--pointpillars-ckpt', default=None)
    parser.add_argument(
        '--centerpoint-config',
        default='configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py',
    )
    parser.add_argument('--centerpoint-ckpt', default=None)
    parser.add_argument(
        '--detector-backbone',
        choices=['centerpoint', 'utonia'],
        default='centerpoint',
        help='Detector backend for refer_model_second. centerpoint uses the pretrained MMDet3D model; utonia uses Utonia -> SECOND -> SECONDFPN -> CenterHead.',
    )
    parser.add_argument(
        '--centerpoint-feature-mode',
        choices=['fused', 'two_levels'],
        default='fused',
        help='CenterPoint features for RMOT: fused SECONDFPN map or two separate backbone levels',
    )
    parser.add_argument('--utonia-ckpt', default=None)
    parser.add_argument('--pointcept-root', default=None)
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
    parser.add_argument('--proposal-queries', type=int, default=150)
    parser.add_argument('--proposal-w-from', choices=['dx', 'dy'], default='dy')

    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--enc-layers', type=int, default=6)
    parser.add_argument('--dec-layers', type=int, default=6)
    parser.add_argument('--dim-feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-feature-levels', type=int, default=3)
    parser.add_argument('--dec-n-points', type=int, default=4)
    parser.add_argument('--enc-n-points', type=int, default=4)
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

    # Loss / matcher settings.
    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.set_defaults(aux_loss=True)
    parser.add_argument('--set-cost-class', type=float, default=2.0)
    parser.add_argument('--set-cost-bbox', type=float, default=5.0)
    parser.add_argument('--set-cost-center', type=float, default=5.0)
    parser.add_argument('--set-cost-refer', type=float, default=2.0)
    parser.add_argument('--set-cost-refer-beta', type=float, default=0.35)
    parser.add_argument('--cls-loss-coef', type=float, default=2.0)
    parser.add_argument('--bbox-loss-coef', type=float, default=5.0)
    parser.add_argument('--giou-loss-coef', type=float, default=2.0)
    parser.add_argument('--refer-loss-coef', type=float, default=2.0)
    parser.add_argument('--refer-focal-gamma', type=float, default=2.0)
    parser.add_argument('--loss-3d-coef', type=float, default=2.0)
    parser.add_argument('--loss-dir-coef', type=float, default=0.2)
    parser.add_argument('--quality-loss-coef', type=float, default=1.0)
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--set-cost-giou', type=float, default=0.5,
                        help='GIoU cost weight in matcher (v3)')
    parser.add_argument('--set-cost-3d', type=float, default=2.0,
                        help='3D attribute L1 cost weight in matcher (v3.2)')
    parser.add_argument('--iou-filter-threshold', type=float, default=0.05,
                        help='Fixed 2D IoU filter threshold in matcher (v3)')
    parser.add_argument('--dga-grid-size', type=int, default=5)
    parser.add_argument('--no-decoder-lang-attn', action='store_true',
                        help='V3.3: disable per-layer language cross-attention in decoder')

    # V2 architecture: curriculum IoU thresholding (only used with refer_model_second_v2)
    parser.add_argument('--iou-threshold-start', type=float, default=0.0,
                        help='Starting IoU threshold for curriculum filtering (v2 only)')
    parser.add_argument('--iou-threshold-end', type=float, default=0.15,
                        help='Final IoU threshold for curriculum filtering (v2 only)')
    parser.add_argument('--iou-threshold-warmup-epochs', type=int, default=10,
                        help='Epochs to ramp IoU threshold from start to end (v2 only)')

    # Debug / observability settings.
    parser.add_argument('--debug-loss-terms', action='store_true')
    parser.add_argument('--debug-loss-every', type=int, default=100)

    # Per-epoch validation subset (speed). A FIXED seeded random subset — the
    # SAME items every epoch and every run — so val_loss curves remain
    # comparable for checkpoint_best selection, early stopping, and cross-run
    # model comparison. (A fresh random subset each epoch would let subset
    # noise dominate the ~0.1-loss differences that decide checkpoint_best.)
    parser.add_argument('--val-fraction', type=float, default=1.0,
                        help='Fraction of the val set used for the per-epoch '
                             'validation pass (e.g. 0.2). Fixed seeded subset.')
    parser.add_argument('--val-subset-seed', type=int, default=42,
                        help='Seed for the fixed val subset selection.')

    # Early stopping on val_loss (patience counted in VALIDATED epochs).
    parser.add_argument('--early-stop-patience', type=int, default=0,
                        help='Stop after N validated epochs without val_loss '
                             'improvement. 0 disables early stopping.')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0,
                        help='Minimum val_loss improvement to reset patience.')

    # GT blacklist (devkit filter_eval_boxes parity — ghosts + bike-rack).
    parser.add_argument('--gt-blacklist', type=str, default=None,
                        help='JSON from build_gt_blacklist.py. Blacklisted '
                             'annotation tokens are excluded from training '
                             'targets and the val-loss pass.')
    parser.add_argument('--ego-range-transform', type=str, default='',
                        help='JSON mapping LIDAR_TOP keyframe -> devkit ego_dist '
                             'transform (8 floats), for exact class-range '
                             'filtering from the EGO pose. Auto-discovered at '
                             '<nuscenes-dataroot>/ego_range_transform.json.')

    args = parser.parse_args()
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
    if args.feature_cache_dir and args.meta_arch not in {'refer_model', 'refer_model_lang_dec', 'refer_model_angle'}:
        parser.error('--feature-cache-dir is currently supported only for --meta-arch refer_model/refer_model_lang_dec/refer_model_angle')
    if args.feature_cache_dir and args.detector_backbone != 'centerpoint':
        parser.error('--feature-cache-dir is only supported with --detector-backbone centerpoint')
    return args


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_distributed() -> tuple[int, int, int]:
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        raise RuntimeError(
            'Distributed environment not initialized. Launch with torchrun, '\
            'for example: torchrun --standalone --nproc_per_node=2 refer_lidar/main_simple_ddp.py ...'
        )

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return rank, world_size, local_rank


def _build_model_args(args: argparse.Namespace) -> SimpleNamespace:
    num_point_features = 5 if args.meta_arch in {'refer_model_second', 'refer_model_second_v2', 'refer_model_second_v3'} else 4
    return SimpleNamespace(
        meta_arch=args.meta_arch,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=False,
        num_queries=args.proposal_queries,
        decoder_cross_self=False,
        sigmoid_attn=False,
        extra_track_attn=False,
        pointpillars_config=args.pointpillars_config,
        pointpillars_ckpt=args.pointpillars_ckpt,
        centerpoint_config=args.centerpoint_config,
        centerpoint_ckpt=args.centerpoint_ckpt,
        detector_backbone=args.detector_backbone,
        centerpoint_feature_mode=args.centerpoint_feature_mode,
        utonia_ckpt=args.utonia_ckpt,
        pointcept_root=args.pointcept_root,
        utonia_bev_size=args.utonia_bev_size,
        utonia_bev_hidden_dim=args.utonia_bev_hidden_dim,
        utonia_compute_normals=args.utonia_compute_normals,
        utonia_normals_k=args.utonia_normals_k,
        proposal_queries=args.proposal_queries,
        proposal_w_from=args.proposal_w_from,
        freeze_pointpillars=True,
        freeze_centerpoint=True,
        freeze_utonia_detector=args.freeze_utonia_detector,
        freeze_utonia_encoder=args.freeze_utonia_encoder,
        refer_data_dir=args.refer_data_dir,
        nuscenes_dataroot=args.nuscenes_dataroot,
        gt_blacklist=args.gt_blacklist,
        ego_range_transform=args.ego_range_transform,
        queries_per_frame=args.queries_per_frame,
        type_weighting=args.type_weighting,
        cbgs=args.cbgs,
        epoch_size=args.epoch_size,
        val_epoch_size=args.val_epoch_size,
        question_types_json=args.question_types_json,
        question_types=args.question_types,
        nuscenes_ann_file=args.nuscenes_ann_file,
        sweeps_num=args.sweeps_num,
        feature_cache_dir=args.feature_cache_dir,
        feature_cache_strict=args.feature_cache_strict,
        seed=args.seed,
        num_point_features=num_point_features,
        point_cloud_range=args.point_cloud_range,
        aux_loss=args.aux_loss,
        set_cost_class=args.set_cost_class,
        set_cost_bbox=args.set_cost_bbox,
        set_cost_center=args.set_cost_center,
        set_cost_refer=args.set_cost_refer,
        set_cost_refer_beta=args.set_cost_refer_beta,
        cls_loss_coef=args.cls_loss_coef,
        bbox_loss_coef=args.bbox_loss_coef,
        giou_loss_coef=args.giou_loss_coef,
        refer_loss_coef=args.refer_loss_coef,
        refer_focal_gamma=args.refer_focal_gamma,
        loss_3d_coef=args.loss_3d_coef,
        loss_dir_coef=args.loss_dir_coef,
        quality_loss_coef=args.quality_loss_coef,
        focal_alpha=args.focal_alpha,
        dga_grid_size=args.dga_grid_size,
        # V2 curriculum IoU thresholding
        iou_threshold_start=args.iou_threshold_start,
        iou_threshold_end=args.iou_threshold_end,
        iou_threshold_warmup_epochs=args.iou_threshold_warmup_epochs,
    )


def _weighted_loss_or_zero(
    outputs: dict,
    loss_dict: dict,
    weight_dict: dict,
    require_grad: bool,
) -> torch.Tensor:
    weighted_terms = [loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict]
    if weighted_terms:
        total = sum(weighted_terms)
        if torch.is_tensor(total) and (not require_grad or total.requires_grad):
            return total
    # Fallback keeps graph valid in training when losses are unexpectedly empty.
    return outputs['pred_boxes'].sum() * 0.0


def _log_line(msg: str, log_fh=None) -> None:
    print(msg)
    if log_fh is not None:
        log_fh.write(msg + '\n')
        log_fh.flush()


def _flatten_targets_for_count(targets: Any) -> list:
    if targets is None:
        return []
    if not isinstance(targets, list):
        return []
    if len(targets) == 0:
        return []
    if isinstance(targets[0], list):
        flat = []
        for t in targets:
            flat.extend(t)
        return flat
    return targets


def _count_target_boxes(targets: Any) -> int:
    total = 0
    for t in _flatten_targets_for_count(targets):
        labels = getattr(t, 'labels', None)
        if labels is not None:
            total += int(labels.numel())
    return total


def _to_device_model_inputs(batch: dict, device: torch.device) -> dict:
    """Move batch payload to device, supporting optional CenterPoint cache tensors."""
    model_inputs = {
        'sentences': batch['sentences'],
    }

    if 'points' in batch:
        points = [
            p.to(device, non_blocking=True) if p is not None else None
            for p in batch['points']
        ]
        if any(p is not None for p in points):
            model_inputs['points'] = points

    if 'centerpoint_srcs' in batch:
        moved_srcs = []
        has_cached = False
        for per_sample in batch['centerpoint_srcs']:
            if per_sample is None:
                moved_srcs.append(None)
                continue
            moved_levels = [lvl.to(device, non_blocking=True) for lvl in per_sample]
            moved_srcs.append(moved_levels)
            has_cached = True
        if has_cached:
            model_inputs['centerpoint_srcs'] = moved_srcs

    if 'centerpoint_props' in batch:
        moved_props = []
        has_cached = False
        for p in batch['centerpoint_props']:
            if p is None:
                moved_props.append(None)
                continue
            moved_props.append(p.to(device, non_blocking=True))
            has_cached = True
        if has_cached:
            model_inputs['centerpoint_props'] = moved_props

    if 'centerpoint_scores' in batch:
        moved_scores = []
        has_cached = False
        for s in batch['centerpoint_scores']:
            if s is None:
                moved_scores.append(None)
                continue
            moved_scores.append(s.to(device, non_blocking=True))
            has_cached = True
        if has_cached:
            model_inputs['centerpoint_scores'] = moved_scores

    if 'pointpillars_srcs' in batch:
        moved_srcs = []
        has_cached = False
        for per_sample in batch['pointpillars_srcs']:
            if per_sample is None:
                moved_srcs.append(None)
                continue
            moved_levels = [lvl.to(device, non_blocking=True) for lvl in per_sample]
            moved_srcs.append(moved_levels)
            has_cached = True
        if has_cached:
            model_inputs['pointpillars_srcs'] = moved_srcs

    if 'pointpillars_props' in batch:
        moved_props = []
        has_cached = False
        for p in batch['pointpillars_props']:
            if p is None:
                moved_props.append(None)
                continue
            moved_props.append(p.to(device, non_blocking=True))
            has_cached = True
        if has_cached:
            model_inputs['pointpillars_props'] = moved_props

    if 'pointpillars_scores' in batch:
        moved_scores = []
        has_cached = False
        for s in batch['pointpillars_scores']:
            if s is None:
                moved_scores.append(None)
                continue
            moved_scores.append(s.to(device, non_blocking=True))
            has_cached = True
        if has_cached:
            model_inputs['pointpillars_scores'] = moved_scores

    # Per-proposal heading (Q,) for the +angle model; absent (None) for a cache
    # built without it.
    if 'pointpillars_yaw' in batch:
        moved = []
        has_cached = False
        for v in batch['pointpillars_yaw']:
            if v is None:
                moved.append(None)
                continue
            moved.append(v.to(device, non_blocking=True))
            has_cached = True
        if has_cached:
            model_inputs['pointpillars_yaw'] = moved

    return model_inputs


def _maybe_log_loss_terms(
    *,
    args: argparse.Namespace,
    rank: int,
    step: int,
    epoch: int,
    total_steps: int,
    loss_dict: dict,
    total_loss: torch.Tensor,
    weight_dict: dict,
    num_target_boxes: int,
    pred_refers_numel: int,
    log_fh,
) -> None:
    if not args.debug_loss_terms or rank != 0:
        return

    every = max(int(args.debug_loss_every), 1)
    if step % every != 0:
        return

    main_keys = ['loss_ce', 'loss_bbox', 'loss_giou', 'loss_refer', 'loss_3d', 'loss_dir', 'loss_quality', 'loss_rad']
    parts = []
    for k in main_keys:
        if k not in loss_dict:
            continue
        raw = float(loss_dict[k].detach().item())
        w = float(weight_dict.get(k, 1.0))
        weighted = raw * w
        parts.append(f'{k}=raw:{raw:.6f},w:{w:.3f},weighted:{weighted:.6f}')

    _log_line(
        f'[LossDebug] epoch={epoch+1} step={step}/{total_steps} '
        f'num_target_boxes={num_target_boxes} pred_refers_numel={pred_refers_numel} '
        f'total_loss={float(total_loss.detach().item()):.6f}',
        log_fh,
    )
    if parts:
        _log_line('[LossDebug] ' + ' | '.join(parts), log_fh)


@torch.no_grad()
def validate_one_epoch(
    model: DDP,
    criterion: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    rank: int,
) -> float:
    model.eval()
    criterion.eval()

    running = 0.0
    weight_dict = getattr(criterion, 'weight_dict', {})
    for batch in loader:
        model_inputs = _to_device_model_inputs(batch, device)
        outputs = model(model_inputs)
        loss_dict = criterion(outputs, batch.get('gt_instances', None))
        total_loss = _weighted_loss_or_zero(
            outputs, loss_dict, weight_dict=weight_dict, require_grad=False)
        running += float(total_loss.detach().item())

    local = torch.tensor([running, float(len(loader))], device=device)
    dist.all_reduce(local, op=dist.ReduceOp.SUM)

    global_running = float(local[0].item())
    global_steps = max(float(local[1].item()), 1.0)
    return global_running / global_steps


def train(args: argparse.Namespace) -> None:
    rank, world_size, local_rank = _init_distributed()
    _set_seed(args.seed + rank)

    os.makedirs(args.output_dir, exist_ok=True)

    model_args = _build_model_args(args)
    if rank == 0:
        print(
            f"[Args] proposal_queries={args.proposal_queries}, "
            f"model_args.proposal_queries={model_args.proposal_queries}, "
            f"resume={'<none>' if not args.resume else args.resume}",
            flush=True,
        )
    model, criterion, _ = build_model(model_args)

    device = torch.device(f'cuda:{local_rank}')
    model.to(device)
    criterion.to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train_dataset = build_dataset('train', model_args)
    val_dataset = build_dataset('val', model_args)

    # Optional fixed seeded val subset (--val-fraction < 1.0) to cut the
    # per-epoch validation cost. Same indices every epoch AND every run.
    if getattr(args, 'val_fraction', 1.0) < 1.0:
        _n_val = len(val_dataset)
        _k_val = max(1, int(round(_n_val * args.val_fraction)))
        _rng = np.random.RandomState(args.val_subset_seed)
        _sub_idx = np.sort(_rng.permutation(_n_val)[:_k_val])
        _base_val = val_dataset
        val_dataset = torch.utils.data.Subset(_base_val, _sub_idx.tolist())
        if hasattr(_base_val, 'set_epoch'):
            # Preserve epoch-seeded query selection on the underlying dataset.
            val_dataset.set_epoch = _base_val.set_epoch
        if rank == 0:
            print(f'[Data] val subset: {_k_val}/{_n_val} items '
                  f'(fraction={args.val_fraction}, seed={args.val_subset_seed}, '
                  f'fixed across epochs/runs)', flush=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=simple_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=simple_collate_fn,
        drop_last=False,
    )

    if rank == 0:
        train_total_queries = len(getattr(train_dataset, '_flat_queries', []))
        val_total_queries = len(getattr(val_dataset, '_flat_queries', []))
        _train_len = len(train_dataset)
        _val_len = len(val_dataset)
        print(
            f'[Data] train_items={_train_len} train_steps_per_epoch={len(train_loader)} '
            f'train_total_queries={train_total_queries}',
            flush=True,
        )
        print(
            f'[Data] val_items={_val_len} val_steps_per_run={len(val_loader)} '
            f'val_interval={args.val_interval} val_total_queries={val_total_queries}',
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    best_val_loss = float('inf')
    start_epoch = 0
    latest_val_loss = None
    epochs_since_best = 0  # early-stop patience counter (validated epochs)

    if args.resume:
        if rank == 0:
            print(f'[Resume] Loading checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.module.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = int(checkpoint.get('epoch', 0))
        best_val_loss = float(checkpoint.get('best_val_loss', float('inf')))
        latest_val_loss = checkpoint.get('val_loss', None)
        epochs_since_best = int(checkpoint.get('epochs_since_best', 0))
        if rank == 0:
            print(f'[Resume] Start epoch: {start_epoch}, best_val_loss: {best_val_loss:.6f}')
    log_fh = None
    metrics_fh = None
    metrics_writer = None
    if rank == 0:
        log_path = os.path.join(args.output_dir, 'train.log')
        metrics_path = os.path.join(args.output_dir, 'metrics.csv')
        log_fh = open(log_path, 'a', encoding='utf-8')
        metrics_fh = open(metrics_path, 'a', newline='', encoding='utf-8')
        metrics_writer = csv.writer(metrics_fh)
        if metrics_fh.tell() == 0:
            metrics_writer.writerow(['epoch', 'train_loss', 'val_loss', 'best_val_loss'])
            metrics_fh.flush()

    try:
        for epoch in range(start_epoch, args.epochs):
            if hasattr(train_dataset, 'set_epoch'):
                train_dataset.set_epoch(epoch)
            if hasattr(val_dataset, 'set_epoch'):
                val_dataset.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            # V2: Update curriculum IoU threshold for matcher
            if hasattr(criterion, 'set_epoch'):
                criterion.set_epoch(epoch)

            model.train()
            criterion.train()

            running = 0.0
            weight_dict = getattr(criterion, 'weight_dict', {})
            for step, batch in enumerate(train_loader):
                gt_instances = batch.get('gt_instances', None)

                model_inputs = _to_device_model_inputs(batch, device)
                outputs = model(model_inputs)
                loss_dict = criterion(outputs, gt_instances)
                total_loss = _weighted_loss_or_zero(
                    outputs, loss_dict, weight_dict=weight_dict, require_grad=True)

                _maybe_log_loss_terms(
                    args=args,
                    rank=rank,
                    step=step,
                    epoch=epoch,
                    total_steps=len(train_loader),
                    loss_dict=loss_dict,
                    total_loss=total_loss,
                    weight_dict=weight_dict,
                    num_target_boxes=_count_target_boxes(gt_instances),
                    pred_refers_numel=int(outputs['pred_refers'].numel()) if 'pred_refers' in outputs else -1,
                    log_fh=log_fh,
                )

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                running += float(total_loss.detach().item())

                if rank == 0 and step % 10 == 0:
                    _log_line(
                        f'[Epoch {epoch+1}/{args.epochs}] '
                        f'step={step}/{len(train_loader)} '
                        f'loss={float(total_loss.detach().item()):.6f}',
                        log_fh,
                    )

            local_train = torch.tensor([running, float(len(train_loader))], device=device)
            dist.all_reduce(local_train, op=dist.ReduceOp.SUM)
            train_mean_loss = float(local_train[0].item()) / max(float(local_train[1].item()), 1.0)

            should_validate = args.val_interval > 0 and (
                ((epoch + 1) % args.val_interval) == 0 or (epoch + 1) == args.epochs
            )
            val_mean_loss = None
            if should_validate:
                val_mean_loss = validate_one_epoch(model, criterion, val_loader, device, rank)
                latest_val_loss = val_mean_loss
            lr_scheduler.step()
            current_lr = float(optimizer.param_groups[0]['lr'])

            if rank == 0:
                if val_mean_loss is None:
                    _log_line(
                        f'[Epoch {epoch+1}/{args.epochs}] '
                        f'train_loss={train_mean_loss:.6f} '
                        f'val_skipped interval={args.val_interval} '
                        f'last_val_loss={"<none>" if latest_val_loss is None else f"{latest_val_loss:.6f}"} '
                        f'lr={current_lr:.2e}',
                        log_fh,
                    )
                else:
                    _log_line(
                        f'[Epoch {epoch+1}/{args.epochs}] '
                        f'train_loss={train_mean_loss:.6f} '
                        f'val_loss={val_mean_loss:.6f} '
                        f'lr={current_lr:.2e}',
                        log_fh,
                    )

                # Early-stop patience: compare against the PREVIOUS best
                # (best_val_loss is updated below). Counted only on validated
                # epochs, so --val-interval > 1 still behaves sensibly.
                if val_mean_loss is not None:
                    if val_mean_loss < best_val_loss - args.early_stop_min_delta:
                        epochs_since_best = 0
                    else:
                        epochs_since_best += 1

                ckpt = {
                    'epoch': epoch + 1,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'train_loss': train_mean_loss,
                    'val_loss': latest_val_loss,
                    'best_val_loss': best_val_loss if val_mean_loss is None else min(best_val_loss, val_mean_loss),
                    'epochs_since_best': epochs_since_best,
                }

                latest_path = os.path.join(args.output_dir, 'checkpoint_latest.pth')
                torch.save(ckpt, latest_path)
                _log_line(f'[Checkpoint] Saved {latest_path}', log_fh)

                if val_mean_loss is not None and val_mean_loss < best_val_loss:
                    best_val_loss = val_mean_loss
                    ckpt['best_val_loss'] = best_val_loss
                    best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
                    torch.save(ckpt, best_path)
                    _log_line(
                        f'[Checkpoint] Saved {best_path} (best val_loss={best_val_loss:.6f})',
                        log_fh,
                    )

                if metrics_writer is not None:
                    metrics_writer.writerow([
                        epoch + 1,
                        f'{train_mean_loss:.6f}',
                        '' if val_mean_loss is None else f'{val_mean_loss:.6f}',
                        f'{best_val_loss:.6f}',
                    ])
                    metrics_fh.flush()

            # Early stopping: decision on rank 0, broadcast so every DDP rank
            # leaves the loop together (a lone break would deadlock NCCL).
            stop_flag = torch.zeros(1, device=device)
            if (rank == 0 and args.early_stop_patience > 0
                    and epochs_since_best >= args.early_stop_patience):
                stop_flag[0] = 1.0
                _log_line(
                    f'[EarlyStop] no val_loss improvement for '
                    f'{epochs_since_best} validated epochs '
                    f'(best={best_val_loss:.6f}) — stopping after epoch {epoch+1}',
                    log_fh,
                )
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() > 0:
                break
    finally:
        if metrics_fh is not None:
            metrics_fh.close()
        if log_fh is not None:
            log_fh.close()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    train(parse_args())
