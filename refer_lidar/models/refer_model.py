import os
from typing import Dict, List, Sequence, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch import nn
from transformers import RobertaModel, RobertaTokenizerFast
import copy

from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.utils import register_all_modules
from util import box_ops
from util.misc import accuracy, get_world_size, is_dist_avail_and_initialized

from .deformable_transformer_plus import (
    FeatureResizer,
    PositionEmbeddingSine2D,
    VisionLanguageFusionModule,
    build_deforamble_transformer,
)
from .matcher import build_bev_matcher

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MLP(nn.Module):
    """Simple multi-layer perceptron used by ReferModel heads."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PointPillarsDetectorBridge(nn.Module):
    """Thin wrapper around a pretrained MMDetection3D PointPillars model.

    It exposes:
    - multi-level FPN feature extraction
    - post-NMS proposals for query initialization
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        point_cloud_range: Sequence[float],
        proposal_count: int = 150,
        proposal_w_from: str = "dy",
        freeze: bool = True,
    ):
        super().__init__()
        self.proposal_count = proposal_count
        self.point_cloud_range = point_cloud_range
        self.proposal_w_from = proposal_w_from

        this_file = Path(__file__).resolve()
        refer_lidar_root = this_file.parents[1]
        repo_root = this_file.parents[2]

        def _resolve_existing_path(p: str) -> str:
            cand = Path(p)
            candidates = []
            if cand.is_absolute():
                candidates.append(cand)
            else:
                candidates.extend([
                    Path.cwd() / cand,
                    refer_lidar_root / cand,
                    repo_root / cand,
                ])
            for c in candidates:
                if c.exists():
                    return str(c)
            return str(candidates[0] if candidates else cand)

        config_path = _resolve_existing_path(config_path)
        checkpoint_path = _resolve_existing_path(checkpoint_path)

        # Ensure MMDet3D custom modules (e.g., Det3DDataPreprocessor) are registered.
        register_all_modules(init_default_scope=True)

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'PointPillars checkpoint not found: {checkpoint_path}')
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f'PointPillars config not found: {config_path}')

        cfg = Config.fromfile(config_path)
        self.detector = MODELS.build(cfg.model)
        _ = load_checkpoint(self.detector, checkpoint_path, map_location="cpu")
        print(f"[PointPillarsBridge] Loaded config: {config_path}", flush=True)
        print(f"[PointPillarsBridge] Loaded checkpoint: {checkpoint_path}", flush=True)
        if freeze:
            self.detector.eval()
            for p in self.detector.parameters():
                p.requires_grad_(False)

    def _make_metas(self, batch_size: int) -> List[dict]:
        return [
            {
                "box_type_3d": LiDARInstance3DBoxes,
            }
            for _ in range(batch_size)
        ]

    @staticmethod
    def _ensure_pointpillars_input(points: List[torch.Tensor]) -> List[torch.Tensor]:
        """Coerce input to the 4-D layout the stock nuScenes PointPillars was
        trained on: [x, y, z, dt].

        The stock checkpoint
        (``hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_*.pth``) was trained via
        ``configs/_base_/datasets/nus-3d.py``, where ``LoadPointsFromMultiSweeps``
        uses its default ``use_dim=[0, 1, 2, 4]`` — dropping intensity (col 3)
        and keeping the sweep-time offset (col 4) at position 3 of the output.
        ``HardVFE.in_channels=4`` consumes that exact layout. **Col 3 here is
        dt, not intensity** — the misleading "Keep intensity" comment that
        used to live here was wrong.

        Our dataset already emits 4-D ``[x, y, z, dt]`` when
        ``num_point_features=4`` (which the live ``refer_model`` path forces),
        so 4-D input passes through unchanged. For 5-D
        ``[x, y, z, intensity, dt]`` (e.g. mistakenly passed in from a
        ``num_point_features=5`` configuration) we drop intensity and keep dt
        at col 3 to match training.
        """
        fixed: List[torch.Tensor] = []
        for p in points:
            if p.shape[-1] == 5:
                # 5-D input [x, y, z, intensity, dt] -> [x, y, z, dt].
                fixed.append(torch.cat([p[:, :3], p[:, 4:5]], dim=-1).contiguous())
            elif p.shape[-1] == 4:
                # Assumed [x, y, z, dt]; matches stock training. Pass through.
                fixed.append(p.contiguous())
            else:
                raise RuntimeError(
                    f'Unexpected point feature dim={p.shape[-1]}; expected 4 or 5.'
                )
        return fixed

    def _normalize_proposals(self, box_tensor: torch.Tensor) -> torch.Tensor:
        """Convert detector boxes [x, y, z, dx, dy, dz, yaw, ...] -> normalized cxcywl."""
        pc = self.point_cloud_range
        x_min, y_min, _, x_max, y_max, _ = pc
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)

        cx = (box_tensor[:, 0] - x_min) / x_span
        cy = (box_tensor[:, 1] - y_min) / y_span

        dx = box_tensor[:, 3]
        dy = box_tensor[:, 4]
        
        if self.proposal_w_from == "dy":
            w, l = dy, dx
        else:
            w, l = dx, dy

        w_norm = w / y_span
        l_norm = l / x_span

        # Stack ONLY the 4 BEV dimensions, discarding z, dz, and yaw
        out = torch.stack([cx, cy, w_norm, l_norm], dim=-1)
        
        return out.clamp(0.0, 1.0)

    @staticmethod
    def _pad_proposals(cxcywl: torch.Tensor, scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = cxcywl.shape[0]
        if n >= k:
            return cxcywl[:k], scores[:k]
        if n == 0:
            pad_box = torch.zeros(k, 4, dtype=cxcywl.dtype, device=cxcywl.device)
            pad_score = torch.zeros(k, dtype=scores.dtype, device=scores.device)
            return pad_box, pad_score
        pad_n = k - n
        pad_box = cxcywl.new_zeros(pad_n, 4)
        pad_box[:, :2] = 0.5
        pad_score = scores.new_zeros(pad_n)
        return torch.cat([cxcywl, pad_box], dim=0), torch.cat([scores, pad_score], dim=0)

    @torch.no_grad()
    def forward(self, points: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        points = self._ensure_pointpillars_input(points)
        batch_size = len(points)
        metas = self._make_metas(batch_size)

        voxel_dict = self.detector.data_preprocessor.voxelize(points, data_samples=[])

        # extract_pts_feat returns the 3 FPN levels natively from pts_neck
        feats = self.detector.extract_pts_feat(
            voxel_dict,
            points=points,
            img_feats=None,
            batch_input_metas=metas,
        )
        
        cls_scores, bbox_preds, dir_cls_preds = self.detector.pts_bbox_head(feats)
        results = self.detector.pts_bbox_head.predict_by_feat(
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            batch_input_metas=metas,
            cfg=self.detector.pts_bbox_head.test_cfg,
            rescale=False,
        )

        prop_boxes = []
        prop_scores = []
        for det in results:
            b = det.bboxes_3d.tensor
            s = det.scores_3d
            if b.numel() == 0:
                norm = b.new_zeros((0, 4))
            else:
                order = torch.argsort(s, descending=True)
                b = b[order]
                s = s[order]
                norm = self._normalize_proposals(b)
            norm, s = self._pad_proposals(norm, s, self.proposal_count)
            prop_boxes.append(norm)
            prop_scores.append(s)

        return list(feats), torch.stack(prop_boxes, dim=0), torch.stack(prop_scores, dim=0)


class ReferModel(nn.Module):
    """Single-frame LiDAR referring detector with PointPillars proposal seeding."""

    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.num_feature_levels = getattr(args, "num_feature_levels", 3)
        self.num_queries = getattr(args, "proposal_queries", 150)
        self.dga_grid_size = int(getattr(args, 'dga_grid_size', 5))
        self.point_cloud_range = getattr(
            args,
            "point_cloud_range",
            [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        )

        self.detector_bridge = PointPillarsDetectorBridge(
            config_path=args.pointpillars_config,
            checkpoint_path=args.pointpillars_ckpt,
            point_cloud_range=self.point_cloud_range,
            proposal_count=self.num_queries,
            proposal_w_from=getattr(args, "proposal_w_from", "dy"),
            freeze=getattr(args, "freeze_pointpillars", True),
        )

        self.transformer = build_deforamble_transformer(args)
        self.class_embed = nn.Linear(self.hidden_dim, 1)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.refer_embed = nn.Linear(self.hidden_dim, 1)
        self.quality_embed = nn.Linear(self.hidden_dim, 1)
        self.head_3d = MLP(self.hidden_dim, self.hidden_dim, 6, 3)

        # Query content is sampled from fused FPN; query position is box-conditioned.
        self.query_pos_mlp = MLP(4, self.hidden_dim, self.hidden_dim, 3)

        self.level_pos_enc = PositionEmbeddingSine2D(normalize=True)
        self.fusion_module = VisionLanguageFusionModule(d_model=self.hidden_dim, nhead=8)
        self.concat_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.num_feature_levels)
            ]
        )

        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        self.txt_proj = FeatureResizer(
            input_feat_size=self.text_encoder.config.hidden_size,
            output_feat_size=self.hidden_dim,
            dropout=0.1,
        )

        num_pred = self.transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for _ in range(num_pred)])
        self.refer_embed = nn.ModuleList([copy.deepcopy(self.refer_embed) for _ in range(num_pred)])
        self.quality_embed = nn.ModuleList([copy.deepcopy(self.quality_embed) for _ in range(num_pred)])
        self.head_3d = nn.ModuleList([copy.deepcopy(self.head_3d) for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = self.bbox_embed

    def forward_text(self, text_queries: List[str], device: torch.device):
        tokenized = self.tokenizer.batch_encode_plus(
            text_queries,
            padding="longest",
            return_tensors="pt",
        ).to(device)
        encoded = self.text_encoder(**tokenized)
        text_features = self.txt_proj(encoded.last_hidden_state)
        text_pad_mask = tokenized.attention_mask.ne(1).bool()
        return text_features, text_pad_mask

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x = x.clamp(min=eps, max=1 - eps)
        return torch.log(x / (1 - x))

    @staticmethod
    def _flatten_sentences(sentences: List, batch_size: int, device: torch.device):
        if len(sentences) > 0 and isinstance(sentences[0], list):
            n_per_frame = [len(s) for s in sentences]
            flat = [q for s in sentences for q in s]
        else:
            if isinstance(sentences, str):
                sentences = [sentences]
            n_per_frame = [1] * batch_size
            flat = list(sentences)
        n_per_frame = torch.as_tensor(n_per_frame, dtype=torch.long, device=device)
        return flat, n_per_frame

    def _expand_frame_to_query_batch(
        self,
        frame_tensors: List[torch.Tensor],
        frame_props: torch.Tensor,
        n_per_frame: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if int(n_per_frame.sum().item()) == frame_tensors[0].shape[0]:
            return frame_tensors, frame_props

        expanded = []
        for src in frame_tensors:
            expanded.append(torch.repeat_interleave(src, n_per_frame, dim=0))
        props = torch.repeat_interleave(frame_props, n_per_frame, dim=0)
        return expanded, props

    def _language_fuse(self, srcs: List[torch.Tensor], text_mem: torch.Tensor, text_mask: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        fused_srcs: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        pos: List[torch.Tensor] = []

        mem = text_mem.permute(1, 0, 2)  # (S, B, D)
        for l, src in enumerate(srcs):
            b, c, h, w = src.shape
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=src.device)
            pos_l = self.level_pos_enc(mask, self.hidden_dim).permute(0, 3, 1, 2)

            src_flat = src.flatten(2).permute(2, 0, 1)  # (HW, B, C)
            fused_flat = self.fusion_module(
                tgt=src_flat,
                memory=mem,
                memory_key_padding_mask=text_mask,
                pos=None,
                query_pos=None,
            )
            fused = fused_flat.permute(1, 2, 0).reshape(b, c, h, w)

            cat = torch.cat([src, fused], dim=1)
            fused_src = self.concat_proj[l](cat)

            fused_srcs.append(fused_src)
            masks.append(mask)
            pos.append(pos_l.to(fused_src.dtype))

        return fused_srcs, masks, pos

    def _build_query_embed(self, fused_srcs: List[torch.Tensor], ref_boxes: torch.Tensor,
                           ref_angles: torch.Tensor = None) -> torch.Tensor:
        # SEED-style idea: sample a local grid inside each proposal instead of
        # only the center, then aggregate to query content. When ref_angles is
        # given, the grid is ROTATED by the proposal heading (SEED's
        # with_rotation), so the k x k footprint aligns to the oriented box.
        src = fused_srcs[0]
        b, c, _, _ = src.shape
        q = ref_boxes.shape[1]
        k = max(self.dga_grid_size, 1)

        cx = ref_boxes[:, :, 0]
        cy = ref_boxes[:, :, 1]
        w = ref_boxes[:, :, 2]
        l = ref_boxes[:, :, 3]

        offsets = torch.linspace(-0.5, 0.5, steps=k, device=src.device, dtype=src.dtype)
        gy, gx = torch.meshgrid(offsets, offsets, indexing='ij')

        # box-local-frame offsets: heading axis scaled by length l, perpendicular by width w
        ox = gx[None, None, :, :] * l[:, :, None, None]
        oy = gy[None, None, :, :] * w[:, :, None, None]
        if ref_angles is not None:
            ang = ref_angles[:, :, None, None].to(ox.dtype)
            cos_a, sin_a = torch.cos(ang), torch.sin(ang)
            ox, oy = ox * cos_a - oy * sin_a, ox * sin_a + oy * cos_a

        grid_x = cx[:, :, None, None] + ox
        grid_y = cy[:, :, None, None] + oy
        grid = torch.stack([grid_x, grid_y], dim=-1).clamp(0.0, 1.0)
        grid = grid * 2.0 - 1.0

        sampled = F.grid_sample(src, grid.view(b, q * k, k, 2), align_corners=False)
        sampled = sampled.view(b, c, q, k, k).mean(dim=(-1, -2)).transpose(1, 2).contiguous()  # (B, Q, C)

        query_pos = self.query_pos_mlp(ref_boxes)
        return torch.cat([query_pos, sampled], dim=-1)

    def _prepare_pointpillars_batch(self, data: Dict):
        """Prepare PointPillars features/proposals from cache and/or raw points.

        Supports three modes:
        1) Raw points only.
        2) Cached features/proposals only (fast path, skips PointPillars inference).
        3) Mixed batch (some cached, some raw points).
        """
        cached_srcs = data.get('pointpillars_srcs')
        cached_props = data.get('pointpillars_props')
        cached_scores = data.get('pointpillars_scores')
        cached_yaw = data.get('pointpillars_yaw')

        # Fast path: no cache payload provided — run PointPillars live.
        if cached_srcs is None or cached_props is None:
            points = data.get('points')
            if points is None:
                raise RuntimeError(
                    'ReferModel expects either `points` or cached '
                    '`pointpillars_srcs` + `pointpillars_props`.'
                )
            if not isinstance(points, list):
                points = [points]
            feats, props, _scores = self.detector_bridge(points)
            return list(feats), props, None   # live path has no cached yaw -> no rotation

        if not isinstance(cached_srcs, list) or not isinstance(cached_props, list):
            raise RuntimeError('Cached PointPillars payload must be list-based (collated batch format).')

        batch_size = len(cached_props)
        if len(cached_srcs) != batch_size:
            raise RuntimeError('Cached feature/proposal batch size mismatch.')
        if cached_scores is not None and len(cached_scores) != batch_size:
            raise RuntimeError('Cached scores batch size mismatch.')

        per_sample_srcs = [None] * batch_size
        per_sample_props = [None] * batch_size
        per_sample_scores = [None] * batch_size
        per_sample_yaw = [None] * batch_size
        needs_detector = []

        for b in range(batch_size):
            src_b = cached_srcs[b]
            prop_b = cached_props[b]
            score_b = cached_scores[b] if cached_scores is not None else None
            yaw_b = cached_yaw[b] if cached_yaw is not None else None

            if src_b is None or prop_b is None:
                needs_detector.append(b)
                continue

            if not isinstance(src_b, list) or len(src_b) == 0:
                raise RuntimeError('Cached `pointpillars_srcs` must be a non-empty list per sample.')

            norm_levels = []
            for lvl in src_b:
                if lvl.dim() == 4 and lvl.shape[0] == 1:
                    lvl = lvl[0]
                if lvl.dim() != 3:
                    raise RuntimeError(f'Cached feature map must be 3D (C,H,W), got shape={tuple(lvl.shape)}.')
                norm_levels.append(lvl)

            if prop_b.dim() == 3 and prop_b.shape[0] == 1:
                prop_b = prop_b[0]
            if prop_b.dim() != 2:
                raise RuntimeError(f'Cached proposals must be 2D (Q, >=8), got shape={tuple(prop_b.shape)}.')

            if score_b is None:
                score_b = prop_b.new_zeros(prop_b.shape[0])
            elif score_b.dim() == 2 and score_b.shape[0] == 1:
                score_b = score_b[0]

            if yaw_b is not None and yaw_b.dim() == 2 and yaw_b.shape[0] == 1:
                yaw_b = yaw_b[0]

            order = torch.argsort(score_b, descending=True)
            prop_b = prop_b[order]
            score_b = score_b[order]
            if yaw_b is not None:
                yaw_b = yaw_b[order]

            # PointPillars natively outputs 8D proposals, keep them for padding
            prop_4d = prop_b[:, :4]
            prop_4d, score_b = self.detector_bridge._pad_proposals(prop_4d, score_b, self.num_queries)

            per_sample_srcs[b] = norm_levels
            per_sample_props[b] = prop_4d
            per_sample_scores[b] = score_b
            if yaw_b is not None:
                per_sample_yaw[b] = self._pad_yaw(yaw_b, self.num_queries)

        if needs_detector:
            points = data.get('points')
            if points is None:
                raise RuntimeError(
                    'Cache miss inside batch but `points` are missing. '
                    'Provide complete cache or disable cache strictness.'
                )
            if not isinstance(points, list):
                points = [points]

            points_subset = [points[b] for b in needs_detector]
            miss_srcs, miss_props, miss_scores = self.detector_bridge(points_subset)
            for i, b in enumerate(needs_detector):
                per_sample_srcs[b] = [lvl[i] for lvl in miss_srcs]
                per_sample_props[b] = miss_props[i]
                per_sample_scores[b] = miss_scores[i]
                per_sample_yaw[b] = None   # live path: no yaw

        num_levels = len(per_sample_srcs[0])
        frame_srcs = [
            torch.stack([per_sample_srcs[b][lvl] for b in range(batch_size)], dim=0)
            for lvl in range(num_levels)
        ]
        frame_props = torch.stack(per_sample_props, dim=0)
        # Only emit a frame_yaw tensor if EVERY sample has yaw (mixed/None -> no rotation).
        if all(y is not None for y in per_sample_yaw):
            frame_yaw = torch.stack(per_sample_yaw, dim=0)
        else:
            frame_yaw = None
        return frame_srcs, frame_props, frame_yaw

    @staticmethod
    def _pad_yaw(yaw: torch.Tensor, k: int) -> torch.Tensor:
        n = yaw.shape[0]
        if n >= k:
            return yaw[:k]
        if n == 0:
            return yaw.new_zeros(k)
        return torch.cat([yaw, yaw.new_zeros(k - n)], dim=0)
    
    def forward(self, data: Dict):
        frame_srcs, frame_props, frame_yaw = self._prepare_pointpillars_batch(data)
        device = frame_srcs[0].device

        batch_size = frame_srcs[0].shape[0]

        flat_sentences, n_per_frame = self._flatten_sentences(data["sentences"], batch_size, device)
        srcs, ref_boxes = self._expand_frame_to_query_batch(frame_srcs, frame_props, n_per_frame)

        # Expand the per-proposal heading to the query batch (same as srcs/boxes).
        ref_angles = None
        if frame_yaw is not None:
            if int(n_per_frame.sum().item()) == frame_yaw.shape[0]:
                ref_angles = frame_yaw
            else:
                ref_angles = torch.repeat_interleave(frame_yaw, n_per_frame, dim=0)

        text_feats, text_mask = self.forward_text(flat_sentences, device)
        fused_srcs, masks, pos = self._language_fuse(srcs, text_feats, text_mask)

        query_embed = self._build_query_embed(fused_srcs, ref_boxes, ref_angles)
        ref_pts_logits = self._inverse_sigmoid(ref_boxes)

        hs, init_ref, inter_refs, _, _ = self.transformer(
            fused_srcs,
            masks,
            pos,
            query_embed=query_embed,
            sentence_embeds=text_feats,
            ref_pts=ref_pts_logits,
            ref_angles=ref_angles,
        )

        outputs_classes = []
        outputs_coords = []
        outputs_refers = []
        outputs_quality = []
        outputs_3d = []

        for lvl in range(hs.shape[0]):
            ref = init_ref if lvl == 0 else inter_refs[lvl - 1]
            ref = self._inverse_sigmoid(ref)

            out_class = self.class_embed[lvl](hs[lvl])
            out_refer = self.refer_embed[lvl](hs[lvl])
            out_quality = self.quality_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if ref.shape[-1] == 4:
                tmp += ref
            else:
                tmp[..., :2] += ref
            out_coord = tmp.sigmoid()

            outputs_classes.append(out_class)
            outputs_coords.append(out_coord)
            outputs_refers.append(out_refer)
            outputs_quality.append(out_quality)
            outputs_3d.append(self.head_3d[lvl](hs[lvl]))

        out = {
            "pred_logits": outputs_classes[-1],
            "pred_boxes": outputs_coords[-1],
            "pred_refers": outputs_refers[-1],
            "pred_quality": outputs_quality[-1],
            "pred_3d": outputs_3d[-1],
            "seed_ref_points": ref_boxes,
        }
        out["aux_outputs"] = [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_refers": r,
                "pred_quality": q,
                "pred_3d": d,
            }
            for a, b, r, q, d in zip(outputs_classes[:-1], outputs_coords[:-1], outputs_refers[:-1], outputs_quality[:-1], outputs_3d[:-1])
        ]
        return out


class SimpleSetCriterion(nn.Module):
    """Criterion for ReferModel using Hungarian matching in BEV."""

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    @staticmethod
    def _flatten_targets(targets):
        if targets is None:
            return []
        if len(targets) == 0:
            return []
        if isinstance(targets[0], list):
            flat = []
            for t in targets:
                flat.extend(t)
            return flat
        return targets

    @staticmethod
    def _move_targets_to_device(targets, device):
        moved = []
        for t in targets:
            if hasattr(t, 'to'):
                moved.append(t.to(device))
            elif isinstance(t, dict):
                moved.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()})
            else:
                moved.append(t)
        return moved

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _swap_wl_for_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert [cx, cy, w(y-span), l(x-span)] -> [cx, cy, w(x-span), h(y-span)]."""
        if boxes.shape[-1] < 4:
            return boxes
        return boxes[..., [0, 1, 3, 2]]

    @staticmethod
    def _sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_boxes

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t.labels[J] for t, (_, J) in zip(targets, indices)], dim=0)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = self._sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=2,
        ) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log and target_classes_o.numel() > 0:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t.boxes[i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(self._swap_wl_for_xyxy(src_boxes)),
                box_ops.box_cxcywh_to_xyxy(self._swap_wl_for_xyxy(target_boxes)),
            )
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_refer(self, outputs, targets, indices, num_boxes):
        pred = outputs['pred_refers'].squeeze(-1)
        target = torch.zeros_like(pred)
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() > 0:
            matched_ref = torch.cat([t.is_ref[j] for t, (_, j) in zip(targets, indices)], dim=0).to(pred.dtype)
            target[idx] = matched_ref
        loss_refer = F.binary_cross_entropy_with_logits(pred, target, reduction='sum') / num_boxes
        return {'loss_refer': loss_refer}

    def loss_3d(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {'loss_3d': outputs['pred_3d'].sum() * 0.0}
        src_3d = outputs['pred_3d'][idx]
        tgt_3d = torch.cat([t.attrs_3d[j] for t, (_, j) in zip(targets, indices)], dim=0).to(src_3d.dtype)
        loss = F.smooth_l1_loss(src_3d, tgt_3d, reduction='sum') / num_boxes
        return {'loss_3d': loss}

    def loss_quality(self, outputs, targets, indices, num_boxes):
        if 'pred_quality' not in outputs:
            return {'loss_quality': outputs['pred_boxes'].sum() * 0.0}
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {'loss_quality': outputs['pred_quality'].sum() * 0.0}

        src_q = outputs['pred_quality'][idx].squeeze(-1)
        src_boxes = outputs['pred_boxes'][idx]
        tgt_boxes = torch.cat([t.boxes[i] for t, (_, i) in zip(targets, indices)], dim=0)

        iou, _ = box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(self._swap_wl_for_xyxy(src_boxes)),
            box_ops.box_cxcywh_to_xyxy(self._swap_wl_for_xyxy(tgt_boxes)),
        )
        tgt_iou = torch.diag(iou).clamp(0.0, 1.0).detach().to(src_q.dtype)
        loss_q = F.binary_cross_entropy_with_logits(src_q, tgt_iou, reduction='sum') / num_boxes
        return {'loss_quality': loss_q}

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'refer': self.loss_refer,
            '3d': self.loss_3d,
            'quality': self.loss_quality,
        }
        assert loss in loss_map, f'Unsupported loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        targets = self._flatten_targets(targets)
        if len(targets) == 0:
            z = outputs['pred_boxes'].sum() * 0.0
            return {
                'loss_ce': z,
                'loss_bbox': z,
                'loss_giou': z,
                'loss_refer': z,
                'loss_3d': z,
                'loss_quality': z,
            }

        targets = self._move_targets_to_device(targets, outputs['pred_boxes'].device)

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t.labels) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_boxes'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss_name in self.losses:
            losses.update(self.get_loss(loss_name, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_out in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_out, targets)
                for loss_name in self.losses:
                    l_dict = self.get_loss(loss_name, aux_out, targets, aux_indices, num_boxes)
                    l_dict = {f'{k}_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build(args):
    """Build proposal-seeded single-frame LiDAR referring model.

    Expected args additions:
    - pointpillars_config
    - pointpillars_ckpt
    - proposal_queries (defaults to 150)
    - proposal_w_from in {'dy','dx'}
    """
    if not hasattr(args, "pointpillars_config"):
        args.pointpillars_config = (
            "configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
        )
    if not hasattr(args, "proposal_queries"):
        args.proposal_queries = 150
    # PointPillars FPN emits 3 levels; keep transformer levels aligned.
    args.num_feature_levels = 3

    # Loss defaults for detection + referring + 3D attributes.
    if not hasattr(args, 'set_cost_class'):
        args.set_cost_class = 2.0
    if not hasattr(args, 'set_cost_bbox'):
        args.set_cost_bbox = 5.0
    if not hasattr(args, 'set_cost_center'):
        args.set_cost_center = 5.0
    if not hasattr(args, 'set_cost_refer'):
        args.set_cost_refer = 2.0
    if not hasattr(args, 'cls_loss_coef'):
        args.cls_loss_coef = 2.0
    if not hasattr(args, 'bbox_loss_coef'):
        args.bbox_loss_coef = 5.0
    if not hasattr(args, 'giou_loss_coef'):
        args.giou_loss_coef = 2.0
    if not hasattr(args, 'refer_loss_coef'):
        args.refer_loss_coef = 2.0
    if not hasattr(args, 'loss_3d_coef'):
        args.loss_3d_coef = 2.0
    if not hasattr(args, 'quality_loss_coef'):
        args.quality_loss_coef = 1.0
    if not hasattr(args, 'focal_alpha'):
        args.focal_alpha = 0.25
    if not hasattr(args, 'aux_loss'):
        args.aux_loss = True

    model = ReferModel(args)
    matcher = build_bev_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_refer': args.refer_loss_coef,
        'loss_3d': args.loss_3d_coef,
        'loss_quality': args.quality_loss_coef,
    }

    if args.aux_loss:
        aux_weight_dict = {}
        # Aux outputs exist for decoder layers except the final one.
        for i in range(model.transformer.decoder.num_layers - 1):
            for k, v in weight_dict.items():
                aux_weight_dict[f'{k}_{i}'] = v
        weight_dict.update(aux_weight_dict)

    criterion = SimpleSetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=['labels', 'boxes', 'refer', '3d', 'quality'],
        focal_alpha=args.focal_alpha,
    )
    return model, criterion, {}
