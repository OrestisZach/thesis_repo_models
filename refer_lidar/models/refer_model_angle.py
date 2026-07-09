"""ReferModel + SEED-style iterative angle refinement (Ablation A).

This meta-arch subclasses :class:`ReferModelLangDec` (the ``+lang`` model) and
adds exactly ONE capability on top of it: a per-decoder-layer angle head that
iteratively refines the heading used to rotate the deformable-attention
sampling grid (SEED's coarse-to-fine orientation), and ties the model's output
heading to that refined angle so mAOE scores the same angle that steered the
sampling.

Nothing in ``refer_model_lang_dec.py`` is modified — the ``+lang`` baseline
stays byte-identical and reproducible. The only shared change is an inert,
gated ``angle_embed`` hook in ``DeformableTransformerDecoderFinal`` (mirroring
the existing ``bbox_embed`` hook), which does nothing unless a model assigns to
it — which only this arch does.

Mechanism (mirrors the box-refinement pattern already in the decoder):
  * Init the reference heading from the frozen detector's proposal yaw
    (``ref_angles``) — already wired by the parent.
  * The decoder rotates layer L's sampling by the current ``ref_angles``, then
    refines it with a residual ``Δθ = angle_embed[L](hs[L])`` and detaches it
    for layer L+1 (deep supervision, no BPTT across layers — like SEED).
  * The model re-applies ``angle_embed[L]`` to ``hs[L]`` (with grad) on top of
    the detached input angle to produce ``pred_yaw[L]`` — the same trick the
    parent uses for ``bbox_embed`` (decoder refines detached refs; the head
    re-applies the shared MLP to ``hs`` for the supervised output).
  * ``angle_embed`` final layer is zero-initialised ⇒ at step 0 every layer's
    Δθ is 0 ⇒ rotation is identical to the fixed-proposal-yaw ``+lang`` model,
    and ``pred_yaw == proposal_yaw``. Training then learns the refinement.

Supervision: a dedicated wrapped-L1 ``loss_rad`` on matched queries. The
matcher and BEV GIoU stay axis-aligned by design, so this is a clean A/B vs
``+lang``. ``loss_3d`` drops the sin/cos dims (heading now comes from
``pred_yaw``); it keeps supervising z and height.
"""
import copy
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .matcher import build_bev_matcher
from .refer_model_lang_dec import MLP, ReferModelLangDec, SimpleSetCriterion


class ReferModelAngle(ReferModelLangDec):
    """``ReferModelLangDec`` + iterative decoder angle refinement."""

    def __init__(self, args):
        super().__init__(args)

        num_pred = self.transformer.decoder.num_layers
        angle_embed = MLP(self.hidden_dim, self.hidden_dim, 1, 3)
        # Zero-init the last layer: Δθ = 0 at start ⇒ identical to +lang's
        # fixed-proposal-yaw rotation until loss_rad moves it.
        nn.init.constant_(angle_embed.layers[-1].weight, 0.0)
        nn.init.constant_(angle_embed.layers[-1].bias, 0.0)
        self.angle_embed = nn.ModuleList(
            [copy.deepcopy(angle_embed) for _ in range(num_pred)]
        )
        # Wire the shared (otherwise inert) decoder hook so the sampling
        # rotation is refined per layer.
        self.transformer.decoder.angle_embed = self.angle_embed

    def forward(self, data: Dict):
        # Mirrors ReferModelLangDec.forward; the ONLY additions are the
        # per-layer pred_yaw head and its emission into out / aux_outputs.
        frame_srcs, frame_props, frame_yaw = self._prepare_pointpillars_batch(data)
        device = frame_srcs[0].device

        batch_size = frame_srcs[0].shape[0]

        flat_sentences, n_per_frame = self._flatten_sentences(
            data["sentences"], batch_size, device)
        srcs, ref_boxes = self._expand_frame_to_query_batch(
            frame_srcs, frame_props, n_per_frame)

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
            text_mask=text_mask,
            ref_angles=ref_angles,
        )

        # (L, Bq, Q) detached per-layer refined headings, or None if the batch
        # carried no proposal yaw (then we leave head_3d's heading untouched).
        inter_ang = self.transformer.decoder.last_ref_angles
        use_yaw = ref_angles is not None and inter_ang is not None

        outputs_classes = []
        outputs_coords = []
        outputs_refers = []
        outputs_quality = []
        outputs_3d = []
        outputs_yaw = []

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

            if use_yaw:
                # Input heading to THIS layer's sampling = proposal yaw (L0) or
                # the detached refined heading from the previous layer. Re-apply
                # the shared angle head to hs[lvl] WITH grad to get pred_yaw.
                ref_ang_in = ref_angles if lvl == 0 else inter_ang[lvl - 1]
                d_ang = self.angle_embed[lvl](hs[lvl]).squeeze(-1)
                outputs_yaw.append(ref_ang_in + d_ang)

        out = {
            "pred_logits": outputs_classes[-1],
            "pred_boxes": outputs_coords[-1],
            "pred_refers": outputs_refers[-1],
            "pred_quality": outputs_quality[-1],
            "pred_3d": outputs_3d[-1],
            "seed_ref_points": ref_boxes,
        }
        if use_yaw:
            out["pred_yaw"] = outputs_yaw[-1]
        out["aux_outputs"] = [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_refers": r,
                "pred_quality": q,
                "pred_3d": d,
                **({"pred_yaw": y} if use_yaw else {}),
            }
            for a, b, r, q, d, y in zip(
                outputs_classes[:-1], outputs_coords[:-1], outputs_refers[:-1],
                outputs_quality[:-1], outputs_3d[:-1],
                (outputs_yaw[:-1] if use_yaw else [None] * (len(outputs_classes) - 1)),
            )
        ]
        return out


class AngleSetCriterion(SimpleSetCriterion):
    """``SimpleSetCriterion`` + wrapped-L1 ``loss_rad``; heading dropped from ``loss_3d``."""

    @staticmethod
    def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))

    def loss_3d(self, outputs, targets, indices, num_boxes):
        # Heading now comes from pred_yaw / loss_rad — supervise only z, height
        # (attrs_3d dims 0,1); drop sin/cos (dims 2,3) and the 0.0 pads (4,5).
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {'loss_3d': outputs['pred_3d'].sum() * 0.0}
        src_3d = outputs['pred_3d'][idx][:, :2]
        tgt_3d = torch.cat(
            [t.attrs_3d[j] for t, (_, j) in zip(targets, indices)], dim=0
        )[:, :2].to(src_3d.dtype)
        loss = F.smooth_l1_loss(src_3d, tgt_3d, reduction='sum') / num_boxes
        return {'loss_3d': loss}

    def loss_rad(self, outputs, targets, indices, num_boxes):
        # Keep angle_embed in the autograd graph even when unsupervised.
        if 'pred_yaw' not in outputs:
            return {'loss_rad': outputs['pred_boxes'].sum() * 0.0}
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {'loss_rad': outputs['pred_yaw'].sum() * 0.0}
        src_yaw = outputs['pred_yaw'][idx]
        tgt_attrs = torch.cat(
            [t.attrs_3d[j] for t, (_, j) in zip(targets, indices)], dim=0
        ).to(src_yaw.dtype)
        gt_yaw = torch.atan2(tgt_attrs[:, 2], tgt_attrs[:, 3])
        diff = self._wrap_to_pi(src_yaw - gt_yaw)
        return {'loss_rad': diff.abs().sum() / num_boxes}

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        if loss == 'rad':
            return self.loss_rad(outputs, targets, indices, num_boxes)
        return super().get_loss(loss, outputs, targets, indices, num_boxes)


def build_angle(args):
    """Build the SEED-style angle-refinement arch (Ablation A).

    Identical defaults to ``refer_model_lang_dec.build_lang_dec`` plus one extra
    loss (``loss_rad``); constructs :class:`ReferModelAngle` +
    :class:`AngleSetCriterion`.
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
    if not hasattr(args, 'loss_rad_coef'):
        args.loss_rad_coef = 2.0
    if not hasattr(args, 'focal_alpha'):
        args.focal_alpha = 0.25
    if not hasattr(args, 'aux_loss'):
        args.aux_loss = True

    model = ReferModelAngle(args)
    matcher = build_bev_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_refer': args.refer_loss_coef,
        'loss_3d': args.loss_3d_coef,
        'loss_quality': args.quality_loss_coef,
        'loss_rad': args.loss_rad_coef,
    }

    if args.aux_loss:
        aux_weight_dict = {}
        # Aux outputs exist for decoder layers except the final one.
        for i in range(model.transformer.decoder.num_layers - 1):
            for k, v in weight_dict.items():
                aux_weight_dict[f'{k}_{i}'] = v
        weight_dict.update(aux_weight_dict)

    criterion = AngleSetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=['labels', 'boxes', 'refer', '3d', 'quality', 'rad'],
        focal_alpha=args.focal_alpha,
    )
    return model, criterion, {}
