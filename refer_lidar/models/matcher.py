# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from models.structures import Instances


def _swap_wl_for_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w(y-span), l(x-span)] -> [cx, cy, w(x-span), h(y-span)]."""
    if boxes.shape[-1] < 4:
        return boxes
    return boxes[..., [0, 1, 3, 2]]


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_refer: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_refer = cost_refer
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets, use_focal=True):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            if use_focal:
                out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            else:
                out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            if isinstance(targets[0], Instances):
                tgt_ids = torch.cat([gt_per_img.labels for gt_per_img in targets])
                tgt_bbox = torch.cat([gt_per_img.boxes for gt_per_img in targets])
            else:
                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            if use_focal:
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            if isinstance(targets[0], Instances):
                sizes = [len(gt_per_img.boxes) for gt_per_img in targets]
            else:
                sizes = [len(v["boxes"]) for v in targets]

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# ── BEV Hungarian Matcher (center L2 distance) ───────────────────────────

class BEVHungarianMatcher(nn.Module):
    """Hungarian matcher using **BEV center L2 distance** instead of
    IoU-based costs.  Used when operating on LiDAR BEV coordinates.

    Predicted and target boxes can have 4 or more columns; the first two
    are always interpreted as normalised BEV centre ``(cx, cy)``.

    Cost =  cost_class * class_cost
          + cost_center * || (cx,cy)_pred − (cx,cy)_gt ||₂
          + cost_bbox   * L1(box_pred, box_gt[:, :box_pred_dim])
    """

    def __init__(self,
                 cost_class: float = 2.0,
                 cost_center: float = 5.0,
                 cost_bbox: float = 2.0,
                 cost_refer: float = 2.0,
                 refer_beta: float = 0.35):
        super().__init__()
        self.cost_class = cost_class
        self.cost_center = cost_center
        self.cost_bbox = cost_bbox
        self.cost_refer = cost_refer
        self.refer_beta = refer_beta

    @torch.no_grad()
    def forward(self, outputs, targets, use_focal=True):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # flatten
        if use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # (B*Q, D)

        if isinstance(targets[0], Instances):
            tgt_ids = torch.cat([t.labels for t in targets])
            tgt_bbox = torch.cat([t.boxes for t in targets])
        else:
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # --- classification cost (focal) ---
        if use_focal:
            alpha, gamma = 0.25, 2.0
            neg = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos[:, tgt_ids] - neg[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # --- BEV centre L2 distance ---
        pred_center = out_bbox[:, :2]           # (B*Q, 2)
        tgt_center = tgt_bbox[:, :2]            # (T, 2)
        cost_center = torch.cdist(pred_center, tgt_center, p=2)  # (B*Q, T)

        # --- optional L1 on full box (size dims) ---
        D = min(out_bbox.shape[-1], tgt_bbox.shape[-1])
        cost_bbox = torch.cdist(out_bbox[:, :D], tgt_bbox[:, :D], p=1)

        # --- geometric quality proxy (GIoU -> [0, 1]) ---
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(_swap_wl_for_xyxy(out_bbox[:, :4])),
            box_cxcywh_to_xyxy(_swap_wl_for_xyxy(tgt_bbox[:, :4])),
        )
        cost_giou = -giou
        geom_quality = giou.clamp(min=0.0, max=1.0)

        # --- language-aware refer quality cost ---
        if 'pred_refers' in outputs:
            pred_refer = outputs['pred_refers'].flatten(0, 1).sigmoid().squeeze(-1)
        else:
            # Fallback for compatibility: use first class channel as proxy.
            pred_refer = out_prob[:, 0]
        pred_refer = pred_refer.clamp(min=1e-6, max=1 - 1e-6)

        if 'pred_quality' in outputs:
            pred_quality = outputs['pred_quality'].flatten(0, 1).sigmoid().squeeze(-1)
            pred_quality = pred_quality.clamp(min=0.0, max=1.0)
            quality_score = (pred_quality[:, None] * geom_quality).clamp(0.0, 1.0)
        else:
            quality_score = geom_quality

        refer_quality = (
            torch.pow(pred_refer[:, None], 1.0 - self.refer_beta)
            * torch.pow(quality_score.clamp(min=1e-6, max=1 - 1e-6), self.refer_beta)
        ).clamp(min=1e-6, max=1 - 1e-6)

        if isinstance(targets[0], Instances):
            tgt_ref = torch.cat([t.is_ref for t in targets]).to(refer_quality.dtype)
        else:
            if 'is_ref' in targets[0]:
                tgt_ref = torch.cat([v['is_ref'] for v in targets]).to(refer_quality.dtype)
            else:
                tgt_ref = torch.ones_like(tgt_ids, dtype=refer_quality.dtype)

        cost_refer = -(
            tgt_ref[None, :] * torch.log(refer_quality)
            + (1.0 - tgt_ref[None, :]) * torch.log(1.0 - refer_quality)
        )

        # --- total ---
        C = (self.cost_class * cost_class
             + self.cost_center * cost_center
             + self.cost_bbox * cost_bbox
             + self.cost_refer * cost_refer
             + 0.5 * cost_giou)
        C = C.view(bs, num_queries, -1).cpu()

        if isinstance(targets[0], Instances):
            sizes = [len(t.boxes) for t in targets]
        else:
            sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args): # 匈牙利匹配器
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            cost_refer=args.set_cost_refer)


def build_bev_matcher(args):
    """Build BEV centre-distance matcher for LiDAR mode."""
    return BEVHungarianMatcher(
        cost_class=getattr(args, 'set_cost_class', 2.0),
        cost_center=getattr(args, 'set_cost_center', 5.0),
        cost_bbox=getattr(args, 'set_cost_bbox', 2.0),
        cost_refer=getattr(args, 'set_cost_refer', 2.0),
        refer_beta=getattr(args, 'set_cost_refer_beta', 0.35),
    )
