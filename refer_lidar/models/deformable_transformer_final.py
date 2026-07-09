# ------------------------------------------------------------------------
# Deformable Transformer FINAL — Per-Query Language Cross-Attention
# ------------------------------------------------------------------------
# This is byte-for-byte the V1 transformer (deformable_transformer_plus.py)
# in EVERY respect except one addition: each decoder layer gains a language
# cross-attention sub-layer where every object query attends to the full
# per-token RoBERTa sequence.
#
# Sub-layer order per decoder layer:
#     self-attention -> (NEW) language cross-attention
#                    -> deformable spatial cross-attention -> FFN
#
# - The encoder, VisionLanguageFusionModule, FeatureResizer,
#   PositionEmbeddingSine2D and all other components are imported UNCHANGED
#   from the V1 module — guaranteeing V1 parity outside the decoder.
# - The decoder threads text_memory (per-token features) + text_mask to
#   ALL layers.
# - When text_memory is None the new sub-layer is skipped, so the decoder
#   reproduces V1 numerically.
# ------------------------------------------------------------------------

from typing import Optional

import torch
from torch import nn

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn

# Re-export original V1 components unchanged (same base as refer_model.py)
from .deformable_transformer_plus import (
    DeformableTransformer,
    DeformableTransformerEncoder,
    DeformableTransformerEncoderLayer,
    FeatureResizer,
    PositionEmbeddingSine2D,
    VisionLanguageFusionModule,
    _get_activation_fn,
    _get_clones,
)


class DeformableTransformerDecoderLayerFinal(nn.Module):
    """V1 decoder layer + per-query language cross-attention.

    Order: self-attention -> language cross-attention
           -> spatial deformable cross-attention -> FFN

    The language cross-attention lets each object query attend to all
    RoBERTa tokens, maintaining a clear language signal through every layer.
    Standard ``nn.MultiheadAttention`` with default (random) init, residual
    add + dedicated LayerNorm, dropout = model dropout.
    """

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 self_cross=True, sigmoid_attn=False, extra_track_attn=False):
        super().__init__()

        self.self_cross = self_cross
        self.num_head = n_heads

        # --- Self-attention (identical to V1) ---
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # --- Language cross-attention (NEW; default random init) ---
        self.lang_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.lang_dropout = nn.Dropout(dropout)
        self.lang_norm = nn.LayerNorm(d_model)

        # --- Spatial deformable cross-attention (identical to V1) ---
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # --- FFN (identical to V1) ---
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # Extra track attention (kept for V1 compatibility, typically False)
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_self_attn(self, tgt, query_pos, attn_mask=None):
        if self.extra_track_attn:
            tgt = self._forward_track_attn(tgt, query_pos)

        q = k = self.with_pos_embed(tgt, query_pos)
        if attn_mask is not None:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
                                  attn_mask=attn_mask)[0].transpose(0, 1)
        else:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)

    def _forward_track_attn(self, tgt, query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        if q.shape[1] > 300:
            tgt2 = self.update_attn(
                q[:, 300:].transpose(0, 1),
                k[:, 300:].transpose(0, 1),
                tgt[:, 300:].transpose(0, 1),
            )[0].transpose(0, 1)
            tgt = torch.cat([tgt[:, :300], self.norm4(tgt[:, 300:] + self.dropout5(tgt2))], dim=1)
        return tgt

    def _forward_lang_cross_attn(self, tgt, query_pos, text_memory, text_mask):
        """Per-query language cross-attention (NEW sub-layer).

        Each object query (Q) attends to all language tokens (K/V).
        - query: object queries + positional encoding (with_pos_embed)
        - key/value: per-token RoBERTa features (NOT pooled/averaged)

        Args:
            tgt: (B, Q, D) object queries
            query_pos: (B, Q, D) positional encoding
            text_memory: (B, S, D) per-token language features
            text_mask: (B, S) bool mask (True = padded, ignore)
        """
        q = self.with_pos_embed(tgt, query_pos)  # (B, Q, D)
        # nn.MultiheadAttention expects (seq, batch, dim)
        tgt2 = self.lang_cross_attn(
            query=q.transpose(0, 1),          # (Q, B, D)
            key=text_memory.transpose(0, 1),  # (S, B, D)
            value=text_memory.transpose(0, 1),
            key_padding_mask=text_mask,
        )[0].transpose(0, 1)  # (B, Q, D)
        tgt = tgt + self.lang_dropout(tgt2)
        return self.lang_norm(tgt)

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes,
                level_start_index, src_padding_mask=None, lvl_pos_embed_flatten=None,
                text_memory=None, text_mask=None, ref_angles=None):
        """Order: self-attn -> lang cross-attn -> spatial deformable cross-attn -> FFN.

        With ``text_memory=None`` the language sub-layer is skipped and this
        reproduces the V1 ``_forward_self_cross`` path exactly.
        """
        # 1. Self-attention (V1)
        tgt = self._forward_self_attn(tgt, query_pos)

        # 2. Language cross-attention (NEW; skipped when no text provided)
        if text_memory is not None:
            tgt = self._forward_lang_cross_attn(tgt, query_pos, text_memory, text_mask)

        # 3. Spatial deformable cross-attention (V1; SEED-rotated when ref_angles given)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src, src_spatial_shapes, level_start_index, src_padding_mask,
            ref_angles=ref_angles,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 4. FFN (V1)
        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableTransformerDecoderFinal(nn.Module):
    """V1 decoder that threads language tokens through every layer."""

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement
        self.bbox_embed = None
        self.class_embed = None
        # Optional SEED-style iterative ANGLE refinement. Inert unless a model
        # assigns a per-layer angle head here (mirrors ``bbox_embed`` above):
        # when set, the reference heading fed to the deformable sampling is
        # refined by a residual delta after every decoder layer. The per-layer
        # refined angles are stashed on ``last_ref_angles`` for the model head.
        self.angle_embed = None
        self.last_ref_angles = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes,
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None,
                lvl_pos_embed_flatten=None, sentence_embeds=None,
                text_mask=None, ref_angles=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        intermediate_ref_angles = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output = layer(
                output, query_pos, reference_points_input,
                src, src_spatial_shapes, src_level_start_index,
                src_padding_mask, lvl_pos_embed_flatten,
                text_memory=sentence_embeds,
                text_mask=text_mask,
                ref_angles=ref_angles,
            )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            # SEED-style iterative angle refinement (inert unless angle_embed set).
            # The heading that rotated THIS layer's deformable sampling is refined
            # by a residual delta and (detached) becomes the heading for the next
            # layer's sampling — mirroring the box-refinement hack above.
            if self.angle_embed is not None and ref_angles is not None:
                d_ang = self.angle_embed[lid](output).squeeze(-1)
                ref_angles = (ref_angles + d_ang).detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_ref_angles.append(ref_angles)

        if self.angle_embed is not None and len(intermediate_ref_angles) > 0 \
                and intermediate_ref_angles[0] is not None:
            self.last_ref_angles = torch.stack(intermediate_ref_angles)
        else:
            self.last_ref_angles = None

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points, query_pos


class DeformableTransformerFinal(DeformableTransformer):
    """FINAL transformer: V1 with a language-aware decoder.

    The encoder and all proposal/query handling are inherited UNCHANGED from
    the V1 ``DeformableTransformer``. Only the decoder layers are upgraded
    with the per-query language cross-attention sub-layer.
    """

    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 decoder_self_cross=True, sigmoid_attn=False,
                 extra_track_attn=False):
        # Initialize parent — this builds the V1 encoder + V1 decoder
        super().__init__(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation,
            return_intermediate_dec=return_intermediate_dec,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points, enc_n_points=enc_n_points,
            two_stage=two_stage,
            two_stage_num_proposals=two_stage_num_proposals,
            decoder_self_cross=decoder_self_cross,
            sigmoid_attn=sigmoid_attn,
            extra_track_attn=extra_track_attn,
        )

        # Replace ONLY the decoder with the language-aware FINAL version.
        decoder_layer = DeformableTransformerDecoderLayerFinal(
            d_model, dim_feedforward,
            dropout, activation,
            num_feature_levels, nhead, dec_n_points,
            decoder_self_cross,
            sigmoid_attn=sigmoid_attn,
            extra_track_attn=extra_track_attn,
        )
        self.decoder = DeformableTransformerDecoderFinal(
            decoder_layer, num_decoder_layers, return_intermediate_dec,
        )

    def forward(self, srcs, masks, pos_embeds, query_embed=None,
                sentence_embeds=None, ref_pts=None, text_mask=None, ref_angles=None):
        """Same as V1 forward but threads text_mask through to the decoder."""
        assert self.two_stage or query_embed is not None

        # Prepare input for encoder (identical to V1)
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # Encoder (identical to V1)
        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index,
            valid_ratios, lvl_pos_embed_flatten, mask_flatten,
        )

        # Prepare input for decoder (identical to V1)
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes,
            )
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            )
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
            )
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            if query_embed.dim() == 2:
                query_embed, tgt = torch.split(query_embed, c, dim=1)
                query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
                tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            elif query_embed.dim() == 3:
                assert query_embed.shape[0] == bs, \
                    f"query_embed batch ({query_embed.shape[0]}) != memory batch ({bs})"
                query_embed, tgt = torch.split(query_embed, c, dim=2)
            else:
                raise ValueError(f"Unsupported query_embed shape: {tuple(query_embed.shape)}")

            if ref_pts is None:
                reference_points = self.reference_points(query_embed).sigmoid()
            else:
                if ref_pts.dim() == 2:
                    reference_points = ref_pts.unsqueeze(0).repeat(bs, 1, 1).sigmoid()
                elif ref_pts.dim() == 3:
                    assert ref_pts.shape[0] == bs, \
                        f"ref_pts batch ({ref_pts.shape[0]}) != memory batch ({bs})"
                    reference_points = ref_pts.sigmoid()
                else:
                    raise ValueError(f"Unsupported ref_pts shape: {tuple(ref_pts.shape)}")
            init_reference_out = reference_points

        # Decoder — FINAL passes per-token text features + mask to every layer
        hs, inter_references = self.decoder(
            tgt, reference_points, memory,
            spatial_shapes, level_start_index, valid_ratios,
            query_embed, mask_flatten, lvl_pos_embed_flatten,
            sentence_embeds=sentence_embeds,
            text_mask=text_mask,
            ref_angles=ref_angles,
        )

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


def build_deformable_transformer_final(args):
    """Build the FINAL transformer (V1 + decoder language cross-attention)."""
    return DeformableTransformerFinal(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        decoder_self_cross=not args.decoder_cross_self,
        sigmoid_attn=args.sigmoid_attn,
        extra_track_attn=args.extra_track_attn,
    )
