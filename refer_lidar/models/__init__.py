# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

def build_model(args):
    """Build one of the three referring architectures from ``--meta-arch``.

    The chain is cumulative: ``refer_model`` (base) -> ``refer_model_lang_dec``
    (+ per-layer language cross-attention) -> ``refer_model_angle`` (+ SEED-style
    iterative orientation refinement). Lazy imports keep an unused model's
    dependencies out of the process.
    """
    if args.meta_arch == 'refer_model':
        from .refer_model import build as build_func
    elif args.meta_arch == 'refer_model_lang_dec':
        from .refer_model_lang_dec import build_lang_dec as build_func
    elif args.meta_arch == 'refer_model_angle':
        from .refer_model_angle import build_angle as build_func
    else:
        raise ValueError(f'invalid meta_arch: {args.meta_arch}')
    return build_func(args)

