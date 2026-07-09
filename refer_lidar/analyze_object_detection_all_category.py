#!/usr/bin/env python3
"""Analyze refer_detection queries and target coverage.

Reports per-class statistics:
- number of prompts per class
- positive/negative prompts (negative = 0 targets)
- total targets per class
- whether each class has prompts
- whether each class has negative prompts

Also reports global stats for zero-target queries across selected question types
(e.g., the same list used by the training script).
"""

import argparse
import json
import os
from glob import glob
from typing import Dict, List, Optional


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


def _canonicalize_class_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip().lower()
    if not s:
        return None

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
        # Full nuScenes names.
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
    if s in set(NUSCENES_DET_CLASSES):
        return s
    if s.endswith('s') and s not in {'bus'}:
        s = s[:-1]
    if s == 'trafficcone':
        return 'traffic_cone'
    if s == 'constructionvehicle':
        return 'construction_vehicle'
    if s in set(NUSCENES_DET_CLASSES):
        return s
    return None


def _infer_class_from_query_text(query: str) -> Optional[str]:
    q = str(query or '').lower().replace('-', ' ').replace('_', ' ')
    if not q:
        return None
    for cname in NUSCENES_DET_CLASSES:
        terms = {cname, cname.replace('_', ' ')}
        if cname.endswith('y'):
            terms.add(cname[:-1] + 'ies')
        else:
            terms.add(cname + 's')
            terms.add(cname.replace('_', ' ') + 's')
        if any(term in q for term in terms):
            return cname
    return None


def _load_shards(refer_data_dir: str, split: str) -> List[dict]:
    shard_paths = sorted(glob(os.path.join(refer_data_dir, '*.json')))
    if not shard_paths:
        raise FileNotFoundError(f'No JSON shards found in: {refer_data_dir}')

    all_entries = []
    has_entry_split = False
    for path in shard_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data and isinstance(data[0], dict) and 'split' in data[0]:
            has_entry_split = True
        all_entries.extend(data)

    if has_entry_split:
        return [e for e in all_entries if e.get('split') == split]

    # Fallback shard split convention used elsewhere in this repo:
    # 0-7 train, 8 val, 9 test.
    split_map = {}
    for i, path in enumerate(shard_paths):
        split_map[path] = 'train' if i < 8 else ('val' if i == 8 else 'test')

    out = []
    for path in shard_paths:
        if split_map[path] != split:
            continue
        with open(path, 'r', encoding='utf-8') as f:
            out.extend(json.load(f))
    return out


def _load_question_types_json(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f'question_types_json not found: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and isinstance(data.get('question_types'), list):
        return [str(x) for x in data['question_types']]
    raise ValueError('question_types_json must be a list or a dict with key "question_types"')


def _init_class_stats() -> Dict[str, dict]:
    return {
        c: {
            'num_prompts': 0,
            'num_positive_prompts': 0,
            'num_negative_prompts': 0,
            'total_targets': 0,
            'unique_prompt_texts': set(),
        }
        for c in NUSCENES_DET_CLASSES
    }


def _collect_selected_query_stats(entries: List[dict], selected_qtypes: Optional[List[str]]) -> Dict[str, object]:
    selected_set = set(selected_qtypes) if selected_qtypes else None

    total_selected_queries = 0
    selected_zero_target_queries = 0
    per_query_type = {}

    for entry in entries:
        for rq in entry.get('refer_queries', []):
            qtype = str(rq.get('query_type', 'unknown'))
            if selected_set is not None and qtype not in selected_set:
                continue

            total_selected_queries += 1
            num_targets = len(rq.get('targets', []))

            if qtype not in per_query_type:
                per_query_type[qtype] = {
                    'num_queries': 0,
                    'num_zero_target_queries': 0,
                    'num_positive_queries': 0,
                    'avg_targets_per_query': 0.0,
                    'total_targets': 0,
                }

            row = per_query_type[qtype]
            row['num_queries'] += 1
            row['total_targets'] += num_targets
            if num_targets == 0:
                row['num_zero_target_queries'] += 1
                selected_zero_target_queries += 1
            else:
                row['num_positive_queries'] += 1

    for qtype in sorted(per_query_type.keys()):
        row = per_query_type[qtype]
        row['avg_targets_per_query'] = row['total_targets'] / max(row['num_queries'], 1)
        row['has_zero_target_queries'] = row['num_zero_target_queries'] > 0

    qtypes_with_zero = [
        qt for qt, row in per_query_type.items() if row['num_zero_target_queries'] > 0
    ]

    return {
        'selected_question_types': sorted(per_query_type.keys()) if selected_set is None else sorted(list(selected_set)),
        'total_queries_selected_types': total_selected_queries,
        'total_zero_target_queries_selected_types': selected_zero_target_queries,
        'zero_target_ratio_selected_types': (
            selected_zero_target_queries / max(total_selected_queries, 1)
        ),
        'query_types_with_zero_target_queries': sorted(qtypes_with_zero),
        'per_question_type': {k: per_query_type[k] for k in sorted(per_query_type.keys())},
    }


def analyze(
    refer_data_dir: str,
    split: str,
    query_type: str,
    question_types_json: Optional[str],
) -> Dict[str, object]:
    entries = _load_shards(refer_data_dir, split)
    selected_qtypes = _load_question_types_json(question_types_json)
    selected_query_stats = _collect_selected_query_stats(entries, selected_qtypes)

    per_class = _init_class_stats()
    total_queries = 0
    unresolved_queries = 0
    unresolved_negative_queries = 0
    multi_target_class_queries = 0

    for entry in entries:
        for rq in entry.get('refer_queries', []):
            if rq.get('query_type') != query_type:
                continue

            total_queries += 1
            query_text = str(rq.get('query', ''))
            targets = rq.get('targets', [])

            target_classes = []
            for t in targets:
                cname = _canonicalize_class_name(t.get('class'))
                if cname is not None:
                    target_classes.append(cname)

            unique_target_classes = set(target_classes)

            if len(unique_target_classes) > 1:
                multi_target_class_queries += 1

            if len(unique_target_classes) == 1:
                assigned = next(iter(unique_target_classes))
            else:
                assigned = _infer_class_from_query_text(query_text)

            if assigned is None:
                unresolved_queries += 1
                if len(targets) == 0:
                    unresolved_negative_queries += 1
                continue

            stat = per_class[assigned]
            stat['num_prompts'] += 1
            stat['unique_prompt_texts'].add(query_text)

            if len(targets) == 0:
                stat['num_negative_prompts'] += 1
            else:
                stat['num_positive_prompts'] += 1

            # Count only targets of assigned class to avoid cross-class inflation.
            stat['total_targets'] += sum(1 for c in target_classes if c == assigned)

    per_class_out: Dict[str, dict] = {}
    for c in NUSCENES_DET_CLASSES:
        st = per_class[c]
        per_class_out[c] = {
            'num_prompts': st['num_prompts'],
            'num_positive_prompts': st['num_positive_prompts'],
            'num_negative_prompts': st['num_negative_prompts'],
            'has_negative_prompts': st['num_negative_prompts'] > 0,
            'total_targets': st['total_targets'],
            'has_prompts': st['num_prompts'] > 0,
            'num_unique_prompt_texts': len(st['unique_prompt_texts']),
            'sample_prompt_texts': sorted(st['unique_prompt_texts'])[:5],
        }

    classes_missing_prompts = [c for c in NUSCENES_DET_CLASSES if not per_class_out[c]['has_prompts']]
    classes_without_negative_prompts = [
        c for c in NUSCENES_DET_CLASSES if not per_class_out[c]['has_negative_prompts']
    ]

    return {
        'split': split,
        'question_types_json': question_types_json,
        'selected_query_type_stats': selected_query_stats,
        'query_type': query_type,
        'num_entries': len(entries),
        'total_queries': total_queries,
        'unresolved_queries': unresolved_queries,
        'unresolved_negative_queries': unresolved_negative_queries,
        'multi_target_class_queries': multi_target_class_queries,
        'classes_missing_prompts': classes_missing_prompts,
        'classes_without_negative_prompts': classes_without_negative_prompts,
        'per_class': per_class_out,
    }


def _print_summary(stats: Dict[str, object]) -> None:
    print('\n[Query-Type Analysis]')
    print(f"  split={stats['split']}")
    print(f"  query_type={stats['query_type']}")
    print(f"  entries={stats['num_entries']}")
    print(f"  total_queries={stats['total_queries']}")
    print(f"  unresolved_queries={stats['unresolved_queries']}")
    print(f"  unresolved_negative_queries={stats['unresolved_negative_queries']}")
    print(f"  multi_target_class_queries={stats['multi_target_class_queries']}")

    sel = stats['selected_query_type_stats']
    print('\n[Selected Question Types - Global Zero-Target Check]')
    print(f"  total_queries_selected_types={sel['total_queries_selected_types']}")
    print(f"  total_zero_target_queries_selected_types={sel['total_zero_target_queries_selected_types']}")
    print(f"  zero_target_ratio_selected_types={sel['zero_target_ratio_selected_types']:.6f}")
    print(
        '  query_types_with_zero_target_queries='
        f"{sel['query_types_with_zero_target_queries'] if sel['query_types_with_zero_target_queries'] else 'none'}"
    )

    print('\n[Per-Class Stats]')
    header = (
        f"{'class':<24} {'prompts':>8} {'positive':>10} {'negative':>10} "
        f"{'targets':>9} {'has_prompt':>11} {'has_neg':>8}"
    )
    print(header)
    print('-' * len(header))

    per_class = stats['per_class']
    for c in NUSCENES_DET_CLASSES:
        row = per_class[c]
        print(
            f"{c:<24} {row['num_prompts']:>8} {row['num_positive_prompts']:>10} "
            f"{row['num_negative_prompts']:>10} {row['total_targets']:>9} "
            f"{str(row['has_prompts']):>11} {str(row['has_negative_prompts']):>8}"
        )

    print('\n[Coverage Checks]')
    missing = stats['classes_missing_prompts']
    no_neg = stats['classes_without_negative_prompts']
    print(f"  classes_missing_prompts={missing if missing else 'none'}")
    print(f"  classes_without_negative_prompts={no_neg if no_neg else 'none'}")

    print('\n[Per Question Type - Selected Set]')
    q_header = (
        f"{'query_type':<40} {'queries':>8} {'zero_target':>12} {'positive':>10} {'avg_targets':>12}"
    )
    print(q_header)
    print('-' * len(q_header))
    for qtype, row in stats['selected_query_type_stats']['per_question_type'].items():
        print(
            f"{qtype:<40} {row['num_queries']:>8} {row['num_zero_target_queries']:>12} "
            f"{row['num_positive_queries']:>10} {row['avg_targets_per_query']:>12.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Analyze per-class stats for object_detection_all_category-style queries.'
    )
    parser.add_argument('--refer-data-dir', required=True, help='Directory with refer_detection JSON shards')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--query-type', default='object_detection_all_category')
    parser.add_argument(
        '--question-types-json',
        default='configs/question_types_det.json',
        help='Training question types JSON used for global zero-target query stats',
    )
    parser.add_argument('--output-json', default=None, help='Optional output JSON path')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = analyze(
        args.refer_data_dir,
        args.split,
        args.query_type,
        args.question_types_json,
    )
    _print_summary(stats)

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved JSON: {args.output_json}")


if __name__ == '__main__':
    main()
