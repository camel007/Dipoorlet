import json
import os
from typing import Any, Dict

import numpy as np

from .deploy_default import deploy_dispatcher
from ..utils import logger


def _to_scalar_min_max(v):
    try:
        vmin = v[0]
        vmax = v[1]
    except Exception:
        # unexpected structure; fallback
        return 0.0, 0.0

    def _scalar(x: Any) -> float:
        try:
            return float(np.min(x)) if hasattr(x, "__len__") else float(x)
        except Exception:
            try:
                return float(x)
            except Exception:
                return 0.0

    min_v = _scalar(vmin)
    max_v = _scalar(vmax)
    if max_v < min_v:
        # ensure valid range; swap or widen minimally
        min_v, max_v = min(max_v, min_v), max(max_v, min_v)
    if max_v - min_v == 0.0:
        max_v = min_v + 1e-2
    return min_v, max_v


def _locate_magik_json(output_dir: str) -> str:
    candidates = [
        os.path.join(output_dir, "magik_node_quant_bit_info.json"),
        os.path.abspath("magik_node_quant_bit_info.json"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "example", "magik_node_quant_bit_info.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


@deploy_dispatcher.register("ingenic")
def gen_ingenic_quant_info(graph, clip_val: Dict[str, Any], args, **kwargs):
    # Find magik_node_quant_bit_info.json
    src_path = _locate_magik_json(args.output_dir)
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"magik_node_quant_bit_info.json not found. Expected at {src_path}."
        )

    # Load
    with open(src_path, "r", encoding="utf-8") as f:
        magik = json.load(f)

    # Enrich with MIN/MAX from clip_val
    updated = False
    updated_cnt = 0
    missing = []
    for tensor_name, info in magik.items():
        if tensor_name in clip_val:
            min_v, max_v = _to_scalar_min_max(clip_val[tensor_name])
            info["MIN"] = float(min_v)
            info["MAX"] = float(max_v)
            updated = True
            updated_cnt += 1
        else:
            missing.append(tensor_name)

    if missing:
        # Warn once with a small sample to avoid noisy logs
        sample = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        logger.warning(
            f"{len(missing)} tensors in magik_node_quant_bit_info.json not found in clip_val: {sample}{more}"
        )

    logger.info(f"Updated MIN/MAX for {updated_cnt} tensors for Ingenic platform.")

    # Always write the enriched file to output_dir; do not overwrite the source template
    out_path = os.path.join(args.output_dir, "magik_node_quant_bit_info.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(magik, f, indent=4, ensure_ascii=False)

    return updated
