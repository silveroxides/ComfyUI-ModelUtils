"""
Utilities for model merging operations.
"""
from typing import Dict, List, Tuple, Any
import re

def get_union_keys(*handlers) -> List[str]:
    """Returns the union of all tensor keys across multiple model handlers."""
    all_keys = set()
    for handler in handlers:
        if handler is not None:
            all_keys.update(handler.keys())
    return sorted(all_keys)


def get_intersection_keys(*handlers) -> List[str]:
    """Returns only keys present in ALL model handlers."""
    handlers = [h for h in handlers if h is not None]
    if not handlers:
        return []
    all_keys = set(handlers[0].keys())
    for handler in handlers[1:]:
        all_keys &= set(handler.keys())
    return sorted(all_keys)


def parse_custom_weights(custom_weights_str: str) -> List[Tuple[re.Pattern, Dict[str, float]]]:
    """Parses multiline string of custom weights.
    Format: pattern:weight OR pattern:param=value,param2=value
    Returns list of (compiled_regex, parameter_dict).
    """
    if not custom_weights_str or not custom_weights_str.strip():
        return []

    results = []
    lines = custom_weights_str.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if ':' not in line:
            continue

        pattern_str, weight_str = line.rsplit(':', 1)
        pattern_str = pattern_str.strip()
        weight_str = weight_str.strip()

        try:
            pattern = re.compile(pattern_str)
        except re.error:
            # Fallback to literal substring if regex invalid
            pattern = re.compile(re.escape(pattern_str))

        params = {}
        if '=' in weight_str:
            # param=value,param2=value
            parts = weight_str.split(',')
            for part in parts:
                if '=' in part:
                    k, v = part.split('=', 1)
                    k = k.strip()
                    # Map common aliases
                    if k in ['a', 'alpha']: k = 'alpha'
                    elif k in ['b', 'beta']: k = 'beta'
                    elif k in ['g', 'gamma']: k = 'gamma'
                    elif k in ['d', 'delta']: k = 'delta'
                    elif k in ['w', 'weight']: k = 'weight'
                    elif re.match(r'^w[1-8]$', k): pass # Keep w1-w8 as is

                    try:
                        params[k] = float(v.strip())
                    except ValueError:
                        pass
        else:
            # single float or comma-separated list of floats (for multi-LoRA)
            if ',' in weight_str:
                parts = weight_str.split(',')
                for i, p in enumerate(parts):
                    try:
                        params[f'w{i+1}'] = float(p.strip())
                    except ValueError:
                        pass
            else:
                try:
                    val = float(weight_str)
                    params['alpha'] = val # default for merger
                    params['weight'] = val # default for lora_merger
                except ValueError:
                    pass

        if params:
            results.append((pattern, params))

    return results


def get_custom_parameters(key: str, parsed_weights: List[Tuple[re.Pattern, Dict[str, float]]], base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Returns updated parameters if key matches any pattern."""
    new_params = base_params.copy()
    for pattern, custom_dict in parsed_weights:
        if pattern.search(key):
            new_params.update(custom_dict)
            break # First match wins
    return new_params
