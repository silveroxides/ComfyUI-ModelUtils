"""
Utilities for model merging operations.
"""
from typing import Dict, List

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
