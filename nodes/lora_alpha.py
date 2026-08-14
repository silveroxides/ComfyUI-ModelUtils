import torch


def lora_alpha_scale(alpha: torch.Tensor | None, rank: int, *, layer: str) -> float:
    """Return the effective LoRA multiplier represented by alpha and rank."""

    if rank <= 0:
        raise ValueError(f"LoRA rank for '{layer}' must be positive.")
    if alpha is None:
        return 1.0
    if alpha.numel() != 1:
        raise ValueError(f"LoRA alpha for '{layer}' must be scalar.")
    return float(alpha.reshape(-1)[0].item()) / rank


def normalize_lora_pair(
    down: torch.Tensor,
    up: torch.Tensor,
    alpha: torch.Tensor | None,
    *,
    layer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize an alpha-free factor pair before rank alignment or merging."""

    if down.ndim < 2 or up.ndim < 2:
        raise ValueError(f"LoRA factors for '{layer}' must have at least two dimensions.")
    rank = int(down.shape[0])
    if int(up.shape[1]) != rank:
        raise ValueError(f"LoRA factor rank mismatch for '{layer}'.")
    scale = lora_alpha_scale(alpha, rank, layer=layer)
    return down, up if scale == 1.0 else up * scale
