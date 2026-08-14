"""Bounded progressive partial SVD for adaptive extraction modes."""

import torch


ADAPTIVE_PARTIAL_MODES = {"ratio", "quantile", "sv_cumulative", "sv_fro"}


def _bounded_rank(singular_values, mode, parameter, output_cap, matrix_fro_sq):
    capped = singular_values[:output_cap]
    if capped.numel() == 0 or capped[0] < 1e-8:
        return 1, True

    if mode == "ratio":
        rank = int(torch.sum(capped > capped[0] / parameter).item()) if parameter > 0 else output_cap
        return max(1, min(rank, output_cap)), rank < output_cap

    if mode in {"quantile", "sv_cumulative"}:
        total = capped.sum()
        if total < 1e-8:
            return 1, True
        rank = int(torch.searchsorted(torch.cumsum(capped, 0) / total, parameter).item()) + 1
        return max(1, min(rank, output_cap)), rank < output_cap

    target_sq = float(parameter) ** 2 * matrix_fro_sq
    cumulative = torch.cumsum(capped.square(), 0)
    reached = bool(cumulative[-1] >= target_sq)
    if reached:
        rank = int(torch.searchsorted(cumulative, target_sq).item()) + 1
        return max(1, min(rank, output_cap)), rank < output_cap
    return output_cap, False


def adaptive_partial_svd(weight, mode, parameter, max_rank, probe_offset, niter):
    """Return a bounded top spectrum and selected output rank."""
    max_possible = min(weight.shape)
    output_cap = max(1, min(int(max_rank), max_possible))
    offset = max(1, int(probe_offset))
    initial_probe = min(max_possible, output_cap + offset)
    probe_limit = min(max_possible, max(initial_probe, output_cap * 2))
    matrix_fro_sq = float(torch.sum(weight.square()).item()) if mode == "sv_fro" else 0.0

    probe_rank = initial_probe
    while True:
        if probe_rank >= max_possible:
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        else:
            U, S, V = torch.svd_lowrank(weight, q=probe_rank, niter=niter)
            Vh = V.T

        rank, resolved_before_cap = _bounded_rank(
            S, mode, parameter, output_cap, matrix_fro_sq
        )
        near_probe_tail = rank >= max(1, probe_rank - offset)
        if resolved_before_cap or not near_probe_tail or probe_rank >= probe_limit:
            return U[:, :rank], S[:rank], Vh[:rank, :], rank
        probe_rank = probe_limit
