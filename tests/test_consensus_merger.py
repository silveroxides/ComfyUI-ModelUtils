import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_cwb_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def cwb():
    return _load_module("nodes.consensus_merger")


def _patch_io(monkeypatch, module, tmp_path, paths):
    monkeypatch.setattr(module.folder_paths, "get_full_path", lambda _, name: paths.get(name))
    monkeypatch.setattr(module.folder_paths, "get_folder_paths", lambda _: [str(tmp_path)])
    monkeypatch.setattr(module.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(module, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "cleanup_after_operation", lambda: None)


def _result_path(tmp_path, category, result):
    return tmp_path / category / result


def _params(output_filename, **overrides):
    settings = {
        "consensus_type": "mean",
        "alignment_method": "index",
        "alignment_threshold": 0.4,
        "similarity_threshold": 0.0,
        "power_alpha": 2.0,
        "diversity_beta": 0.0,
        "rescale_norm": False,
        "global_scale": 1.0,
        "dynamic_similarity_contrast": False,
        "soft_comfort_bandpass": False,
        "position_weight": 0.0,
        "preserve_common_prefix": False,
    }
    for key in list(settings):
        if key in overrides:
            settings[key] = overrides.pop(key)
    params = {
        "cwb_preset": "broad_sim_medn_rn_softcb",
        "cwb_config": _load_module("nodes.consensus_merger").build_custom_cwb_settings(
            **settings
        ),
        "mismatch_mode": "skip",
        "output_filename": output_filename,
        "save_dtype": "fp16",
        "process_device": "cpu",
        "exclude_patterns": "",
        "discard_patterns": "",
        "glob_patterns": False,
        "lazy_load": True,
        "force_clear_cache": False,
        "override_dtype": False,
        "include_1d_diffs": False,
    }
    params.update(overrides)
    return params


def _settings(cwb, **overrides):
    values = {
        "consensus_type": "mean",
        "alignment_method": "index",
        "alignment_threshold": 0.4,
        "similarity_threshold": 0.0,
        "power_alpha": 2.0,
        "diversity_beta": 0.0,
        "rescale_norm": False,
        "global_scale": 1.0,
        "dynamic_similarity_contrast": False,
        "soft_comfort_bandpass": False,
        "position_weight": 0.0,
        "preserve_common_prefix": False,
    }
    values.update(overrides)
    return cwb.build_custom_cwb_settings(**values)


def test_preset_resolution_matches_extended_contract(cwb):
    default = cwb.resolve_cwb_settings(
        "broad_sim_medn_rn_softcb", cwb.LORA_CWB_PRESETS
    )
    assert default.consensus_type == "median"
    assert default.alignment_method == "similarity"
    assert default.alignment_threshold == pytest.approx(0.0)
    assert default.similarity_threshold == pytest.approx(0.0)
    assert default.power_alpha == pytest.approx(2.0)
    assert default.diversity_beta == pytest.approx(4.0)
    assert default.rescale_norm is True
    assert default.dynamic_similarity_contrast is False
    assert default.soft_comfort_bandpass is True
    assert default.position_weight == pytest.approx(0.05)

    with pytest.raises(ValueError, match="Unsupported CWB preset"):
        cwb.resolve_cwb_settings("baseline", cwb.LORA_CWB_PRESETS)


def test_preset_construction_rejects_omitted_fields(cwb):
    with pytest.raises(TypeError):
        cwb._preset(
            consensus_type="median",
            alignment_method="similarity",
            alignment_threshold=0.0,
        )


def test_lora_alignment_presets_use_sweep_verified_thresholds(cwb):
    assert cwb.LORA_CWB_PRESETS["broad_sim_medn_rn_softcb"][
        "alignment_threshold"
    ] == pytest.approx(0.0)
    assert cwb.LORA_CWB_PRESETS["moderate_sim_medn_rn_softcb"][
        "alignment_threshold"
    ] == pytest.approx(0.0005)
    assert cwb.LORA_CWB_PRESETS["conservative_sim_medn_rn_softcb"][
        "alignment_threshold"
    ] == pytest.approx(0.0025)


@pytest.mark.parametrize(
    ("registry_name", "preset", "alignment_threshold", "similarity_threshold"),
    [
        (registry_name, preset, alignment_threshold, similarity_threshold)
        for registry_name in ("EMBEDDING_CWB_PRESETS", "LORA_CWB_PRESETS")
        for preset, alignment_threshold, similarity_threshold in (
            ("focused_strong_sim_medn", 0.85, 0.60),
            ("focused_balance_sim_medn", 0.75, 0.55),
            ("focused_soft_sim_medn", 0.55, 0.50),
            ("focused_weak_sim_medn", 0.25, 0.35),
        )
    ],
)
def test_focused_median_similarity_presets(
    cwb, registry_name, preset, alignment_threshold, similarity_threshold
):
    settings = getattr(cwb, registry_name)[preset]
    assert settings == {
        "consensus_type": "median",
        "alignment_method": "similarity",
        "alignment_threshold": alignment_threshold,
        "similarity_threshold": similarity_threshold,
        "power_alpha": 1.25,
        "diversity_beta": 0.0,
        "rescale_norm": False,
        "global_scale": 1.0,
        "dynamic_similarity_contrast": False,
        "soft_comfort_bandpass": False,
        "position_weight": 0.20,
        "preserve_common_prefix": False,
    }


def test_cwb_math_index_prefix_and_scale(cwb):
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="index",
        global_scale=1.0,
    )
    first = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    second = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    torch.testing.assert_close(
        cwb.merge_cwb_tensors([first, second], settings),
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
    )

    prefix_settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="index",
        global_scale=2.0,
        preserve_common_prefix=True,
    )
    prefixed = cwb.merge_cwb_tensors(
        [torch.tensor([[9.0, 9.0], [1.0, 0.0]]),
         torch.tensor([[9.0, 9.0], [3.0, 0.0]])],
        prefix_settings,
    )
    torch.testing.assert_close(prefixed[0], torch.tensor([9.0, 9.0]))
    torch.testing.assert_close(prefixed[1], torch.tensor([4.0, 0.0]))


def test_similarity_alignment_reorders_matching_rows(cwb):
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="similarity",
        alignment_threshold=0.5,
    )
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reversed_source = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    torch.testing.assert_close(
        cwb.merge_cwb_tensors([reference, reversed_source], settings),
        torch.tensor([[1.5, 0.0], [0.0, 1.5]]),
    )

    scores = torch.ones((2, 2))
    biased = cwb._position_biased_scores(scores, 1.0)
    assert biased[0, 0] > biased[0, 1]


def test_fixed_coordinate_merge_never_reorders_model_rows(cwb):
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="similarity",
        alignment_threshold=0.5,
    )
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reversed_source = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    expected = torch.stack([
        cwb.merge_consensus_group(
            torch.stack([reference[row], reversed_source[row]]), settings
        )
        for row in range(reference.shape[0])
    ])
    torch.testing.assert_close(
        cwb.merge_cwb_tensors(
            [reference, reversed_source],
            settings,
            allow_similarity_alignment=False,
        ),
        expected,
    )


def test_lora_similarity_alignment_pairs_a_rows_with_b_columns(cwb):
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="similarity",
        alignment_threshold=0.5,
    )
    reference_down = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reference_up = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    source_down = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    source_up = torch.tensor([[0.0, -2.0], [3.0, 0.0]])

    merged_down, merged_up = cwb.merge_cwb_lora_pairs(
        [reference_down, source_down],
        [reference_up, source_up],
        settings,
        reference_index=0,
    )
    torch.testing.assert_close(merged_down, reference_down)
    torch.testing.assert_close(merged_up, reference_up)


def test_lora_krea_shape_scores_only_rank_components(monkeypatch, cwb):
    rank = 256
    output_features = 36864
    down = torch.ones((rank, 2))
    up = torch.ones((output_features, rank))
    matrix_shapes = []

    def bounded_mm(left, right):
        matrix_shapes.append((left.shape, right.shape))
        return torch.eye(left.shape[0], right.shape[1], dtype=left.dtype)

    monkeypatch.setattr(cwb.torch, "mm", bounded_mm)
    monkeypatch.setattr(
        cwb,
        "merge_consensus_group",
        lambda stacked, settings, **kwargs: stacked[0].clone(),
    )
    settings = _settings(
        cwb,
        alignment_method="similarity",
        alignment_threshold=0.5,
    )
    merged_down, merged_up = cwb.merge_cwb_lora_pairs(
        [down, down], [up, up], settings, reference_index=0
    )

    assert merged_down.shape == down.shape
    assert merged_up.shape == up.shape
    assert matrix_shapes
    assert all(left[0] == rank and right[1] == rank for left, right in matrix_shapes)


def test_mixed_rank_similarity_searches_full_reference_without_padding(
    monkeypatch, cwb
):
    reference_down = torch.eye(3)
    reference_up = torch.eye(3)
    source_down = torch.tensor([[0.0, 0.0, 2.0]])
    source_up = torch.tensor([[0.0], [0.0], [4.0]])
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="similarity",
        alignment_threshold=0.5,
        rescale_norm=True,
    )
    diagnostics = cwb.CWBDiagnostics()
    matrix_shapes = []
    original_mm = cwb.torch.mm

    def record_mm(left, right):
        matrix_shapes.append((tuple(left.shape), tuple(right.shape)))
        return original_mm(left, right)

    monkeypatch.setattr(cwb.torch, "mm", record_mm)
    merged_down, merged_up = cwb.merge_cwb_lora_pairs(
        [reference_down, source_down],
        [reference_up, source_up],
        settings,
        reference_index=0,
        diagnostics=diagnostics,
    )

    assert matrix_shapes == [((3, 3), (3, 1)), ((3, 3), (3, 1))]
    assert diagnostics.alignment_matches == 1
    assert diagnostics.anchor_only_groups == 2
    torch.testing.assert_close(merged_down[:2], reference_down[:2])
    torch.testing.assert_close(merged_up[:, :2], reference_up[:, :2])
    torch.testing.assert_close(merged_down[2], torch.tensor([0.0, 0.0, 1.5]))
    torch.testing.assert_close(merged_up[:, 2], torch.tensor([0.0, 0.0, 2.5]))


def test_mixed_rank_index_does_not_rescale_absent_components(cwb):
    reference_down = torch.eye(3)
    reference_up = torch.eye(3)
    source_down = torch.tensor([[2.0, 0.0, 0.0]])
    source_up = torch.tensor([[2.0], [0.0], [0.0]])
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="index",
        rescale_norm=True,
    )

    merged_down, merged_up = cwb.merge_cwb_lora_pairs(
        [reference_down, source_down],
        [reference_up, source_up],
        settings,
        reference_index=0,
    )

    torch.testing.assert_close(merged_down[1:], reference_down[1:])
    torch.testing.assert_close(merged_up[:, 1:], reference_up[:, 1:])
    assert torch.linalg.vector_norm(merged_down[0]).item() == pytest.approx(1.5)
    assert torch.linalg.vector_norm(merged_up[:, 0]).item() == pytest.approx(1.5)


def test_genuine_zero_component_is_not_treated_as_structural_padding(cwb):
    reference_down = torch.eye(2)
    reference_up = torch.eye(2)
    source_down = torch.zeros((1, 2))
    source_up = torch.zeros((2, 1))
    settings = _settings(
        cwb,
        alignment_method="similarity",
        alignment_threshold=0.0,
    )
    diagnostics = cwb.CWBDiagnostics()

    cwb.merge_cwb_lora_pairs(
        [reference_down, source_down],
        [reference_up, source_up],
        settings,
        reference_index=0,
        diagnostics=diagnostics,
    )

    assert diagnostics.alignment_matches == 1
    assert diagnostics.anchor_only_groups == 1


def test_explicit_mismatch_zero_applies_to_every_reference_component(cwb):
    down = torch.eye(2)
    up = torch.eye(2)
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="index",
        rescale_norm=True,
    )
    diagnostics = cwb.CWBDiagnostics()

    merged_down, merged_up, rank = cwb._merge_lora_pair_to_cpu(
        [cwb.LoRAPairSource(0, down, up), cwb.LoRAPairSource.zero(1)],
        settings,
        "cpu",
        torch.float32,
        diagnostics=diagnostics,
    )

    assert rank == 2
    torch.testing.assert_close(merged_down, down * 0.5)
    torch.testing.assert_close(merged_up, up * 0.5)
    assert diagnostics.explicit_zero_contributors == 1
    assert diagnostics.structural_rank_slots_excluded == 0


def test_mixed_rank_diagnostics_count_only_genuine_components(cwb):
    settings = _settings(cwb, alignment_method="index")
    diagnostics = cwb.CWBDiagnostics()
    sources = [
        cwb.LoRAPairSource(0, torch.ones((4, 2)), torch.ones((2, 4))),
        cwb.LoRAPairSource(1, torch.ones((2, 2)), torch.ones((2, 2))),
        cwb.LoRAPairSource(2, torch.ones((3, 2)), torch.ones((2, 3))),
    ]

    _, _, rank = cwb._merge_lora_pair_to_cpu(
        sources,
        settings,
        "cpu",
        torch.float32,
        diagnostics=diagnostics,
    )

    assert rank == 4
    assert diagnostics.lora_groups == 1
    assert diagnostics.mixed_rank_lora_groups == 1
    assert diagnostics.genuine_lora_components == 9
    assert diagnostics.structural_rank_slots_excluded == 3
    report = diagnostics.render("custom", 3, True)
    assert "Mixed-rank LoRA groups: 1" in report
    assert "Genuine LoRA components: 9" in report
    assert "Structural rank slots excluded: 3" in report
    assert "CWB LORA DIMENSION GROUPS" in report
    assert "Input ranks: (4, 2, 3)" in report
    assert "Matched source components: 5/5 (100.00%)" in report


def test_lora_report_groups_dimensions_and_summarizes_rectangular_similarity(cwb):
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="similarity",
        alignment_threshold=0.5,
    )
    diagnostics = cwb.CWBDiagnostics()
    sources = [
        cwb.LoRAPairSource(0, torch.eye(3), torch.eye(3)),
        cwb.LoRAPairSource(
            1,
            torch.tensor([[0.0, 0.0, 2.0]]),
            torch.tensor([[0.0], [0.0], [2.0]]),
        ),
    ]

    for layer in ("blocks.1.attn", "blocks.2.attn"):
        cwb._merge_lora_pair_to_cpu(
            sources,
            settings,
            "cpu",
            torch.float32,
            operation_label=layer,
            diagnostics=diagnostics,
        )

    report = diagnostics.render("custom", 2, True)
    assert "Group 1: 2 layer(s)" in report
    assert "Alignment: similarity; matrices: 3x1" in report
    assert "Matched source components: 2/2 (100.00%)" in report
    assert "Reference-only components: 4/6" in report
    assert "Structural rank slots excluded: 4" in report
    assert "Candidate similarity count: 6" in report
    assert "Matched similarity p05/median/p95: 1 / 1 / 1" in report
    assert "blocks.1.attn" in report
    assert "blocks.2.attn" in report
    assert all(
        not isinstance(value, torch.Tensor)
        for layer_report in diagnostics.lora_layer_reports
        for value in vars(layer_report).values()
    )


def test_lora_alpha_and_global_scale_apply_once_to_pair(cwb):
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="index",
        global_scale=3.0,
    )
    down = torch.ones((1, 1))
    up = torch.ones((1, 1))
    merged_down, merged_up, rank = cwb._merge_lora_pair_to_cpu(
        [
            cwb.LoRAPairSource(0, down, up, 2.0),
            cwb.LoRAPairSource(1, down, up, 2.0),
        ],
        settings,
        "cpu",
        torch.float32,
    )
    assert rank == 1
    torch.testing.assert_close(merged_down, torch.ones((1, 1)))
    torch.testing.assert_close(merged_up, torch.full((1, 1), 6.0))


def test_lora_pair_alignment_supports_convolution_factors(cwb):
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="index",
    )
    down = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1)
    up = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1, 1)
    merged_down, merged_up = cwb.merge_cwb_lora_pairs(
        [down, down], [up, up], settings, reference_index=0
    )
    torch.testing.assert_close(merged_down, down)
    torch.testing.assert_close(merged_up, up)


def test_lora_cuda_oom_retries_current_pair_on_cpu(monkeypatch, caplog, cwb):
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="index",
    )
    original_to_compute = cwb._to_compute
    attempted_devices = []

    def fail_cuda(tensor, device):
        attempted_devices.append(device)
        if str(device).startswith("cuda"):
            raise torch.OutOfMemoryError("injected allocation failure")
        return original_to_compute(tensor, device)

    monkeypatch.setattr(cwb, "_to_compute", fail_cuda)
    monkeypatch.setattr(cwb, "_release_failed_cuda_operation", lambda: None)
    reference_down = torch.eye(2)
    reference_up = torch.eye(2)
    source_down = torch.ones((1, 2))
    source_up = torch.ones((2, 1))
    diagnostics = cwb.CWBDiagnostics()
    with caplog.at_level("WARNING"):
        merged_down, merged_up, rank = cwb._merge_lora_pair_to_cpu(
            [
                cwb.LoRAPairSource(0, reference_down, reference_up),
                cwb.LoRAPairSource(1, source_down, source_up),
            ],
            settings,
            "cuda",
            torch.float32,
            operation_label="diffusion_model.foo",
            diagnostics=diagnostics,
        )

    assert rank == 2
    assert merged_down.shape == reference_down.shape
    assert merged_up.shape == reference_up.shape
    assert attempted_devices[0] == "cuda"
    assert "cpu" in attempted_devices
    assert "retrying this layer on CPU" in caplog.text
    assert "diffusion_model.foo" in caplog.text
    assert diagnostics.cpu_fallbacks == 1
    assert diagnostics.lora_groups == 1
    assert diagnostics.structural_rank_slots_excluded == 1


def test_mixed_rank_common_prefix_is_limited_to_smallest_rank(cwb):
    reference_down = torch.eye(3)
    reference_up = torch.eye(3)
    source_down = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    source_up = torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    settings = _settings(
        cwb,
        consensus_type="mean",
        alignment_method="index",
        preserve_common_prefix=True,
    )

    merged_down, merged_up = cwb.merge_cwb_lora_pairs(
        [reference_down, source_down],
        [reference_up, source_up],
        settings,
        reference_index=0,
    )

    torch.testing.assert_close(merged_down[0], reference_down[0])
    torch.testing.assert_close(merged_up[:, 0], reference_up[:, 0])
    torch.testing.assert_close(merged_down[2], reference_down[2])
    torch.testing.assert_close(merged_up[:, 2], reference_up[:, 2])


def test_norm_rescale_and_dsc_bandpass_are_finite(cwb):
    rescale = _settings(
        cwb,
        consensus_type="mean",
        rescale_norm=True,
    )
    merged = cwb.merge_consensus_group(
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]), rescale
    )
    assert torch.linalg.vector_norm(merged).item() == pytest.approx(2.0)

    dsc = _settings(
        cwb,
        consensus_type="median",
        diversity_beta=1.5,
        dynamic_similarity_contrast=True,
        soft_comfort_bandpass=True,
    )
    result = cwb.merge_consensus_group(
        torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]), dsc
    )
    assert torch.isfinite(result).all()


def test_all_cwb_schemas_and_compact_control_contract(cwb):
    assert len(cwb.CWB_MERGER_NODES) == 14
    schemas = [node.define_schema() for node in cwb.CWB_MERGER_NODES]
    assert len({schema.node_id for schema in schemas}) == 14
    advanced = {
        "consensus_type", "alignment_method", "alignment_threshold",
        "similarity_threshold", "power_alpha", "diversity_beta",
        "rescale_norm", "global_scale", "dynamic_similarity_contrast",
        "soft_comfort_bandpass", "position_weight", "preserve_common_prefix",
    }
    config_schema = cwb.CWBCustomConfiguration.define_schema()
    assert {value.id for value in config_schema.inputs} == advanced

    for node in cwb.CWB_MERGER_NODES:
        schema = node.define_schema()
        ids = [value.id for value in schema.inputs]
        assert schema.description
        assert all(value.tooltip for value in schema.inputs)
        if node is cwb.CWBCustomConfiguration:
            continue
        assert [output.display_name for output in schema.outputs] == [
            "output_filename", "documentation", "cwb_report"
        ]
        assert schema.outputs[0].get_io_type() == "*"
        assert schema.outputs[1].get_io_type() == "STRING"
        assert schema.outputs[2].get_io_type() == "STRING"
        assert advanced.isdisjoint(ids)
        assert "cwb_config" in ids
        if node is cwb.CWBLoRAMultiMerger:
            assert ids[:4] == ["execution_mode", "lora_count", "lora_1", "lora_2"]
            assert schema.inputs[1].default == "2"
            assert ids[-3:] == [
                "include_1d_diffs",
                "counterfactual_weight_sweep",
                "include_mode",
            ]
            continue
        if node in (cwb.CWBEmbeddingSelfCoalesce, cwb.CWBEmbeddingMultiMerger):
            assert "vision_boundary_embeddings" in ids
            assert "legacy_boundary_search" in ids
            assert "boundary_reference_embedding" in ids
            assert "boundary_similarity_threshold" in ids
            assert "target_vector_count" in ids
            continue
        assert ids[:3] == ["execution_mode", "model_a", "model_b"]
        if node.INPUT_COUNT == 3:
            assert ids[3] == "model_c"
        if node.LORA_MODE:
            assert schema.inputs[-2].id == "include_1d_diffs"
        else:
            assert "include_1d_diffs" not in ids
        assert schema.inputs[-1].id == "include_mode"
        assert schema.inputs[-1].default is False


def test_use_case_preset_names_match_complete_settings(cwb):
    fields = set(cwb.CWBSettings.__dataclass_fields__)
    for registry in (
        cwb.DENSE_CWB_PRESETS,
        cwb.EMBEDDING_CWB_PRESETS,
        cwb.LORA_CWB_PRESETS,
    ):
        assert len(registry) >= 6
        for name, values in registry.items():
            assert set(values) == fields
            assert ("_rn" in name) is values["rescale_norm"]
            assert ("_dsc" in name) is values["dynamic_similarity_contrast"]
            assert ("_softcb" in name) is values["soft_comfort_bandpass"]
            assert ("_pcp" in name) is values["preserve_common_prefix"]
            if registry is not cwb.DENSE_CWB_PRESETS:
                expected_alignment = "similarity" if "_sim_" in name else "index"
                assert values["alignment_method"] == expected_alignment
                expected_consensus = "median" if "_medn" in name else "mean"
                assert values["consensus_type"] == expected_consensus


def test_connected_custom_config_completely_overrides_preset(cwb):
    custom = _settings(
        cwb,
        consensus_type="median",
        alignment_method="similarity",
        alignment_threshold=0.17,
        similarity_threshold=-0.25,
        diversity_beta=7.0,
        dynamic_similarity_contrast=True,
    )
    resolved = cwb.resolve_cwb_settings(
        "direct_idx_medn_rn_softcb", cwb.LORA_CWB_PRESETS, custom
    )
    assert resolved is custom


def test_diagnostics_distinguish_alignment_and_weight_fallbacks(cwb):
    alignment_diagnostics = cwb.CWBDiagnostics()
    settings = _settings(
        cwb, alignment_method="similarity", alignment_threshold=0.9
    )
    cwb.merge_cwb_tensors(
        [torch.eye(2), -torch.eye(2)],
        settings,
        diagnostics=alignment_diagnostics,
    )
    assert alignment_diagnostics.alignment_matches == 0
    assert alignment_diagnostics.anchor_only_groups == 2

    weighting_diagnostics = cwb.CWBDiagnostics()
    cwb.merge_consensus_group(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        _settings(cwb, similarity_threshold=0.9),
        diagnostics=weighting_diagnostics,
    )
    assert weighting_diagnostics.all_rejected_fallbacks == 1
    assert weighting_diagnostics.weighting_contributors == 2
    assert weighting_diagnostics.accepted_weighting_contributors == 0
    assert weighting_diagnostics.dominant_weight_min == pytest.approx(0.5)
    assert weighting_diagnostics.dominant_weight_max == pytest.approx(0.5)
    assert weighting_diagnostics.effective_contributor_min == pytest.approx(2.0)
    assert weighting_diagnostics.effective_contributor_max == pytest.approx(2.0)


def test_weight_diagnostics_measure_parameter_influence(cwb):
    diagnostics = cwb.CWBDiagnostics()
    cwb.merge_consensus_group(
        torch.tensor([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]]),
        _settings(
            cwb,
            similarity_threshold=0.0,
            power_alpha=2.0,
            diversity_beta=0.0,
        ),
        diagnostics=diagnostics,
    )

    assert diagnostics.weighting_groups == 1
    assert diagnostics.weighting_contributors == 3
    assert diagnostics.accepted_weighting_contributors == 2
    assert diagnostics.dominant_weight_max > 0.5
    assert 1.0 < diagnostics.effective_contributor_min < 2.0
    report = diagnostics.render("test", 3, custom=False)
    assert "Accepted weighting contributors: 2/3" in report
    assert "Dominant normalized weight min/mean/max:" in report
    assert "Effective contributors min/mean/max:" in report


def test_counterfactual_weight_sweep_reuses_similarity_vectors(cwb):
    sweep = cwb.CWBWeightSweepDiagnostics()
    similarities = torch.tensor([0.25, 0.5, 0.75])
    sweep._record_similarities("median", similarities)

    baseline = sweep.stats[("median", 0.0, 2.0, 0.0, False, False)]
    expected = similarities.square()
    expected /= expected.sum()
    assert baseline.groups == 1
    assert baseline.accepted == 3
    assert baseline.dominant_sum == pytest.approx(float(expected.max()))
    assert baseline.effective_sum == pytest.approx(
        float(1.0 / expected.square().sum())
    )
    assert len(sweep.stats) == 180
    assert all(not isinstance(value, torch.Tensor) for value in sweep.__dict__.values())


def test_default_lora_preset_changes_perturbed_factors(cwb):
    settings = cwb.resolve_cwb_settings(
        "broad_sim_medn_rn_softcb", cwb.LORA_CWB_PRESETS
    )
    reference_down = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reference_up = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    source_down = torch.tensor([[1.0, 0.1], [0.1, 1.0]])
    source_up = torch.tensor([[1.1, 0.1], [0.1, 0.9]])
    diagnostics = cwb.CWBDiagnostics()
    merged_down, merged_up = cwb.merge_cwb_lora_pairs(
        [reference_down, source_down],
        [reference_up, source_up],
        settings,
        reference_index=0,
        diagnostics=diagnostics,
    )
    assert diagnostics.alignment_matches == 2
    assert diagnostics.anchor_only_groups == 0
    assert not torch.equal(merged_down, reference_down)
    assert not torch.equal(merged_up, reference_up)


def test_eight_input_lora_merge_normalizes_before_variable_rank_alignment(
    monkeypatch, tmp_path, cwb
):
    paths = {}
    events = []
    original_normalize = cwb.normalize_lora_pair
    original_merge = cwb.merge_cwb_lora_pairs
    original_mark_processed = cwb.MemoryEfficientSafeOpen.mark_processed
    received_ranks = []
    processed = []
    for index in range(8):
        rank = 1 if index < 4 else 2
        name = f"lora_{index}"
        path = tmp_path / f"{name}.safetensors"
        save_file({
            "diffusion_model.foo.lora_A.weight": torch.full(
                (rank, 2), 1.0 + index * 0.01
            ),
            "diffusion_model.foo.lora_B.weight": torch.full(
                (2, rank), 1.0 + index * 0.02
            ),
            "diffusion_model.foo.alpha": torch.tensor(float(rank) * 0.5),
        }, str(path))
        paths[name] = str(path)

    def record_normalize(*args, **kwargs):
        events.append("normalize")
        return original_normalize(*args, **kwargs)

    def record_merge(downs, ups, *args, **kwargs):
        events.append("merge")
        received_ranks.extend(down.shape[0] for down in downs)
        for down, up in zip(downs, ups):
            assert down.shape[0] == up.shape[1]
        return original_merge(downs, ups, *args, **kwargs)

    def record_processed(loader, key):
        processed.append(key)
        return original_mark_processed(loader, key)

    monkeypatch.setattr(cwb, "normalize_lora_pair", record_normalize)
    monkeypatch.setattr(cwb, "merge_cwb_lora_pairs", record_merge)
    monkeypatch.setattr(
        cwb.MemoryEfficientSafeOpen, "mark_processed", record_processed
    )
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        list(paths), "loras", _params("eight_input"), lora_mode=True
    )
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))
    assert tensors["diffusion_model.foo.lora_A.weight"].shape[0] == 2
    assert "diffusion_model.foo.alpha" not in tensors
    assert events.count("normalize") == 8
    assert events.index("merge") > max(
        index for index, event in enumerate(events) if event == "normalize"
    )
    assert received_ranks == [1, 1, 1, 1, 2, 2, 2, 2]
    assert len(processed) == 24
    assert processed.count("diffusion_model.foo.lora_A.weight") == 8
    assert processed.count("diffusion_model.foo.lora_B.weight") == 8
    assert processed.count("diffusion_model.foo.alpha") == 8


def test_lora_mismatch_zeros_remains_explicit_across_full_output_rank(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "mismatch_zero_a.safetensors"
    b = tmp_path / "mismatch_zero_b.safetensors"
    save_file({
        "diffusion_model.foo.lora_A.weight": torch.eye(2),
        "diffusion_model.foo.lora_B.weight": torch.eye(2),
    }, str(a))
    save_file({"unrelated": torch.tensor([1.0])}, str(b))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b)})

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "loras",
        _params(
            "mismatch_zero",
            mismatch_mode="zeros",
            consensus_type="mean",
            alignment_method="index",
            rescale_norm=True,
            save_dtype="fp32",
        ),
        lora_mode=True,
    )
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))

    torch.testing.assert_close(
        tensors["diffusion_model.foo.lora_A.weight"], torch.eye(2) * 0.5
    )
    torch.testing.assert_close(
        tensors["diffusion_model.foo.lora_B.weight"], torch.eye(2) * 0.5
    )


def test_lora_multi_wrapper_validates_count_and_forwards_equal_prior_names(
    monkeypatch, cwb
):
    kwargs = {
        "execution_mode": "MERGE",
        "lora_count": "3",
        "lora_1": "a.safetensors",
        "lora_2": "b.safetensors",
        "lora_3": "None",
        "lora_4": "None",
        "lora_5": "None",
        "lora_6": "None",
        "lora_7": "None",
        "lora_8": "None",
        "cwb_preset": "broad_sim_medn_rn_softcb",
        "cwb_config": None,
    }
    monkeypatch.setattr(cwb, "load_documentation_from_file", lambda _: "DOCS")
    with pytest.raises(ValueError, match="unselected input.*3"):
        cwb.CWBLoRAMultiMerger.execute(**kwargs)

    captured = {}

    def execute(names, model_type, params, **options):
        captured.update(
            names=names,
            model_type=model_type,
            params=params,
            options=options,
        )
        return "merged.safetensors", "REPORT"

    monkeypatch.setattr(cwb.ConsensusMergerLogic, "execute", execute)
    kwargs["lora_3"] = "c.safetensors"
    result = cwb.CWBLoRAMultiMerger.execute(**kwargs)
    assert captured["names"] == [
        "a.safetensors", "b.safetensors", "c.safetensors"
    ]
    assert captured["model_type"] == "loras"
    assert captured["options"] == {"lora_mode": True}
    assert result.result == ("merged.safetensors", "DOCS", "REPORT")


def test_embedding_coalescing_uses_mutual_local_pairs(cwb):
    rows = torch.tensor([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
    ])
    merged, pairs = cwb.coalesce_embedding_rows(
        rows,
        _settings(cwb),
        target_vector_count=2,
        similarity_threshold=0.95,
        position_window=1.0,
    )
    assert merged.shape == (2, 2)
    assert pairs == ((0, 1),)
    torch.testing.assert_close(merged[1], rows[2])


def test_visual_boundary_preparation_validates_and_searches_legacy_rows(cwb):
    start = torch.tensor([1.0, 0.0])
    end = torch.tensor([0.0, 1.0])
    body = torch.tensor([[0.6, 0.4]])
    legacy = torch.stack((torch.tensor([-1.0, 0.0]), start, body[0], end, torch.tensor([0.0, -1.0])))
    prepared, scores, trimmed = cwb._prepare_vision_embedding(
        legacy,
        start_reference=start,
        end_reference=end,
        legacy_search=True,
        threshold=0.99,
    )
    torch.testing.assert_close(prepared, body)
    assert scores == pytest.approx((1.0, 1.0))
    assert trimmed == 2

    with pytest.raises(ValueError, match="Vision boundary validation failed"):
        cwb._prepare_vision_embedding(
            torch.stack((start, body[0], -end)),
            start_reference=start,
            end_reference=end,
            legacy_search=False,
            threshold=0.99,
        )


def test_embedding_self_coalesce_streams_and_restores_visual_boundaries(
    monkeypatch, tmp_path, cwb
):
    source = tmp_path / "source.safetensors"
    start = torch.tensor([1.0, 0.0])
    end = torch.tensor([0.0, 1.0])
    save_file({"visual": torch.stack((start, torch.tensor([0.99, 0.01]), torch.tensor([0.98, 0.02]), end))}, str(source))
    _patch_io(monkeypatch, cwb, tmp_path, {"source": str(source)})
    monkeypatch.setattr(cwb, "load_documentation_from_file", lambda _: "DOCS")
    result = cwb.CWBEmbeddingSelfCoalesce.execute(
        embedding="source",
        execution_mode="MERGE",
        cwb_preset="balanced_sim_mean",
        cwb_config=_settings(cwb),
        target_vector_count=1,
        coalesce_similarity_threshold=0.95,
        coalesce_position_window=1.0,
        vision_boundary_embeddings=True,
        legacy_boundary_search=False,
        boundary_reference_embedding="None",
        boundary_similarity_threshold=0.99,
        output_filename="visual_coalesced",
        save_dtype="fp32",
        process_device="cpu",
        lazy_load=True,
        force_clear_cache=False,
        override_dtype=False,
    )
    output = load_file(str(_result_path(tmp_path, "embeddings", result.result[0])))
    assert output["visual"].shape == (3, 2)
    torch.testing.assert_close(output["visual"][0], start)
    torch.testing.assert_close(output["visual"][-1], end)
    assert "CWB EMBEDDING COALESCING" in result.result[2]
    assert "Similarity threshold: 0.9500" in result.result[2]


def test_embedding_multi_wrapper_forwards_selected_inputs(monkeypatch, cwb):
    kwargs = {
        "execution_mode": "MERGE",
        "embedding_count": "3",
        "embedding_1": "a.safetensors",
        "embedding_2": "b.safetensors",
        "embedding_3": "c.safetensors",
        "embedding_4": "None",
        "embedding_5": "None",
        "embedding_6": "None",
        "embedding_7": "None",
        "embedding_8": "None",
        "cwb_preset": "balanced_sim_mean",
        "cwb_config": None,
        "target_vector_count": 0,
        "coalesce_similarity_threshold": 0.95,
        "coalesce_position_window": 0.1,
        "vision_boundary_embeddings": False,
        "legacy_boundary_search": False,
        "boundary_reference_embedding": "None",
        "boundary_similarity_threshold": 0.95,
        "output_filename": "merged",
        "save_dtype": "fp32",
        "process_device": "cpu",
        "lazy_load": True,
        "force_clear_cache": False,
        "override_dtype": False,
    }
    monkeypatch.setattr(cwb, "load_documentation_from_file", lambda _: "DOCS")
    captured = {}

    def execute(names, model_type, params, **options):
        captured.update(names=names, model_type=model_type, params=params, options=options)
        return "merged.safetensors", "REPORT"

    monkeypatch.setattr(cwb.ConsensusMergerLogic, "execute", execute)
    result = cwb.CWBEmbeddingMultiMerger.execute(**kwargs)
    assert captured["names"] == ["a.safetensors", "b.safetensors", "c.safetensors"]
    assert captured["model_type"] == "embeddings"
    assert captured["options"] == {"embedding_union": True}
    assert captured["params"]["embedding_coalesce"] is True
    assert result.result == ("merged.safetensors", "DOCS", "REPORT")


def test_streaming_dtype_nonfloat_and_secondary_preservation(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    save_file({
        "shared": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "only_a": torch.tensor([5.0, 6.0], dtype=torch.float16),
        "metadata_tensor": torch.tensor([1, 2], dtype=torch.int64),
    }, str(a), metadata={"source": "A"})
    save_file({
        "shared": torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float16),
        "extra": torch.ones((2, 2)),
        "metadata_tensor": torch.tensor([9, 9], dtype=torch.int64),
    }, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "diffusion_models", _params("anchored")
    )
    tensors = load_file(str(_result_path(tmp_path, "diffusion_models", result)))
    assert set(tensors) == {"shared", "only_a", "extra", "metadata_tensor"}
    torch.testing.assert_close(tensors["shared"], torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    assert tensors["shared"].dtype == torch.float32
    torch.testing.assert_close(tensors["only_a"], torch.tensor([5.0, 6.0], dtype=torch.float16))
    torch.testing.assert_close(tensors["metadata_tensor"], torch.tensor([1, 2]))
    torch.testing.assert_close(tensors["extra"], torch.ones((2, 2)))

    overridden = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "diffusion_models",
        _params("anchored_override", override_dtype=True),
    )
    overridden_tensors = load_file(
        str(_result_path(tmp_path, "diffusion_models", overridden))
    )
    assert overridden_tensors["shared"].dtype == torch.float16
    assert overridden_tensors["metadata_tensor"].dtype == torch.int64


def test_failed_merge_preserves_existing_output_and_removes_temporary_file(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "atomic_a.safetensors"
    b = tmp_path / "atomic_b.safetensors"
    output = tmp_path / "diffusion_models" / "atomic_output.safetensors"
    output.parent.mkdir(parents=True)
    save_file({"layer": torch.ones((2, 2))}, str(a))
    save_file({"layer": torch.full((2, 2), 2.0)}, str(b))
    save_file({"existing": torch.tensor([7.0])}, str(output))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)

    def fail_merge(*args, **kwargs):
        raise RuntimeError("injected CWB failure")

    monkeypatch.setattr(cwb, "merge_cwb_tensors", fail_merge)
    with pytest.raises(RuntimeError, match="injected CWB failure"):
        cwb.ConsensusMergerLogic.execute(
            ["a", "b"], "diffusion_models", _params("atomic_output")
        )

    existing = load_file(str(output))
    assert set(existing) == {"existing"}
    torch.testing.assert_close(existing["existing"], torch.tensor([7.0]))
    assert not list(output.parent.glob(".atomic_output.safetensors.*.tmp"))


def test_embedding_union_uses_longest_first_dimension(monkeypatch, tmp_path, cwb):
    a = tmp_path / "embed_a.safetensors"
    b = tmp_path / "embed_b.safetensors"
    save_file({"emb": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}, str(a))
    save_file({
        "emb": torch.tensor([[3.0, 0.0], [0.0, 3.0], [4.0, 4.0]]),
        "secondary_only": torch.tensor([[7.0, 0.0]]),
    }, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "embeddings",
        _params("embedding_union"),
        embedding_union=True,
    )
    tensors = load_file(str(_result_path(tmp_path, "embeddings", result)))
    assert tensors["emb"].shape == (3, 2)
    torch.testing.assert_close(tensors["emb"][:2], torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    torch.testing.assert_close(tensors["emb"][2], torch.tensor([4.0, 4.0]))
    torch.testing.assert_close(tensors["secondary_only"], torch.tensor([[7.0, 0.0]]))


def test_output_name_preserves_category_relative_subdirectory(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "relative_a.safetensors"
    b = tmp_path / "relative_b.safetensors"
    save_file({"layer": torch.tensor([[1.0]])}, str(a))
    save_file({"layer": torch.tensor([[1.0]])}, str(b))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b)})

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "loras",
        _params("groupfolder/modelname"),
    )

    assert result == "groupfolder/modelname.safetensors"
    assert (tmp_path / "loras" / "groupfolder" / "modelname.safetensors").is_file()


def test_three_input_streaming_merge(monkeypatch, tmp_path, cwb):
    paths = {}
    for name, value in (("a", 1.0), ("b", 3.0), ("c", 5.0)):
        path = tmp_path / f"three_{name}.safetensors"
        save_file({"layer": torch.tensor([[value, 0.0]])}, str(path))
        paths[name] = str(path)
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b", "c"], "diffusion_models", _params("three_inputs")
    )
    tensor = load_file(str(_result_path(tmp_path, "diffusion_models", result)))["layer"]
    torch.testing.assert_close(tensor, torch.tensor([[3.0, 0.0]]))


@pytest.mark.parametrize("model_count", [2, 3])
@pytest.mark.parametrize("prefixed_input", ["a", "b"])
def test_diffusion_cwb_matches_prefixed_quantized_and_bare_layers(
    monkeypatch, tmp_path, cwb, model_count, prefixed_input,
):
    names = ["a", "b", "c"][:model_count]
    paths = {}
    for index, name in enumerate(names):
        path = tmp_path / f"mixed_{name}.safetensors"
        key = f"{'model.diffusion_model.' if name == prefixed_input else ''}layer.weight"
        values = torch.tensor([2.0, 4.0]) * (index + 1)
        if name == prefixed_input:
            stem = key[:-len(".weight")]
            save_file({
                key: values.to(torch.int8),
                f"{stem}.weight_scale": torch.tensor(1.0),
                f"{stem}.comfy_quant": torch.tensor(
                    list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8
                ),
            }, str(path))
        else:
            save_file({key: values}, str(path))
        paths[name] = str(path)
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        names, "diffusion_models", _params("mixed_cwb")
    )
    tensors = load_file(str(_result_path(tmp_path, "diffusion_models", result)))
    assert list(tensors) == ["layer.weight"]
    torch.testing.assert_close(tensors["layer.weight"].float(), torch.tensor([2.0, 4.0]) * (model_count + 1) / 2)


def test_diffusion_cwb_excluded_quantized_layer_is_saved_dense(monkeypatch, tmp_path, cwb):
    a = tmp_path / "excluded_quant_a.safetensors"
    b = tmp_path / "excluded_quant_b.safetensors"
    save_file({
        "model.diffusion_model.layer.weight": torch.tensor([1, 2], dtype=torch.int8),
        "model.diffusion_model.layer.weight_scale": torch.tensor(2.0),
        "model.diffusion_model.layer.comfy_quant": torch.tensor(
            list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8
        ),
    }, str(a))
    save_file({"layer.weight": torch.tensor([9.0, 9.0])}, str(b))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b)})
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "diffusion_models", _params("excluded_quant", include_mode=True),
    )
    tensors = load_file(str(_result_path(tmp_path, "diffusion_models", result)))
    assert list(tensors) == ["layer.weight"]
    torch.testing.assert_close(tensors["layer.weight"].float(), torch.tensor([2.0, 4.0]))


def test_diffusion_cwb_guarded_quantized_primary_is_saved_dense(monkeypatch, tmp_path, cwb):
    a = tmp_path / "guarded_quant_a.safetensors"
    b = tmp_path / "guarded_quant_b.safetensors"
    save_file({
        "model.diffusion_model.layer.weight": torch.tensor([1, 2], dtype=torch.int8),
        "model.diffusion_model.layer.weight_scale": torch.tensor(2.0),
        "model.diffusion_model.layer.comfy_quant": torch.tensor(
            list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8
        ),
    }, str(a))
    save_file({"layer.weight": torch.tensor([9, 9], dtype=torch.uint8)}, str(b))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b)})
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "diffusion_models", _params("guarded_quant"),
    )
    tensors = load_file(str(_result_path(tmp_path, "diffusion_models", result)))
    assert list(tensors) == ["layer.weight"]
    torch.testing.assert_close(tensors["layer.weight"].float(), torch.tensor([2.0, 4.0]))


def test_two_input_preserves_secondary_only_tensor_exactly(monkeypatch, tmp_path, cwb):
    a = tmp_path / "secondary_a.safetensors"
    b = tmp_path / "secondary_b.safetensors"
    save_file({"shared": torch.tensor([1.0])}, str(a))
    save_file({
        "shared": torch.tensor([1.0]),
        "secondary": torch.tensor([2.0, 3.0], dtype=torch.float32),
    }, str(b))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b)})

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "diffusion_models",
        _params(
            "secondary_exact",
            global_scale=2.0,
            override_dtype=True,
            save_dtype="fp16",
        ),
    )
    tensor = load_file(str(_result_path(tmp_path, "diffusion_models", result)))["secondary"]
    torch.testing.assert_close(tensor, torch.tensor([2.0, 3.0]))
    assert tensor.dtype == torch.float32


def test_three_input_merges_available_secondary_sources(monkeypatch, tmp_path, cwb):
    a = tmp_path / "union_a.safetensors"
    b = tmp_path / "union_b.safetensors"
    c = tmp_path / "union_c.safetensors"
    save_file({"shared": torch.tensor([1.0])}, str(a))
    save_file({
        "shared": torch.tensor([1.0]),
        "secondary_shared": torch.tensor([2.0]),
        "b_only": torch.tensor([7.0]),
    }, str(b))
    save_file({
        "shared": torch.tensor([1.0]),
        "secondary_shared": torch.tensor([4.0]),
        "c_only": torch.tensor([9.0]),
    }, str(c))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b), "c": str(c)})

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b", "c"], "diffusion_models", _params("secondary_union")
    )
    tensors = load_file(str(_result_path(tmp_path, "diffusion_models", result)))
    torch.testing.assert_close(tensors["secondary_shared"], torch.tensor([3.0]))
    torch.testing.assert_close(tensors["b_only"], torch.tensor([7.0]))
    torch.testing.assert_close(tensors["c_only"], torch.tensor([9.0]))


def test_secondary_shape_conflict_skip_preserves_earliest_source(monkeypatch, tmp_path, cwb):
    a = tmp_path / "shape_a.safetensors"
    b = tmp_path / "shape_b.safetensors"
    c = tmp_path / "shape_c.safetensors"
    save_file({"shared": torch.tensor([1.0])}, str(a))
    save_file({"secondary": torch.tensor([2.0, 3.0])}, str(b))
    save_file({"secondary": torch.tensor([[8.0, 9.0]])}, str(c))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b), "c": str(c)})

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b", "c"],
        "diffusion_models",
        _params("secondary_shape_skip", mismatch_mode="skip"),
    )
    tensor = load_file(str(_result_path(tmp_path, "diffusion_models", result)))["secondary"]
    torch.testing.assert_close(tensor, torch.tensor([2.0, 3.0]))


@pytest.mark.parametrize("lora_mode", [False, True])
@pytest.mark.parametrize("patterns, glob_patterns, selected", [
    (r"\.selected", False, True),
    ("*.selected*", True, True),
    ("", False, False),
    ("no_match", False, False),
])
def test_include_filter_preserves_nonmatches_and_discards_first(
    monkeypatch, tmp_path, cwb, lora_mode, patterns, glob_patterns, selected,
):
    paths = {name: str(tmp_path / f"{name}.safetensors") for name in ("a", "b")}
    for name, value in (("a", 2.0), ("b", 6.0)):
        save_file({
            f"diffusion_model.{layer}.diff": torch.full((2, 2), value)
            for layer in ("selected", "other", "selected_discard")
        }, paths[name])
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    category = "loras" if lora_mode else "diffusion_models"
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], category,
        _params("include_filter", exclude_patterns=patterns, include_mode=True,
                glob_patterns=glob_patterns, discard_patterns="*discard*" if glob_patterns else "discard",
                power_alpha=0.0),
        lora_mode=lora_mode,
    )
    tensors = load_file(str(_result_path(tmp_path, category, result)))
    prefix = "diffusion_model." if lora_mode else ""
    assert set(tensors) == {f"{prefix}selected.diff", f"{prefix}other.diff"}
    torch.testing.assert_close(tensors[f"{prefix}selected.diff"], torch.full((2, 2), 4.0 if selected else 2.0))
    torch.testing.assert_close(tensors[f"{prefix}other.diff"], torch.full((2, 2), 2.0))


def test_secondary_only_filters_and_low_bit_preservation(monkeypatch, tmp_path, cwb):
    a = tmp_path / "filter_a.safetensors"
    b = tmp_path / "filter_b.safetensors"
    save_file({"shared": torch.tensor([1.0])}, str(a))
    save_file({
        "secondary.keep": torch.tensor([4, 5], dtype=torch.uint8),
        "secondary.drop": torch.tensor([6.0]),
    }, str(b))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b)})

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "diffusion_models",
        _params("secondary_filters", discard_patterns=r"secondary\.drop"),
    )
    tensors = load_file(str(_result_path(tmp_path, "diffusion_models", result)))
    assert "secondary.drop" not in tensors
    torch.testing.assert_close(
        tensors["secondary.keep"], torch.tensor([4, 5], dtype=torch.uint8)
    )


def test_lora_cross_format_companion_group_preserves_model_a_canonically(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "lora_a.safetensors"
    b = tmp_path / "lora_b.safetensors"
    save_file({
        "diffusion_model.foo.lora_A.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "diffusion_model.foo.lora_B.weight": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        "diffusion_model.foo.dora_scale": torch.tensor([1.0, 1.0]),
    }, str(a))
    save_file({
        "lora_unet_foo.lora_down.weight": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        "lora_unet_foo.lora_up.weight": torch.tensor([[3.0, 5.0], [4.0, 6.0]]),
        "lora_unet_secondary.lora_down.weight": torch.ones((1, 2)),
        "lora_unet_secondary.lora_up.weight": torch.ones((2, 1)),
    }, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "loras", _params("lora_cross_format"), lora_mode=True
    )
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))
    assert set(tensors) == {
        "diffusion_model.foo.lora_A.weight",
        "diffusion_model.foo.lora_B.weight",
        "diffusion_model.foo.dora_scale",
        "lora_unet_secondary.lora_A.weight",
        "lora_unet_secondary.lora_B.weight",
    }
    assert tensors["diffusion_model.foo.lora_A.weight"].shape == (1, 2)
    assert tensors["diffusion_model.foo.lora_B.weight"].shape == (2, 1)
    assert tensors["diffusion_model.foo.lora_A.weight"].dtype == torch.float32
    torch.testing.assert_close(
        tensors["diffusion_model.foo.dora_scale"], torch.tensor([1.0, 1.0])
    )
    torch.testing.assert_close(
        tensors["lora_unet_secondary.lora_A.weight"], torch.ones((1, 2))
    )
    torch.testing.assert_close(
        tensors["lora_unet_secondary.lora_B.weight"], torch.ones((2, 1))
    )


def test_lora_secondary_only_pair_is_alpha_normalized(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "lora_secondary_a.safetensors"
    b = tmp_path / "lora_secondary_b.safetensors"
    save_file({
        "diffusion_model.base.diff": torch.tensor([[1.0]]),
    }, str(a))
    save_file({
        "transformer.extra.lora_down.weight": torch.tensor([[2.0, 3.0]]),
        "transformer.extra.lora_up.weight": torch.tensor([[4.0], [5.0]]),
        "transformer.extra.alpha": torch.tensor(0.5),
    }, str(b))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b)})

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "loras",
        _params(
            "lora_secondary_exact",
            global_scale=2.0,
            override_dtype=True,
            save_dtype="fp16",
        ),
        lora_mode=True,
    )
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))
    torch.testing.assert_close(
        tensors["diffusion_model.extra.lora_A.weight"],
        torch.tensor([[2.0, 3.0]]),
    )
    torch.testing.assert_close(
        tensors["diffusion_model.extra.lora_B.weight"],
        torch.tensor([[2.0], [2.5]]),
    )
    assert tensors["diffusion_model.extra.lora_A.weight"].dtype == torch.float32
    assert "diffusion_model.extra.alpha" not in tensors


def test_lora_three_input_merges_b_and_c_group_without_a(monkeypatch, tmp_path, cwb):
    a = tmp_path / "lora_union_a.safetensors"
    b = tmp_path / "lora_union_b.safetensors"
    c = tmp_path / "lora_union_c.safetensors"
    save_file({"diffusion_model.base.diff": torch.tensor([[1.0]])}, str(a))
    save_file({
        "diffusion_model.extra.lora_A.weight": torch.full((1, 2), 1.0),
        "diffusion_model.extra.lora_B.weight": torch.full((2, 1), 1.0),
    }, str(b))
    save_file({
        "transformer.extra.lora_down.weight": torch.full((1, 2), 3.0),
        "transformer.extra.lora_up.weight": torch.full((2, 1), 3.0),
    }, str(c))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b), "c": str(c)})

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b", "c"], "loras", _params("lora_secondary_union"), lora_mode=True
    )
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))
    torch.testing.assert_close(
        tensors["diffusion_model.extra.lora_A.weight"], torch.full((1, 2), 2.0)
    )
    torch.testing.assert_close(
        tensors["diffusion_model.extra.lora_B.weight"], torch.full((2, 1), 2.0)
    )


def test_lora_secondary_direct_and_companion_groups_are_atomic(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "lora_atomic_a.safetensors"
    b = tmp_path / "lora_atomic_b.safetensors"
    save_file({"diffusion_model.base.diff": torch.tensor([[1.0]])}, str(a))
    save_file({
        "diffusion_model.norm.diff": torch.tensor([2.0, 3.0], dtype=torch.bfloat16),
        "diffusion_model.extra.lora_A.weight": torch.tensor([[4.0, 5.0]]),
        "diffusion_model.extra.lora_B.weight": torch.tensor([[6.0], [7.0]]),
        "diffusion_model.extra.dora_scale": torch.tensor([8.0, 9.0]),
    }, str(b))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b)})

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "loras",
        _params("lora_secondary_atomic", global_scale=3.0),
        lora_mode=True,
    )
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))
    torch.testing.assert_close(
        tensors["diffusion_model.norm.diff"],
        torch.tensor([2.0, 3.0], dtype=torch.bfloat16),
    )
    assert tensors["diffusion_model.norm.diff"].dtype == torch.bfloat16
    torch.testing.assert_close(
        tensors["diffusion_model.extra.lora_A.weight"], torch.tensor([[4.0, 5.0]])
    )
    torch.testing.assert_close(
        tensors["diffusion_model.extra.lora_B.weight"], torch.tensor([[6.0], [7.0]])
    )
    torch.testing.assert_close(
        tensors["diffusion_model.extra.dora_scale"], torch.tensor([8.0, 9.0])
    )


def test_lora_rejects_duplicate_normalized_cores_in_secondary_input(cwb):
    with pytest.raises(ValueError, match="same logical layer 'foo'.*input 2"):
        cwb._secondary_lora_map(
            {
                "diffusion_model.foo": {},
                "transformer.foo": {},
            },
            input_index=1,
        )


def test_lora_peft_prefix_matching_and_existing_alpha(monkeypatch, tmp_path, cwb):
    a = tmp_path / "peft_a.safetensors"
    b = tmp_path / "peft_b.safetensors"
    save_file({
        "base_model.model.diffusion_model.foo.lora_A.weight": torch.ones((1, 2)),
        "base_model.model.diffusion_model.foo.lora_B.weight": torch.ones((2, 1)),
        "base_model.model.diffusion_model.foo.alpha": torch.tensor(1.0),
    }, str(a))
    save_file({
        "diffusion_model.foo.lora_down.weight": torch.ones((2, 2)),
        "diffusion_model.foo.lora_up.weight": torch.ones((2, 2)),
        "diffusion_model.foo.alpha": torch.tensor(2.0),
    }, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "loras", _params("peft_prefix"), lora_mode=True
    )
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))
    assert set(tensors) == {
        "diffusion_model.foo.lora_A.weight",
        "diffusion_model.foo.lora_B.weight",
    }
    assert "diffusion_model.foo.alpha" not in tensors


def test_lora_mochi_inputs_emit_preferred_canonical_output(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "mochi_a.safetensors"
    b = tmp_path / "mochi_b.safetensors"
    for path, value in ((a, 1.0), (b, 3.0)):
        save_file({
            "diffusion_model.foo.lora_A": torch.full((1, 2), value),
            "diffusion_model.foo.lora_B": torch.full((2, 1), value),
            "diffusion_model.foo.alpha": torch.tensor(1.0),
        }, str(path))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "loras", _params("mochi_canonical"), lora_mode=True
    )
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))

    assert set(tensors) == {
        "diffusion_model.foo.lora_A.weight",
        "diffusion_model.foo.lora_B.weight",
    }


def test_lora_1d_direct_switch_is_default_fallback_and_fp32(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "direct_a.safetensors"
    b = tmp_path / "direct_b.safetensors"
    save_file({"diffusion_model.norm.diff": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}, str(a))
    save_file({"diffusion_model.norm.diff": torch.tensor([3.0, 4.0], dtype=torch.bfloat16)}, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)

    default = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "loras", _params("direct_default"), lora_mode=True
    )
    default_tensor = load_file(str(_result_path(tmp_path, "loras", default)))[
        "diffusion_model.norm.diff"
    ]
    torch.testing.assert_close(default_tensor, torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    assert default_tensor.dtype == torch.bfloat16

    enabled = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "loras",
        _params("direct_enabled", include_1d_diffs=True, override_dtype=True),
        lora_mode=True,
    )
    enabled_tensor = load_file(str(_result_path(tmp_path, "loras", enabled)))[
        "diffusion_model.norm.diff"
    ]
    torch.testing.assert_close(enabled_tensor, torch.tensor([2.0, 3.0]))
    assert enabled_tensor.dtype == torch.float32


def test_invalid_quantization_sidecar_errors_before_output(monkeypatch, tmp_path, cwb):
    a = tmp_path / "quant_a.safetensors"
    b = tmp_path / "quant_b.safetensors"
    save_file({
        "layer.weight": torch.ones((2, 2)),
        "layer.comfy_quant": torch.tensor([1], dtype=torch.uint8),
    }, str(a))
    save_file({"layer.weight": torch.ones((2, 2))}, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    with pytest.raises(ValueError, match="Invalid quantization sidecar"):
        cwb.ConsensusMergerLogic.execute(
            ["a", "b"], "diffusion_models", _params("must_not_exist")
        )
    assert not (tmp_path / "diffusion_models" / "must_not_exist.safetensors").exists()


def test_isolated_low_bit_preserves_anchored_tensor(monkeypatch, tmp_path, cwb):
    a = tmp_path / "guard_a.safetensors"
    b = tmp_path / "guard_b.safetensors"
    save_file({"layer": torch.tensor([1.0, 2.0])}, str(a))
    save_file({"layer": torch.tensor([8, 9], dtype=torch.uint8)}, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "diffusion_models", _params("guarded_generic")
    )
    tensor = load_file(str(_result_path(tmp_path, "diffusion_models", result)))["layer"]
    torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0]))
    assert tensor.dtype == torch.float32


def test_isolated_low_bit_preserves_complete_lora_layer(monkeypatch, tmp_path, cwb):
    a = tmp_path / "guard_lora_a.safetensors"
    b = tmp_path / "guard_lora_b.safetensors"
    save_file({
        "diffusion_model.foo.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "diffusion_model.foo.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        "diffusion_model.foo.alpha": torch.tensor(1.0),
        "diffusion_model.foo.dora_scale": torch.tensor([5.0, 6.0]),
    }, str(a))
    save_file({
        "diffusion_model.foo.lora_A.weight": torch.tensor([[9.0, 9.0]]),
        "diffusion_model.foo.lora_B.weight": torch.tensor([[9.0], [9.0]]),
        "diffusion_model.foo.alpha": torch.tensor(1, dtype=torch.uint8),
    }, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "loras", _params("guarded_lora"), lora_mode=True
    )
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))
    torch.testing.assert_close(
        tensors["diffusion_model.foo.lora_A.weight"], torch.tensor([[1.0, 2.0]])
    )
    torch.testing.assert_close(
        tensors["diffusion_model.foo.lora_B.weight"], torch.tensor([[3.0], [4.0]])
    )
    assert "diffusion_model.foo.alpha" not in tensors
    torch.testing.assert_close(
        tensors["diffusion_model.foo.dora_scale"], torch.tensor([5.0, 6.0])
    )


def test_alpha_bearing_low_bit_lora_factors_are_rejected(monkeypatch, tmp_path, cwb):
    a = tmp_path / "alpha_low_bit_a.safetensors"
    b = tmp_path / "alpha_low_bit_b.safetensors"
    save_file({
        "diffusion_model.foo.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "diffusion_model.foo.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        "diffusion_model.foo.alpha": torch.tensor(1.0),
    }, str(a))
    save_file({
        "diffusion_model.foo.lora_A.weight": torch.tensor([[1, 2]], dtype=torch.uint8),
        "diffusion_model.foo.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        "diffusion_model.foo.alpha": torch.tensor(1.0),
    }, str(b))
    _patch_io(monkeypatch, cwb, tmp_path, {"a": str(a), "b": str(b)})

    with pytest.raises(ValueError, match="Cannot alpha-normalize low-bit LoRA factors"):
        cwb.ConsensusMergerLogic.execute(
            ["a", "b"], "loras", _params("alpha_low_bit"), lora_mode=True
        )
