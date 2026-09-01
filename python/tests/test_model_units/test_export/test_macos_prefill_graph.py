# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the optional ``prefill`` entrypoint on macOS exports.

A model that sets ``exports_prefill_graph`` gets a second entrypoint from the same
signature: traced again with prefill mode on when the model is eager, or the one decode
trace staged again with its non-state outputs trimmed when it arrives flattened. Either
way it must declare no outputs and bind exactly the inputs and states ``main`` does,
because that is the contract the Swift runner validates in ``loadPrefillGraph``. Beyond
that shape contract, the KV cache it writes is its only product, so the tests here
execute both flavours and check that cache numerically against eager torch.

Exports a tiny randomly-initialised model, so no HuggingFace weights are needed, but
it does run a real conversion and therefore needs ``coreai-torch``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.utils._pytree as pytree

from coreai_models._constants import (
    KEY_CACHE_NAME,
    MAIN_GRAPH_NAME,
    PREFILL_GRAPH_NAME,
    QUANT_TRACE_OFFSET,
    VALUE_CACHE_NAME,
)

try:
    import coreai.runtime as rt
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from coreai_models.export.externalize import patch_model_for_externalization
    from coreai_models.export.macos import _drop_user_outputs, export_macos_model
    from coreai_models.models.base import TraceSpec
    from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM
    from coreai_models.primitives.macos.cache import KVCache

    # Imported here rather than at module scope: `testing_utils` imports
    # `coreai_torch` unguarded, so a top-level import would break collection in
    # environments without the toolchain, which the skip below exists to tolerate.
    from tests._runner_infra.testing_utils import assert_close

    HAS_COREAI = True
except ImportError:  # pragma: no cover - depends on the installed toolchain
    HAS_COREAI = False

MAX_CTX = 256

# Prompt length for the parity test. Longer than one token so prefill is doing real
# multi-token work, and short enough that `QUANT_TRACE_OFFSET + PREFILL_LEN` stays inside
# the traced `position_ids` bound of MAX_CTX - 1.
PREFILL_LEN = 32

pytestmark = pytest.mark.skipif(not HAS_COREAI, reason="coreai-torch not available")


def _tiny_config() -> Qwen3Config:
    """Smallest Qwen3 that still exercises attention, an MLP and a KV cache."""
    return Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=MAX_CTX,
        tie_word_embeddings=True,
    )


def _export(*, opts_in: bool) -> tuple[torch.nn.Module, object]:
    """Export a tiny random model, opting the prefill graph in or out.

    Subclassing to flip the flag keeps the real ``forward`` -- including its
    ``prefill_mode`` guard -- so the opt-out case proves the flag alone decides,
    not the presence of the guard.
    """

    class _Model(Qwen3ForCausalLM):
        exports_prefill_graph = opts_in

    config = _tiny_config()
    # Seeded so the parity test's error margin is reproducible run to run.
    torch.manual_seed(0)
    model = _Model(config).to(torch.float16).eval()
    # `compute_precision` has to agree with the dtype above: the exporter resolves the
    # trace dtype from it, not from the model's parameters.
    export_config = SimpleNamespace(max_context_length=MAX_CTX, compute_precision="float16")
    program = export_macos_model(model, config, export_config)
    return model, program


async def _function_descriptors(program) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """Save the program and read back one descriptor per entrypoint."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.aimodel"
        program.save_asset(path, rt.AIModelAssetMetadata())
        model = await rt.AIModel.load(path)
        return {name: model.load_function(name).desc for name in model.function_names}


def _prefill_inputs(config: Qwen3Config) -> tuple[torch.Tensor, torch.Tensor]:
    """One prefill chunk of ``PREFILL_LEN`` tokens landing at ``QUANT_TRACE_OFFSET``.

    ``position_ids`` has to come out strictly longer than ``input_ids``. The authoring
    derives ``offset = len(position_ids) - len(input_ids)`` and marks it size-like, so a
    program exported from it guards on a nonzero offset and refuses a whole prompt at
    offset 0 when it is run in eager -- even though the eager model itself is happy to.
    Prefilling at the offset the exporter traced with stays inside that envelope, the same
    way ``_prep_calib_inputs`` builds ``position_ids`` for calibration. Both sides of the
    comparison see identical inputs, so the offset costs this test nothing.
    """
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(
        1, config.vocab_size, (1, PREFILL_LEN), dtype=torch.int32, generator=generator
    )
    position_ids = torch.arange(QUANT_TRACE_OFFSET + PREFILL_LEN, dtype=torch.int32).unsqueeze(0)
    return input_ids, position_ids


def _zeroed_caches() -> tuple[torch.Tensor, torch.Tensor]:
    """A fresh, empty cache at the exported context length."""
    return KVCache.create_cache_tensors(_tiny_config(), dtype=torch.float16, seq_len=MAX_CTX)


def _torch_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager reference: run ``forward`` in prefill mode and return the caches it filled.

    This is the same authoring the prefill graph is traced from, in prefill mode, which
    is the only reference available: HuggingFace has no prefill-only forward to compare
    against. So this pins the conversion, not the authoring.

    Also serves the trimmed flattened graph, which resolved ``prefill_mode`` back when it
    was captured and so has no flag left to toggle -- the same reason the exporter reaches
    for the setter through ``getattr``.
    """
    k_cache, v_cache = _zeroed_caches()
    set_prefill_mode = getattr(model, "set_prefill_mode", None)
    if set_prefill_mode is not None:
        set_prefill_mode(True)
    try:
        with torch.no_grad():
            assert model(input_ids, position_ids, k_cache, v_cache) == ()
    finally:
        if set_prefill_mode is not None:
            set_prefill_mode(False)
    return k_cache, v_cache


@pytest.mark.asyncio
async def test_prefill_graph_is_emitted_when_opted_in() -> None:
    model, program = _export(opts_in=True)
    descs = await _function_descriptors(program)

    assert set(descs) == {MAIN_GRAPH_NAME, PREFILL_GRAPH_NAME}

    prefill, main = descs[PREFILL_GRAPH_NAME], descs[MAIN_GRAPH_NAME]

    # No outputs at all: the KV cache writes are the graph's only product. The runner
    # rejects a prefill graph that declares any, so this is the load-bearing assertion.
    assert list(prefill.output_names) == []
    assert list(main.output_names) == ["logits"]

    # Same bindings as `main`, by name -- the runner feeds both from one code path.
    assert list(prefill.input_names) == list(main.input_names)
    assert set(prefill.state_names) == set(main.state_names)
    assert set(prefill.state_names) == {KEY_CACHE_NAME, VALUE_CACHE_NAME}

    # The exporter must not leave a shared model in prefill mode.
    assert model.prefill_mode is False


@pytest.mark.asyncio
async def test_prefill_graph_is_absent_when_opted_out() -> None:
    model, program = _export(opts_in=False)
    descs = await _function_descriptors(program)

    assert set(descs) == {MAIN_GRAPH_NAME}
    assert PREFILL_GRAPH_NAME not in descs
    assert list(descs[MAIN_GRAPH_NAME].output_names) == ["logits"]
    assert model.prefill_mode is False


def _export_flattened() -> tuple[torch.export.ExportedProgram, object]:
    """Export the way graph-mode quantization does: mark, capture, hand over the graph.

    Returns the captured program -- the thing the exporter trims for its prefill
    entrypoint, and so the thing the numerics test needs -- alongside the export.
    """
    config = _tiny_config()
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).to(torch.float16).eval()
    spec = TraceSpec(max_context_length=MAX_CTX, cache_seq_len=MAX_CTX)
    reference_inputs = model.build_reference_inputs(config, torch.float16, spec)[MAIN_GRAPH_NAME]
    dynamic_shapes = model.build_dynamic_shapes(config, spec)[MAIN_GRAPH_NAME]
    # Marking before capture is what graph-mode quantization does, and it is what leaves
    # the composite call sites in the flattened graph for the exporter to externalize.
    patch_model_for_externalization(model)
    with torch.no_grad():
        flattened = torch.export.export(
            model, args=tuple(reference_inputs.values()), dynamic_shapes=dynamic_shapes
        )

    assert model.exports_prefill_graph
    export_config = SimpleNamespace(max_context_length=MAX_CTX, compute_precision="float16")
    return flattened, export_macos_model(
        flattened.module(), config, export_config, externalized_model=model
    )


def _get_trimmed_flattened() -> torch.nn.Module:
    """The exporter's trimmed prefill program, made runnable in eager.

    ``_drop_user_outputs`` rewrites ``graph_signature``, which is the only view the
    converter reads, but ``module_call_graph`` still carries the pytree spec captured
    before the trim -- it describes the ``logits`` the trim just deleted.
    ``ExportedProgram.module()`` builds its flatten/unflatten calls from that spec, so
    without this repair the wrapper unflattens the graph's zero outputs into a one-leaf
    spec and raises. The export path never trips over it because it never makes the
    trimmed program eagerly callable, so the repair belongs here rather than in the
    exporter.
    """
    flattened, _ = _export_flattened()
    trimmed = _drop_user_outputs(flattened)
    root = next(entry for entry in trimmed.module_call_graph if entry.fqn == "")
    root.signature.out_spec = pytree.tree_structure(())
    return trimmed.module()


@pytest.mark.asyncio
async def test_flattened_model_gets_a_trimmed_prefill_graph() -> None:
    """Graph-mode quantization hands the exporter a flattened module, which resolved
    ``prefill_mode`` when it was captured, so there is no second trace to take. The
    exporter stages the one decode trace again with its non-state outputs trimmed, which
    leaves the LM head dead. The result has to satisfy the same runner contract as the
    twice-traced eager one.
    """
    _, program = _export_flattened()

    descs = await _function_descriptors(program)
    assert set(descs) == {MAIN_GRAPH_NAME, PREFILL_GRAPH_NAME}

    prefill, main = descs[PREFILL_GRAPH_NAME], descs[MAIN_GRAPH_NAME]
    assert list(prefill.output_names) == []
    assert list(main.output_names) == ["logits"]
    assert list(prefill.input_names) == list(main.input_names)
    assert set(prefill.state_names) == set(main.state_names) == {KEY_CACHE_NAME, VALUE_CACHE_NAME}


@pytest.mark.asyncio
async def test_flattened_vs_model_numerics() -> None:
    """Test base torch module with prefill_mode = True against trimmed flattened GraphModule"""
    model, _ = _export(opts_in=False)
    flattened = _get_trimmed_flattened()
    input_ids, position_ids = _prefill_inputs(_tiny_config())
    k_torch_base, v_torch_base = _torch_prefill(model, input_ids, position_ids)
    k_torch_flattened, v_torch_flattened = _torch_prefill(flattened, input_ids, position_ids)

    for name, torch_cache_base, torch_cache_flattened in (
        (KEY_CACHE_NAME, k_torch_base, k_torch_flattened),
        (VALUE_CACHE_NAME, v_torch_base, v_torch_flattened),
    ):
        print(f"comparing {name}")
        assert_close(torch_cache_base.float(), torch_cache_flattened.float(), atol=1e-9, rtol=1e-9)
