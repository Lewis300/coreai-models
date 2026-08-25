# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the optional ``prompt`` (prefill) entrypoint on macOS exports.

A model that sets ``exports_prompt_graph`` gets a second entrypoint traced from the
same signature with prefill mode on. It must declare no outputs and bind exactly the
inputs and states ``main`` does, because that is the contract the Swift runner
validates in ``loadPromptGraph``.

Exports a tiny randomly-initialised model, so no HuggingFace weights are needed, but
it does run a real conversion and therefore needs ``coreai-torch``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from coreai_models._constants import (
    KEY_CACHE_NAME,
    MAIN_GRAPH_NAME,
    PROMPT_GRAPH_NAME,
    VALUE_CACHE_NAME,
)

try:
    import coreai.runtime as rt
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from coreai_models.export.macos import export_macos_model
    from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM

    HAS_COREAI = True
except ImportError:  # pragma: no cover - depends on the installed toolchain
    HAS_COREAI = False

MAX_CTX = 256

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
    """Export a tiny random model, opting the prompt graph in or out.

    Subclassing to flip the flag keeps the real ``forward`` -- including its
    ``prefill_mode`` guard -- so the opt-out case proves the flag alone decides,
    not the presence of the guard.
    """

    class _Model(Qwen3ForCausalLM):
        exports_prompt_graph = opts_in

    config = _tiny_config()
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


@pytest.mark.asyncio
async def test_prompt_graph_is_emitted_when_opted_in() -> None:
    model, program = _export(opts_in=True)
    descs = await _function_descriptors(program)

    assert set(descs) == {MAIN_GRAPH_NAME, PROMPT_GRAPH_NAME}

    prompt, main = descs[PROMPT_GRAPH_NAME], descs[MAIN_GRAPH_NAME]

    # No outputs at all: the KV cache writes are the graph's only product. The runner
    # rejects a prompt graph that declares any, so this is the load-bearing assertion.
    assert list(prompt.output_names) == []
    assert list(main.output_names) == ["logits"]

    # Same bindings as `main`, by name -- the runner feeds both from one code path.
    assert list(prompt.input_names) == list(main.input_names)
    assert set(prompt.state_names) == set(main.state_names)
    assert set(prompt.state_names) == {KEY_CACHE_NAME, VALUE_CACHE_NAME}

    # The exporter must not leave a shared model in prefill mode.
    assert model.prefill_mode is False


@pytest.mark.asyncio
async def test_prompt_graph_is_absent_when_opted_out() -> None:
    model, program = _export(opts_in=False)
    descs = await _function_descriptors(program)

    assert set(descs) == {MAIN_GRAPH_NAME}
    assert PROMPT_GRAPH_NAME not in descs
    assert list(descs[MAIN_GRAPH_NAME].output_names) == ["logits"]
    assert model.prefill_mode is False
