#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Standalone export recipe for the Gemma 4 PLE variants (E2B / E4B), iOS.

The Per-Layer-Embeddings variants don't fit the generic ``coreai.llm.export``
pipeline, so they ship as a standalone recipe. This recipe is the only iOS path
for Gemma 4 -- see ``models/gemma4/README.md``.

Runs in the workspace environment; no dependency overlay is needed, since the
workspace pins ``transformers>=5.5,<6.0`` which carries ``transformers.models.gemma4``:

    cd models/gemma4
    uv run export.py --model google/gemma-4-E2B-it --max-context-length 32768

The export is a per-context blocked ladder of *statically*-shaped programs:

* Every context bucket is its own pair of entrypoints (``extend_{ctx}`` /
  ``prompt_opt_{ctx}``), each further specialized by query length, because the
  flash block loop is unrolled into the graph.
* A flat global KV cache (one dynamic write offset) paired with a fixed-depth
  sliding-window ring, plus the shared ``load_embeddings`` /
  ``gather_embeddings`` helper functions.
* RoPE arrives precomputed as ``rope_cos``/``rope_sin`` inputs instead of
  ``position_ids`` (a 131k position overflows a 16-bit input).
* An externalized INT8 Per-Layer Embeddings (PLE) sidecar written next to the
  asset, plus per-function hardware constraints and static shape configs.
* Compression is k-means palettization, driven by a coreai-opt YAML
  (``gemma4_4bit_palettized.yaml``). ``--compression`` selects a named
  palettization preset instead.

The recipe reuses the shared building blocks (``TorchConverter``, the
compression helpers, ``bundle_llm_asset``) so the resulting bundle layout
matches every other LLM export.

Developer overrides (comma-separated flags): ``--dev-extend-qlens`` /
``--dev-prompt-qlens`` (per-function query-length ladder), ``--dev-ladder-only``
(restrict to specific context buckets). See ``DevOverrides``.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import torch
import yaml
from coreai.authoring import AIProgram
from coreai.authoring.types import AllocationType, HardwareConstraints
from coreai_opt.palettization.config.palettization_config import KMeansPalettizerConfig
from coreai_torch import TorchConverter
from transformers import AutoConfig, GenerationConfig

from coreai_models._constants import (
    CAUSAL_MASK_INPUT_NAME,
    DEFAULT_INCLUDE_DEBUG_INFO,
    EMBEDDING_TABLE_INPUT_NAME,
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
    IN_STEP_INPUT_NAME,
    KEY_CACHE_INPUT_NAME,
    KEY_CACHE_OUTPUT_NAME,
    LOAD_EMBEDDINGS_FUNCTION_NAME,
    PROMPT_OPT_FUNCTION_NAME,
    TOKEN_IDS_INPUT_NAME,
    TRANSFORMER_INPUT_NAME,
    VALUE_CACHE_INPUT_NAME,
    VALUE_CACHE_OUTPUT_NAME,
)
from coreai_models.export.bundle import bundle_llm_asset
from coreai_models.export.compression import (
    palettize_pytorch_model,
)
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.export.mlir_ops import (
    register_custom_torch_lowering,
    remove_functionalization,
)
from coreai_models.export.pipeline import ExportConfig, _generate_output_name
from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.models.ios.gemma4_text import (
    PLE_EMBEDDINGS_INPUT_NAME,
    QUERY_LENGTHS,
    ROPE_COS_INPUT_NAME,
    ROPE_SIN_INPUT_NAME,
    SLIDING_CAUSAL_MASK_INPUT_NAME,
    SLIDING_IN_STEP_INPUT_NAME,
    SLIDING_KEY_CACHE_INPUT_NAME,
    SLIDING_KEY_CACHE_OUTPUT_NAME,
    SLIDING_VALUE_CACHE_INPUT_NAME,
    SLIDING_VALUE_CACHE_OUTPUT_NAME,
    Gemma4ForCausalLMForiOS,
    sliding_ring_size,
)

# The iOS KV-cache interleave factor moved onto the iOS base class when the export
# contract was introduced; the Gemma4 ladder still constrains its caches by hand.
KV_CACHE_INTERLEAVE_FACTOR = BaseForCausalLMForiOS.KV_CACHE_INTERLEAVE_FACTOR

logger = logging.getLogger("gemma4.export")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IOS_COMPRESSION_CONFIG = (
    Path(__file__).resolve().parent / "gemma4_4bit_palettized.yaml"
)

# Checkpoints whose exports have been accuracy- and performance-verified. These
# are the ones the model card advertises as supported. Other Gemma 4 checkpoints
# are accepted with a warning -- they may well trace and export, but nothing about
# the resulting artifact has been verified.
SUPPORTED_MODELS = (
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
)

PRECISIONS = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Gemma 4 checkpoints are multimodal: the text decoder is nested under
# `text_config`.
HF_CONFIG_ATTR = "text_config"


# ===========================================================================
# Shared helpers
# ===========================================================================


def _resolve_eos_token_ids(hf_model_id: str, text_config: Any) -> list[int]:
    """Collect end-of-generation token ids from the generation config.

    The tokenizer exposes only a single ``eos_token`` (``<eos>``), but Gemma chat
    models also stop on ``<end_of_turn>``. Carry the full ``generation_config``
    eos list into the bundle metadata so the runner halts cleanly. Falls back to
    the model config's ``eos_token_id``. Always returns a de-duplicated list.
    """
    ids: list[int] = []

    def _add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for v in value:
                _add(v)
        elif isinstance(value, int) and value not in ids:
            ids.append(value)

    try:
        _add(GenerationConfig.from_pretrained(hf_model_id).eos_token_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not load generation config for eos tokens: {exc}")

    _add(getattr(text_config, "eos_token_id", None))
    return ids


def _ios_metadata_extras(text_config: Any) -> dict[str, Any]:
    """Extra ``language`` metadata keys the iOS runner needs.

    * ``sliding_window`` -- the runner builds the windowed mask for the sliding
      KV cache from it.
    * ``rope`` -- the iOS graph takes precomputed ``rope_cos``/``rope_sin``
      instead of ``position_ids`` (a 131k position overflows a 16-bit input and a
      32-bit input fails to compile), so the runner needs
      the two head dims, the two RoPE bases, and the global partial-rotary factor
      to build the combined table rows.
    * ``final_logit_softcapping`` -- ``tanh`` is best run on the CPU rather than in the
      graph, so the iOS graph omits the ``c * tanh(logits / c)`` cap and the runner
      applies it on the CPU between reading the logits and sampling.
    """
    extras: dict[str, Any] = {}

    sliding_window = getattr(text_config, "sliding_window", None)
    if isinstance(sliding_window, int) and sliding_window > 0:
        extras["sliding_window"] = sliding_window

    softcap = getattr(text_config, "final_logit_softcapping", None)
    if isinstance(softcap, (int, float)) and softcap > 0:
        extras["final_logit_softcapping"] = float(softcap)

    rope_parameters = getattr(text_config, "rope_parameters", None)
    global_head_dim = getattr(text_config, "global_head_dim", None)
    if (
        isinstance(rope_parameters, dict)
        and "sliding_attention" in rope_parameters
        and "full_attention" in rope_parameters
        and isinstance(global_head_dim, int)
    ):
        full = rope_parameters["full_attention"]
        extras["rope"] = {
            "sliding_head_dim": text_config.head_dim,
            "global_head_dim": global_head_dim,
            "sliding_rope_theta": rope_parameters["sliding_attention"]["rope_theta"],
            "global_rope_theta": full["rope_theta"],
            "partial_rotary_factor": full.get("partial_rotary_factor", 0.25),
        }
    return extras


def _patch_language_metadata(
    bundle_path: Path,
    hf_model_id: str,
    text_config: Any,
    extras: Optional[dict[str, Any]] = None,
    assets: Optional[dict[str, Any]] = None,
) -> None:
    """Add Gemma4-specific keys to a written bundle.

    ``bundle_llm_asset`` writes the generic 0.2-schema metadata; the keys added
    here are Gemma4-only, so they are merged in afterwards rather than
    special-cased inside the shared bundler. ``extras`` lands in the ``language``
    block; ``assets`` lands in the top-level role map, which is where sidecar
    artifacts belong so the runner resolves them through one generic path.
    """
    patch: dict[str, Any] = dict(extras or {})
    asset_patch: dict[str, Any] = dict(assets or {})

    eos_token_ids = _resolve_eos_token_ids(hf_model_id, text_config)
    if eos_token_ids:
        patch["eos_token_ids"] = eos_token_ids

    if not patch and not asset_patch:
        return

    metadata_path = bundle_path / "metadata.json"
    with metadata_path.open() as fh:
        metadata = json.load(fh)
    metadata["language"].update(patch)
    metadata.setdefault("assets", {}).update(asset_patch)
    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(f"Recorded {sorted(patch) + sorted(asset_patch)} in {metadata_path}")


def _text_config(hf_model_id: str) -> Any:
    """Load the Gemma 4 text-decoder sub-config."""
    raw_config = AutoConfig.from_pretrained(hf_model_id)
    return getattr(raw_config, HF_CONFIG_ATTR, raw_config)


def _resolve_bundle_paths(
    output_dir: str, output_name: str, overwrite: bool
) -> tuple[Path, Path]:
    """Create the bundle directory and clear a stale asset. Returns (bundle, asset)."""
    bundle_path = Path(output_dir) / output_name
    aimodel_path = bundle_path / f"{output_name}.aimodel"
    if aimodel_path.exists():
        if not overwrite:
            raise SystemExit(
                f"{aimodel_path} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(aimodel_path)
    bundle_path.mkdir(parents=True, exist_ok=True)
    return bundle_path, aimodel_path


# ===========================================================================
# Compression
# ===========================================================================


def _load_compression_yaml(yaml_path: Path):  # type: ignore[no-untyped-def]
    """Load a coreai-opt compression YAML and validate it.

    iOS compresses via k-means palettization, so the YAML's single top-level
    key must be ``kmeans_palettization_config``. Returned as a prebuilt
    ``KMeansPalettizerConfig``.

    A YAML for another mechanism is rejected rather than silently ignored.
    """
    with yaml_path.open() as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise SystemExit(f"{yaml_path}: expected a YAML mapping at top level.")

    pipeline_level_options = data.pop("coreai_models", {}) or {}
    if len(data) != 1:
        raise SystemExit(
            f"{yaml_path}: expected exactly one coreai-opt top-level key "
            f"('kmeans_palettization_config'), got {sorted(data)}."
        )
    top_key = next(iter(data))
    inner = data[top_key]

    if top_key == "kmeans_palettization_config":
        if pipeline_level_options:
            raise SystemExit(
                f"{yaml_path}: palettization recipes do not support the "
                "'coreai_models' block."
            )
        return KMeansPalettizerConfig.from_dict({top_key: inner})

    raise SystemExit(
        f"{yaml_path}: unknown top-level key '{top_key}'. "
        "Expected 'kmeans_palettization_config'."
    )


# ===========================================================================
# iOS: per-context blocked ladder of statically-shaped programs
# ===========================================================================

# Flash chunk width for the blocked global attention (not a cache dimension). The
# global cache stays FLAT (one dynamic write offset) and BlockedSDPA walks it in
# block_size chunks so each score op's key axis stays within the accelerator's
# per-dimension size limit.
DEFAULT_KV_BLOCK_SIZE = 8192

# Shipping per-function-type query lengths. Decode (``extend``) only needs q=8 (a
# single new token); prefill (``prompt_opt``) runs at q=64, and the runner routes a
# <=64-token initial prompt/tail through ``prompt_opt`` too, so an ``extend`` q=64
# variant is redundant today (and dropping it halves the extend function count,
# keeping the program under the accelerator's per-program I/O cap).
SHIPPING_EXTEND_QLENS = [8]
SHIPPING_PROMPT_QLENS = [64]

# Smallest context bucket. Short prompts pay only this context (a single ctx-wide
# flash chunk).
MIN_CONTEXT_LENGTH = 1024

# Shipping context ladder: a SPARSE set of buckets (<=4 up to 131072). A dense
# power-of-two ladder produces too many functions and blows past the accelerator's
# per-program I/O cap; this sparse ladder keeps every bucket (including 131072)
# accelerator-resident, at the cost of coarser decode-speed tiering.
SHIPPING_CONTEXT_LADDER = [1024, 8192, 32768, 131072]

# Largest context the iOS path supports. `context_ladder` rounds the requested
# length up to the next power of two and emits that as its own bucket, so asking
# for more than the top shipping bucket would synthesize an untested bucket (and
# push the program past the accelerator's per-program I/O cap). This is also the
# default, so an export covers the model's full context unless `--max-context-length`
# asks for less.
IOS_MAX_CONTEXT_LENGTH = SHIPPING_CONTEXT_LADDER[-1]

# Calibration trace query length (extend runs a single new token; q=8 covers it).
_CALIB_QUERY_LEN = 8


def _int_list(raw: str) -> tuple[int, ...]:
    """argparse type for a comma-separated positive-int list."""
    try:
        values = tuple(int(x) for x in raw.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {raw!r}"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"values must be positive, got {list(values)}")
    return values


@dataclass(frozen=True)
class DevOverrides:
    """Developer-only narrowing of what the ladder emits, from the ``--dev-*`` flags.

    A full export traces every context bucket at every query length, which is slow
    enough that debugging one rung is impractical. These cut the matrix down. A
    shipping export leaves all three unset, which is what :data:`SHIPPING_EXTEND_QLENS`,
    :data:`SHIPPING_PROMPT_QLENS` and :data:`SHIPPING_CONTEXT_LADDER` describe.
    """

    extend_qlens: Optional[tuple[int, ...]] = None
    prompt_qlens: Optional[tuple[int, ...]] = None
    ladder_only: Optional[tuple[int, ...]] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DevOverrides":
        return cls(
            extend_qlens=args.dev_extend_qlens,
            prompt_qlens=args.dev_prompt_qlens,
            ladder_only=args.dev_ladder_only,
        )

    def log_if_set(self) -> None:
        """Log any narrowing, so a debug artifact is never mistaken for a shipping one."""
        for label, value in (
            ("extend query lengths", self.extend_qlens),
            ("prompt query lengths", self.prompt_qlens),
            ("context buckets", self.ladder_only),
        ):
            if value:
                logger.warning(f"DEV OVERRIDE: {label} restricted to {list(value)}")


def _extend_query_lengths(dev: DevOverrides = DevOverrides()) -> list[int]:
    """Q-lengths to specialize the decode (``extend_*``) functions for.

    ``--dev-extend-qlens`` overrides; defaults to :data:`SHIPPING_EXTEND_QLENS`.
    """
    return list(dev.extend_qlens or SHIPPING_EXTEND_QLENS)


def _prompt_query_lengths(dev: DevOverrides = DevOverrides()) -> list[int]:
    """Q-lengths to specialize the prefill (``prompt_opt_*``) functions for.

    ``--dev-prompt-qlens`` overrides; defaults to :data:`SHIPPING_PROMPT_QLENS`.
    """
    return list(dev.prompt_qlens or SHIPPING_PROMPT_QLENS)


def context_ladder(
    max_context_length: int, block_size: int, dev: DevOverrides = DevOverrides()
) -> list[int]:
    """Context-length buckets for the flat-cache flash ladder: the sparse
    :data:`SHIPPING_CONTEXT_LADDER` entries that fit under the smallest power of two
    covering ``max_context_length`` (always including that cap).

    Each context size is its own statically-shaped program (the flash block loop is
    unrolled, so ``ceil(ctx / block_size)`` is baked into the graph).
    ``--dev-ladder-only`` emits only those buckets.
    """
    ctx_max = 1
    while ctx_max < max_context_length:
        ctx_max *= 2
    buckets = [b for b in SHIPPING_CONTEXT_LADDER if b < ctx_max]
    buckets.append(ctx_max)
    buckets = sorted(set(buckets))

    if dev.ladder_only:
        want = set(dev.ladder_only)
        buckets = [b for b in buckets if b in want] or [ctx_max]
    return buckets


def _head_dim(config) -> int:
    if hasattr(config, "head_dim") and isinstance(config.head_dim, int):
        return config.head_dim
    return config.hidden_size // config.num_attention_heads


def _ple_total_dim(config) -> int:
    """Width of one token's externalized Per-Layer Embeddings row.

    Every supported checkpoint (E2B, E4B) ships per-layer embeddings, so the
    ``ple_embeddings`` graph input is unconditional — see the export contract on
    :class:`Gemma4ForCausalLMForiOS`.
    """
    return config.num_hidden_layers * config.hidden_size_per_layer_input


def _ios_decomp_table():
    """iOS decomposition table: keep ``silu`` as-is (the accelerator has a fused op)."""
    decomp_table = torch.export.default_decompositions()
    decomp_table.pop(torch.ops.aten.silu.default)
    decomp_table.pop(torch.ops.aten.silu.out)
    return decomp_table


def _export_forward_pair(
    model: torch.nn.Module,
    forward_inputs: dict,
    forward_dynamic_shapes: dict,
    decomp_table,
) -> tuple:
    """Export the (extend decode, prompt prefill) pair for one set of inputs.

    Resets to decode mode before the extend export so this is safe to call in a
    loop (one pair per context bucket).
    """
    with torch.no_grad():
        model.set_prefill_mode(False)
        logger.info("Exporting extend module...")
        extend_program = torch.export.export(
            model.extend,
            args=(),
            kwargs=forward_inputs,
            dynamic_shapes=forward_dynamic_shapes,
        ).run_decompositions(decomp_table)
        remove_functionalization(extend_program)

        model.set_prefill_mode(True)
        logger.info("Exporting extend module (prefill mode)...")
        prompt_program = torch.export.export(
            model.extend,
            args=(),
            kwargs=forward_inputs,
            dynamic_shapes=forward_dynamic_shapes,
        ).run_decompositions(decomp_table)
        remove_functionalization(prompt_program)
    return extend_program, prompt_program


def _export_aux_programs(
    model: torch.nn.Module,
    embed_tokens_inputs: tuple,
    embed_tokens_dynamic_shapes: dict,
) -> tuple:
    """Export the (gather_embeddings, load_embeddings) programs (bucket-independent)."""
    with torch.no_grad():
        logger.info("Exporting gather_embeddings module...")
        gather_program = torch.export.export(
            model.gather_embeddings,
            args=embed_tokens_inputs,
            dynamic_shapes=embed_tokens_dynamic_shapes,
        )
        logger.info("Exporting load_embeddings module...")
        load_program = torch.export.export(model.load_embeddings, args=tuple())
    return gather_program, load_program


def _build_ios_reference_inputs(
    model: torch.nn.Module, config, max_context_length: int, vocab_size: int, ctx: int
) -> dict:
    """Build reference input tensors for one context bucket of the Gemma4 ladder.

    The global cache is FLAT ``(n_global_storing, 1, C_g, 1, ctx)`` (a single
    dynamic write offset); the sliding cache is a fixed-depth ring. RoPE arrives as
    precomputed ``rope_cos``/``rope_sin`` rows instead of ``position_ids``.
    """
    batch_size = 1
    query_len = 8

    input_ids = torch.randint(1, vocab_size, (batch_size, query_len), dtype=torch.int32)
    in_step = torch.zeros((1,), dtype=torch.int32)
    sliding_in_step = torch.zeros((1,), dtype=torch.int32)

    head_dim = _head_dim(config)
    global_head_dim = config.global_head_dim
    n_kv = config.num_key_value_heads
    n_global_storing = model.extend.model.n_global_storing
    n_sliding_storing = model.extend.model.n_sliding_storing
    sliding_ring = sliding_ring_size(config.sliding_window, max(QUERY_LENGTHS))

    key_cache = torch.zeros(
        n_global_storing, 1, n_kv * global_head_dim, 1, ctx, dtype=torch.float16
    )
    value_cache = key_cache.clone()
    sliding_key_cache = torch.zeros(
        n_sliding_storing, 1, n_kv * head_dim, 1, sliding_ring, dtype=torch.float16
    )
    sliding_value_cache = sliding_key_cache.clone()
    causal_mask = torch.zeros(1, ctx, 1, query_len, dtype=torch.float16)
    sliding_causal_mask = torch.zeros(
        1, sliding_ring, 1, query_len, dtype=torch.float16
    )
    rope_width = head_dim + global_head_dim
    rope_cos = torch.zeros(1, query_len, rope_width, dtype=torch.float16)
    rope_sin = torch.zeros(1, query_len, rope_width, dtype=torch.float16)

    embedding_table = model.load_embeddings.embedding_table
    transformer_input = model.gather_embeddings(input_ids, embedding_table)

    forward_inputs = {
        TRANSFORMER_INPUT_NAME: transformer_input,
        ROPE_COS_INPUT_NAME: rope_cos,
        ROPE_SIN_INPUT_NAME: rope_sin,
        IN_STEP_INPUT_NAME: in_step,
        SLIDING_IN_STEP_INPUT_NAME: sliding_in_step,
        CAUSAL_MASK_INPUT_NAME: causal_mask,
        SLIDING_CAUSAL_MASK_INPUT_NAME: sliding_causal_mask,
        KEY_CACHE_INPUT_NAME: key_cache,
        VALUE_CACHE_INPUT_NAME: value_cache,
        SLIDING_KEY_CACHE_INPUT_NAME: sliding_key_cache,
        SLIDING_VALUE_CACHE_INPUT_NAME: sliding_value_cache,
        EMBEDDING_TABLE_INPUT_NAME: embedding_table,
    }

    forward_inputs[PLE_EMBEDDINGS_INPUT_NAME] = torch.randint(
        -128, 127, (batch_size, query_len, 1, _ple_total_dim(config)), dtype=torch.int8
    )

    seq_len_dim = torch.export.Dim("seq_len", max=max_context_length)
    # ctx (global cache seq) and the sliding ring depth are STATIC per bucket; only
    # the query/seq dim is dynamic.
    forward_dynamic_shapes: dict = {
        TRANSFORMER_INPUT_NAME: {1: seq_len_dim},
        ROPE_COS_INPUT_NAME: {1: seq_len_dim},
        ROPE_SIN_INPUT_NAME: {1: seq_len_dim},
        IN_STEP_INPUT_NAME: None,
        SLIDING_IN_STEP_INPUT_NAME: None,
        CAUSAL_MASK_INPUT_NAME: {3: seq_len_dim},
        SLIDING_CAUSAL_MASK_INPUT_NAME: {3: seq_len_dim},
        KEY_CACHE_INPUT_NAME: None,
        VALUE_CACHE_INPUT_NAME: None,
        SLIDING_KEY_CACHE_INPUT_NAME: None,
        SLIDING_VALUE_CACHE_INPUT_NAME: None,
        EMBEDDING_TABLE_INPUT_NAME: None,
        PLE_EMBEDDINGS_INPUT_NAME: {1: seq_len_dim},
    }

    embed_tokens_inputs = (input_ids, embedding_table)
    embed_tokens_dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        EMBEDDING_TABLE_INPUT_NAME: None,
    }

    return {
        "forward_inputs": forward_inputs,
        "embed_tokens_inputs": embed_tokens_inputs,
        "forward_dynamic_shapes": forward_dynamic_shapes,
        "embed_tokens_dynamic_shapes": embed_tokens_dynamic_shapes,
    }


async def _convert_blocked_ladder_to_coreai(
    ladder: list,
    gather_embeddings_program,
    load_embeddings_program,
    kv_cached_embed_size: int,
    hidden_size: int,
    ple_total_dim: int,
    sliding_ring: int,
    n_sliding_storing: int,
    sliding_channels: int,
    n_global_storing: int,
    rope_width: int,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
    dev: DevOverrides = DevOverrides(),
) -> AIProgram:
    """Convert a flat-global-cache context ladder to a single multi-function AIProgram.

    ``ladder`` is a list of ``(ctx, extend_program, prompt_program)``, one per
    context bucket. Each is registered as its own entrypoint ``extend_{ctx}`` /
    ``prompt_opt_{ctx}`` with ``q_len`` as the static-shape specialization, so the
    emitted functions are ``extend_{ctx}_{q}`` / ``prompt_opt_{ctx}_{q}``. Shared
    weights are referenced (not copied) across programs; the runner right-sizes the
    global cache to the running bucket's ctx.
    """
    converter = TorchConverter(
        mode=TorchConverter.Mode.DEBUG
        if include_debug_info
        else TorchConverter.Mode.RELEASE
    )
    register_custom_torch_lowering(converter)

    # The graph I/O contract lives on the model class, so the recipe and the
    # eager model cannot drift apart. Each name hook is keyed by graph; the
    # ladder registers every bucket's pair of entrypoints against the single
    # EXTEND entry, since all rungs share one signature and differ only in shape.
    names = Gemma4ForCausalLMForiOS.export_input_names()
    states = Gemma4ForCausalLMForiOS.export_state_names()
    outputs = Gemma4ForCausalLMForiOS.export_output_names()

    converter.add_exported_program(
        load_embeddings_program,
        input_names=list(names[LOAD_EMBEDDINGS_FUNCTION_NAME]),
        output_names=list(outputs[LOAD_EMBEDDINGS_FUNCTION_NAME]),
        entrypoint_name=LOAD_EMBEDDINGS_FUNCTION_NAME,
    )
    converter.add_exported_program(
        gather_embeddings_program,
        input_names=list(names[GATHER_EMBEDDINGS_FUNCTION_NAME]),
        output_names=list(outputs[GATHER_EMBEDDINGS_FUNCTION_NAME]),
        entrypoint_name=GATHER_EMBEDDINGS_FUNCTION_NAME,
    )

    input_names = list(names[EXTEND_FUNCTION_NAME])
    state_names = list(states[EXTEND_FUNCTION_NAME])
    output_names = list(outputs[EXTEND_FUNCTION_NAME])

    entrypoints: list[tuple[str, int]] = []  # (entrypoint_name, ctx)
    for ctx, extend_program, prompt_program in ladder:
        extend_name = f"{EXTEND_FUNCTION_NAME}_{ctx}"
        prompt_name = f"{PROMPT_OPT_FUNCTION_NAME}_{ctx}"
        converter.add_exported_program(
            extend_program,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
            entrypoint_name=extend_name,
        )
        converter.add_exported_program(
            prompt_program,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
            entrypoint_name=prompt_name,
        )
        entrypoints.append((extend_name, ctx))
        entrypoints.append((prompt_name, ctx))

    coreai_program: AIProgram = converter.to_coreai()

    # ----- Static shape configs (query-length specialization within each bucket) -----
    extend_qlens = _extend_query_lengths(dev)
    prompt_qlens = _prompt_query_lengths(dev)
    gather_qlens = sorted(set(extend_qlens) | set(prompt_qlens))
    gather_static_cfg = {
        f'"{q_len}"': {TOKEN_IDS_INPUT_NAME: (1, q_len)} for q_len in gather_qlens
    }
    coreai_program.set_static_shape_config(
        GATHER_EMBEDDINGS_FUNCTION_NAME, gather_static_cfg
    )

    def _forward_static_cfg(ctx: int, q_lens: list[int]) -> dict:
        cfg_by_q: dict[str, dict[str, tuple[int, ...]]] = {}
        for q_len in q_lens:
            cfg = {
                TRANSFORMER_INPUT_NAME: (1, q_len, 1, hidden_size),
                ROPE_COS_INPUT_NAME: (1, q_len, rope_width),
                ROPE_SIN_INPUT_NAME: (1, q_len, rope_width),
                CAUSAL_MASK_INPUT_NAME: (1, ctx, 1, q_len),
                KEY_CACHE_INPUT_NAME: (
                    n_global_storing,
                    1,
                    kv_cached_embed_size,
                    1,
                    ctx,
                ),
                VALUE_CACHE_INPUT_NAME: (
                    n_global_storing,
                    1,
                    kv_cached_embed_size,
                    1,
                    ctx,
                ),
                SLIDING_CAUSAL_MASK_INPUT_NAME: (1, sliding_ring, 1, q_len),
                SLIDING_KEY_CACHE_INPUT_NAME: (
                    n_sliding_storing,
                    1,
                    sliding_channels,
                    1,
                    sliding_ring,
                ),
                SLIDING_VALUE_CACHE_INPUT_NAME: (
                    n_sliding_storing,
                    1,
                    sliding_channels,
                    1,
                    sliding_ring,
                ),
                PLE_EMBEDDINGS_INPUT_NAME: (1, q_len, 1, ple_total_dim),
            }
            cfg_by_q[f'"{q_len}"'] = cfg
        return cfg_by_q

    # ----- Hardware constraints (per-bucket ctx; channel interleave + seq-stride) -----
    emb_table_constraints = HardwareConstraints(
        AllocationType.IOSurface, interleave=[8, 1, 1], alignments=[1, 1, 1, 1]
    )
    sliding_cache_constraints = HardwareConstraints(
        AllocationType.IOSurface,
        interleave=[1, 1, KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
        alignments=[1, 1, 1, 1, KV_CACHE_INTERLEAVE_FACTOR * sliding_ring, 1],
    )

    def _forward_constraints(ctx: int) -> dict:
        # Per-bucket seq-stride alignment = THIS bucket's ctx (channel stride =
        # ctx*interleave). The runner right-sizes/grows the global KV cache to match
        # the running bucket's ctx exactly; the sliding ring is fixed-size.
        cache_constraints = HardwareConstraints(
            AllocationType.IOSurface,
            interleave=[1, 1, KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
            alignments=[1, 1, 1, 1, KV_CACHE_INTERLEAVE_FACTOR * ctx, 1],
        )
        return {
            EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints,
            KEY_CACHE_INPUT_NAME: cache_constraints,
            KEY_CACHE_OUTPUT_NAME: cache_constraints,
            VALUE_CACHE_INPUT_NAME: cache_constraints,
            VALUE_CACHE_OUTPUT_NAME: cache_constraints,
            SLIDING_KEY_CACHE_INPUT_NAME: sliding_cache_constraints,
            SLIDING_KEY_CACHE_OUTPUT_NAME: sliding_cache_constraints,
            SLIDING_VALUE_CACHE_INPUT_NAME: sliding_cache_constraints,
            SLIDING_VALUE_CACHE_OUTPUT_NAME: sliding_cache_constraints,
        }

    gather_constraints = {EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints}
    load_constraints = {EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints}

    logger.info("Applying optimization passes...")
    coreai_program.set_hardware_constraints(
        LOAD_EMBEDDINGS_FUNCTION_NAME, load_constraints
    )
    coreai_program.set_hardware_constraints(
        GATHER_EMBEDDINGS_FUNCTION_NAME, gather_constraints
    )
    for entrypoint_name, ctx in entrypoints:
        q_lens = (
            prompt_qlens
            if entrypoint_name.startswith(PROMPT_OPT_FUNCTION_NAME)
            else extend_qlens
        )
        coreai_program.set_static_shape_config(
            entrypoint_name, _forward_static_cfg(ctx, q_lens)
        )
        coreai_program.set_hardware_constraints(
            entrypoint_name, _forward_constraints(ctx)
        )
    coreai_program.optimize()

    return coreai_program


async def _export_blocked_ladder(
    model: torch.nn.Module,
    config,
    max_context_length: int,
    vocab_size: int,
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
    dev: DevOverrides = DevOverrides(),
) -> AIProgram:
    """Export the Gemma4 model as a per-context blocked-flash ladder AIProgram."""
    head_dim = _head_dim(config)
    global_head_dim = config.global_head_dim
    n_kv = config.num_key_value_heads

    block_size = getattr(model.extend.model, "kv_block_size", DEFAULT_KV_BLOCK_SIZE)
    buckets = context_ladder(max_context_length, block_size, dev)
    logger.info(
        f"iOS flat-cache context ladder: block_size={block_size} contexts={buckets}"
    )

    decomp_table = _ios_decomp_table()
    ladder: list = []
    gather_program = None
    load_program = None
    for ctx in buckets:
        logger.info(
            f"Exporting context bucket: ctx={ctx} "
            f"(flash chunks={(ctx + block_size - 1) // block_size})..."
        )
        inputs = _build_ios_reference_inputs(
            model, config, max_context_length, vocab_size, ctx
        )
        extend_program, prompt_program = _export_forward_pair(
            model,
            inputs["forward_inputs"],
            inputs["forward_dynamic_shapes"],
            decomp_table,
        )
        ladder.append((ctx, extend_program, prompt_program))
        if gather_program is None:
            gather_program, load_program = _export_aux_programs(
                model,
                inputs["embed_tokens_inputs"],
                inputs["embed_tokens_dynamic_shapes"],
            )

    return await _convert_blocked_ladder_to_coreai(
        ladder=ladder,
        gather_embeddings_program=gather_program,
        load_embeddings_program=load_program,
        kv_cached_embed_size=n_kv * global_head_dim,
        hidden_size=config.hidden_size,
        ple_total_dim=_ple_total_dim(config),
        sliding_ring=sliding_ring_size(config.sliding_window, max(QUERY_LENGTHS)),
        n_sliding_storing=model.extend.model.n_sliding_storing,
        sliding_channels=n_kv * head_dim,
        n_global_storing=model.extend.model.n_global_storing,
        rope_width=head_dim + global_head_dim,
        include_debug_info=include_debug_info,
        dev=dev,
    )


def _build_palettization_inputs(
    model: torch.nn.Module, config, max_context_length: int
) -> tuple:
    """Construct the forward-trace inputs the palettizer calibrates against.

    Mirrors the blocked-ladder graph contract; the global cache uses the smallest
    context bucket (calibration only needs representative activations for weight
    statistics).
    """
    batch_size = 1
    q = _CALIB_QUERY_LEN
    vocab_size = config.vocab_size

    head_dim = _head_dim(config)
    global_head_dim = config.global_head_dim
    n_kv = config.num_key_value_heads
    n_global_storing = model.extend.model.n_global_storing
    n_sliding_storing = model.extend.model.n_sliding_storing
    sliding_ring = sliding_ring_size(config.sliding_window, max(QUERY_LENGTHS))
    ctx = min(MIN_CONTEXT_LENGTH, max_context_length)

    input_ids = torch.randint(1, vocab_size, (batch_size, q), dtype=torch.int32)
    in_step = torch.zeros((1,), dtype=torch.int32)
    sliding_in_step = torch.zeros((1,), dtype=torch.int32)
    key_cache = torch.zeros(
        n_global_storing, 1, n_kv * global_head_dim, 1, ctx, dtype=torch.float16
    )
    value_cache = key_cache.clone()
    sliding_key_cache = torch.zeros(
        n_sliding_storing, 1, n_kv * head_dim, 1, sliding_ring, dtype=torch.float16
    )
    sliding_value_cache = sliding_key_cache.clone()
    causal_mask = torch.zeros(1, ctx, 1, q, dtype=torch.float16)
    sliding_causal_mask = torch.zeros(1, sliding_ring, 1, q, dtype=torch.float16)
    rope_width = head_dim + global_head_dim
    rope_cos = torch.zeros(1, q, rope_width, dtype=torch.float16)
    rope_sin = torch.zeros(1, q, rope_width, dtype=torch.float16)

    inputs: tuple = (
        input_ids,
        rope_cos,
        rope_sin,
        in_step,
        sliding_in_step,
        causal_mask,
        sliding_causal_mask,
        key_cache,
        value_cache,
        sliding_key_cache,
        sliding_value_cache,
    )
    ple_embeddings = torch.randint(
        -128, 127, (batch_size, q, 1, _ple_total_dim(config)), dtype=torch.int8
    )
    return (*inputs, ple_embeddings)


async def _export_ios(args: argparse.Namespace) -> str:
    hf_model_id: str = args.model
    target_dtype = PRECISIONS[args.compute_precision]
    max_ctx = args.max_context_length or IOS_MAX_CONTEXT_LENGTH
    dev = DevOverrides.from_args(args)
    dev.log_if_set()

    # Palettization comes from a coreai-opt YAML: DEFAULT_IOS_COMPRESSION_CONFIG
    # unless --compression-config overrides it. `--compression none` leaves it
    # unset. Resolved before any weights are loaded so a bad recipe fails fast.
    palettization_config = None
    if args.compression_config is not None:
        palettization_config = _load_compression_yaml(args.compression_config)
        compression = args.compression_config.stem
    else:
        compression = "none"

    output_name = args.output_name or _generate_output_name(
        ExportConfig(
            hf_model_id=hf_model_id,
            variant="iOS",
            compression=compression,
            compression_config_object=palettization_config,
        )
    )
    bundle_path, aimodel_path = _resolve_bundle_paths(
        args.output_dir, output_name, args.overwrite
    )

    logger.info(
        f"Loading {hf_model_id} (iOS, dtype={target_dtype}, max_ctx={max_ctx})..."
    )

    # Move loaded weights to disk-backed mmap tensors so the OS can evict weight
    # pages during palettization and the long blocked-ladder conversion. The temp
    # dir must outlive every read of the weights, so it wraps the whole model
    # lifetime (through ``del model``); it is cleaned up on scope exit.
    with tempfile.TemporaryDirectory(prefix="gemma4_export_") as temp_dir:
        hf_config = _text_config(hf_model_id)

        # Same guard the macOS path applies: the ladder cap is checked in
        # `_resolve_platform_defaults` (it is a static property of the shipping
        # ladder), but the model's own window is only known once its config is
        # loaded.
        native_max_ctx = getattr(hf_config, "max_position_embeddings", None)
        if native_max_ctx is not None and max_ctx > native_max_ctx:
            raise SystemExit(
                f"--max-context-length ({max_ctx}) exceeds the model's "
                f"max_position_embeddings ({native_max_ctx}). "
                f"Choose a value <= {native_max_ctx}."
            )

        model = Gemma4ForCausalLMForiOS.from_hf(
            hf_model_id,
            max_context_length=max_ctx,
            target_dtype=target_dtype,
            mmap_path=os.path.join(temp_dir, "weights"),
        ).eval()
        hf_config.max_position_embeddings = max_ctx

        if not (hasattr(model, "extend") and hasattr(model.extend, "sliding_cache")):
            raise SystemExit(
                f"'{hf_model_id}' is not a Gemma4 sliding-cache model — use "
                "`coreai.llm.export` for other models."
            )

        # ---- Palettization (skipped for --compression none) ----
        if palettization_config is not None:
            logger.info(f"Applying palettization ({compression})...")
            inputs = _build_palettization_inputs(model, hf_config, max_ctx)
            model = palettize_pytorch_model(model, inputs, palettization_config)

        # ---- Blocked-ladder export ----
        coreai_program = await _export_blocked_ladder(
            model,
            hf_config,
            max_ctx,
            hf_config.vocab_size,
            include_debug_info=args.include_debug_info,
            dev=dev,
        )

        # ---- PLE sidecar (while the model is still in memory) ----
        logger.info("Dumping Per-Layer Embeddings (PLE) artifact...")
        ple_path = model.dump_ple_embedding(str(bundle_path), output_name)
        logger.info(f"Wrote PLE artifact to {ple_path}")

        del model

        logger.info(f"Saving model to {aimodel_path}...")
        await asyncio.to_thread(
            coreai_program.save_asset, aimodel_path, build_aimodel_metadata(hf_model_id)
        )

        bundle_llm_asset(
            bundle_path=bundle_path,
            hf_model_id=hf_model_id,
            hf_config=hf_config,
            compression=compression,
            name=output_name,
        )
        # The iOS runner needs the sliding window and the dual-RoPE table
        # parameters from `language`; the PLE sidecar is declared in the generic
        # `assets` role map, which is the single path the runner resolves through.
        extras = _ios_metadata_extras(hf_config)
        assets = {"per_layer_embeddings": Path(ple_path).name}
        _patch_language_metadata(
            bundle_path, hf_model_id, hf_config, extras, assets=assets
        )

    logger.info(f"Export complete: {bundle_path}")
    return str(bundle_path)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Gemma 4 text decoder to a Core AI bundle (iOS).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID. Verified on: " + ", ".join(SUPPORTED_MODELS),
    )
    parser.add_argument(
        "--platform",
        choices=["iOS"],
        default="iOS",
        help="Target platform. iOS only on this branch; the flag is kept so "
        "invocations that name the platform explicitly keep working.",
    )
    compression_group = parser.add_mutually_exclusive_group()
    compression_group.add_argument(
        "--compression-config",
        type=Path,
        default=None,
        help="coreai-opt compression YAML: a 'kmeans_palettization_config' recipe "
        f"(default: {DEFAULT_IOS_COMPRESSION_CONFIG.name}, alongside this script)",
    )
    compression_group.add_argument(
        "--compression",
        choices=["none"],
        default=None,
        help="Only 'none' is accepted, which exports at full precision. The named "
        "presets are not used for Gemma 4: the shipped YAML recipes are "
        "mixed-precision, which the presets cannot express. Pass "
        "--compression-config <yaml> for a custom recipe.",
    )
    parser.add_argument(
        "--compute-precision",
        choices=sorted(PRECISIONS),
        default=None,
        help="Compute precision for the model weights (iOS requires float16)",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help=f"Maximum context length (default and max: {IOS_MAX_CONTEXT_LENGTH})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "exports"),
        help="Output directory for the bundle (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Custom bundle name (without extension)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing bundle"
    )
    parser.add_argument(
        "--include-debug-info",
        action="store_true",
        help="Embed debug information in the exported .aimodel for debugging a conversion. "
        "Default: off, which embeds minimum debug information and makes the exported "
        "asset smaller.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logging"
    )

    # Developer-only narrowing. A full ladder traces every context bucket at every
    # query length; these cut that down so one rung can be debugged in minutes.
    # Shipping exports pass none of them -- see `DevOverrides`.
    dev_group = parser.add_argument_group("developer overrides")
    dev_group.add_argument(
        "--dev-extend-qlens",
        type=_int_list,
        metavar="N[,N...]",
        help="Query lengths to specialize the decode (extend_*) functions for, "
        f"instead of the shipping {list(SHIPPING_EXTEND_QLENS)}.",
    )
    dev_group.add_argument(
        "--dev-prompt-qlens",
        type=_int_list,
        metavar="N[,N...]",
        help="Query lengths to specialize the prefill (prompt_opt_*) functions for, "
        f"instead of the shipping {list(SHIPPING_PROMPT_QLENS)}.",
    )
    dev_group.add_argument(
        "--dev-ladder-only",
        type=_int_list,
        metavar="CTX[,CTX...]",
        help="Emit only these context buckets. Buckets outside the ladder for the "
        "requested --max-context-length are ignored; if none remain, the cap alone "
        "is emitted.",
    )
    return parser


def _resolve_platform_defaults(args: argparse.Namespace) -> None:
    """Fill in the defaults and reject invalid combinations.

    Compression defaults to the ``kmeans_palettization_config`` YAML shipped
    alongside this script. ``--compression-config <yaml>`` swaps in a custom
    recipe. ``--compression`` only accepts ``none``, which skips compression
    entirely.
    """
    if args.compute_precision is None:
        args.compute_precision = "float16"

    if args.compute_precision != "float16":
        raise SystemExit(
            f"--platform iOS requires --compute-precision float16 "
            f"(got '{args.compute_precision}')."
        )
    if (
        args.max_context_length is not None
        and args.max_context_length > IOS_MAX_CONTEXT_LENGTH
    ):
        raise SystemExit(
            f"--platform iOS supports at most {IOS_MAX_CONTEXT_LENGTH} tokens "
            f"(got {args.max_context_length}); the static-shape ladder tops out "
            "at that bucket."
        )

    if args.compression is None and args.compression_config is None:
        args.compression_config = DEFAULT_IOS_COMPRESSION_CONFIG

    if args.compression_config is not None and not args.compression_config.is_file():
        raise SystemExit(
            f"--compression-config: file not found: {args.compression_config}"
        )


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    _resolve_platform_defaults(args)

    if args.model not in SUPPORTED_MODELS:
        logger.warning(
            f"{args.model} is not one of the verified checkpoints "
            f"({', '.join(SUPPORTED_MODELS)}). Exporting anyway, but the accuracy and "
            "performance of the result have not been verified."
        )

    print(f"Export complete: {asyncio.run(_export_ios(args))}")


if __name__ == "__main__":
    if sys.version_info < (3, 11):
        raise SystemExit("Gemma 4 export requires Python 3.11+")
    main()
