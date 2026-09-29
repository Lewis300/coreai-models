# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for iOS Gemma4 two-cache sliding-window attention parity with HuggingFace.

The iOS Gemma4 decoder uses two compacted KV caches — a full-context global
cache and a small sliding-window ring (depth S) — and applies windowed attention
via a runner-built ``sliding_causal_mask``. These tests mimic the Swift runner's
chunked prefill against persistent caches and compare per-position logits to a
single HF forward (which applies sliding-window attention internally). The key
case is a prompt longer than both W and S so the ring wraps.
"""

import pytest
import torch

pytest.importorskip("transformers")

try:  # transformers>=5.5 (gemma4)
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4 import Gemma4ForCausalLM
except Exception:  # pragma: no cover - requires transformers>=5.5
    Gemma4TextConfig = None
    Gemma4ForCausalLM = None

from coreai_models.models.ios.gemma4_text import (  # noqa: E402
    Gemma4ForCausalLMForiOS,
    _compute_kv_layout,
    sliding_ring_size,
)
from coreai_models.primitives.ios.rope import RoPECache  # noqa: E402
from tests._runner_infra._deps import _hf_hub_reachable  # noqa: E402

DTYPE = torch.float32
NEG = float("-inf")

pytestmark = pytest.mark.skipif(
    Gemma4ForCausalLM is None, reason="gemma4 requires transformers>=5.5"
)


# Numerical reference for the combined sliding+global RoPE rows the runner builds.
class Gemma4CombinedRoPE(RoPECache):
    """Single RoPE cache for Gemma4's dual head dims.

    Gemma4 has two RoPE variants — standard for sliding-attention layers (head_dim 256)
    and proportional (0.25 rotary) for global layers (head_dim 512). Both variants' cos/sin
    tables are concatenated along the head dim into a single
    ``[max_pos, sliding_hd + global_hd]`` cache and gathered once. Callers slice
    ``[:sliding_hd]`` for sliding layers and ``[sliding_hd:]`` for global layers.
    """

    def __init__(
        self,
        sliding_head_dim: int,
        global_head_dim: int,
        max_cache_size: int,
        sliding_base: float,
        global_base: float,
        partial_rotary_factor: float = 0.25,
    ) -> None:
        self._sliding_head_dim = sliding_head_dim
        self._global_head_dim = global_head_dim
        self._sliding_base = sliding_base
        self._global_base = global_base
        self._partial_rotary_factor = partial_rotary_factor
        super().__init__(sliding_head_dim + global_head_dim, max_cache_size, sliding_base)

    @staticmethod
    def _emb(theta: torch.Tensor, max_cache_size: int) -> torch.Tensor:
        seq_idx = torch.arange(end=max_cache_size, dtype=torch.int32)
        freqs = seq_idx[:, None] * theta
        return torch.concatenate((freqs, freqs), dim=-1)

    def _compute_sin_and_cos(self, dtype: torch.dtype = torch.float32) -> None:
        with torch.device("cpu"):
            # Sliding (standard RoPE).
            s_theta = 1.0 / (
                self._sliding_base
                ** (
                    torch.arange(0, self._sliding_head_dim, 2, dtype=torch.float32)
                    / self._sliding_head_dim
                )
            )
            s_emb = self._emb(s_theta, self._max_cache_size)

            # Global (proportional RoPE: only partial_rotary_factor of dims rotate).
            hd = self._global_head_dim
            rope_angles = int(self._partial_rotary_factor * hd // 2)
            nope_angles = hd // 2 - rope_angles
            inv_freq = 1.0 / (
                self._global_base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / hd)
            )
            if nope_angles > 0:
                g_theta = torch.cat(
                    [inv_freq, torch.zeros(nope_angles, dtype=torch.float32)], dim=0
                )
            else:
                g_theta = inv_freq
            g_emb = self._emb(g_theta, self._max_cache_size)

            cos = torch.cat([torch.cos(s_emb), torch.cos(g_emb)], dim=-1)
            sin = torch.cat([torch.sin(s_emb), torch.sin(g_emb)], dim=-1)
            self.cos_cached = torch.nn.Buffer(cos.to(dtype=dtype), persistent=False)
            self.sin_cached = torch.nn.Buffer(sin.to(dtype=dtype), persistent=False)


def _make_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        hidden_size_per_layer_input=16,
        num_kv_shared_layers=2,
        sliding_window=8,
        max_position_embeddings=128,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        rms_norm_eps=1e-6,
        use_double_wide_mlp=False,
        tie_word_embeddings=True,
    )


def _build_ios_model(cfg, hf_sd):
    sd = dict(hf_sd)
    model = Gemma4ForCausalLMForiOS(cfg, model_device="cpu", disable_embedding_quantization=True)
    model.to(DTYPE).eval()
    model._mutate_state_dict(sd)
    model.load_state_dict(sd, assign=True, strict=True)
    return model


def _ple_input_fp(hf_model, cfg, token_ids):
    # Build the fp ``ple_embeddings`` graph input the way export quantizes it
    # (ple_weight[token] * sqrt(ple_dim)), but kept fp to avoid INT8 noise.
    w = hf_model.state_dict()["model.embed_tokens_per_layer.weight"]
    ple_dim = cfg.hidden_size_per_layer_input
    total = cfg.num_hidden_layers * ple_dim
    rows = w[:, :total][token_ids].to(DTYPE) * (float(ple_dim) ** 0.5)
    return rows.reshape(1, len(token_ids), 1, total)


def _global_mask(ctx, q_len, aligned_step):
    m = torch.full((1, ctx, 1, q_len), NEG, dtype=DTYPE)
    for i in range(q_len):
        m[0, : aligned_step + i + 1, 0, i] = 0.0
    return m


def _sliding_mask(S, q_len, aligned_step, window):
    m = torch.full((1, S, 1, q_len), NEG, dtype=DTYPE)
    for i in range(q_len):
        p = aligned_step + i
        for pos in range(max(0, p - window + 1), p + 1):
            m[0, pos % S, 0, i] = 0.0
    return m


def _chunked_prefill_logits(ios, hf, cfg, token_ids, q_len, S, ctx):
    n_kv = cfg.num_key_value_heads
    sliding_storing, global_storing, _ = _compute_kv_layout(cfg)
    seq = len(token_ids)
    rope = _combined_rope(cfg, ctx, DTYPE)

    key_cache = torch.zeros(len(global_storing), 1, n_kv * cfg.global_head_dim, 1, ctx, dtype=DTYPE)
    value_cache = key_cache.clone()
    skey_cache = torch.zeros(len(sliding_storing), 1, n_kv * cfg.head_dim, 1, S, dtype=DTYPE)
    svalue_cache = skey_cache.clone()

    out_logits = torch.zeros(seq, cfg.vocab_size, dtype=DTYPE)
    for start in range(0, seq, q_len):
        chunk = token_ids[start : start + q_len]
        ids = chunk.reshape(1, q_len)
        pos = torch.arange(start, start + q_len, dtype=torch.int32).reshape(1, q_len)
        rope_cos, rope_sin = rope.gather_cos_sin(pos)
        in_step = torch.tensor([start], dtype=torch.int32)
        sliding_in_step = torch.tensor([start % S], dtype=torch.int32)
        with torch.no_grad():
            out = ios(
                ids,
                rope_cos,
                rope_sin,
                in_step,
                sliding_in_step,
                _global_mask(ctx, q_len, start),
                _sliding_mask(S, q_len, start, cfg.sliding_window),
                key_cache,
                value_cache,
                skey_cache,
                svalue_cache,
                _ple_input_fp(hf, cfg, chunk),
            )
        out_logits[start : start + q_len] = out.reshape(q_len, cfg.vocab_size)
    return out_logits


def test_kv_layout_dead_slot_compaction():
    """Layout keeps only storing layers; shared layers reuse the source slot."""
    cfg = _make_config()
    sliding_storing, global_storing, layout = _compute_kv_layout(cfg)
    assert sliding_storing == [0, 1, 3]
    assert global_storing == [2]
    # Shared sliding layer 4 -> source 3 -> sliding slot 2; shared global 5 -> global slot 0.
    assert layout[4] == (True, True, 2)
    assert layout[5] == (False, True, 0)


def test_sliding_parity_with_ring_wrap():
    """Chunked prefill over a prompt longer than both W and S matches HF."""
    torch.manual_seed(0)
    cfg = _make_config()
    hf = Gemma4ForCausalLM(cfg).to(DTYPE).eval()
    ios = _build_ios_model(cfg, dict(hf.state_dict()))

    q_len = 4
    S = sliding_ring_size(cfg.sliding_window, q_len)  # 12
    ctx = 32
    seq = 24  # > S (ring wraps) and > W (windowing active)
    token_ids = torch.randint(0, cfg.vocab_size, (seq,))

    with torch.no_grad():
        hf_logits = hf(
            input_ids=token_ids.reshape(1, seq),
            position_ids=torch.arange(seq).reshape(1, seq),
        ).logits[0]

    ios_logits = _chunked_prefill_logits(ios, hf, cfg, token_ids, q_len, S, ctx)

    assert (ios_logits.argmax(-1) == hf_logits.argmax(-1)).all()
    torch.testing.assert_close(ios_logits, hf_logits, atol=1e-3, rtol=1e-3)


def test_final_logit_softcap_left_to_runner():
    """The iOS forward emits *uncapped* logits; the runner applies the cap on CPU.

    ``tanh`` is best run on the CPU rather than in the graph, so the iOS decoder no longer applies
    ``c·tanh(logits/c)`` (see ``models/ios/gemma4_text.py``); the Swift runner does it
    instead, between reading ``out_logits`` and sampling (``LogitSoftcap``). This pins
    both halves of that split: our forward reproduces HF's *pre-cap* logits, and capping
    our output afterwards reproduces HF's capped logits.
    """
    torch.manual_seed(2)
    cfg = _make_config()
    # The synthetic model's logits are all |x| < 0.5, so the released cap (30.0) would be
    # indistinguishable from the identity here. Pick a cap small enough that ``tanh``
    # actually bends this data — the guard below fails if it doesn't.
    cap = 0.1
    cfg.final_logit_softcapping = cap
    hf = Gemma4ForCausalLM(cfg).to(DTYPE).eval()
    ios = _build_ios_model(cfg, dict(hf.state_dict()))

    q_len = 4
    S = sliding_ring_size(cfg.sliding_window, q_len)
    ctx = 32
    seq = 12
    token_ids = torch.randint(0, cfg.vocab_size, (seq,))

    with torch.no_grad():
        hf_logits = hf(
            input_ids=token_ids.reshape(1, seq),
            position_ids=torch.arange(seq).reshape(1, seq),
        ).logits[0]

    ios_logits = _chunked_prefill_logits(ios, hf, cfg, token_ids, q_len, S, ctx)

    # Guard: the cap must actually bite on this data, or the assertions below are vacuous.
    assert not torch.allclose(ios_logits, hf_logits, atol=1e-2), (
        "cap had no measurable effect — pick a smaller cap or different seed"
    )

    # Our forward leaves the logits uncapped: applying the cap ourselves lands on HF.
    # (Had the forward still capped, this would be a double cap and would not match.)
    torch.testing.assert_close(torch.tanh(ios_logits / cap) * cap, hf_logits, atol=1e-3, rtol=1e-3)

    # The cap is monotonic, so it cannot move an argmax — which is why greedy sampling
    # is unaffected by *where* it runs, and only the logit values need the runner's pass.
    assert (ios_logits.argmax(-1) == hf_logits.argmax(-1)).all()


# ---------------------------------------------------------------------------
# Full E2B end-to-end: real weights, HF reference vs our iOS torch model.
#
# The tests above run the iOS forward against a tiny synthetic config. This one
# loads the *real* ``google/gemma-4-E2B-it`` through ``from_hf`` (with fp
# embeddings, so the compare isn't muddied by INT8 embedding noise) and mimics the
# Swift runner's chunked prefill — precomputed dual RoPE rows, the flat global
# cache + sliding ring, and the fp PLE sidecar — then compares per-position logits
# to a single HuggingFace forward. It is heavy (loads a ~2B model twice) and needs
# network + weights + transformers>=5.5, so it is opt-in via ``-m slow``.
# ---------------------------------------------------------------------------

E2B_MODEL_ID = "google/gemma-4-E2B-it"

# Decode (``extend``) query width the export ladder specializes for; also sizes
# the sliding ring. Matches ``models/gemma4/export.py``.
_E2B_MAX_QUERY_LEN = 64


def _combined_rope(cfg, max_ctx: int, dtype: torch.dtype) -> Gemma4CombinedRoPE:
    """The dual (sliding + global) RoPE table the runner precomputes and feeds in
    as ``rope_cos``/``rope_sin`` rows."""
    return Gemma4CombinedRoPE(
        sliding_head_dim=cfg.head_dim,
        global_head_dim=cfg.global_head_dim,
        max_cache_size=max_ctx,
        sliding_base=cfg.rope_parameters["sliding_attention"]["rope_theta"],
        global_base=cfg.rope_parameters["full_attention"]["rope_theta"],
        partial_rotary_factor=cfg.rope_parameters["full_attention"].get(
            "partial_rotary_factor", 0.25
        ),
    ).to(dtype)


def _e2b_prefill_logits(model, rope, cfg, token_ids, q_len, S, ctx, dtype):
    """Chunked prefill of the real iOS model against persistent caches.

    Returns per-position logits ``(seq, vocab)`` in fp32. Mirrors the Swift runner:
    flat global cache with a single
    absolute write offset, a fixed-depth sliding ring, and the fp PLE input built as
    ``ple_weight[token] * sqrt(ple_dim)``.
    """
    n_kv = cfg.num_key_value_heads
    n_g = model.extend.model.n_global_storing
    n_s = model.extend.model.n_sliding_storing
    ple_dim = cfg.hidden_size_per_layer_input
    ple_total = cfg.num_hidden_layers * ple_dim
    ple_scale = float(ple_dim) ** 0.5
    window = cfg.sliding_window
    seq = len(token_ids)

    key_cache = torch.zeros(n_g, 1, n_kv * cfg.global_head_dim, 1, ctx, dtype=dtype)
    value_cache = key_cache.clone()
    skey_cache = torch.zeros(n_s, 1, n_kv * cfg.head_dim, 1, S, dtype=dtype)
    svalue_cache = skey_cache.clone()

    out_logits = torch.zeros(seq, cfg.vocab_size, dtype=torch.float32)
    for start in range(0, seq, q_len):
        chunk = token_ids[start : start + q_len]
        n_real = len(chunk)
        if n_real < q_len:  # pad the final partial chunk; padded cols are never read
            chunk = chunk + [0] * (q_len - n_real)
        ids = torch.tensor(chunk, dtype=torch.int32).reshape(1, q_len)
        pos = torch.arange(start, start + q_len, dtype=torch.int32).reshape(1, q_len)
        rope_cos, rope_sin = rope.gather_cos_sin(pos)
        in_step = torch.tensor([start], dtype=torch.int32)
        sliding_in_step = torch.tensor([start % S], dtype=torch.int32)
        ple_rows = model._ple_weight[torch.tensor(chunk)].to(dtype) * ple_scale
        ple = ple_rows.reshape(1, q_len, 1, ple_total)
        with torch.no_grad():
            out = model(
                ids,
                rope_cos,
                rope_sin,
                in_step,
                sliding_in_step,
                _global_mask(ctx, q_len, start).to(dtype),
                _sliding_mask(S, q_len, start, window).to(dtype),
                key_cache,
                value_cache,
                skey_cache,
                svalue_cache,
                ple,
            )
        chunk_logits = out.reshape(q_len, cfg.vocab_size).float()
        out_logits[start : start + n_real] = chunk_logits[:n_real]
    return out_logits


@pytest.mark.slow
@pytest.mark.flaky(reruns=0)
def test_e2b_full_parity_hf_vs_torch():
    """Real E2B: our iOS torch model's chunked-prefill logits match a HF forward."""
    if Gemma4ForCausalLM is None:
        pytest.skip("gemma4 requires transformers>=5.5")
    if not _hf_hub_reachable(E2B_MODEL_ID):
        pytest.skip(f"HuggingFace Hub unreachable for {E2B_MODEL_ID!r}")
    try:
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
        )
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"transformers gemma4 symbols unavailable: {exc}")

    dtype = torch.float32  # tightest parity; the shipped path is fp16 + palettized
    q_len = 8
    max_ctx = 1024  # smallest shipping bucket; one flash block (< block_size)

    # A short, unambiguous prompt keeps the HF full-attention reference cheap and
    # the greedy next token deterministic. No chat template so HF and iOS see
    # identical ids.
    try:
        tok = AutoTokenizer.from_pretrained(E2B_MODEL_ID)
        model = Gemma4ForCausalLMForiOS.from_hf(
            E2B_MODEL_ID,
            max_context_length=max_ctx,
            target_dtype=dtype,
            disable_embedding_quantization=True,  # fp embeddings -> clean HF parity
        ).eval()
    except OSError as exc:  # pragma: no cover - gated/download failures
        pytest.skip(f"could not load E2B weights: {exc}")

    cfg = model.config
    token_ids = tok("The capital of France is", return_tensors="pt")["input_ids"][0].tolist()
    assert len(token_ids) <= max_ctx

    S = sliding_ring_size(cfg.sliding_window, _E2B_MAX_QUERY_LEN)
    rope = _combined_rope(cfg, max_ctx, dtype)
    ios_logits = _e2b_prefill_logits(model, rope, cfg, token_ids, q_len, S, max_ctx, dtype)
    del model  # free the iOS model before loading the HF reference (peak = one model)

    # Match HF's math to what the iOS model actually computes. The iOS decoder applies
    # neither the attention logit cap nor ``final_logit_softcapping`` (the latter moved to
    # the Swift runner, since ``tanh`` is best run on the CPU rather than in the graph),
    # so: disable the
    # attention cap on the HF reference, and stand in for the runner by applying the final
    # cap to *our* logits below. Without both, no sane atol/rtol holds. Nulling via the
    # config before construction ensures it sticks even if a module caches the value at
    # init. (The tiny-config tests leave the attention cap unset in their synthetic config.)
    hf_cfg = AutoConfig.from_pretrained(E2B_MODEL_ID)
    for _c in (hf_cfg, getattr(hf_cfg, "text_config", None)):
        if _c is None:
            continue
        for _attr in ("attention_logit_cap", "attn_logit_softcapping"):
            if getattr(_c, _attr, None) is not None:
                setattr(_c, _attr, None)

    # The cap the Swift runner reads out of bundle metadata and applies on the CPU.
    final_cap = getattr(cfg, "final_logit_softcapping", None)
    if final_cap:
        ios_logits = torch.tanh(ios_logits / final_cap) * final_cap

    hf = Gemma4ForConditionalGeneration.from_pretrained(
        E2B_MODEL_ID, config=hf_cfg, torch_dtype=dtype
    ).eval()
    with torch.no_grad():
        hf_logits = (
            hf(
                input_ids=torch.tensor(token_ids).reshape(1, -1),
                position_ids=torch.arange(len(token_ids)).reshape(1, -1),
            )
            .logits[0]
            .float()
        )
    del hf

    assert ios_logits.shape == hf_logits.shape
    max_abs = (ios_logits - hf_logits).abs().max().item()
    agreement = (ios_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()

    # The decode-relevant token (greedy next token) must match exactly.
    assert ios_logits[-1].argmax() == hf_logits[-1].argmax(), (
        f"last-token argmax diverged (max_abs={max_abs:.3f}, agreement={agreement:.3f})"
    )
    # Per-position top-1 agreement across the whole prompt.
    assert agreement >= 0.9, (
        f"per-position top-1 agreement {agreement:.3f} too low (max_abs={max_abs:.3f})"
    )
    # Numerical closeness, with the attention cap disabled on HF above and fp
    # embeddings (``disable_embedding_quantization``), so the only spread vs HF is fp32
    # op ordering through the deep stack + the flash online-softmax recurrence. Looser
    # than the tiny-config's 1e-3 because a real 2B model accumulates more.
    torch.testing.assert_close(ios_logits, hf_logits, atol=1e-2, rtol=1e-2)
