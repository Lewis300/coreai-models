# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The Gemma 4 export recipe, run end to end through Core AI conversion.

The recipe is co-located at ``models/gemma4/export.py`` (not part of the
installed package), so it is loaded by path. A tiny synthetic Gemma 4 is
exported as a one-bucket ladder, converted, saved, and loaded back, and the
emitted functions are checked against the names the runner looks up.
"""

import asyncio
import importlib.util
import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from coreai_models._constants import (  # noqa: E402
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
)
from coreai_models.models.ios.gemma4_text import Gemma4ForCausalLMForiOS  # noqa: E402
from tests._runner_infra._deps import _HAS_COREAI, _MSG_COREAI_NOT_FOUND  # noqa: E402
from tests.test_model_units.test_models.test_ios_layers.test_gemma4 import (  # noqa: E402
    Gemma4ForCausalLM,
    _build_ios_model,
    _make_config,
)

pytestmark = [
    pytest.mark.skipif(Gemma4ForCausalLM is None, reason="gemma4 requires transformers>=5.5"),
    pytest.mark.skipif(not _HAS_COREAI, reason=_MSG_COREAI_NOT_FOUND),
]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_spec = importlib.util.spec_from_file_location(
    "gemma4_export", _REPO_ROOT / "models/gemma4/export.py"
)
export_g4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_g4)


def test_ladder_converts_and_loads():
    """Every ladder function converts, and loads under the name and I/O the runner uses."""
    torch.manual_seed(0)
    cfg = _make_config()
    cfg.sliding_window = Gemma4ForCausalLMForiOS.SLIDING_WINDOW
    model = _build_ios_model(cfg, dict(Gemma4ForCausalLM(cfg).state_dict())).half()
    ctx = 1024

    program = asyncio.run(export_g4._export_blocked_ladder(model, model.config, ctx))

    inputs = Gemma4ForCausalLMForiOS.export_input_names()
    states = Gemma4ForCausalLMForiOS.export_state_names()
    expected = {
        "load_embeddings": ((), ()),
        **{
            f"gather_embeddings_{q}": (inputs[GATHER_EMBEDDINGS_FUNCTION_NAME], ()) for q in (8, 64)
        },
        f"extend_{ctx}_8": (inputs[EXTEND_FUNCTION_NAME], states[EXTEND_FUNCTION_NAME]),
        f"prompt_opt_{ctx}_64": (inputs[EXTEND_FUNCTION_NAME], states[EXTEND_FUNCTION_NAME]),
    }

    async def load() -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            asset = program.save_asset(Path(tmpdir) / "gemma4.aimodel")
            async with asset.executable() as aimodel:
                loaded = {}
                for name in aimodel.function_names:
                    desc = aimodel.load_function(name).desc
                    loaded[name] = (tuple(desc.input_names), tuple(desc.state_names))
                return loaded

    assert asyncio.run(load()) == expected
