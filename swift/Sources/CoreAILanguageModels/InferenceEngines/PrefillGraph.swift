// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

/// Name of the optional prefill entrypoint. Exported beside `main` (see
/// `export/macos.py`) with the same inputs and states, but no LM head and no outputs:
/// it only fills the KV cache.
let prefillGraphFunctionName = "prefill"

/// Load the prefill graph, or nil if the asset has none.
///
/// It must take the same inputs and states as `main` and declare no outputs, because that
/// is how callers bind it. A graph that disagrees is a stale asset, so this throws instead
/// of falling back.
func loadPrefillGraph(
    from model: AIModel,
    matching main: InferenceFunctionDescriptor,
    mainName: String
) throws -> InferenceFunction? {
    guard let prefill = model.functionDescriptor(for: prefillGraphFunctionName) else { return nil }

    guard prefill.inputNames == main.inputNames else {
        throw InferenceRuntimeError.invalidInputType(
            "'\(prefillGraphFunctionName)' graph inputs \(prefill.inputNames) do not match "
                + "'\(mainName)' inputs \(main.inputNames)")
    }
    guard Set(prefill.stateNames) == Set(main.stateNames) else {
        throw InferenceRuntimeError.invalidOutputType(
            "'\(prefillGraphFunctionName)' graph states \(prefill.stateNames) do not match "
                + "'\(mainName)' states \(main.stateNames)")
    }
    guard prefill.outputNames.isEmpty else {
        throw InferenceRuntimeError.invalidOutputType(
            "'\(prefillGraphFunctionName)' graph declares outputs \(prefill.outputNames); "
                + "expected none. Re-export the model.")
    }
    guard let loaded = try model.loadFunction(named: prefillGraphFunctionName) else {
        throw InferenceRuntimeError.genericError(
            "Cannot load function '\(prefillGraphFunctionName)'")
    }
    return loaded
}
