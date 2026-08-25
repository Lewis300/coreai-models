// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

/// Name of the optional prefill entrypoint. Exported beside `main` (see
/// `export/macos.py`) with the same inputs and states, but no LM head and no outputs:
/// it only fills the KV cache.
let promptGraphFunctionName = "prompt"

/// Load the prompt graph, or nil if the asset has none.
///
/// It must take the same inputs and states as `main` and declare no outputs, because that
/// is how callers bind it. A graph that disagrees is a stale asset, so this throws instead
/// of falling back.
func loadPromptGraph(
    from model: AIModel,
    matching main: InferenceFunctionDescriptor,
    mainName: String
) throws -> InferenceFunction? {
    guard let prompt = model.functionDescriptor(for: promptGraphFunctionName) else { return nil }

    guard prompt.inputNames == main.inputNames else {
        throw InferenceRuntimeError.invalidInputType(
            "'\(promptGraphFunctionName)' graph inputs \(prompt.inputNames) do not match "
                + "'\(mainName)' inputs \(main.inputNames)")
    }
    guard Set(prompt.stateNames) == Set(main.stateNames) else {
        throw InferenceRuntimeError.invalidOutputType(
            "'\(promptGraphFunctionName)' graph states \(prompt.stateNames) do not match "
                + "'\(mainName)' states \(main.stateNames)")
    }
    guard prompt.outputNames.isEmpty else {
        throw InferenceRuntimeError.invalidOutputType(
            "'\(promptGraphFunctionName)' graph declares outputs \(prompt.outputNames); "
                + "expected none. Re-export the model.")
    }
    guard let loaded = try model.loadFunction(named: promptGraphFunctionName) else {
        throw InferenceRuntimeError.genericError(
            "Cannot load function '\(promptGraphFunctionName)'")
    }
    return loaded
}
