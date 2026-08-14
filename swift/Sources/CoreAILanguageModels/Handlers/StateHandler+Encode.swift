// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Metal

/// Encode an inference step with KV cache states, optional additional MTLBuffer
/// states, and a single output.
///
/// The output is the graph's logits for `main`, or the unread by-product of a
/// prefill-only `prompt` graph — hence the neutral `output*` naming and the explicit
/// scalar type.
func encodeWithStates(
    function: InferenceFunction,
    inputs: [String: InferenceFunction.AsyncValue],
    keyState: inout InferenceFunction.AsyncMutableValue,
    keyCacheName: String,
    valState: inout InferenceFunction.AsyncMutableValue,
    valueCacheName: String,
    additionalStates: FixedMTLBufferState?,
    outputBuffer: MTLBuffer,
    outputName: String,
    outputShape: [Int],
    outputStrides: [Int],
    outputScalarType: NDArray.ScalarType = .float16,
    computeStream: ComputeStream
) throws {
    var asyncStates = InferenceFunction.AsyncMutableViews()
    asyncStates.insert(&keyState, for: keyCacheName)
    asyncStates.insert(&valState, for: valueCacheName)
    additionalStates?.bind(into: &asyncStates)

    var output = unsafe InferenceFunction.AsyncMutableValue(
        unsafeBuffer: outputBuffer, byteOffset: 0,
        scalarType: outputScalarType, shape: outputShape, strides: outputStrides)
    var asyncOutputs = InferenceFunction.AsyncMutableViews()
    asyncOutputs.insert(&output, for: outputName)
    let _ = try function.encode(
        inputs: inputs, states: consume asyncStates,
        outputViews: consume asyncOutputs, to: computeStream)
}
