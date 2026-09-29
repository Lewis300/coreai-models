// Sliding-window ring-cache inputs for static-shape models.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Supplies the mask and write offset for a fixed-depth sliding-window ring cache.
///
/// Models that interleave sliding-window attention with global attention keep the
/// windowed layers' keys and values in a compact ring of depth `S` rather than a
/// full-context cache: the entry for absolute position `p` lives at slot `p % S`.
/// That needs two inputs the flat causal path doesn't have:
///
/// - `sliding_causal_mask` — like the causal mask, but restricted to the last
///   `window` keys and indexed into the ring by absolute position.
/// - `sliding_in_step` — the ring write offset, `alignedStep % S`, computed here
///   so the graph needs no in-graph remainder op.
public struct SlidingWindowInputHandler: StaticInputHandler {
    public static let maskInputName = "sliding_causal_mask"
    public static let stepInputName = "sliding_in_step"

    public let inputNames: [String]

    private let maskDescriptors: BucketedInputDescriptors
    private let stepDescriptors: BucketedInputDescriptors

    /// Attention window in tokens.
    private let window: Int
    /// Ring depth `S`, from the sliding cache's sequence dimension.
    private let ringDepth: Int

    public init(
        window: Int,
        ringDepth: Int,
        maskDescriptors: BucketedInputDescriptors,
        stepDescriptors: BucketedInputDescriptors
    ) throws {
        guard ringDepth > 0 else {
            throw InferenceRuntimeError.invalidState(
                "Graph declares sliding-window inputs but no `sliding_key_cache` state to size the ring")
        }
        // Mask is `(1, S, 1, q_len)`. Its ring must be the cache's, and deep enough
        // that a chunk's in-window keys never collide: `S >= window + q_len - 1`.
        for descriptor in maskDescriptors.descriptors {
            let shape = descriptor.shape
            guard shape.count == 4, shape[1] == ringDepth, ringDepth >= window + shape[3] - 1 else {
                throw InferenceRuntimeError.invalidState(
                    "'\(Self.maskInputName)' has shape \(shape) but the sliding cache ring depth is "
                        + "\(ringDepth) and the window is \(window) — expected (1, \(ringDepth), 1, q_len) "
                        + "with \(ringDepth) >= window + q_len - 1")
            }
        }

        self.window = window
        self.ringDepth = ringDepth
        self.maskDescriptors = maskDescriptors
        self.stepDescriptors = stepDescriptors

        var names: [String] = []
        if !maskDescriptors.isEmpty { names.append(Self.maskInputName) }
        if !stepDescriptors.isEmpty { names.append(Self.stepInputName) }
        self.inputNames = names
    }

    public func registerBuffers(into buffers: inout InputBuffers) {
        maskDescriptors.registerBuffers(name: Self.maskInputName, into: &buffers)
        stepDescriptors.registerBuffers(name: Self.stepInputName, into: &buffers)
    }

    public func fill(_ context: InputContext, into buffers: inout InputBuffers) throws {
        let key = StaticBucketKey(batchSize: context.batchSize, contextBucket: context.contextBucket)

        if !maskDescriptors.isEmpty {
            let span = InstrumentsProfiler.beginMaskBuild()
            let descriptor = try maskDescriptors.require(key, input: Self.maskInputName)
            buffers.ensureCapacity(name: Self.maskInputName, descriptor: descriptor)
            let tokensInBatch = context.tokens.count
            let alignedStep = context.alignedStep
            let window = self.window
            let ringDepth = self.ringDepth
            try buffers.withMutableBuffer(Self.maskInputName) { array in
                // Mask shape is `(1, S, 1, q_len)`: one row per ring slot. Start
                // fully masked with the fp16-safe `-inf` sentinel, then unmask, for
                // each query at position `p = alignedStep + query`, exactly the
                // in-window causal keys — positions `[max(0, p - window + 1), p]` —
                // at their ring slots. Because the exporter sizes
                // `S >= window + q_len - 1`, those `window` positions map to
                // distinct slots (no collisions) and keys written by later queries
                // in the same chunk stay masked.
                array.mutableView(as: LogitsScalarType.self)
                    .withUnsafeMutablePointer { ptr, shape, strides in
                        for slot in 0..<ringDepth {
                            for query in 0..<shape[3] {
                                let offset = slot &* strides[1] &+ query &* strides[3]
                                ptr[offset] = causalMaskSentinel
                            }
                        }
                        for query in 0..<tokensInBatch {
                            let queryPosition = alignedStep + query
                            let lowerPosition = max(0, queryPosition &- window &+ 1)
                            for position in lowerPosition...queryPosition {
                                let slot = position % ringDepth
                                let offset = slot &* strides[1] &+ query &* strides[3]
                                ptr[offset] = 0
                            }
                        }
                    }
            }
            span.end()
        }

        if !stepDescriptors.isEmpty {
            let descriptor = try stepDescriptors.require(key, input: Self.stepInputName)
            buffers.ensureCapacity(name: Self.stepInputName, descriptor: descriptor)
            let offset = context.alignedStep % ringDepth
            try buffers.withMutableBuffer(Self.stepInputName) { array in
                fillNDArray(&array, as: Int32.self, count: 1) { _ in Int32(offset) }
            }
        }
    }
}
