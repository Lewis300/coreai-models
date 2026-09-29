// Per-Layer Embeddings input for static-shape models.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Gathers INT8 Per-Layer Embedding rows into the `ple_embeddings` graph input.
///
/// Models with per-layer embeddings carry one INT8 row of
/// `numLayers · perLayerDim` values per vocabulary token. That table is multiple
/// gigabytes, so it is externalized into a sidecar artifact instead of baked into
/// the graph (see ``PerLayerEmbeddings``); each step the runner copies the rows
/// for the current batch's tokens into the input and the graph dequantizes them
/// in place with the scale and zero-point baked in at export time.
struct PerLayerEmbeddingsInputHandler: StaticInputHandler {
    static let inputName = "ple_embeddings"

    let inputNames: [String] = [inputName]

    private let table: PerLayerEmbeddings
    private let descriptors: BucketedInputDescriptors

    init(table: PerLayerEmbeddings, descriptors: BucketedInputDescriptors) {
        self.table = table
        self.descriptors = descriptors
    }

    func registerBuffers(into buffers: inout InputBuffers) {
        descriptors.registerBuffers(name: Self.inputName, into: &buffers)
    }

    func fill(_ context: InputContext, into buffers: inout InputBuffers) throws {
        let key = StaticBucketKey(batchSize: context.batchSize, contextBucket: context.contextBucket)
        let descriptor = try descriptors.require(key, input: Self.inputName)

        let rowWidth = descriptor.shape.last ?? table.rowWidth
        guard rowWidth == table.rowWidth else {
            throw InferenceRuntimeError.invalidState(
                "PLE row width mismatch: graph expects \(rowWidth), table has \(table.rowWidth)")
        }

        buffers.ensureCapacity(name: Self.inputName, descriptor: descriptor)

        let elementCount = descriptor.shape.reduce(1, *)
        let tokenIDs = Array(context.tokens)
        let batchSize = context.batchSize
        let table = self.table

        let span = InstrumentsProfiler.beginPLEGather()

        try buffers.withMutableBuffer(Self.inputName) { array in
            let view = array.mutableView(as: Int8.self)
            // The flat row-major gather assumes a contiguous buffer. The
            // ple_embeddings input is exported without interleave so it is, but
            // verify rather than silently write to wrong offsets.
            guard view.isContiguous else {
                throw InferenceRuntimeError.invalidState(
                    "ple_embeddings array has non-contiguous layout")
            }
            view.withUnsafeMutablePointer { ptr, _, _ in
                // Zero first: a partial final batch leaves the padding slots
                // holding the previous step's rows otherwise.
                ptr.update(repeating: 0, count: elementCount)
                let buffer = UnsafeMutableBufferPointer(start: ptr, count: elementCount)
                table.gather(tokenIDs: tokenIDs, batchSize: batchSize, into: buffer)
            }
        }
        span.end()
    }
}
