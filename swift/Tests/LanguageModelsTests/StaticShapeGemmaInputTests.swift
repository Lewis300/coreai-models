// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAILanguageModels

// MARK: - Sliding-window ring rewind

@Suite("Sliding-window ring rewind")
struct SlidingRingRewindTests {
    // Gemma 4's ring: window 512, depth 576.
    let ringDepth = 576
    let window = 512

    @Test("A rewind survives while at most ringDepth - window + 1 positions were written past it")
    func boundary() {
        let limit = ringDepth - window + 1
        #expect(
            StaticShapeEngine.ringAllowsRewind(
                processed: 1000, to: 1000 - limit, ringDepth: ringDepth, window: window))
        #expect(
            !StaticShapeEngine.ringAllowsRewind(
                processed: 1000, to: 1000 - limit - 1, ringDepth: ringDepth, window: window))
    }

    @Test("Rewinding one token, the chat-extension case, is always fine")
    func oneToken() {
        #expect(
            StaticShapeEngine.ringAllowsRewind(
                processed: 50_000, to: 49_999, ringDepth: ringDepth, window: window))
    }

    @Test("A full reset is always allowed")
    func fullReset() {
        #expect(
            StaticShapeEngine.ringAllowsRewind(
                processed: 50_000, to: 0, ringDepth: ringDepth, window: window))
    }
}

// MARK: - Sliding-window mask

@Suite("Sliding-window ring mask")
struct SlidingWindowMaskTests {
    /// Fills a contiguous `(1, S, 1, qLen)` mask and returns it as `[slot][query]`.
    private func fill(
        ringDepth: Int, window: Int, qLen: Int, alignedStep: Int, tokensInBatch: Int
    ) -> [[Float16]] {
        var mask = [Float16](repeating: 1, count: ringDepth * qLen)
        mask.withUnsafeMutableBufferPointer { buffer in
            SlidingWindowInputHandler.fillMask(
                buffer.baseAddress!, slotStride: qLen, queryStride: 1, queryColumns: qLen,
                ringDepth: ringDepth, window: window, alignedStep: alignedStep,
                tokensInBatch: tokensInBatch)
        }
        return (0..<ringDepth).map { Array(mask[($0 * qLen)..<(($0 + 1) * qLen)]) }
    }

    @Test(
        "Each query sees exactly its in-window keys at their ring slots",
        arguments: [0, 4, 9, 20, 44])
    func unmasksWindowAtRingSlots(alignedStep: Int) {
        let ringDepth = 12
        let window = 8
        let qLen = 4
        let mask = fill(
            ringDepth: ringDepth, window: window, qLen: qLen, alignedStep: alignedStep,
            tokensInBatch: qLen)
        for query in 0..<qLen {
            let position = alignedStep + query
            let visible = Set((max(0, position - window + 1)...position).map { $0 % ringDepth })
            #expect(visible.count == min(window, position + 1), "slots collided")
            for slot in 0..<ringDepth {
                let expected: Float16 = visible.contains(slot) ? 0 : causalMaskSentinel
                #expect(
                    mask[slot][query] == expected,
                    "step \(alignedStep) query \(query) slot \(slot)")
            }
        }
    }

    @Test("Padding query columns stay fully masked")
    func paddingColumnsMasked() {
        let mask = fill(ringDepth: 12, window: 8, qLen: 4, alignedStep: 30, tokensInBatch: 2)
        for slot in 0..<12 {
            #expect(mask[slot][2] == causalMaskSentinel)
            #expect(mask[slot][3] == causalMaskSentinel)
        }
    }
}

// MARK: - Handler validation

@Suite("Gemma input handler validation")
struct GemmaInputHandlerValidationTests {
    @Test("The sliding-window handler rejects a non-positive window")
    func rejectsNonPositiveWindow() {
        #expect(throws: InferenceRuntimeError.self) {
            try SlidingWindowInputHandler(
                window: 0, ringDepth: 576, maskDescriptors: .init([:]), stepDescriptors: .init([:]))
        }
    }

    @Test("The sliding-window handler needs a sliding cache to size the ring")
    func rejectsMissingRing() {
        #expect(throws: InferenceRuntimeError.self) {
            try SlidingWindowInputHandler(
                window: 512, ringDepth: 0, maskDescriptors: .init([:]), stepDescriptors: .init([:]))
        }
    }

    @Test("The dual-RoPE handler rejects odd or zero head dims", arguments: [(256, 511), (0, 512)])
    func rejectsBadHeadDims(dims: (Int, Int)) {
        let rope = RoPEConfig(
            slidingHeadDim: dims.0, globalHeadDim: dims.1, slidingRopeTheta: 10_000,
            globalRopeTheta: 1_000_000, partialRotaryFactor: 0.25)
        #expect(throws: InferenceRuntimeError.self) {
            try DualRoPEInputHandler(
                rope: rope, cosDescriptors: .init([:]), sinDescriptors: .init([:]))
        }
    }
}

// MARK: - Per-layer embeddings sidecar

@Suite("Per-layer embeddings sidecar parsing")
struct PerLayerEmbeddingsParsingTests {
    /// Writes a safetensors file holding one INT8 `embed_tokens_per_layer` tensor.
    private func writeTable(
        vocab: Int = 4, rowWidth: Int = 3, dtype: String = "I8",
        offsets: [Int]? = nil, dataBytes: Int? = nil, headerLength: UInt64? = nil
    ) throws -> URL {
        let offsets = offsets ?? [0, vocab * rowWidth]
        let header = try JSONSerialization.data(withJSONObject: [
            "embed_tokens_per_layer": [
                "dtype": dtype, "shape": [vocab, rowWidth], "data_offsets": offsets,
            ]
        ])
        var file = Data()
        var length = headerLength ?? UInt64(header.count)
        withUnsafeBytes(of: &length) { file.append(contentsOf: $0) }
        file.append(header)
        file.append(contentsOf: (0..<(dataBytes ?? vocab * rowWidth)).map { UInt8($0 % 100) })
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ple-\(UUID().uuidString).safetensors")
        try file.write(to: url)
        return url
    }

    @Test("A valid table gathers each token's row")
    func gathersRows() throws {
        let url = try writeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        let table = try PerLayerEmbeddings(contentsOf: url)
        #expect(table.vocabSize == 4 && table.rowWidth == 3)

        var rows = [Int8](repeating: -1, count: 2 * 3)
        rows.withUnsafeMutableBufferPointer { buffer in
            table.gather(tokenIDs: [2, 0] as [Int32], batchSize: 2, into: buffer)
        }
        #expect(rows == [6, 7, 8, 0, 1, 2])
    }

    @Test(
        "Malformed tables are rejected",
        arguments: [
            "truncated data", "wrong dtype", "negative offset", "oversized header",
        ])
    func rejectsMalformed(kind: String) throws {
        let url: URL
        switch kind {
        case "truncated data": url = try writeTable(dataBytes: 5)
        case "wrong dtype": url = try writeTable(dtype: "F16")
        case "negative offset": url = try writeTable(offsets: [-4, 8])
        default: url = try writeTable(headerLength: UInt64.max)
        }
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: (any Error).self) { try PerLayerEmbeddings(contentsOf: url) }
    }
}
