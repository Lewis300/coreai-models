// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Loads an externalized INT8 Per-Layer Embeddings (PLE) table and gathers
/// per-token rows to feed the `ple_embeddings` graph input.
///
/// The table (one INT8 row of `numLayers * perLayerDim` values per vocabulary
/// token) is multiple gigabytes, so the export writes it to a sidecar rather than
/// into the graph. It is mmapped here; the graph dequantizes the gathered rows
/// with the scale and zero point baked in at export time.
///
/// ## Safetensors layout
/// `[8-byte little-endian header length][JSON header][raw tensor bytes]`. The
/// `embed_tokens_per_layer` entry is a 2-D INT8 tensor of shape
/// `[vocabSize, rowWidth]`.
struct PerLayerEmbeddings: Sendable {
    /// The mmapped file contents (header + raw INT8 rows).
    private let data: Data
    /// Byte offset where the INT8 tensor data begins.
    private let dataStart: Int
    /// Number of vocabulary rows.
    let vocabSize: Int
    /// INT8 elements per token row (`numLayers * perLayerDim`).
    let rowWidth: Int

    private static let tensorKey = "embed_tokens_per_layer"

    /// Locates this sidecar through the bundle's `assets` role map, or nil when
    /// the bundle doesn't ship one.
    static func resolveURL(in bundle: ModelBundle) -> URL? {
        bundle.modelURL(for: EngineOptions.AssetKey.perLayerEmbeddings)
    }

    enum PLEError: Error, CustomStringConvertible {
        case tooSmall
        case badHeader(String)
        case missingTensor

        var description: String {
            switch self {
            case .tooSmall: return "PLE file is too small to contain a safetensors header"
            case .badHeader(let m): return "PLE safetensors header invalid: \(m)"
            case .missingTensor: return "PLE file missing '\(PerLayerEmbeddings.tensorKey)' tensor"
            }
        }
    }

    init(contentsOf url: URL) throws {
        // Always map: `.mappedIfSafe` silently reads the multi-GB file into memory when
        // it judges mapping unsafe.
        let mapped = try Data(contentsOf: url, options: .alwaysMapped)
        guard mapped.count >= 8 else { throw PLEError.tooSmall }

        // First 8 bytes: little-endian uint64 JSON header length.
        var len: UInt64 = 0
        for i in 0..<8 {
            len |= UInt64(mapped[mapped.startIndex + i]) << (8 * i)
        }
        guard let headerLength = Int(exactly: len), headerLength <= mapped.count - 8 else {
            throw PLEError.tooSmall
        }

        let headerData = mapped.subdata(in: (mapped.startIndex + 8)..<(mapped.startIndex + 8 + headerLength))
        guard
            let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any],
            let tensorDict = json[Self.tensorKey] as? [String: Any]
        else {
            throw PLEError.missingTensor
        }
        guard
            let shape = tensorDict["shape"] as? [Int], shape.count == 2, shape.allSatisfy({ $0 > 0 }),
            let offsets = tensorDict["data_offsets"] as? [Int], offsets.count == 2,
            (0...mapped.count).contains(offsets[0])
        else {
            throw PLEError.badHeader("missing/invalid shape or data_offsets for \(Self.tensorKey)")
        }
        // The bounds check below sizes the tensor at 1 byte/element, which holds
        // for INT8 alone. State the requirement here so a re-quantized export
        // fails on its dtype rather than on a byte count that looks arbitrary.
        guard tensorDict["dtype"] as? String == "I8" else {
            throw PLEError.badHeader("expected INT8 (I8) data for \(Self.tensorKey)")
        }

        self.data = mapped
        self.vocabSize = shape[0]
        self.rowWidth = shape[1]
        self.dataStart = 8 + headerLength + offsets[0]

        // The tensor data must actually fit in the mapped file — otherwise a
        // valid token id could index past the mmap (SIGBUS) during gather.
        let (expectedBytes, overflow) = vocabSize.multipliedReportingOverflow(by: rowWidth)  // INT8
        guard !overflow, offsets[1] - offsets[0] == expectedBytes,
            expectedBytes <= mapped.count - dataStart
        else {
            throw PLEError.badHeader(
                "PLE tensor data out of bounds: shape \(shape), offsets \(offsets), "
                    + "file \(mapped.count) bytes")
        }
    }

    /// Copies the PLE rows for `tokenIDs` into `dest`, a buffer holding
    /// `batchSize * rowWidth` INT8 values laid out row-major (token-major).
    ///
    /// Tokens beyond `tokenIDs.count` (padding up to `batchSize`) are left as
    /// whatever `dest` already contains (callers pass a zeroed buffer).
    func gather(
        tokenIDs: some Collection<Int32>, batchSize: Int, into dest: UnsafeMutableBufferPointer<Int8>
    ) {
        precondition(dest.count >= batchSize * rowWidth, "PLE destination buffer too small")
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress else { return }
            let src = base.advanced(by: dataStart).assumingMemoryBound(to: Int8.self)
            for (i, tokenID) in tokenIDs.prefix(batchSize).enumerated() {
                let token = Int(tokenID)
                guard token >= 0, token < vocabSize else { continue }
                let srcRow = src.advanced(by: token * rowWidth)
                let dstRow = dest.baseAddress!.advanced(by: i * rowWidth)
                dstRow.update(from: srcRow, count: rowWidth)
            }
        }
    }
}
