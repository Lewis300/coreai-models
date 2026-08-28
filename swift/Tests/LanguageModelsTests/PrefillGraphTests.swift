// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// Covers the decisions the engines make around the optional `prefill` entrypoint: whether
/// to chunk, how wide a chunk may be, how a prompt splits, and whether a descriptor may be
/// bound at all. These call the same functions the engines call, so a behaviour change in
/// either engine shows up here.
@Suite("Prefill Graph")
struct PrefillGraphTests {
    // MARK: - Query length clamp

    @Test("Query length is the chunk size when it fits the context")
    func queryLengthUsesChunkSize() {
        #expect(prefillQueryLength(prefillChunkSize: 64, maxContextLength: 4096) == 64)
    }

    @Test("Query length is clamped to the context")
    func queryLengthClampedToContext() {
        // A chunk wider than the context could never be traced, let alone run.
        #expect(prefillQueryLength(prefillChunkSize: 512, maxContextLength: 256) == 256)
    }

    @Test("Query length is at least one token")
    func queryLengthFloorsAtOne() {
        // A zero or negative config must not produce an empty or negative chunk width:
        // prefillChunkSizes would then loop forever or emit nonsense.
        #expect(prefillQueryLength(prefillChunkSize: 0, maxContextLength: 4096) == 1)
        #expect(prefillQueryLength(prefillChunkSize: -8, maxContextLength: 4096) == 1)
        #expect(prefillQueryLength(prefillChunkSize: 64, maxContextLength: 0) == 1)
    }

    // MARK: - Whether to chunk

    @Test("With a prefill graph, prefill chunks at any size")
    func chunksAtAnySizeWithPrefillGraph() {
        // No LM head per chunk, so there is no threshold to clear.
        for count in [1, 2, 17, 4096] {
            #expect(
                shouldChunkPrefill(
                    tokenCount: count, hasPrefillGraph: true, chunkThreshold: 512),
                "expected chunking at \(count) tokens")
        }
    }

    @Test("Without a prefill graph, chunking waits for the threshold")
    func honoursThresholdWithoutPrefillGraph() {
        #expect(
            !shouldChunkPrefill(tokenCount: 512, hasPrefillGraph: false, chunkThreshold: 512),
            "at the threshold, not past it")
        #expect(
            shouldChunkPrefill(tokenCount: 513, hasPrefillGraph: false, chunkThreshold: 512))
    }

    // MARK: - How the prompt splits

    @Test("A prompt shorter than one chunk prefills in a single chunk")
    func singleChunk() {
        // 40 tokens, one held back for `main`, so 39 go through the prefill graph at once.
        #expect(prefillChunkSizes(tokenCount: 40, chunkSize: 64, heldBack: 1) == [39])
    }

    @Test("A long prompt splits into full chunks plus a remainder")
    func multipleChunks() {
        // 400 tokens, 399 to prefill: six 64s and a 15.
        let sizes = prefillChunkSizes(tokenCount: 400, chunkSize: 64, heldBack: 1)
        #expect(sizes == [64, 64, 64, 64, 64, 64, 15])
        #expect(sizes.reduce(0, +) == 399)
    }

    @Test("An exact multiple leaves no remainder chunk")
    func exactMultiple() {
        // 129 tokens, 128 to prefill, so two full chunks and nothing trailing.
        #expect(prefillChunkSizes(tokenCount: 129, chunkSize: 64, heldBack: 1) == [64, 64])
    }

    @Test("The held-back token is always excluded from the plan")
    func heldBackExcluded() {
        // The invariant the engines depend on: chunks cover exactly count - heldBack, so the
        // caller's suffix(1) is never prefilled twice.
        for count in [1, 2, 5, 63, 64, 65, 128, 400] {
            let sizes = prefillChunkSizes(tokenCount: count, chunkSize: 64, heldBack: 1)
            #expect(sizes.reduce(0, +) == count - 1, "at \(count) tokens")
        }
    }

    @Test("A one-token prompt prefills nothing")
    func oneTokenPromptPrefillsNothing() {
        // The single token is the one that has to reach `main` for logits, so there is
        // nothing left for the prefill graph and the loop must not run.
        #expect(prefillChunkSizes(tokenCount: 1, chunkSize: 64, heldBack: 1).isEmpty)
    }

    @Test("An empty prompt prefills nothing")
    func emptyPromptPrefillsNothing() {
        // Callers guard on isEmpty, but underflowing to -1 here would hang the loop.
        #expect(prefillChunkSizes(tokenCount: 0, chunkSize: 64, heldBack: 1).isEmpty)
    }

    @Test("With nothing held back, chunks cover the whole prompt")
    func noneHeldBackCoversEverything() {
        // The no-prefill-graph path: the trailing chunk carries the logits, so every token
        // is chunked and none is held back.
        let sizes = prefillChunkSizes(tokenCount: 400, chunkSize: 64, heldBack: 0)
        #expect(sizes.reduce(0, +) == 400)
        #expect(sizes == [64, 64, 64, 64, 64, 64, 16])
    }

    @Test("A degenerate chunk size still terminates")
    func degenerateChunkSizeTerminates() {
        // Belt and braces: prefillQueryLength already floors at 1, but a direct caller
        // passing 0 must not spin forever.
        #expect(prefillChunkSizes(tokenCount: 4, chunkSize: 0, heldBack: 1) == [1, 1, 1])
    }

    @Test("Chunks never exceed the query length the graph was traced for")
    func chunksNeverExceedQueryLength() {
        // Running a chunk wider than the traced query length would fail at encode time.
        let width = prefillQueryLength(prefillChunkSize: 64, maxContextLength: 256)
        for count in [1, 65, 200, 257, 1000] {
            for chunk in prefillChunkSizes(tokenCount: count, chunkSize: width, heldBack: 1) {
                #expect(chunk <= width, "chunk \(chunk) exceeds width \(width) at \(count)")
                #expect(chunk >= 1)
            }
        }
    }

    // MARK: - Descriptor validation

    private let mainInputs = ["input_ids", "position_ids"]
    private let mainStates = ["keyCache", "valueCache"]

    @Test("A matching descriptor validates")
    func matchingDescriptorValidates() throws {
        try validatePrefillShape(
            prefillInputs: mainInputs,
            prefillStates: mainStates,
            prefillOutputs: [],
            mainInputs: mainInputs,
            mainStates: mainStates,
            mainName: "main")
    }

    @Test("States may be in a different order")
    func stateOrderDoesNotMatter() throws {
        // States are bound by name, so order carries no meaning -- compared as sets.
        try validatePrefillShape(
            prefillInputs: mainInputs,
            prefillStates: ["valueCache", "keyCache"],
            prefillOutputs: [],
            mainInputs: mainInputs,
            mainStates: mainStates,
            mainName: "main")
    }

    @Test("Mismatched inputs are rejected")
    func mismatchedInputsRejected() {
        #expect(throws: InferenceRuntimeError.self) {
            try validatePrefillShape(
                prefillInputs: ["input_ids"],
                prefillStates: mainStates,
                prefillOutputs: [],
                mainInputs: mainInputs,
                mainStates: mainStates,
                mainName: "main")
        }
    }

    @Test("Input order is significant")
    func inputOrderIsSignificant() {
        // Inputs bind positionally, so a reordering is a real mismatch, unlike states.
        #expect(throws: InferenceRuntimeError.self) {
            try validatePrefillShape(
                prefillInputs: ["position_ids", "input_ids"],
                prefillStates: mainStates,
                prefillOutputs: [],
                mainInputs: mainInputs,
                mainStates: mainStates,
                mainName: "main")
        }
    }

    @Test("Mismatched states are rejected")
    func mismatchedStatesRejected() {
        #expect(throws: InferenceRuntimeError.self) {
            try validatePrefillShape(
                prefillInputs: mainInputs,
                prefillStates: ["keyCache"],
                prefillOutputs: [],
                mainInputs: mainInputs,
                mainStates: mainStates,
                mainName: "main")
        }
    }

    @Test("A declared output is rejected")
    func declaredOutputRejected() {
        // The load-bearing one: a `prefill` graph carrying `logits` is an asset exported
        // before the LM head was dropped, and binding it would silently cost the head.
        #expect(throws: InferenceRuntimeError.self) {
            try validatePrefillShape(
                prefillInputs: mainInputs,
                prefillStates: mainStates,
                prefillOutputs: ["logits"],
                mainInputs: mainInputs,
                mainStates: mainStates,
                mainName: "main")
        }
    }

    @Test("The rejection message names the entrypoint and the fix")
    func rejectionMessageIsActionable() {
        // A stale asset is fixed by re-exporting, so the error has to say so.
        do {
            try validatePrefillShape(
                prefillInputs: mainInputs,
                prefillStates: mainStates,
                prefillOutputs: ["logits"],
                mainInputs: mainInputs,
                mainStates: mainStates,
                mainName: "main")
            Issue.record("expected a throw")
        } catch {
            let message = "\(error)"
            #expect(message.contains(prefillGraphFunctionName))
            #expect(message.contains("Re-export"))
        }
    }
}
