// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// The Gemma4 iOS graph emits *uncapped* logits — `tanh` does not place on the Neural
/// Engine — so the runner applies `c · tanh(logits / c)` on the CPU before sampling.
/// These cover the arithmetic and the metadata → config plumbing that feeds it; a
/// dropped `final_logit_softcapping` key fails silently (uncapped logits, plausible
/// output, wrong distribution), so both halves are worth pinning.
@Suite("LogitSoftcap")
struct LogitSoftcapTests {
    /// Reference implementation, in Double, to compare the fp16 pipeline against.
    private func reference(_ x: Double, cap: Double) -> Double {
        tanh(x / cap) * cap
    }

    @Test("Matches c·tanh(logits/c) elementwise")
    func matchesReference() {
        let cap: Float = 30
        var logits: [LogitsScalarType] = [-100, -30, -7.5, -1, 0, 1, 7.5, 30, 100]
        let inputs = logits.map { Double($0) }

        LogitSoftcap.apply(cap: cap, to: &logits)

        for (actual, input) in zip(logits, inputs) {
            let expected = reference(input, cap: Double(cap))
            // Tolerance covers the round back to LogitsScalarType (Float16 on ARM),
            // whose spacing near 30 is ~0.016.
            #expect(abs(Double(actual) - expected) < 0.02)
        }
    }

    @Test("Bounds every value within ±cap")
    func boundsOutput() {
        let cap: Float = 5
        var logits: [LogitsScalarType] = [-1000, -50, 0, 50, 1000]
        LogitSoftcap.apply(cap: cap, to: &logits)

        for value in logits {
            #expect(abs(Float(value)) <= cap)
        }
    }

    @Test("Preserves ordering, so greedy sampling is unaffected")
    func preservesOrdering() {
        let cap: Float = 30
        var logits: [LogitsScalarType] = [1, 42, 3, 41.5, 2]
        let argmaxBefore = logits.indices.max(by: { logits[$0] < logits[$1] })

        LogitSoftcap.apply(cap: cap, to: &logits)

        let argmaxAfter = logits.indices.max(by: { logits[$0] < logits[$1] })
        #expect(argmaxBefore == argmaxAfter)
        // Monotonic: the whole ordering survives, not just the top element.
        #expect(logits[1] > logits[3])
        #expect(logits[3] > logits[2])
        #expect(logits[2] > logits[4])
        #expect(logits[4] > logits[0])
    }

    @Test("Saturates large magnitudes toward ±cap")
    func saturates() {
        let cap: Float = 30
        var logits: [LogitsScalarType] = [500, -500]
        LogitSoftcap.apply(cap: cap, to: &logits)

        #expect(abs(Float(logits[0]) - cap) < 0.05)
        #expect(abs(Float(logits[1]) + cap) < 0.05)
    }

    @Test("The scratch overload reuses its buffer and matches the one-shot form")
    func scratchOverloadMatches() {
        let cap: Float = 5
        let input: [LogitsScalarType] = [-20, -3, 0, 2.5, 40]
        var expected = input
        LogitSoftcap.apply(cap: cap, to: &expected)

        var scratch: [Float] = []
        for _ in 0..<2 {
            var logits = input
            LogitSoftcap.apply(cap: cap, to: &logits, scratch: &scratch)
            #expect(logits == expected)
        }
        #expect(scratch.count == input.count)
    }

    @Test("Non-positive cap is a no-op")
    func nonPositiveCapIsNoOp() {
        let original: [LogitsScalarType] = [-3, 0, 7]

        for cap in [Float(0), -1] {
            var logits = original
            LogitSoftcap.apply(cap: cap, to: &logits)
            #expect(logits == original)
        }
    }

    @Test("Empty buffer is a no-op")
    func emptyBufferIsNoOp() {
        var logits: [LogitsScalarType] = []
        LogitSoftcap.apply(cap: 30, to: &logits)
        #expect(logits.isEmpty)
    }
}

// MARK: - Metadata plumbing

@Suite("Final logit softcap config plumbing")
struct FinalLogitSoftcapConfigTests {
    @Test("LanguageConfig decodes language.final_logit_softcapping")
    func languageConfigDecodes() throws {
        let json = """
            {
                "tokenizer": "google/gemma-4-E2B-it",
                "vocab_size": 262144,
                "max_context_length": 131072,
                "sliding_window": 512,
                "final_logit_softcapping": 30.0
            }
            """
        let config = try JSONDecoder().decode(LanguageConfig.self, from: Data(json.utf8))
        #expect(config.finalLogitSoftcapping == 30.0)
    }

    @Test("LanguageConfig leaves the cap nil when the key is absent")
    func languageConfigDefaultsToNil() throws {
        let json = """
            {
                "tokenizer": "Qwen/Qwen3-0.6B",
                "vocab_size": 151936,
                "max_context_length": 4096
            }
            """
        let config = try JSONDecoder().decode(LanguageConfig.self, from: Data(json.utf8))
        #expect(config.finalLogitSoftcapping == nil)
    }

    @Test("ModelConfig decodes final_logit_softcapping")
    func modelConfigDecodes() throws {
        let json = """
            {
                "name": "gemma-4-e2b-it_static",
                "tokenizer": "google/gemma-4-E2B-it",
                "vocab_size": 262144,
                "max_context_length": 131072,
                "serialized_model": ["gemma-4-e2b-it_static.aimodel"],
                "function": "main",
                "final_logit_softcapping": 30.0
            }
            """
        let config = try ModelConfig(parsing: Data(json.utf8))
        #expect(config.finalLogitSoftcapping == 30.0)
    }

    @Test("ModelConfig rejects a non-positive cap", arguments: ["0.0", "-30.0"])
    func modelConfigRejectsNonPositiveCap(cap: String) {
        let json = """
            {
                "name": "m", "tokenizer": "t", "vocab_size": 8, "max_context_length": 64,
                "serialized_model": ["m.aimodel"], "final_logit_softcapping": \(cap)
            }
            """
        #expect(throws: (any Error).self) { try ModelConfig(parsing: Data(json.utf8)) }
    }

    @Test("ModelConfig round-trips the cap through Codable")
    func modelConfigRoundTrips() throws {
        // The runner hands the engine factory an encoded ModelConfig, so an encode/decode
        // gap here would drop the cap between bundle load and engine construction.
        let config = ModelConfig(
            name: "gemma-4-e2b-it_static",
            tokenizer: "google/gemma-4-E2B-it",
            vocabSize: 262144,
            maxContextLength: 131072,
            serializedModel: ["gemma-4-e2b-it_static.aimodel"],
            function: "main",
            finalLogitSoftcapping: 30.0
        )

        let decoded = try ModelConfig(parsing: try JSONEncoder().encode(config))
        #expect(decoded.finalLogitSoftcapping == 30.0)
    }
}
