// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILanguageModels
import Foundation

// MARK: - Raw Token Scoring

/// Decides whether a `--raw-tokens` run scores a forced continuation, and reads the ids it scores.
///
/// `--raw-tokens` is always the context. A run scores only when `--raw-continuation` names the ids
/// to score, which mirrors `--prompt` and `--continuation` in the text mode. Scoring used to follow
/// from `--raw-tokens` plus a logits flag alone and took `tokens[0]` as the whole context, so no run
/// could generate from raw tokens and save the logits of what it generated.
enum RawTokenScoring {
    /// The ids a `--raw-continuation` run scores.
    struct Request: Equatable {
        /// Every id in `--raw-tokens`.
        let context: [Int32]
        /// Every id in `--raw-continuation`; each one is scored.
        let continuation: [Int32]
    }

    /// - Parameters:
    ///   - rawContinuationPath: The `--raw-continuation` file, in the `{"tokens": [...]}` format of
    ///     `--raw-tokens`, or `nil` when the run generates.
    ///   - rawTokensPath: The `--raw-tokens` file that gives the context, or `nil`.
    ///   - textContinuationGiven: Whether the text `--continuation` is also given.
    ///   - logitsRequested: Whether `--print-logits` or `--save-logits` is given.
    ///   - conflictingFlags: Flags given on this run that take their own path and would ignore
    ///     `--raw-continuation` (`--batch-prototype`, `--image`, `--video`).
    /// - Returns: The context and the ids to score, or `nil` when the run generates from
    ///   `--raw-tokens`.
    /// - Throws: `ContinuationOptionError` when `--raw-continuation` is given with the text
    ///   `--continuation` or a conflicting flag, without `--raw-tokens`, without a logits output, or
    ///   when either file cannot be read or holds no ids. Both files are read here so the run
    ///   refuses before the model loads.
    static func request(
        rawContinuationPath: String?,
        rawTokensPath: String?,
        textContinuationGiven: Bool,
        logitsRequested: Bool,
        conflictingFlags: [String] = []
    ) throws -> Request? {
        guard let path = rawContinuationPath else { return nil }
        guard !textContinuationGiven else {
            throw ContinuationOptionError.rawContinuationConflicts(flag: "--continuation")
        }
        if let flag = conflictingFlags.first {
            throw ContinuationOptionError.rawContinuationConflicts(flag: flag)
        }
        guard let contextPath = rawTokensPath else {
            throw ContinuationOptionError.rawContinuationRequiresRawTokens
        }
        guard logitsRequested else {
            throw ContinuationOptionError.rawContinuationRequiresLogitsOutput
        }
        let context = try readTokens(at: contextPath)
        guard !context.isEmpty else {
            throw ContinuationOptionError.emptyRawContext(path: contextPath)
        }
        let continuation = try readTokens(at: path)
        guard !continuation.isEmpty else {
            throw ContinuationOptionError.emptyRawContinuation(path: path)
        }
        return Request(context: context, continuation: continuation)
    }

    /// Refuses ids the model cannot take: one outside its vocabulary, or more ids in total than its
    /// context holds. Needs only the bundle's config, so the run refuses before the model loads.
    static func checkFits(_ request: Request, vocabSize: Int, maxContextLength: Int) throws {
        for (flag, ids) in [("--raw-tokens", request.context), ("--raw-continuation", request.continuation)] {
            if let id = ids.first(where: { $0 < 0 || Int($0) >= vocabSize }) {
                throw ContinuationOptionError.tokenOutsideVocabulary(flag: flag, id: id, vocabSize: vocabSize)
            }
        }
        let total = request.context.count + request.continuation.count
        guard total <= maxContextLength else {
            throw ContinuationOptionError.exceedsContextLength(
                context: request.context.count, continuation: request.continuation.count,
                maxContextLength: maxContextLength)
        }
    }

    /// Scores under `--json-schema`: masks each position's logits with the schema's grammar before
    /// they are scored, as constrained generation masks them before it samples. The context is not
    /// constrained, only the continuation, so log-probabilities are over the ids the schema allows.
    ///
    /// - Throws: `ContinuationOptionError` when a continuation id is one the schema does not allow
    ///   at its position, or comes after the schema's JSON is complete.
    static func maskedByGrammar(
        _ result: ContinuationEvaluationResult, session: inout ConstrainedGenerationSession
    ) throws -> ContinuationEvaluationResult {
        var masked: [[LogitsScalarType]] = []
        masked.reserveCapacity(result.logits.count)
        for (position, (logits, id)) in zip(result.logits, result.continuationTokens).enumerated() {
            var row = logits
            guard session.applyMask(to: &row) else {
                throw ContinuationOptionError.continuationPastSchemaEnd(position: position, id: id)
            }
            guard session.acceptToken(id) else {
                throw ContinuationOptionError.continuationViolatesSchema(position: position, id: id)
            }
            masked.append(row)
        }
        return ContinuationEvaluationResult(
            contextTokens: result.contextTokens, continuationTokens: result.continuationTokens, logits: masked)
    }

    private static func readTokens(at path: String) throws -> [Int32] {
        let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        do {
            return try JSONDecoder().decode(RawTokensInput.self, from: Data(contentsOf: url)).tokens
        } catch {
            throw ContinuationOptionError.unreadableRawTokens(path: path, reason: error.localizedDescription)
        }
    }
}

// MARK: - Errors

/// Why an `llm-runner` run refuses its continuation options.
enum ContinuationOptionError: Error, LocalizedError, Equatable {
    case requiresDisabledChatTemplate
    case requiresLogitsOutput
    case engineDoesNotSupportLogits
    case emptyContinuation
    case rawTokensNotSupported
    case rawContinuationRequiresRawTokens
    case rawContinuationRequiresLogitsOutput
    case rawContinuationConflicts(flag: String)
    case unreadableRawTokens(path: String, reason: String)
    case emptyRawContext(path: String)
    case emptyRawContinuation(path: String)
    case tokenOutsideVocabulary(flag: String, id: Int32, vocabSize: Int)
    case exceedsContextLength(context: Int, continuation: Int, maxContextLength: Int)
    case continuationViolatesSchema(position: Int, id: Int32)
    case continuationPastSchemaEnd(position: Int, id: Int32)

    var errorDescription: String? {
        switch self {
        case .requiresDisabledChatTemplate:
            return "--continuation requires --apply-chat-template=false"
        case .requiresLogitsOutput:
            return "--continuation requires --print-logits or --save-logits"
        case .engineDoesNotSupportLogits:
            return "The current inference engine does not support returning logits"
        case .emptyContinuation:
            return "Continuation string cannot be empty"
        case .rawTokensNotSupported:
            return "--continuation requires text prompt (--prompt or --prompt-file), not --raw-tokens"
        case .rawContinuationRequiresRawTokens:
            return "--raw-continuation requires --raw-tokens"
        case .rawContinuationRequiresLogitsOutput:
            return "--raw-continuation requires --print-logits or --save-logits"
        case .rawContinuationConflicts(let flag):
            return "--raw-continuation cannot be used with \(flag)"
        case .unreadableRawTokens(let path, let reason):
            return "Cannot read token file \(path): \(reason) "
                + "Expected format: {\"tokens\": [1, 2, 3, ...]}"
        case .emptyRawContext(let path):
            return "--raw-tokens file \(path) holds no tokens, so there is no context to score against"
        case .emptyRawContinuation(let path):
            return "--raw-continuation file \(path) holds no tokens"
        case .tokenOutsideVocabulary(let flag, let id, let vocabSize):
            return "\(flag) holds token id \(id), outside the model's vocabulary of \(vocabSize)"
        case .exceedsContextLength(let context, let continuation, let maxContextLength):
            return "--raw-tokens (\(context) ids) plus --raw-continuation (\(continuation) ids) is "
                + "\(context + continuation) ids, more than the model's context length of \(maxContextLength)"
        case .continuationViolatesSchema(let position, let id):
            return "--raw-continuation id \(id) at position \(position) is not allowed by --json-schema"
        case .continuationPastSchemaEnd(let position, let id):
            return "--raw-continuation id \(id) at position \(position) comes after the --json-schema JSON is complete"
        }
    }
}
