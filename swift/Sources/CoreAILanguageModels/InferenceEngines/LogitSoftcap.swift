// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

/// Final-logit soft capping, applied on the CPU after the forward pass.
///
/// Gemma-family models squash their output logits with `c · tanh(logits / c)`. The
/// `tanh` is best run on the CPU rather than in the graph, so the iOS export leaves it out of the
/// graph (see `models/ios/gemma4_text.py`) and the runner applies it here instead,
/// between reading `out_logits` and sampling.
///
/// Applying it runner-side keeps sampling *and* any `--save-logits` / `--print-logits`
/// output on the same capped values the reference implementation produces, so parity
/// comparisons stay meaningful.
enum LogitSoftcap {
    /// Applies `cap · tanh(logits / cap)` to `logits` in place.
    ///
    /// The arithmetic runs in `Float` even when ``LogitsScalarType`` is `Float16`:
    /// `tanh` of a half-precision quotient loses too much of the small-difference
    /// structure that sampling depends on. Results are rounded back to
    /// ``LogitsScalarType`` on the way out — which is where any remaining divergence
    /// from the reference fp32 implementation comes from.
    ///
    /// No-ops for a non-positive `cap` or an empty buffer.
    static func apply(cap: Float, to logits: inout [LogitsScalarType]) {
        var scratch: [Float] = []
        apply(cap: cap, to: &logits, scratch: &scratch)
    }

    /// ``apply(cap:to:)`` with a caller-owned `scratch`, reused across calls so a
    /// decode loop doesn't allocate a vocab-sized buffer every token.
    static func apply(cap: Float, to logits: inout [LogitsScalarType], scratch: inout [Float]) {
        guard cap > 0, !logits.isEmpty else { return }

        let count = logits.count
        let inverseCap = 1 / cap

        // vForce's tanh wants Float32, so widen (and pre-divide) into scratch, transform
        // in place, then narrow back. The scratch allocation is ~1 MB at Gemma's vocab
        // size — negligible next to the forward pass that produced these logits.
        if scratch.count != count {
            scratch = [Float](repeating: 0, count: count)
        }
        for i in 0..<count {
            scratch[i] = Float(logits[i]) * inverseCap
        }

        var elementCount = Int32(count)
        scratch.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            vvtanhf(base, base, &elementCount)
        }

        for i in 0..<count {
            logits[i] = LogitsScalarType(scratch[i] * cap)
        }
    }
}
