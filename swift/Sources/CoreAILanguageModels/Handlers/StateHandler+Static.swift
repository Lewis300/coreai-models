// Persistent state handlers for static-shape (bucketed) engines.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// MARK: - Protocol

/// Persistent model state carried across steps by a static-shape engine.
///
/// A static-shape asset is a *ladder* of per-context-bucket programs, and each
/// bucket's program declares its own state shapes. That makes the static case
/// different from the dynamic one that ``SyncStateHandler`` serves:
///
/// - Binding is per-function. The engine passes the running function's
///   descriptor so the handler can slice its backing storage down to the shape
///   that bucket declares.
/// - Storage may have to be re-laid-out when the running bucket changes. A
///   state whose shape scales with the context bucket is compiled with per-ctx
///   sequence strides, so a buffer laid out for one bucket cannot simply be
///   sliced for another.
///
/// Handlers are classes so they own their `NDArray`s at refcount 1 and
/// `bind(into:for:)` can call `mutableRawView()` without triggering COW.
public protocol StaticStateHandler: AnyObject {
    /// Names of the states this handler manages.
    var stateNames: [String] { get }

    /// Lay out backing storage for `contextBucket`, preserving the first
    /// `writtenTokenCount` sequence positions.
    ///
    /// Called before every step with the bucket the engine is about to run.
    /// Returns `true` when storage was actually re-laid-out.
    @discardableResult
    func prepare(contextBucket: Int, writtenTokenCount: Int) throws -> Bool

    /// Insert this handler's states into `views`, sliced to the shapes
    /// `descriptor` declares. States the function doesn't declare are skipped.
    @_lifetime(views: borrow self)
    func bind(
        into views: inout InferenceFunction.MutableViews,
        for descriptor: InferenceFunctionDescriptor
    )

    /// Zero all backing storage and rewind to the initial layout.
    func reset()
}

// MARK: - Shared storage

/// The storage every static state handler owns, plus the two operations that do
/// not vary between them: binding the arrays into a running function's views, and
/// zeroing them.
///
/// A class, and the sole owner of its `NDArray`s, so ``bind(into:for:)`` calls
/// `mutableRawView()` at refcount 1 and does not trigger COW. That ownership is
/// why this is a base class rather than a free function over the dictionary:
/// passing `[String: NDArray]` by value would bump the refcount and copy a
/// multi-hundred-megabyte KV cache on every step.
///
/// Subclasses differ only in how storage is created and re-laid-out — see
/// ``FixedStaticState`` and ``BucketedStaticState``.
public class StaticStateStorage {
    public let stateNames: [String]

    /// Backing storage per state. `var` so `bind` can take a mutable raw view.
    var arrays: [String: NDArray]

    init(stateNames: [String], arrays: [String: NDArray]) {
        self.stateNames = stateNames
        self.arrays = arrays
    }

    /// Insert every state this handler owns into `views`, sliced to the shape the
    /// running function declares. States the function does not declare are skipped.
    ///
    /// The slice is what lets one buffer serve several buckets: a fixed state
    /// allocated at the maximum context binds as the leading `ctx` positions of
    /// itself, with the strides the program was compiled against.
    @_lifetime(views: borrow self)
    public func bind(
        into views: inout InferenceFunction.MutableViews,
        for descriptor: InferenceFunctionDescriptor
    ) {
        for name in stateNames {
            guard case .ndArray(let stateDescriptor) = descriptor.stateDescriptor(of: name) else {
                continue
            }
            let view = _overrideLifetime(
                arrays[name]!.mutableRawView().slice(at: stateDescriptor.shape.map { 0..<$0 }),
                borrowing: Void())
            views.insert(view, for: name)
        }
    }

    /// Zero all backing storage.
    ///
    /// Chunked-flash attention reads *every* key position each step, including
    /// positions past the write cursor, with an additive mask. Garbage there feeds
    /// `q @ k` and can overflow fp16 to `Inf`, which adding `-40000` cannot
    /// suppress, yielding NaN logits.
    public func reset() {
        for name in stateNames { zeroFillNDArray(&arrays[name]!) }
    }
}

// MARK: - Fixed

/// Static state whose backing buffer does not vary with the context bucket.
///
/// Allocated once from a reference (largest-context) descriptor and zero-filled.
/// Covers both the ordinary `key_cache` / `value_cache` of models whose buckets
/// all share one cache buffer, and fixed-size auxiliary caches such as a
/// sliding-window ring.
///
/// A fixed state's declared *shape* may still shrink with the bucket: a model
/// compiled against one max-context cache gives every bucket the max-context
/// strides and declares only the first `ctx` sequence positions. ``bind(into:for:)``
/// slices to whatever the running function declares, so that case needs no
/// re-layout — the storage underneath is already the one the program indexes.
///
/// Zeroing matters: chunked-flash attention reads *every* key position each step
/// — including positions past the write cursor — with an additive mask. Garbage
/// there feeds `q @ k` and can overflow fp16 to `Inf`, which adding `-40000`
/// cannot suppress, yielding NaN logits.
public final class FixedStaticState: StaticStateStorage, StaticStateHandler {
    public init(states: [(name: String, descriptor: NDArrayDescriptor)]) {
        var arrays: [String: NDArray] = [:]
        for (name, descriptor) in states {
            var array = NDArray(descriptor: descriptor)
            zeroFillNDArray(&array)
            arrays[name] = array
            CLILogger.log(
                "Static state '\(name)' allocated: \(descriptor.minimumByteCount) bytes "
                    + "(fixed, shape \(descriptor.shape))")
        }
        super.init(stateNames: states.map(\.name), arrays: arrays)
    }

    /// Nothing to do: one buffer serves every bucket, and `bind` slices it.
    @discardableResult
    public func prepare(contextBucket: Int, writtenTokenCount: Int) throws -> Bool { false }
}

// MARK: - Bucketed

/// Static state whose backing buffer scales with the context bucket.
///
/// Right-sizes: allocates at the session's current bucket rather than the model
/// maximum, and re-lays-out the written prefix when decode crosses into a larger
/// bucket. On a ladder like 4096 / 32768 / 131072 that is the difference between
/// paying for the 131072 cache from the first token and paying for it only once
/// the conversation actually gets there.
///
/// Re-layout is required, not merely an optimization: each bucket's program is
/// compiled with its own sequence stride (`ctx · interleave`), so a buffer laid
/// out for one bucket is not a valid slice of another.
///
/// ``forwardGraph`` selects the smallest bucket greater than the current
/// position, so within a session the bucket is monotonic non-decreasing and this
/// only ever grows. After a reset the first step shrinks it back with nothing to
/// copy.
public final class BucketedStaticState: StaticStateStorage, StaticStateHandler {
    /// Context bucket → per-state descriptor for that bucket.
    private let descriptorsByContext: [Int: [String: NDArrayDescriptor]]
    private let smallestContext: Int

    /// The bucket the backing storage is currently laid out for.
    public private(set) var currentContextBucket: Int

    public enum LayoutError: Error, CustomStringConvertible {
        case noBuckets
        case missingDescriptor(state: String, context: Int)
        case sequenceDimensionNotLast(state: String, context: Int, shape: [Int])
        case interleaveOutsideSequence(state: String, dimension: Int, sequenceDimension: Int)
        case paddedBuffer(state: String, context: Int, expected: Int, actual: Int)
        case unsupportedScalarType(state: String, type: String)

        public var description: String {
            switch self {
            case .noBuckets:
                return "BucketedStaticState needs at least one context bucket"
            case .missingDescriptor(let state, let context):
                return "State '\(state)' is not declared by the ctx \(context) program"
            case .sequenceDimensionNotLast(let state, let context, let shape):
                return
                    "State '\(state)' has shape \(shape) at ctx \(context) — expected the context "
                    + "length as the last dimension, so the written prefix cannot be re-laid-out"
            case .interleaveOutsideSequence(let state, let dimension, let sequenceDimension):
                return
                    "State '\(state)' interleaves dim \(dimension), which is not inside the "
                    + "sequence dim \(sequenceDimension) — prefix re-layout would reorder elements"
            case .paddedBuffer(let state, let context, let expected, let actual):
                return
                    "State '\(state)' at ctx \(context) is padded (\(actual) bytes for \(expected) "
                    + "bytes of elements) — per-group run copy would land at wrong offsets"
            case .unsupportedScalarType(let state, let type):
                return "State '\(state)' has unsupported scalar type \(type) for prefix re-layout"
            }
        }
    }

    /// - Parameters:
    ///   - stateNames: The states this handler owns.
    ///   - descriptorsByContext: Every context bucket's descriptor for each state.
    /// - Throws: ``LayoutError`` when the physical layout cannot support a
    ///   written-prefix re-layout. Callers must not fall back to allocating at
    ///   the maximum bucket — for a state whose stride scales with ctx, that
    ///   binds a buffer the program will index incorrectly.
    public init(stateNames: [String], descriptorsByContext: [Int: [String: NDArrayDescriptor]]) throws {
        guard let smallest = descriptorsByContext.keys.min() else { throw LayoutError.noBuckets }

        for (context, byName) in descriptorsByContext {
            for name in stateNames {
                guard let descriptor = byName[name] else {
                    throw LayoutError.missingDescriptor(state: name, context: context)
                }
                try Self.validateLayout(descriptor, state: name, context: context)
            }
        }

        self.descriptorsByContext = descriptorsByContext
        self.smallestContext = smallest
        self.currentContextBucket = smallest

        var arrays: [String: NDArray] = [:]
        for name in stateNames {
            let descriptor = descriptorsByContext[smallest]![name]!
            var array = NDArray(descriptor: descriptor)
            zeroFillNDArray(&array)
            arrays[name] = array
            CLILogger.log(
                "Static state '\(name)' allocated: \(descriptor.minimumByteCount) bytes "
                    + "(bucketed, starting at ctx \(smallest))")
        }
        super.init(stateNames: stateNames, arrays: arrays)
    }

    // MARK: Layout validation

    private static func byteWidth(_ type: NDArray.ScalarType) -> Int? {
        switch type {
        case .float16, .bfloat16: return 2
        case .float32: return 4
        default: return nil
        }
    }

    private static func validateLayout(
        _ descriptor: NDArrayDescriptor, state: String, context: Int
    ) throws {
        let shape = descriptor.shape
        guard shape.last == context else {
            throw LayoutError.sequenceDimensionNotLast(state: state, context: context, shape: shape)
        }
        let sequenceDimension = shape.count - 1
        if let interleave = descriptor.interleaveLayout, interleave.dimension >= sequenceDimension {
            throw LayoutError.interleaveOutsideSequence(
                state: state, dimension: interleave.dimension,
                sequenceDimension: sequenceDimension)
        }
        guard let width = byteWidth(descriptor.scalarType) else {
            throw LayoutError.unsupportedScalarType(
                state: state, type: String(describing: descriptor.scalarType))
        }
        // copyPrefix reinterprets storage through `LogitsScalarType` while taking
        // strides from the logical shape, so a bucketed state must share that
        // element width. fp16/bf16 are both 2 bytes; fp32 would re-lay-out at the
        // wrong offsets and silently corrupt the carried prefix on a crossing.
        guard width == MemoryLayout<LogitsScalarType>.stride else {
            throw LayoutError.unsupportedScalarType(
                state: state, type: String(describing: descriptor.scalarType))
        }
        let expected = shape.reduce(1, *) * width
        guard descriptor.minimumByteCount == expected else {
            throw LayoutError.paddedBuffer(
                state: state, context: context, expected: expected,
                actual: descriptor.minimumByteCount)
        }
    }

    // MARK: StaticStateHandler

    @discardableResult
    public func prepare(contextBucket: Int, writtenTokenCount: Int) throws -> Bool {
        guard contextBucket != currentContextBucket else { return false }
        guard let byName = descriptorsByContext[contextBucket] else {
            throw LayoutError.missingDescriptor(state: stateNames.first ?? "", context: contextBucket)
        }

        // Clamp to both layouts: growing is the normal path, but a shrink (after
        // a reset that left a cursor behind) must not read or write past either end.
        let copyLength = min(writtenTokenCount, currentContextBucket, contextBucket)

        var totalBytes = 0
        for name in stateNames {
            let descriptor = byName[name]!
            var replacement = NDArray(descriptor: descriptor)
            zeroFillNDArray(&replacement)
            if copyLength > 0 {
                Self.copyPrefix(from: arrays[name]!, to: &replacement, copyLength: copyLength)
            }
            arrays[name] = replacement
            totalBytes += descriptor.minimumByteCount
        }

        CLILogger.log(
            "Static state re-laid-out: ctx \(currentContextBucket) → \(contextBucket) "
                + "(copied \(copyLength) positions, \(totalBytes) bytes across \(stateNames.count) states)")
        currentContextBucket = contextBucket
        return true
    }

    // MARK: Prefix re-layout

    /// Copies the first `copyLength` sequence positions from `source` to
    /// `destination`, re-laying-out for the destination's context length.
    ///
    /// The state is `[…, ctx]` with an optional channel interleave `(dim, factor)`
    /// inside the sequence dim: physically `[…, ctx, factor]` row-major, with the
    /// interleaved elements innermost and the sequence next. So for each
    /// `groupCount = product(shape) / ctx / factor` group, positions
    /// `[0, copyLength)` across the interleaved channels form ONE contiguous run
    /// of `copyLength · factor` elements at group base `g · (ctx · factor)`.
    /// Source and destination share interleave and group order and differ only in
    /// `ctx` — the sequence stride scale — so a per-group run copy is correct
    /// without knowing the interleave details. ``validateLayout`` enforces the
    /// preconditions this relies on.
    static func copyPrefix(from source: NDArray, to destination: inout NDArray, copyLength: Int) {
        let sourceShape = source.shape
        let destinationShape = destination.shape
        let sequenceDimension = sourceShape.count - 1
        let sourceSequence = sourceShape[sequenceDimension]
        let destinationSequence = destinationShape[sequenceDimension]
        let factor = source.interleaveLayout?.factor ?? 1
        precondition(
            copyLength <= sourceSequence && copyLength <= destinationSequence,
            "copyPrefix overflow: \(copyLength) into \(sourceSequence) → \(destinationSequence)")

        let groupCount = sourceShape.reduce(1, *) / sourceSequence / factor
        let sourceGroupStride = sourceSequence * factor
        let destinationGroupStride = destinationSequence * factor
        let runElements = copyLength * factor

        let sourceView = source.view(as: LogitsScalarType.self)
        sourceView.withUnsafePointer { sourcePointer, _, _ in
            let destinationView = destination.mutableView(as: LogitsScalarType.self)
            destinationView.withUnsafeMutablePointer { destinationPointer, _, _ in
                for group in 0..<groupCount {
                    destinationPointer.advanced(by: group * destinationGroupStride)
                        .update(
                            from: sourcePointer.advanced(by: group * sourceGroupStride),
                            count: runElements)
                }
            }
        }
    }
}

// MARK: - Handler set

/// The static states of one asset, split by lifecycle.
///
/// Two concrete slots rather than `[any StaticStateHandler]` because binding is
/// lifetime-dependent: views inserted from a `for` loop escape the loop
/// variable's scope, so the handlers have to be named. Classification produces
/// at most these two groups, so nothing is lost. ``SyncStateHandlerSet`` splits
/// the dynamic engines' states the same way and for the same reason.
public struct StaticStateSet {
    /// States whose backing buffer scales with the context bucket, right-sized per session.
    public let bucketed: BucketedStaticState?
    /// States with one buffer across every bucket, allocated at the maximum.
    public let fixed: FixedStaticState?

    public init(bucketed: BucketedStaticState?, fixed: FixedStaticState?) {
        self.bucketed = bucketed
        self.fixed = fixed
    }

    public var isEmpty: Bool { bucketed == nil && fixed == nil }

    public var stateNames: [String] {
        (bucketed?.stateNames ?? []) + (fixed?.stateNames ?? [])
    }

    /// Lay out every handler for the bucket about to run.
    public func prepare(contextBucket: Int, writtenTokenCount: Int) throws {
        try bucketed?.prepare(contextBucket: contextBucket, writtenTokenCount: writtenTokenCount)
        try fixed?.prepare(contextBucket: contextBucket, writtenTokenCount: writtenTokenCount)
    }

    public func reset() {
        bucketed?.reset()
        fixed?.reset()
    }
}

// MARK: - Factory

/// Builds the ``StaticStateSet`` for a static-shape asset.
///
/// Classification is derived from the asset itself: a state whose *backing
/// buffer* is identical across every context bucket is fixed; a state whose
/// buffer varies with the bucket is bucketed and gets right-sized storage. That
/// gives the right answer with no metadata — models whose buckets all share one
/// cache buffer keep their single max-context allocation, while a per-bucket
/// ladder is right-sized automatically.
///
/// The buffer, not the declared shape, is what decides. A model compiled against
/// one max-context cache declares a *shrinking shape* per bucket (`[…, 256]`,
/// `[…, 512]`, … up to `[…, 4096]`) over unchanging max-context strides and byte
/// count: every bucket views the same allocation, so it is fixed and binding just
/// slices it. Classifying that on shape alone would call it bucketed and try to
/// re-lay-out a buffer that never changes.
///
/// `metadata.json`'s `language.states` block biases the heuristic when a model
/// needs it: `sliding_cache` / `fixed` force fixed. `kv_cache` asserts bucketed
/// rather than forcing it — whether per-bucket storage is representable at all is
/// a physical property of the asset, so declaring it on a state whose buffer does
/// not vary is an export bug and throws at load rather than being honored. Left
/// to force, a stale or copy-pasted entry would reintroduce exactly the
/// misclassification the footprint rule exists to prevent.
public enum StaticStateFactory {
    /// - Parameters:
    ///   - descriptorsByContext: Context bucket → a representative function
    ///     descriptor for that bucket (any query length; states don't vary with it).
    ///   - referenceDescriptor: The largest-context descriptor, used to enumerate
    ///     state names and to size fixed states.
    ///   - stateKinds: Optional explicit classification from bundle metadata.
    public static func makeStateSet(
        descriptorsByContext: [Int: InferenceFunctionDescriptor],
        referenceDescriptor: InferenceFunctionDescriptor,
        stateKinds: [String: StateKind]? = nil
    ) throws -> StaticStateSet {
        let names = referenceDescriptor.stateNames
        guard !names.isEmpty else { return StaticStateSet(bucketed: nil, fixed: nil) }

        var bucketedNames: [String] = []
        var fixedStates: [(name: String, descriptor: NDArrayDescriptor)] = []

        for name in names {
            guard case .ndArray(let reference) = referenceDescriptor.stateDescriptor(of: name) else {
                continue
            }

            // The asset decides: a state that is one buffer across every bucket
            // cannot be bucketed, whatever the metadata says.
            let varies = storageVariesByContext(name: name, descriptorsByContext: descriptorsByContext)
            let isBucketed: Bool
            switch stateKinds?[name] {
            case .slidingCache, .fixed:
                isBucketed = false
            case .kvCache:
                guard varies else {
                    throw InferenceRuntimeError.invalidState(
                        "State '\(name)' is declared `kv_cache` in metadata.json but its buffer is "
                            + "identical across every context bucket (\(reference.minimumByteCount) "
                            + "bytes) — it is one max-context allocation the buckets slice, not a "
                            + "per-bucket ladder. Drop the entry or declare it `fixed`.")
                }
                isBucketed = true
            case nil:
                isBucketed = varies
            }

            if isBucketed {
                bucketedNames.append(name)
            } else {
                fixedStates.append((name, reference))
            }
        }

        var bucketed: BucketedStaticState?
        if !bucketedNames.isEmpty {
            var perContext: [Int: [String: NDArrayDescriptor]] = [:]
            for (context, descriptor) in descriptorsByContext {
                var byName: [String: NDArrayDescriptor] = [:]
                for name in bucketedNames {
                    guard case .ndArray(let stateDescriptor) = descriptor.stateDescriptor(of: name) else {
                        continue
                    }
                    byName[name] = stateDescriptor
                }
                perContext[context] = byName
            }
            CLILogger.log("Bucketed (right-sized) states: \(bucketedNames.sorted())")
            bucketed = try BucketedStaticState(
                stateNames: bucketedNames, descriptorsByContext: perContext)
        }

        var fixed: FixedStaticState?
        if !fixedStates.isEmpty {
            CLILogger.log("Fixed states: \(fixedStates.map(\.name).sorted())")
            fixed = FixedStaticState(states: fixedStates)
        }

        return StaticStateSet(bucketed: bucketed, fixed: fixed)
    }

    /// Whether a state needs its own buffer per context bucket.
    ///
    /// Compares physical layout — byte count and strides — not the declared
    /// shape. A state that is one allocation viewed at a shrinking extent reports
    /// the same layout at every bucket and is therefore fixed.
    private static func storageVariesByContext(
        name: String,
        descriptorsByContext: [Int: InferenceFunctionDescriptor]
    ) -> Bool {
        var layouts: [StorageLayout] = []
        for (_, descriptor) in descriptorsByContext {
            guard case .ndArray(let stateDescriptor) = descriptor.stateDescriptor(of: name) else {
                continue
            }
            layouts.append(
                StorageLayout(
                    byteCount: stateDescriptor.minimumByteCount,
                    strides: stateDescriptor.preferredStrides))
        }
        return footprintVaries(layouts)
    }

    /// One bucket's physical footprint for a state: what it costs and how it is
    /// addressed. Deliberately excludes `shape`, which is the *view* extent the
    /// running program declares and shrinks with the bucket even when the buffer
    /// underneath does not.
    struct StorageLayout: Equatable {
        var byteCount: Int
        var strides: [Int]
    }

    /// The classification rule, over plain footprints.
    ///
    /// Split out from ``storageVariesByContext(name:descriptorsByContext:)``
    /// because `InferenceFunctionDescriptor` cannot be constructed in a test —
    /// which is why this rule went uncovered and regressed. Taking footprints
    /// keeps it pinnable without a model.
    static func footprintVaries(_ layouts: [StorageLayout]) -> Bool {
        guard let first = layouts.first else { return false }
        return layouts.contains { $0 != first }
    }
}

// MARK: - Run helper

/// Run one static-shape step with the asset's states bound.
///
/// Mirrors ``runWithStates`` for the dynamic engines: binding and the `run` call
/// stay in one straight-line scope, and `_unsafeEscapeMutableViews` detaches the
/// lifetime dependency so the views survive the `await`.
func runStaticStep(
    function: InferenceFunction,
    descriptor: InferenceFunctionDescriptor,
    inputs: [String: NDArray],
    states stateSet: StaticStateSet
) async throws -> InferenceFunction.Outputs {
    var states = InferenceFunction.MutableViews()
    stateSet.bucketed?.bind(into: &states, for: descriptor)
    stateSet.fixed?.bind(into: &states, for: descriptor)
    return try await function.run(
        inputs: inputs,
        states: _unsafeEscapeMutableViews(consume states),
        outputViews: InferenceFunction.MutableViews())
}
