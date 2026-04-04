# Fixed Root Area GC Design

## Goal

Move Beagle to a GC/root model where stack scanning never consults stack maps.

Instead:

- every Beagle frame contains an embedded GC-linked-list node
- that node points to the previous frame's node
- that node records a fixed `root_count`
- the frame contains a contiguous root area immediately associated with that node
- GC follows the linked list and scans exactly `root_count` slots per frame

This must work for:

- active stack scanning
- detached continuation-segment scanning
- continuation capture/invoke
- moving GC pointer updates

## Desired Invariant

At every GC safepoint, every live GC-visible Beagle value for a frame must already be stored in that frame's fixed root area.

That includes:

- user locals
- temporary intermediate values that can survive a call, allocation, safepoint, continuation capture, or resume
- values currently in registers if they are live across a safepoint

That excludes:

- machine scratch state
- callee-saved register spill buffers used only for builtin marshalling
- non-GC native temporaries
- any stack scratch area not intentionally part of the frame's root window

If this invariant holds, GC only needs:

1. the linked-list head
2. `next`
3. `root_count`
4. the root area base

No return-address lookup is needed for scanning.

## What The Runtime Does Today

Today the implementation is mixed:

- GC already finds frames by following the embedded GC linked list rooted at `GC_FRAME_TOP`.
- frame headers currently encode an upper bound on traced slots
- detached continuation scans still narrow that slot count using stack-map metadata
- stack-map metadata currently distinguishes semantic locals from other frame storage

This is better than raw stack walking, but it is still not the desired end state because "which slots are roots?" is partly answered by compile-time stack maps rather than by the frame's own fixed root layout.

## End State Model

Each compiled Beagle frame should have a layout conceptually like:

```text
[FP+8]   saved return address
[FP+0]   saved caller FP
[FP-8]   frame header / gc metadata
[FP-16]  prev gc frame pointer
[FP-24]  root[0]
[FP-32]  root[1]
...
[FP-N]   root[root_count-1]
         non-root frame storage below this point, if any
```

The exact encoding can vary, but these properties must hold:

- `root_count` is fixed for the function's frame shape
- the root area is contiguous
- the root area is the only stack region GC scans for that frame
- continuation capture copies this exact root area as part of the frame bytes
- continuation resume restores it verbatim

## Compiler Consequences

This design is mostly a compiler/codegen change, not just a GC change.

### 1. Distinguish Root Slots From Non-Root Slots

Today a frame effectively mixes together:

- semantic locals
- live eval-stack values
- regalloc spill slots
- special-purpose save/marshalling slots

The new design requires a hard separation:

- root slots: always GC-scanned
- non-root slots: never GC-scanned

This means the compiler/backend must stop treating generic frame slots as one interchangeable pool.

### 2. Give Every GC-Visible Temporary a Home in the Root Area

Any Beagle value that can be live across a safepoint must be materialized into a root slot before the safepoint.

That includes:

- call arguments that remain live after the call
- intermediate expression values that survive allocation
- values live across continuation capture and perform/resume paths
- closure/environment values if represented on the stack during calls

The key rule is:

- a live GC value may exist in registers for execution efficiency
- but if a safepoint can happen, the value must also already have an authoritative home in the root area

### 3. Remove GC Semantics From Regalloc Spill Slots

Regalloc spills should become purely non-root storage unless the compiler explicitly chooses to place a value in a root slot.

That means:

- the allocator cannot silently create new GC-scanned stack slots by spilling
- "spill" and "root home" must become separate concepts

Possible implementation approaches:

- reserve fixed root homes per IR value class and let regalloc use non-root spill space separately
- lower all GC-visible values to explicit root-slot loads/stores around safepoints
- keep SSA/register allocation as-is for execution, but require safepoint lowering to synchronize all live roots into their root homes

### 4. Remove Dependence On `current_stack_size`

The current detached-scan model uses `number_of_locals + current_stack_size`.

In the desired model, there is no dynamic "live eval stack" region for GC scanning.

Instead:

- temporary eval-stack values that matter to GC are already in root slots
- `root_count` is fixed
- scanning does not depend on the return address or current machine position within the function

### 5. Make Safepoints a Synchronization Problem, Not a Metadata Lookup Problem

Today stack maps answer: "which temporary stack values are live at this return address?"

In the target model, the compiler must instead guarantee:

- before any safepoint, the fixed root area is already authoritative

That shifts complexity from GC metadata lookup into code generation and safepoint lowering.

## Runtime / GC Consequences

### 1. Active Stack Scanning Becomes Simpler

The active scanner should do:

1. read current frame header
2. read `prev`
3. scan `root_count` contiguous slots
4. continue

No stack maps.
No return-address lookup.
No special live-slot reconstruction.

### 2. Detached Continuation Segment Scanning Also Becomes Simpler

Detached continuation scanning should do the same thing on copied frames:

1. read copied frame header
2. follow copied `prev`
3. scan `root_count`

This is one of the main benefits of the design. Detached frames no longer need reconstruction logic based on resume-site metadata.

### 3. Pointer Updating For Moving GC Becomes More Direct

Updating pointers inside continuation segments becomes:

- follow copied frame chain
- update the fixed root windows

The GC no longer needs to know which safepoint within the function the frame corresponds to.

### 4. Continuation Capture/Resume Semantics Improve

Because the fixed root area is part of the frame bytes:

- capture copies the exact root state
- resume restores the exact root state
- there is no separate notion of "live slots inferred from stack map"

This is a better fit for heap-backed continuation segments.

## Backend / Frame Layout Changes

The current frame header uses metadata derived from "locals + eval stack" and is interpreted differently in different contexts.

For the fixed-root-area model, the frame metadata should instead directly encode:

- `root_count`
- any optional frame flags
- optionally any continuation-mark metadata or other per-frame metadata already encoded in the header

The backend then needs two separate layout notions:

- root window size
- total physical frame size

These are not the same thing.

For example:

- root window might be 12 slots
- total frame might be 24 slots because of native scratch space, spills, alignment, saved registers, or temporary call setup

Only the first 12 slots are GC-visible.

## Migration Plan

This should be done in phases.

### Phase 1: Define the Frame Contract Explicitly

Document and enforce:

- where the root area starts
- how many slots it contains
- which storage classes are allowed inside it
- which storage classes are forbidden inside it

Add assertions/debug checks around frame construction and detached scanning.

### Phase 2: Split Root Slots From Scratch Slots In Codegen

Teach the backend and IR lowering that:

- root slots are a fixed window
- spill slots are separate
- builtin marshalling space is separate

This is the phase where most accidental GC coupling should be removed.

### Phase 3: Synchronize Live GC Values Into Root Homes At Safepoints

Before calls, allocations, `perform`, continuation capture, and other safepoints:

- all live GC values must be written to their root homes

This likely needs explicit compiler machinery.

### Phase 4: Stop Recording/Using Stack Maps For Scanning

Once the compiler invariant is strong enough:

- active scanning should ignore stack maps
- detached scanning should ignore stack maps
- continuation-segment pointer updates should ignore stack maps

At this point stack maps may still exist for debugging or other uses, but not for GC root discovery.

### Phase 5: Simplify Continuation Scanning Code

After stack maps are no longer used for scanning:

- remove detached-frame live-slot reconstruction
- collapse active and detached scanners toward the same algorithm
- remove special-case logic that exists only to reconcile frame layout with stack-map liveness

### Phase 6: Remove Obsolete Metadata Paths

Delete or demote:

- stack-map-dependent slot-count logic for GC scanning
- code paths that interpret return addresses to discover live roots
- any frame-header semantics that only exist to support the old mixed model

## Important Design Questions

These need to be decided before implementation goes far.

### 1. Is `root_count` Truly Constant Per Function?

If yes:

- simplest scanning model
- easier continuation capture
- more stack space used

If no:

- the design drifts back toward runtime liveness metadata

The desired end state described in this document assumes "yes."

### 2. How Are Temporaries Assigned Root Homes?

Candidates:

- every IR temporary that can survive a safepoint gets a dedicated root home
- a safepoint-aware liveness pass assigns reusable root homes
- root homes are per-basic-block or per-safepoint but still within a fixed frame window

This is the biggest compiler-design choice.

### 3. How Aggressively Should Registers Mirror Root Homes?

Possible rule:

- registers are only performance caches of values already stored in root slots

That is the simplest correctness model, but it may increase stores.

### 4. What Happens To Eval-Stack-Based IR Assumptions?

If the current IR/backend model relies on an eval-stack-like transient slot region, that model must either:

- keep existing execution behavior but mirror GC-visible values into the root window
- or be redesigned so GC-visible temporaries are root-slot-based from the start

## Risks

### 1. Performance Regressions

Always homing live GC values into root slots can increase:

- stores before calls
- loads after calls
- frame size
- register pressure

This should be expected and measured.

### 2. Partial Migration Is Dangerous

The worst state is a hybrid where:

- some live values only live in scratch space
- scanning assumes the root window is authoritative

That would silently lose roots.

### 3. Builtin and Continuation Paths Are the Sharpest Edges

`perform`, capture, resume, exception handling, and async builtins are where:

- values cross safepoints frequently
- stack state is copied or reconstructed
- GC interactions are most timing-sensitive

Those paths should be treated as first-class validation targets during migration.

## Validation Strategy

The migration needs tests that specifically prove the invariant rather than only testing end programs.

Useful checks:

- assert in debug builds that detached scanning never consults stack maps
- assert that continuation-segment scanners only scan the fixed root window
- add compiler/backend debug dumps that show root-window size separately from total frame size
- add tests where live values exist in registers right before `perform`/capture/calls and verify GC still updates them correctly
- add tests where scratch spill slots contain stale heap-looking values and verify GC ignores them
- keep existing `--gc-always` continuation and async stress programs in the validation set

## Suggested Intermediate Debugging Aids

While migrating, it would help to have:

- a per-function dump of `root_count` vs total frame slots
- a debug mode that poisons non-root scratch slots with heap-looking garbage
- a debug mode that asserts every live GC value at a safepoint has an assigned root home
- continuation-segment dumps that annotate which slots are roots vs non-roots

## Summary

The fixed-root-area design is viable and simpler than the current stack-map-assisted model, but it requires a stronger compiler invariant:

- the frame itself must fully describe its GC roots
- every live GC-visible value must already live in the frame's fixed root area at every safepoint
- GC scanning should become pure linked-list traversal plus fixed-width root scanning

The main work is not in the collector. The main work is making the compiler and frame layout honest enough that the collector can become simple.
