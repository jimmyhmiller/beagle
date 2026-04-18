# Porting reset/shift and effect handlers to the x86-64 backend

## Status

x86-64 is broken for anything that captures or resumes a continuation.
Everything that compiles without touching reset/shift/handle/perform
still works: arithmetic, closures, structs, enums, namespaces, exceptions
without resume, the basic stdlib.

Anything that reaches `k(value)` panics with

> "continuation invocation is not implemented on the x86-64 backend"

from `invoke_continuation_runtime` in `src/builtins/continuations.rs`.
That function is the placeholder the ARM64 port left in for the x86-64
generated trampoline in `src/main.rs::compile_continuation_trampolines`
to call. The generated trampoline itself still exists and still calls
into this stub, but it is written against the **v1** reset/shift design
(the one deleted in commits `4ad1ae7`, `fe2463c`, `ea56785`) and has
been dead since then.

The fix is to bring x86-64 up to parity with ARM64 under the post-
rewrite design. This doc describes what that entails.

## Design contract the port must satisfy

The new design lives in `src/builtins/reset_shift.rs`. It is deliberately
architecture-agnostic *except* at four touch points, each of which
assumes an ARM64-shaped stack frame today:

1. **Frame layout.** Every Beagle function's frame header is
   `[fp+0] = saved caller FP`, `[fp+8] = saved caller return address`,
   `[fp-8] = GC prev-header link`. ARM64 achieves this with
   `stp x29, x30, [sp, #-16]!; mov x29, sp`. x86-64 System V happens to
   produce the same header: `push rbp; mov rbp, rsp` puts saved RBP at
   `[rbp+0]` and the `call`-pushed return address at `[rbp+8]`. So the
   header layout is already compatible — **no codegen change required**.

2. **Callee-saved registers.** Captured continuations do not replay
   callee-saved values; the codegen spills any live callee-saveds into
   root slots and restores from there across a `shift`. This is
   architecture-independent and already works on x86-64 (the save-loop
   lives in `src/compiler.rs`). No work here.

3. **The byte copy.** `capture_continuation_runtime_inner` copies
   `[current_sp, outermost_body_fp + 16)` to a heap segment;
   `continuation_restore_on_scratch` copies it back. Pure memory
   operations, no arch specifics. No work here.

4. **The four JIT trampolines.** This is where x86-64 is missing
   parity. They're short ARM64 functions installed in
   `src/main.rs::compile_arm_continuation_return_stub` that the
   Rust-side `continuation_trampoline` / `return_from_shift_*` /
   `pop_top_tag_and_return_stub` functions call by name. Their
   bodies are architecture-specific; their names and ABIs are not.

Everything the ARM64 port does — FP-chain walk, segment relocation,
prompt-tag side stack, `__reset__` frame sentinel — works verbatim on
x86-64 because it's all Rust. The only deliverable is rewriting the
four trampolines.

## The four trampolines to port

All four currently live in `compile_arm_continuation_return_stub` in
`src/main.rs`. The x86-64 versions belong in a parallel
`compile_x86_continuation_return_stub`, dispatched from the same
`#[cfg]` branch in `main` that currently calls
`compile_continuation_trampolines`.

### 1. `beagle.builtin/return-jump`

**What it does.** Unified "set SP/FP/LR and branch" primitive. Called
from `continuation_restore_on_scratch` (final step of a resume), from
`return_from_shift_runtime_inner` (end of a plain `shift`'s body),
from `return_from_shift_tagged_runtime` (end of a perform that returns
a value without resuming), and from `pop_top_tag_and_return_stub`
(tagged-resume stub).

**ARM64 ABI** (source of truth — don't change the signature for
x86-64, match it):

```
X0 = new_sp
X1 = new_fp            -> written into X29
X2 = new_lr            -> written into X30 (unused by all current callers; they pass 0)
X3 = jump_target       -> BR here
X4 = callee_saved_ptr  -> NULL means skip restore
X5 = value             -> placed in X0 immediately before BR
```

**x86-64 equivalent.** Use System V AMD64 ABI: arguments in
`RDI, RSI, RDX, RCX, R8, R9`. Map:

```
RDI = new_sp           -> mov rsp, rdi
RSI = new_fp           -> mov rbp, rsi
RDX = new_lr           -> ignored (zero-valued in practice)
RCX = jump_target      -> jmp rcx
R8  = callee_saved_ptr -> NULL means skip; otherwise restore rbx,r12,r13,r14,r15
R9  = value            -> mov rax, r9 before jmp
```

Do not use `ret` here — the callee-saved pointer is NULL in every
current use and there is no return address on the stack to pop.
Unconditional `jmp rcx` is correct.

The already-existing `return-jump` generated in
`compile_continuation_trampolines` does something *similar* but takes
7 arguments including a "frame_src + frame_size" byte-copy loop that
belongs to v1 semantics. Replace with a 6-arg version that matches the
ARM ABI above.

### 2. `beagle.builtin/read-fp`

**What it does.** Returns the caller's frame pointer. Two instructions
on ARM (`mov x0, x29; ret`). Used by `continuation_trampoline` to
discover `trampoline_fp` without depending on Rust's own frame layout.

**x86-64 equivalent.**

```
mov rax, rbp
ret
```

Two bytes plus the ret. Trivial.

### 3. `beagle.builtin/read-sp`  and  `beagle.builtin/read-sp-fp`

`read-sp` returns `sp` in `x0`/`rax`. `read-sp-fp` returns
`(sp, fp)` in `(x0, x1)` / `(rax, rdx)`.

Note the ABI asymmetry: ARM can return a second value in `X1` without
ceremony. On x86-64 System V, a struct return of two 8-byte
pointer-sized values is returned in `rax` and `rdx`. The one caller
in `reset_shift.rs` that uses `read-sp-fp` is on ARM-only paths
today; if x86-64 ends up needing it, match System V struct-return
convention.

**x86-64 `read-sp`.** RSP on entry still points at the return address
(since `call` pushed it), so subtract 8 to get the caller's SP:

```
lea rax, [rsp + 8]
ret
```

**x86-64 `read-sp-fp`.**

```
lea rax, [rsp + 8]
mov rdx, rbp
ret
```

### 4. `beagle.builtin/stack-switch`

**What it does.** Switch to a different stack, then tail-call a target.
Used by `continuation_trampoline` to flip onto the per-thread
`CONTINUATION_SCRATCH_STACK` before performing the byte copy — the
copy would otherwise overlap Rust's own stack frame.

**ARM64 ABI.** `X0 = new_stack_top`, `X1 = target_fn`. Sets `SP = X0`,
then `BLR X1`, then `BRK` (target never returns).

**x86-64 equivalent.**

```
mov rsp, rdi
call rsi        ; preserves System V 16-byte alignment: new stack_top is 16-aligned,
                ; call pushes 8-byte return address, so callee sees unaligned sp
int3
```

One subtlety: the scratch stack's top is computed in Rust as
`stack.as_mut_ptr() + stack.len()` masked to 16. On x86-64, after
`mov rsp, rdi; call rsi`, RSP is `new_top - 8`, which is *mis-aligned*
for System V (callees assume `rsp % 16 == 8` on entry, i.e. 16-aligned
*after* the call pushes the return address). So
`stack_switch(scratch_top, ...)` needs `scratch_top` to already be
16-aligned; the ARM64 caller already masks it with `& !0xf`. Good.

The sole target of `stack-switch` is `continuation_restore_on_scratch`
in Rust, which gets an `extern "C" fn(usize, usize) -> !` signature.
Its first two args are dummies on both architectures — the real
state travels through `CONTINUATION_RESTORE_PLAN` TLS. That handoff
is architecture-independent.

## x86-64-specific hazards

The ARM port has a small set of comments in `reset_shift.rs` about
LLVM inliner decisions. Each one gets easier on x86-64, harder, or
re-appears differently. Tagging the ones the porter should watch:

### (a) Duplicated `capture_continuation` body

`reset_shift.rs:~730` says:

> Factoring the body out into a helper changes Rust's call-stack
> shape enough that `save_gc_context!` reads the wrong saved LR.

This is about `save_gc_context!` (in `src/builtins/mod.rs`) reading
the caller's LR through a compile-time-known offset. On ARM64, the
offset assumes the Rust frame has one `stp x29, x30` at its top.
Inlining more code changes what ends up at that offset. The same
concern applies on x86-64: `save_gc_context!` will need a matching
x86-64 implementation. Grep `save_gc_context` — its expansion is
already arch-gated, but the *assumptions* are not fully documented;
verify the offsets on x86-64 before trusting any helper extraction.

### (b) `LocalKey::with` can generate out-of-line calls

`reset_shift.rs:~1070` explains why `gc_frame_top_slot` is resolved
*before* the no-call critical section. `LocalKey::with` on macOS ARM64
is usually a direct TLS access (via
`__thread_local_variable_accessor`) but may be emitted as a function
call depending on the optimizer's mood. If a call happens, it pushes
a frame into `dst` and corrupts the copy in progress.

This concern transfers to x86-64 with a wrinkle: macOS x86-64 TLS
uses a helper call (`tlv_get_addr`) by design — it is *always*
out-of-line. Caching `GC_FRAME_TOP.with(|cell| cell.as_ptr())` before
the critical section is mandatory on x86-64, not merely prudent. The
existing code already does this; the x86-64 port just has to preserve
the property.

### (c) Red zone

System V AMD64 reserves a 128-byte **red zone** below RSP that
leaf functions can use without adjusting RSP. `continuation_trampoline`
writes to addresses *above* its own RBP (overlaying the invoker's
frame area), not below RSP, so the red zone doesn't affect the copy
itself. But
`continuation_restore_on_scratch` — which runs on the scratch stack —
might. Be cautious: if the compiler emits a leaf that uses the red
zone for a local in the critical section, and then you switch
stacks, the red-zone-relative local is gone. The mitigation is the
same as on ARM: no Rust function calls inside the write-then-jump
critical section. The current design already ensures this.

### (d) Frame-pointer omission

Release builds of Beagle pass `-C force-frame-pointers` (check
`.cargo/config.toml` and `Cargo.toml` profile settings). The trampoline
reads `rbp` via `read-fp`; if rustc omits the frame pointer for
`continuation_trampoline`, `read-fp` returns the *caller's* frame
pointer, not the trampoline's. The fix is either `#[inline(never)]`
plus `-Cforce-frame-pointers=yes` scoped to this crate (already set
for ARM64 today), or explicitly stash RBP on entry. Verify the
rustc flags on x86-64 match what ARM64 assumes.

## Recommended order of work

1. **Smoke test.** Move `invoke_continuation_runtime` aside — turn the
   whole `compile_continuation_trampolines` function into a `todo!()`
   body and confirm the x86-64 cross-compile at least links and runs
   non-continuation tests.

2. **Port `read-fp`, `read-sp`, `read-sp-fp`** first. Trivial,
   verifiable in isolation with a small test that calls them from
   Rust and asserts the results.

3. **Port `return-jump`** with the 6-arg ARM signature. Delete the
   v1 7-arg generator in `compile_continuation_trampolines`.

4. **Port `stack-switch`.** Also small. Verify by writing a test that
   switches stacks, runs a scratch function, and `return-jump`s back.

5. **Delete `invoke_continuation_runtime` and
   `compile_continuation_trampolines`** entirely. The x86-64 backend
   should now use the same pure-Rust `continuation_trampoline` as
   ARM64, reached via the `beagle.builtin/continuation-trampoline`
   builtin registered in `install.rs`. No per-arch trampoline
   generation required.

6. **Run the full test suite** under
   `cargo test-x86` (alias in `.cargo/config.toml`). Expect most
   reset/shift and handle/perform tests to pass; anything that fails
   is almost certainly one of hazards (a)–(d). Hit them one at a
   time.

7. **Delete this document** once x86-64 is at parity. Leave a
   one-line note in `docs/continuation-approaches.md` that both
   backends use the same Rust trampoline and differ only in the
   four JIT helpers.

## What you can skip

- **Nothing in `capture_continuation_runtime_inner`** needs porting.
  The ARM-specific references in that function are already generic
  (they use the frame-header layout described in §1, which is the
  same on x86-64 under System V).

- **Nothing in the effect-handler dispatch path** needs porting.
  `perform_dispatch_and_return_runtime` and the
  `pop_top_tag_and_return_stub` logic are all Rust; they call
  `return-jump` but don't contain any ARM assembly themselves.

- **The GC stack walker** (`src/gc/stack_walker.rs`) is already
  architecture-independent — it walks the GC prev chain, which is
  set up the same way on both architectures.

## Rough estimate

With all four trampolines being < 20 instructions each and the Rust
side unchanged, the mechanical port is a day of work. The fiddly part
is hazards (a)–(d), which can each absorb a day of their own if the
reproducer is a multishot-under-gc-stress failure. Budget 3–5 days
for confident parity including test-suite green.
