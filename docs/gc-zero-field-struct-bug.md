# GC "Zero-Field Struct" Bug

## Summary

The root cause was not a special-case bug in the generational copier for 8-byte structs.

The real bug was that the active effect-handler registry stored a raw heap pointer to the handler object in thread-local state, but that pointer was not itself a GC root. Under `// gc-always`, the handler could be moved or reclaimed while still logically installed. Later `perform`/`resume` paths then read a stale pointer from the registry and treated whatever now lived at that address as the handler.

That failure showed up most reliably with `struct H {}` because zero-field handlers are the smallest possible heap objects, so `gc-always` made their allocation, promotion, and address reuse pattern much tighter. The object size was a trigger, not the root cause.

## What Was Wrong

Before the fix, the handler registry entry was:

- `protocol_key: String`
- `handler_instance: usize`
- `tag: u64`

That meant `push_handler_builtin` saved the tagged handler pointer directly in TLS, and `find_handler_builtin` returned that same raw value later. The GC never traced or rewrote that TLS field.

Relevant pre-fix code:

- `src/builtins/effects.rs` from the parent of `44088d7`
  - `push_handler_builtin` stored `handler_instance` directly
  - `find_handler_builtin` returned `e.handler_instance`
- `src/runtime.rs` from the parent of `44088d7`
  - `HandlerRegistryEntry` contained `handler_instance: usize`

## Failure Chain

1. `handle ... with h` pushed `h` into the thread-local handler registry.
2. The registry kept only the raw tagged heap pointer.
3. A GC ran while the handler was still installed.
4. The collector moved or reclaimed the handler object, but the registry entry was unchanged because the GC did not scan that TLS pointer.
5. A later `perform` looked up the handler and got the stale address.
6. Dispatch/resume then executed against corrupted or unrelated heap data, which produced the "zero-field struct" crashes and silent exits.

This also explains the old observations:

- `mark-and-sweep` often passed because the object was not moved, so the stale pointer still happened to be usable.
- Adding a field often passed because the larger object changed reuse timing and made the stale-pointer race harder to hit.
- `// gc-always` made the bug deterministic because every allocation stressed the untraced registry entry.

## The Fix

The handler registry now stores a GC root slot id instead of the raw pointer.

Code:

- [src/builtins/effects.rs](/Users/jimmyhmiller/Documents/Code/beagle/src/builtins/effects.rs:24)
  - `push_handler_builtin` calls `register_temporary_root(handler_instance)`
  - `pop_handler_builtin` calls `unregister_temporary_root(...)`
  - `find_handler_builtin` reads the current pointer with `peek_temporary_root(...)`
- [src/runtime.rs](/Users/jimmyhmiller/Documents/Code/beagle/src/runtime.rs:3655)
  - `HandlerRegistryEntry` now stores `handler_root_id`
- [src/runtime.rs](/Users/jimmyhmiller/Documents/Code/beagle/src/runtime.rs:5107)
  - temporary roots are backed by handle-root slots
- [src/runtime.rs](/Users/jimmyhmiller/Documents/Code/beagle/src/runtime.rs:5433)
  - handle roots live in the per-thread `GlobalObjectBlock` chain that the GC already traces and updates

After this change, the GC rewrites the root slot when the handler moves, and handler lookup always returns the current address.

## Verification

The previously representative zero-field handler repro now passes:

```bash
cargo run --release -- test resources/gc_handler_minimal_test.bg
```

Current result in this workspace:

```text
pass  resources/gc_handler_minimal_test.bg
```

## Scope

This explains the reproducible zero-field handler failure that originally motivated this document.

It does **not** explain the remaining `gc_frame_chain_continuation_test.bg` / `gc_frame_chain_multishot_test.bg` continuation-chain crashes. Those still fail in the current workspace and are a separate GC/continuation interaction.
