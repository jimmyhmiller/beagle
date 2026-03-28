# Raylib Game Crash Report: Struct Redefinition + Phantom Field Arithmetic

## Summary

Hot-reloading a struct definition with new fields, then redefining a function to do **arithmetic or comparison** on those new fields, causes a `TypeError` crash. The root cause is that old struct instances (allocated with fewer fields) are read past their bounds when accessing the new phantom fields, and the adjacent heap memory contains values with unexpected type tags.

## Error

```
Uncaught exception:
SystemError.TypeError { message: "Type mismatch in arithmetic operation. To mix integers and floats, use to-float() to convert integers: e.g., 3.14 * to-float(2)", location: null }
Stack trace:
  at raylib_game/game-loop (raylib_game.bg:151)
  at raylib_game/main (raylib_game.bg:162)
```

## Root Cause Analysis

1. **Old struct instance on the heap**: The running game has a `Game` value with 5 fields (`player_x, balls, score, missed, frame`), allocated as a 5-slot object.

2. **Struct redefined to 7 fields**: Adding `bullets` and `shoot_cooldown` mutates the struct definition in-place (same struct ID). The runtime now believes all `Game` instances have 7 fields.

3. **Phantom field access reads past allocation**: Accessing `game.shoot_cooldown` (index 6) on the old 5-slot allocation reads whatever is in adjacent heap memory.

4. **In calm REPL conditions**: The memory beyond the struct is typically zeroed, so phantom fields return `0` (tagged as `Int`). Arithmetic on `0` works fine — **this is why the bug doesn't reproduce in simple REPL tests**.

5. **In the active game loop**: The heap is under constant allocation pressure (Ball structs, lists, raylib state). The memory adjacent to the old Game struct contains arbitrary tagged values (could be floats, strings, lists, struct pointers, etc.).

6. **Arithmetic type guard fails**: The JIT-compiled `>` or `-` operator checks the type tag of both operands. When the phantom field's memory has a non-integer tag, the guard jumps to `throw_type_error`.

## Key Finding: Assignment vs Arithmetic

- `let sc = game.shoot_cooldown` → **No crash** (assignment doesn't check type tags)
- `game.shoot_cooldown > 0` → **Crash** (comparison checks that both tags match)
- `game.shoot_cooldown - 1` → **Crash** (arithmetic checks tag is int or float)

This is why the error message says "Type mismatch in arithmetic" rather than "field not found" — the field access *succeeds* (reading garbage), and the arithmetic *fails* on the garbage value.

## Steps to Reproduce

### Minimal repro (crashes every time)

1. Start the raylib game:
   ```
   beagle_run /Users/jimmyhmiller/Documents/Code/beagle/examples/raylib_game.bg
   ```

2. Redefine the Game struct with extra fields (game keeps running fine):
   ```
   struct Game {
       player_x
       balls
       score
       missed
       frame
       bullets          // new
       shoot_cooldown   // new
   }
   ```

3. Redefine `update-player` to do **arithmetic on the new field** (crashes):
   ```
   fn update-player(game) {
       let dx = if IsKeyDown(KEY_LEFT) != 0 { 0 - SPEED } else { 0 }
       let dx = if IsKeyDown(KEY_RIGHT) != 0 { dx + SPEED } else { dx }
       let new_x = game.player_x + dx
       let clamped = if new_x < 0 { 0 }
           else if new_x + PLAYER_W > WIDTH { WIDTH - PLAYER_W }
           else { new_x }
       let new_cooldown = if game.shoot_cooldown > 0 { game.shoot_cooldown - 1 } else { 0 }
       Game { ...game, player_x: clamped, shoot_cooldown: new_cooldown }
   }
   ```

### What does NOT crash

- Redefining the struct alone (step 2 only) — the old functions construct 7-field Games with phantom 0s, which works
- Redefining `update-player` identically (no new field access) — no out-of-bounds read
- Reading a phantom field without arithmetic: `let sc = game.shoot_cooldown` — assignment doesn't type-check

### What DOES crash

- Any arithmetic or comparison on a phantom field: `game.shoot_cooldown > 0`, `game.shoot_cooldown - 1`
- The crash is **non-deterministic in its exact trigger point** — it depends on what happens to be in heap memory adjacent to the old struct allocation

## Possible Fixes

1. **Bounds-check on field access**: When accessing field index N on a struct, verify N < the instance's actual allocated slot count. If out of bounds, return a proper default (null/0) or throw a clear "field not found on old instance" error.

2. **Struct versioning**: Track a version/generation on each struct instance. When the struct definition changes, detect stale instances and either migrate them (copy + extend with defaults) or reject field access with a clear error.

3. **Better error message**: At minimum, detect the condition and throw something like: `"Field 'shoot_cooldown' does not exist on this Game instance (struct was redefined — restart required)"` instead of the misleading arithmetic type mismatch.
