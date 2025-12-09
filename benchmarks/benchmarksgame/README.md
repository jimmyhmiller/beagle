# Benchmarks Game Implementation for Beagle

This directory contains Beagle implementations of the [Computer Language Benchmarks Game](https://benchmarksgame-team.pages.debian.net/benchmarksgame/index.html) benchmarks.

## Overview

The Benchmarks Game is a well-known collection of micro-benchmarks used to compare programming language performance. We're implementing single-threaded versions of each benchmark to measure Beagle's performance against other languages.

## Benchmark Status

| Benchmark | Status | Notes |
|-----------|--------|-------|
| [binary-trees](#binary-trees) | Ready | Existing implementation needs output format adjustment |
| [fannkuch-redux](#fannkuch-redux) | Ready | Feasible with current language features |
| [fasta](#fasta) | Ready | Feasible with current language features |
| [n-body](#n-body) | Blocked | Requires `sqrt()` function |
| [spectral-norm](#spectral-norm) | Blocked | Requires `sqrt()` function |
| [mandelbrot](#mandelbrot) | Ready | Feasible with current language features |
| [pidigits](#pidigits) | Blocked | Requires arbitrary precision integers |
| [k-nucleotide](#k-nucleotide) | Partial | Requires hash map (persistent_map available) |
| [reverse-complement](#reverse-complement) | Ready | Feasible with current language features |
| [regex-redux](#regex-redux) | Blocked | Requires regex library |

## Benchmark Descriptions

### binary-trees

**Status:** Ready

Allocate and deallocate many binary trees to stress test memory allocation and garbage collection.

**Algorithm:**
1. Create a "stretch" tree of depth N+1 and compute its checksum
2. Create a long-lived tree of depth N
3. For depths 4 to N (step 2), create many trees and sum their checksums
4. Output the long-lived tree's checksum

**Expected Output (N=10):**
```
stretch tree of depth 11	 check: 4095
1024	 trees of depth 4	 check: 31744
256	 trees of depth 6	 check: 32512
64	 trees of depth 8	 check: 32704
16	 trees of depth 10	 check: 32752
long lived tree of depth 10	 check: 2047
```

**Beagle Features Required:** structs, recursion, basic arithmetic

---

### fannkuch-redux

**Status:** Ready

Count permutations and track maximum "pancake flips" (reversing prefixes).

**Algorithm:**
1. Generate all permutations of [1..N]
2. For each permutation, count flips needed to bring 1 to front
3. Track maximum flips and alternating sum of flip counts

**Expected Output (N=7):**
```
228
Pfannkuchen(7) = 16
```

**Beagle Features Required:** arrays, while loops, array mutation

---

### fasta

**Status:** Ready

Generate DNA sequences using repeat and random selection algorithms.

**Algorithm:**
1. Output a repeated sequence (ALU) for "ONE"
2. Output random nucleotides with IUB frequencies for "TWO"
3. Output random nucleotides with human frequencies for "THREE"

**Expected Output (N=1000):** FASTA format DNA sequences (~5000 characters)

**Beagle Features Required:** arrays, string concatenation, modular arithmetic, floating-point

---

### n-body

**Status:** Blocked - needs `sqrt()`

Simulate the solar system using Newtonian physics.

**Algorithm:**
1. Initialize 5 bodies (Sun, Jupiter, Saturn, Uranus, Neptune)
2. Offset momentum to center of mass
3. Advance N timesteps using symplectic integrator
4. Output initial and final system energy

**Expected Output (N=1000):**
```
-0.169075164
-0.169087605
```

**Beagle Features Required:** structs, floating-point math, `sqrt()` function

**Blocker:** Beagle doesn't have a built-in `sqrt()` function. Options:
- Implement via FFI (call C's `sqrt`)
- Implement Newton-Raphson square root in pure Beagle

---

### spectral-norm

**Status:** Blocked - needs `sqrt()`

Calculate an eigenvalue using the power method.

**Algorithm:**
1. Initialize vector u = [1, 1, ..., 1]
2. Repeatedly multiply by matrix A and its transpose
3. Compute sqrt(vBv / vv) as the spectral norm

**Expected Output (N=100):**
```
1.274219991
```

**Beagle Features Required:** arrays, floating-point math, `sqrt()` function

**Blocker:** Same as n-body - needs `sqrt()`

---

### mandelbrot

**Status:** Ready

Generate a Mandelbrot set image in PBM format.

**Algorithm:**
1. For each pixel, iterate z = z² + c until escape or max iterations
2. Pack 8 pixels into each byte
3. Output as binary PBM image

**Expected Output (N=200):** Binary PBM image (200x200 pixels)

**Beagle Features Required:** floating-point math, bit operations, binary output

**Note:** May need to output as text-based representation if binary output is problematic

---

### pidigits

**Status:** Blocked - needs arbitrary precision integers

Calculate digits of pi using a streaming spigot algorithm.

**Algorithm:**
1. Use the unbounded spigot algorithm with matrix transformations
2. Extract digits one at a time
3. Output in groups of 10

**Expected Output (N=27):**
```
3141592653	:10
5897932384	:20
6264338   	:27
```

**Beagle Features Required:** Arbitrary precision integer arithmetic (bigint)

**Blocker:** Beagle only has 62-bit integers. This benchmark fundamentally requires unbounded integers for the matrix multiplication.

---

### k-nucleotide

**Status:** Partial

Count occurrences of nucleotide subsequences in DNA.

**Algorithm:**
1. Read DNA sequence from input
2. Count all 1-mers and 2-mers, output by frequency
3. Count specific sequences (GGT, GGTA, etc.)

**Expected Output:** Frequency tables and specific sequence counts

**Beagle Features Required:** hash maps, string slicing, sorting

**Note:** Beagle has `persistent_map` which should work, but may need additional features like iteration over map entries.

---

### reverse-complement

**Status:** Ready

Compute the reverse complement of DNA sequences.

**Algorithm:**
1. Read FASTA sequences
2. Reverse each sequence
3. Complement each nucleotide (A↔T, G↔C, etc.)
4. Output in FASTA format

**Beagle Features Required:** string manipulation, character mapping

---

### regex-redux

**Status:** Blocked - needs regex library

Count regex pattern matches and perform substitutions.

**Algorithm:**
1. Read DNA sequence
2. Count matches for 9 specific patterns
3. Perform 5 substitutions
4. Output match counts and sequence lengths

**Beagle Features Required:** Regular expression library

**Blocker:** Beagle doesn't have regex support. Would need to either:
- Implement regex via FFI
- Write a regex engine in Beagle (complex)

---

## Missing Language Features

To implement all benchmarks, Beagle needs:

### High Priority (blocks multiple benchmarks)

1. **`sqrt()` function** - Blocks n-body and spectral-norm
   - Option A: Add as builtin (call libc sqrt)
   - Option B: Implement Newton-Raphson in Beagle
   - Option C: Add FFI wrapper for libm

### Medium Priority

2. **Arbitrary precision integers** - Blocks pidigits
   - Would need a bigint library implementation

3. **Regex support** - Blocks regex-redux
   - Could use FFI to wrap PCRE or similar

### Low Priority (workarounds exist)

4. **Binary file output** - For mandelbrot
   - Can output text-based representation instead

5. **stdin reading** - For k-nucleotide, reverse-complement, regex-redux
   - Can use file input as alternative

## Running Benchmarks

```bash
# Run a specific benchmark
cargo run -- benchmarks/benchmarksgame/binary_trees.bg

# Run with timing
cargo run -- --show-times benchmarks/benchmarksgame/binary_trees.bg

# Compare with reference implementation
ruby /path/to/benchmarksgame-sourcecode/binarytrees/binarytrees.ruby 10
```

## Reference Implementations

Reference implementations for comparison are available at:
- Official repo: https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
- Local copy: `/Users/jimmyhmiller/Downloads/benchmarksgame-sourcecode/`

## Performance Goals

Based on Beagle's current performance characteristics:
- Target: 2x Ruby performance (already achieved for some workloads)
- Stretch goal: Within 50% of Node.js performance
