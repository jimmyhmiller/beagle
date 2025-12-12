# Benchmarks Game Implementation for Beagle

This directory contains Beagle implementations of the [Computer Language Benchmarks Game](https://benchmarksgame-team.pages.debian.net/benchmarksgame/index.html) benchmarks.

## Overview

The Benchmarks Game is a well-known collection of micro-benchmarks used to compare programming language performance. We're implementing single-threaded versions of each benchmark to measure Beagle's performance against other languages.

## Benchmark Status

| Benchmark | Status | Notes |
|-----------|--------|-------|
| [binary-trees](#binary-trees) | ✅ Complete | `binary_trees.bg` |
| [fannkuch-redux](#fannkuch-redux) | ✅ Complete | `fannkuch_redux.bg` |
| [n-body](#n-body) | ✅ Complete | `nbody.bg` |
| [spectral-norm](#spectral-norm) | ✅ Complete | `spectral_norm.bg` |
| [fasta](#fasta) | ✅ Complete | `fasta.bg` |
| [reverse-complement](#reverse-complement) | ✅ Complete | `revcomp.bg` |
| [k-nucleotide](#k-nucleotide) | ✅ Complete | `knucleotide.bg` |
| [mandelbrot](#mandelbrot) | ⏳ Blocked | Float comparison bug prevents correct execution |
| [pidigits](#pidigits) | ❌ Not Feasible | Requires arbitrary precision integers |
| [regex-redux](#regex-redux) | ❌ Not Feasible | Requires regex library |

## Implemented Benchmarks

### binary-trees

**Status:** ✅ Complete

Allocate and deallocate many binary trees to stress test memory allocation and garbage collection.

```bash
cargo run -- benchmarks/benchmarksgame/binary_trees.bg
```

**Output (N=10):**
```
stretch tree of depth 11	 check: 4095
1024	 trees of depth 4	 check: 31744
256	 trees of depth 6	 check: 32512
64	 trees of depth 8	 check: 32704
16	 trees of depth 10	 check: 32752
long lived tree of depth 10	 check: 2047
```

---

### fannkuch-redux

**Status:** ✅ Complete

Count permutations and track maximum "pancake flips" (reversing prefixes).

```bash
cargo run -- benchmarks/benchmarksgame/fannkuch_redux.bg
```

**Output (N=7):**
```
228
Pfannkuchen(7) = 16
```

---

### n-body

**Status:** ✅ Complete

Simulate the solar system using Newtonian physics.

```bash
cargo run -- benchmarks/benchmarksgame/nbody.bg
```

**Output (N=1000):**
```
-0.16907516382852442
-0.16908760523460603
```

---

### spectral-norm

**Status:** ✅ Complete

Calculate an eigenvalue using the power method.

```bash
cargo run -- benchmarks/benchmarksgame/spectral_norm.bg
```

**Output (N=100):**
```
1.2742199912349306
```

---

### fasta

**Status:** ✅ Complete

Generate DNA sequences using repeat and random selection algorithms.

```bash
cargo run -- benchmarks/benchmarksgame/fasta.bg
```

**Output (N=10):**
```
>ONE Homo sapiens alu
GGCCGGGCGCGGTGGCTCAC
>TWO IUB ambiguity codes
cttBtatcatatgctaKggNcataaaSatg
>THREE Homo sapiens frequency
taaatcttgtgcttcgttagaagtctcgactacgtgtagcctagtgtttg
```

**Implementation Notes:**
- Uses integer thresholds (ceiling values) to work around Beagle's float comparison bug
- String indexing (`s[i]`) provides character-level access
- Linear Congruential Generator (LCG) for random number generation

---

### reverse-complement

**Status:** ✅ Complete

Read DNA sequences from stdin, compute the reverse complement, and output with 60-char line wrapping.

```bash
cat input.fasta | cargo run -- benchmarks/benchmarksgame/revcomp.bg
```

**Implementation Notes:**
- Uses `read_line` builtin for stdin reading (returns null on EOF)
- Uses `print_byte` builtin for binary output
- Uses persistent_vector for efficient sequence accumulation
- Complement table lookup for character transformation

---

### k-nucleotide

**Status:** ✅ Complete

Count occurrences of nucleotide subsequences in DNA sequences.

```bash
cat input.fasta | cargo run -- benchmarks/benchmarksgame/knucleotide.bg
```

**Output (1000-line input):**
```
T 31.520
A 29.600
C 19.480
G 19.400

AT 9.922
...
54	GGT
24	GGTA
4	GGTATT
0	GGTATTTTAATT
0	GGTATTTTAATTTATAGT
```

**Implementation Notes:**
- Uses `read_line` builtin for stdin reading
- Uses persistent_map for frequency counting
- Bubble sort for sorting results (simple but functional)
- Integer rounding workaround for percentage formatting

---

## Blocked Benchmarks

### mandelbrot

**Status:** ⏳ Blocked

Generate a Mandelbrot set image in PBM format.

**Blocker:** Float comparison bug in Beagle. Float-to-float comparisons (`a > b`, `a < b`) return incorrect results. The mandelbrot algorithm requires `mag > 4.0` to check for escape, but this comparison doesn't work correctly.

**Code Status:** Implementation exists in `mandelbrot.bg` but produces incorrect output due to the bug.

---

## Not Feasible

### pidigits

**Status:** ❌ Not Feasible

Calculate digits of pi using a streaming spigot algorithm.

**Blocker:** Requires arbitrary precision integers. Beagle only has 62-bit integers.

---

### regex-redux

**Status:** ❌ Not Feasible

Count regex pattern matches and perform substitutions.

**Blocker:** Requires a regex library. Would need FFI wrapper for PCRE or similar.

---

## Added Builtins

To support these benchmarks, the following builtin functions were added to Beagle:

**Math functions:**
- `sqrt(x)` - Square root of a float
- `floor(x)` - Floor of a float
- `ceil(x)` - Ceiling of a float
- `abs(x)` - Absolute value of a float
- `sin(x)` - Sine of a float (radians)
- `cos(x)` - Cosine of a float (radians)
- `to_float(n)` - Convert an integer to a float

**I/O functions:**
- `read_line()` - Read a line from stdin (returns null on EOF)
- `print_byte(n)` - Output a single byte (for binary output)

**String functions:**
- `char_code(s)` - Get ASCII code of first character in string
- `char_from_code(n)` - Create single-character string from ASCII code

---

## Running Benchmarks

```bash
# Run a specific benchmark
cargo run -- benchmarks/benchmarksgame/binary_trees.bg

# Run with timing
cargo run -- --show-times benchmarks/benchmarksgame/binary_trees.bg

# Run all tests (includes benchmarks)
cargo run -- --all-tests
```

## Reference Implementations

Reference implementations for comparison are available at:
- Official repo: https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
- Local copy: `/Users/jimmyhmiller/Downloads/benchmarksgame-sourcecode/`

## Performance Goals

Based on Beagle's current performance characteristics:
- Target: 2x Ruby performance (already achieved for some workloads)
- Stretch goal: Within 50% of Node.js performance
