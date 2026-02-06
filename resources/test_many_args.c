#include <stdint.h>

// A function that takes 15 integer arguments and returns their sum.
// This tests that the FFI trampoline correctly handles stack overflow args
// (ARM64 only has 8 integer registers, so args 9-15 go on the stack).
int64_t sum_15(int64_t a1, int64_t a2, int64_t a3, int64_t a4,
               int64_t a5, int64_t a6, int64_t a7, int64_t a8,
               int64_t a9, int64_t a10, int64_t a11, int64_t a12,
               int64_t a13, int64_t a14, int64_t a15) {
    return a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 +
           a9 + a10 + a11 + a12 + a13 + a14 + a15;
}

// A function that takes 10 args (2 overflow on ARM64)
int64_t sum_10(int64_t a1, int64_t a2, int64_t a3, int64_t a4,
               int64_t a5, int64_t a6, int64_t a7, int64_t a8,
               int64_t a9, int64_t a10) {
    return a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10;
}

// A function that takes 9 args (1 overflow on ARM64)
int64_t sum_9(int64_t a1, int64_t a2, int64_t a3, int64_t a4,
              int64_t a5, int64_t a6, int64_t a7, int64_t a8,
              int64_t a9) {
    return a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9;
}

// A function that verifies each arg has the expected value (1-15).
// Returns 1 if all correct, 0 if any mismatch.
int64_t verify_15(int64_t a1, int64_t a2, int64_t a3, int64_t a4,
                  int64_t a5, int64_t a6, int64_t a7, int64_t a8,
                  int64_t a9, int64_t a10, int64_t a11, int64_t a12,
                  int64_t a13, int64_t a14, int64_t a15) {
    return (a1 == 1 && a2 == 2 && a3 == 3 && a4 == 4 &&
            a5 == 5 && a6 == 6 && a7 == 7 && a8 == 8 &&
            a9 == 9 && a10 == 10 && a11 == 11 && a12 == 12 &&
            a13 == 13 && a14 == 14 && a15 == 15) ? 1 : 0;
}
