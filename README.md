# simd_granodi.h

### Easy 128-bit AArch64 NEON / x64 SSE2 SIMD in a single header file, for both plain C99, and C++ with operator overloading

Currently has full test coverage, but not used in production yet. Code examples to come soon. Every function also has a "generic" non-vector implementation that will work on any platform (or can be forced with `SSE_GRANODI_FORCE_GENERIC`), which also serves as documentation.

`simd_granodi.h` is the only file you need.  
`simd_granodi_sse2_on_neon.h` contains some macros used by the author of this project, but not needed in new projects.  
`test_simd_granodi.c` contains the test code.  

## Example code (C++)

The following example code computes 4 approximate `log2(x)` values at the same time, using a cubic approximation in the interval [1, 2]. (A smooth but inaccurate approximation, useful for envelope generation in music etc). This uses the C++ classes, but could be re-written (in a few more lines) using the equivalent C macros from the simd_granodi.h. (It also returns 0 for "invalid" input, whereas a more mathematically correct version might return -inf for 0, etc).

```
Vec_ps log2_p3(const Vec_ps& x) {
    Vec_ps exponent = ((x.bitcast_to_pi32().shift_rl_imm<23>() & 0xff) - 127)
        .convert_to_ps(),
    mantissa = ((x.bitcast_to_pi32() & 0x807fffff) | 0x3f800000)
        .bitcast_to_ps();

    mantissa = ((1.6404256133344508e-1f*mantissa + -1.0988652862227437f)*mantissa +
        3.1482979293341158f)*mantissa + -2.2134752044448169f;

    return (x > 0.0f).choose_else_zero(exponent + mantissa);
}
```
