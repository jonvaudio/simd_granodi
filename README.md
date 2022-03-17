# SIMD Granodi

### Easy 128-bit AArch64 NEON / x64 SSE2 SIMD in a single header file, for both plain C99, and C++ with operator overloading

Currently has full test coverage, but not used in production yet. Code examples to come soon. Every function also has a "generic" non-vector implementation that will work on any platform (or can be forced with `SSE_GRANODI_FORCE_GENERIC`), which also serves as documentation.

`simd_granodi.h` is the only file you need.  
`simd_granodi_sse2_on_neon.h` contains some macros used by the author of this project, but not needed in new projects.  
`test_simd_granodi.c` contains the test code.  
