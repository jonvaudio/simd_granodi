#ifndef SIMD_GRANODI_H
#define SIMD_GRANODI_H

/*

SIMD GRANODI
Copyright (c) 2021-2022 Jon Ville

OBJECTIVES:

Thin layer to abstract ARM64/NEON and x64/SSE2, mostly focusing on
packed 4 x float and 2 x double but with support for packed 32 & 64 bit
signed integers.

Target C99 and C++11.

The C implementation only uses typedefs, with no new structs (other than for
the generic implementation).

The C++ classes for operator overloading are written in terms of the C
implementation.

Separate typedefs / classes for the result of a vector comparison.

Avoid UB (obey strict aliasing rules & use memcpy where necessary). No unions

Generic, SSE, and NEON definitions interleaved for comparison, education,
documentation, reference etc

Assume high confidence in modern compilers to inline, elide copies,
pre-calculate constants etc. Written to compile with -O3

Behaviour of corner cases may NOT be identical on separate platforms
(eg min / max of signed floating point zero, abs(INT_MIN), some conversion
corner cases, rounding when converting double -> float)

In some cases, separate multiply->add intrinsics may get optimized into fma,
giving slightly different results on different platforms (this happens on
ARM64 gcc with -O3)

Naming conventions are based on SSE2


NON-VECTOR FUNCTIONS:
(ie, might switch to general purpose registers, stall pipeline etc)

Non-vector on SSE2 only:
sg_cvt_pi64_ps
sg_cvt_pi64_pd
sg_cvt_ps_pi64
sg_cvtt_ps_pi64
sg_cvtf_ps_pi64
sg_cvt_pd_pi64
sg_cvtt_pd_pi64
sg_cvtf_pd_pi64
sg_cmplt_pi64
sg_cmplte_pi64
sg_cmpgte_pi64
sg_cmpgt_pi64
sg_abs_pi64
sg_min_pi64
sg_max_pi64

SSE2 in vector registers, but slower:
- sg_sl_pi32, sg_sl_pi64, sg_srl_pi32, sg_srl_pi64, sg_sra_pi32, shift one
  element at a time
- sg_mul_pi32 combines two _mm_mul_epu32 (unsigned mul) with a lot of other ops

Non-vector on NEON:
sg_srl_pi32
sg_srl_pi64
sg_sra_pi32

Non-vector on SSE2 and NEON:
sg_mul_pi64
sg_div_pi32
sg_div_pi64
sg_safediv_pi32
sg_safediv_pi64
sg_sra_pi64

PLATFORM DETECTION
32 bit ARM NEON hardware does not have all the intrinsic instructions used here,
but can compile using SIMD_GRANODI_FORCE_GENERIC, and can disable denormals.

32-bit x86 can work, but you cannot disable denormal numbers on x87,
so it is recommended to set your compiler to generate SSE2 code for all floats
if you wish to disable denormal numbers with mixed intrinsic / scalar code.

TODO:
- Add class methods for shifting by in-register values
- Set lane
- Find efficient right shifting implementations for NEON
- Add truncate, and round, intrinsics
- Investigate rounding for sg_cvt_pd_ps()
- sg_abs_pi64() on SSE2 might be easy to implement in-vector
- Load / store intrinsics

*/

// Sanity check
#if defined (SIMD_GRANODI_SSE2) || defined (SIMD_GRANODI_NEON) || \
    defined (SIMD_GRANODI_ARCH_SSE) || \
    defined (SIMD_GRANODI_ARCH_ARM64) || \
    defined (SIMD_GRANODI_ARCH_ARM32)
#error "A SIMD_GRANODI macro was defined before it should be"
#endif

#if defined (__GNUC__) || defined (__clang__)
    #if defined (__x86_64__) || (defined (__i386__) && defined (__SSE2__))
        // Warning: on x86 (32-bit), this doesn't guarantee that x87
        // instructions won't also be generated. Check your compiler options!
        #define SIMD_GRANODI_SSE2
        #define SIMD_GRANODI_ARCH_SSE
    #elif defined (__aarch64__)
        #define SIMD_GRANODI_NEON
        #define SIMD_GRANODI_ARCH_ARM64
    #elif defined (__arm__)
        #define SIMD_GRANODI_FORCE_GENERIC
        #define SIMD_GRANODI_ARCH_ARM32
    #else
        #define SIMD_GRANODI_FORCE_GENERIC
    #endif
#elif defined (_MSC_VER)
    #if defined (_M_AMD64) || (defined (_M_IX86) && (_M_IX86_FP == 2))
        #define SIMD_GRANODI_SSE2
        #define SIMD_GRANODI_ARCH_SSE
    #else
        #define SIMD_GRANODI_FORCE_GENERIC
    #endif
#endif

// In case the user wants to FORCE_GENERIC, the #ifdefs below should prioritize
// this, but just in case there is an error
#ifdef SIMD_GRANODI_FORCE_GENERIC
#undef SIMD_GRANODI_SSE2
#undef SIMD_GRANODI_NEON
#endif

#ifdef __cplusplus
#include <algorithm> // for std::min(), std::max()
#endif

#ifndef __cplusplus
#include <stdbool.h>
#endif
#include <stdint.h>
#include <stdlib.h> // for abs(int32/64)
#include <string.h> // for memcpy

// For rint, rintf, fabs, fabsf
// SSE2 uses rintf() for sg_cvt_ps_pi64()
// Now we always include for wrapping std math lib functions (sin etc)
//#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
#include <math.h>
//#endif

#ifdef SIMD_GRANODI_ARCH_SSE
#include <emmintrin.h>
#elif defined SIMD_GRANODI_NEON
#include <arm_neon.h>
#endif

// These are always needed for testing. Using different member names for each
// type to avoid uncaught bugs.
// These structs are laid out the same way as the registers are stored
// in memory. We don't use memcpy to read and write to them, but users could
// (with caution).
typedef struct { int32_t i0, i1, i2, i3; } sg_generic_pi32;
typedef struct { int64_t l0, l1; } sg_generic_pi64;
typedef struct { float f0, f1, f2, f3; } sg_generic_ps;
typedef struct { double d0, d1; } sg_generic_pd;

// Generic comparison structs could be implemented as bit-fields. But
// the hope is that they get optimized out of existence.
typedef struct { bool b0, b1, b2, b3; } sg_generic_cmp4;
typedef struct { bool b0, b1; } sg_generic_cmp2;

#define sg_allset_u32 (0xffffffffu)
#define sg_allset_u64 (0xffffffffffffffffu)
#define sg_allset_s32 (-1)
#define sg_allset_s64 (-1)
#define sg_fp_signmask_s32 (0x7fffffff)
#define sg_fp_signmask_s64 (0x7fffffffffffffff)

// All basic types are 128-bit:
// sg_pi32: 4 x int32_t
// sg_pi64: 2 x int64_t
// sg_ps:   4 x float  (32-bit float)
// sg_pd:   2 x double (64-bit float)

// All comparison types are 128 bit masks, but with different
// types depending on platform.
// sg_cmp_TYPE is the mask resulting from comparing two registers both
// containing sg_TYPE

#ifdef SIMD_GRANODI_FORCE_GENERIC
typedef sg_generic_pi32 sg_pi32;
typedef sg_generic_pi64 sg_pi64;
typedef sg_generic_ps sg_ps;
typedef sg_generic_pd sg_pd;

typedef sg_generic_cmp4 sg_cmp_pi32;
typedef sg_generic_cmp2 sg_cmp_pi64;
typedef sg_generic_cmp4 sg_cmp_ps;
typedef sg_generic_cmp2 sg_cmp_pd;

#elif defined SIMD_GRANODI_SSE2
typedef __m128i sg_pi32;
typedef __m128i sg_pi64;
typedef __m128 sg_ps;
typedef __m128d sg_pd;

typedef __m128i sg_cmp_pi32;
typedef __m128i sg_cmp_pi64;
typedef __m128 sg_cmp_ps;
typedef __m128d sg_cmp_pd;

#elif defined SIMD_GRANODI_NEON
typedef int32x4_t sg_pi32;
typedef int64x2_t sg_pi64;
typedef float32x4_t sg_ps;
typedef float64x2_t sg_pd;

typedef uint32x4_t sg_cmp_pi32;
typedef uint64x2_t sg_cmp_pi64;
typedef uint32x4_t sg_cmp_ps;
typedef uint64x2_t sg_cmp_pd;
#endif

// Declare a function used in different order from definitions
#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi32 sg_choose_pi32(const sg_cmp_pi32, const sg_pi32,
    const sg_pi32);
#endif


#ifdef SIMD_GRANODI_SSE2
#define sg_sse2_allset_si128 _mm_set_epi64x(sg_allset_s64, sg_allset_s64)
#define sg_sse2_allset_ps _mm_castsi128_ps(sg_sse2_allset_si128)
#define sg_sse2_allset_pd _mm_castsi128_pd(sg_sse2_allset_si128)
#define sg_sse2_signbit_ps _mm_set1_ps(-0.0f)
#define sg_sse2_signbit_pd _mm_set1_pd(-0.0)
#define sg_sse2_signmask_ps _mm_castsi128_ps( \
    _mm_set1_epi32(sg_fp_signmask_s32))
#define sg_sse2_signmask_pd _mm_castsi128_pd( \
    _mm_set1_epi64x(sg_fp_signmask_s64))
#endif

// Bitcasts (reinterpret, no conversion takes place)

// Scalar bitcasts

static inline int32_t sg_bitcast_u32x1_s32x1(const uint32_t a) {
    int32_t result; memcpy(&result, &a, sizeof(int32_t)); return result;
}
static inline uint32_t sg_bitcast_s32x1_u32x1(const int32_t a) {
    uint32_t result; memcpy(&result, &a, sizeof(uint32_t)); return result;
}
static inline int64_t sg_bitcast_u64x1_s64x1(const uint64_t a) {
    int64_t result; memcpy(&result, &a, sizeof(int64_t)); return result;
}
static inline uint64_t sg_bitcast_s64x1_u64x1(const int64_t a) {
    uint64_t result; memcpy(&result, &a, sizeof(uint64_t)); return result;
}

static inline float sg_bitcast_u32x1_f32x1(const uint32_t a) {
    float result; memcpy(&result, &a, sizeof(float)); return result;
}
static inline float sg_bitcast_s32x1_f32x1(const int32_t a) {
    float result; memcpy(&result, &a, sizeof(float)); return result;
}
static inline uint32_t sg_bitcast_f32x1_u32x1(const float a) {
    uint32_t result; memcpy(&result, &a, sizeof(uint32_t)); return result;
}
static inline int32_t sg_bitcast_f32x1_s32x1(const float a) {
    int32_t result; memcpy(&result, &a, sizeof(int32_t)); return result;
}

static inline double sg_bitcast_u64x1_f64x1(const uint64_t a) {
    double result; memcpy(&result, &a, sizeof(double)); return result;
}
static inline double sg_bitcast_s64x1_f64x1(const int64_t a) {
    double result; memcpy(&result, &a, sizeof(double)); return result;
}
static inline uint64_t sg_bitcast_f64x1_u64x1(const double a) {
    uint64_t result; memcpy(&result, &a, sizeof(uint64_t)); return result;
}
static inline int64_t sg_bitcast_f64x1_s64x1(const double a) {
    int64_t result; memcpy(&result, &a, sizeof(int64_t)); return result;
}

// For the generic versions, we directly define casts for:
// - Same size, but different type
// - Different sized integer
// And then write the rest in terms of those

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_bitcast_pi32_pi64(const sg_pi32 a) {
    sg_pi64 result;
    result.l0 = sg_bitcast_u64x1_s64x1(
            (((uint64_t) sg_bitcast_s32x1_u32x1(a.i1)) << 32) |
            ((uint64_t) sg_bitcast_s32x1_u32x1(a.i0)));
    result.l1 = sg_bitcast_u64x1_s64x1(
            (((uint64_t) sg_bitcast_s32x1_u32x1(a.i3)) << 32) |
            ((uint64_t) sg_bitcast_s32x1_u32x1(a.i2)));
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_pi32_pi64(a) (a)
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_pi32_pi64 vreinterpretq_s64_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_bitcast_pi64_pi32(const sg_pi64 a) {
    const uint64_t u0 = sg_bitcast_s64x1_u64x1(a.l0),
        u1 = sg_bitcast_s64x1_u64x1(a.l1);
    sg_pi32 result;
    result.i0 = sg_bitcast_u32x1_s32x1((uint32_t) (u0 & 0xffffffffu));
    result.i1 = sg_bitcast_u32x1_s32x1((uint32_t) (u0 >> 32u));
    result.i2 = sg_bitcast_u32x1_s32x1((uint32_t) (u1 & 0xffffffffu));
    result.i3 = sg_bitcast_u32x1_s32x1((uint32_t) (u1 >> 32u));
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_pi64_pi32(a) (a)
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_pi64_pi32 vreinterpretq_s32_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_bitcast_pi32_ps(const sg_pi32 a) {
    sg_ps result;
    result.f0 = sg_bitcast_s32x1_f32x1(a.i0); result.f1 = sg_bitcast_s32x1_f32x1(a.i1);
    result.f2 = sg_bitcast_s32x1_f32x1(a.i2); result.f3 = sg_bitcast_s32x1_f32x1(a.i3);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_pi32_ps _mm_castsi128_ps
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_pi32_ps vreinterpretq_f32_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_bitcast_ps_pi32(const sg_ps a) {
    sg_pi32 result;
    result.i0 = sg_bitcast_f32x1_s32x1(a.f0); result.i1 = sg_bitcast_f32x1_s32x1(a.f1);
    result.i2 = sg_bitcast_f32x1_s32x1(a.f2); result.i3 = sg_bitcast_f32x1_s32x1(a.f3);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_ps_pi32 _mm_castps_si128
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_ps_pi32 vreinterpretq_s32_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_bitcast_pi64_pd(const sg_pi64 a) {
    sg_pd result;
    result.d0 = sg_bitcast_s64x1_f64x1(a.l0); result.d1 = sg_bitcast_s64x1_f64x1(a.l1);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_pi64_pd _mm_castsi128_pd
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_pi64_pd vreinterpretq_f64_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_bitcast_pd_pi64(const sg_pd a) {
    sg_pi64 result;
    result.l0 = sg_bitcast_f64x1_s64x1(a.d0); result.l1 = sg_bitcast_f64x1_s64x1(a.d1);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_pd_pi64 _mm_castpd_si128
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_pd_pi64 vreinterpretq_s64_f64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_bitcast_pi32_pd(const sg_pi32 a) {
    return sg_bitcast_pi64_pd(sg_bitcast_pi32_pi64(a));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_pi32_pd _mm_castsi128_pd
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_pi32_pd vreinterpretq_f64_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_bitcast_pi64_ps(const sg_pi64 a) {
    return sg_bitcast_pi32_ps(sg_bitcast_pi64_pi32(a));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_pi64_ps _mm_castsi128_ps
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_pi64_ps vreinterpretq_f32_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_bitcast_ps_pi64(const sg_ps a) {
    return sg_bitcast_pi32_pi64(sg_bitcast_ps_pi32(a));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_ps_pi64 _mm_castps_si128
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_ps_pi64 vreinterpretq_s64_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_bitcast_ps_pd(const sg_ps a) {
    return sg_bitcast_pi64_pd(sg_bitcast_pi32_pi64(sg_bitcast_ps_pi32(a)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_ps_pd _mm_castps_pd
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_ps_pd vreinterpretq_f64_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_bitcast_pd_pi32(const sg_pd a) {
    return sg_bitcast_pi64_pi32(sg_bitcast_pd_pi64(a));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_pd_pi32 _mm_castpd_si128
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_pd_pi32 vreinterpretq_s32_f64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_bitcast_pd_ps(const sg_pd a) {
    return sg_bitcast_pi32_ps(sg_bitcast_pi64_pi32(sg_bitcast_pd_pi64(a)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_bitcast_pd_ps _mm_castpd_ps
#elif defined SIMD_GRANODI_NEON
#define sg_bitcast_pd_ps vreinterpretq_f32_f64
#endif

// Shuffle

// Generate immediate values for SSE shuffling, but also used for the switch
// statements on all platforms
#define sg_sse2_shuffle32_imm(src3, src2, src1, src0) \
    (((src3)<<6)|((src2)<<4)|((src1)<<2)|(src0))
#define sg_sse2_shuffle64_imm(src1, src0) ((src0)|((src1)<<1))

// 4x32 shuffles on NEON are generated by a Java program that conducts a brute
// force search of all possible combinations of vector manipulations, and picks
// the combination with the fewest ops and fewest temporary vars. These are
// typically 1 or 2 ops, but 3 in the worst case.
// Switch statements get optimized out when src args are compile time constants.

#define sg_shuffle_pi32(a, src3_compile_time_constant, \
    src2_compile_time_constant, \
    src1_compile_time_constant, \
    src0_compile_time_constant) \
    sg_shuffle_pi32_switch_(a, sg_sse2_shuffle32_imm( \
        src3_compile_time_constant, \
        src2_compile_time_constant, \
        src1_compile_time_constant, \
        src0_compile_time_constant))
static inline sg_pi32 sg_shuffle_pi32_switch_(const sg_pi32 a,
    const int imm8_compile_time_constant)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    const int32_t array[] = { a.i0, a.i1, a.i2, a.i3 };
    sg_pi32 result;
    result.i0 = array[imm8_compile_time_constant & 3];
    result.i1 = array[(imm8_compile_time_constant >> 2) & 3];
    result.i2 = array[(imm8_compile_time_constant >> 4) & 3];
    result.i3 = array[(imm8_compile_time_constant >> 6) & 3];
    return result;
    #elif defined SIMD_GRANODI_SSE2
switch (imm8_compile_time_constant & 0xff) {
case 0: return _mm_shuffle_epi32(a, 0); case 1: return _mm_shuffle_epi32(a, 1);
case 2: return _mm_shuffle_epi32(a, 2); case 3: return _mm_shuffle_epi32(a, 3);
case 4: return _mm_shuffle_epi32(a, 4); case 5: return _mm_shuffle_epi32(a, 5);
case 6: return _mm_shuffle_epi32(a, 6); case 7: return _mm_shuffle_epi32(a, 7);
case 8: return _mm_shuffle_epi32(a, 8); case 9: return _mm_shuffle_epi32(a, 9);
case 10: return _mm_shuffle_epi32(a, 10);
case 11: return _mm_shuffle_epi32(a, 11);
case 12: return _mm_shuffle_epi32(a, 12);
case 13: return _mm_shuffle_epi32(a, 13);
case 14: return _mm_shuffle_epi32(a, 14);
case 15: return _mm_shuffle_epi32(a, 15);
case 16: return _mm_shuffle_epi32(a, 16);
case 17: return _mm_shuffle_epi32(a, 17);
case 18: return _mm_shuffle_epi32(a, 18);
case 19: return _mm_shuffle_epi32(a, 19);
case 20: return _mm_shuffle_epi32(a, 20);
case 21: return _mm_shuffle_epi32(a, 21);
case 22: return _mm_shuffle_epi32(a, 22);
case 23: return _mm_shuffle_epi32(a, 23);
case 24: return _mm_shuffle_epi32(a, 24);
case 25: return _mm_shuffle_epi32(a, 25);
case 26: return _mm_shuffle_epi32(a, 26);
case 27: return _mm_shuffle_epi32(a, 27);
case 28: return _mm_shuffle_epi32(a, 28);
case 29: return _mm_shuffle_epi32(a, 29);
case 30: return _mm_shuffle_epi32(a, 30);
case 31: return _mm_shuffle_epi32(a, 31);
case 32: return _mm_shuffle_epi32(a, 32);
case 33: return _mm_shuffle_epi32(a, 33);
case 34: return _mm_shuffle_epi32(a, 34);
case 35: return _mm_shuffle_epi32(a, 35);
case 36: return _mm_shuffle_epi32(a, 36);
case 37: return _mm_shuffle_epi32(a, 37);
case 38: return _mm_shuffle_epi32(a, 38);
case 39: return _mm_shuffle_epi32(a, 39);
case 40: return _mm_shuffle_epi32(a, 40);
case 41: return _mm_shuffle_epi32(a, 41);
case 42: return _mm_shuffle_epi32(a, 42);
case 43: return _mm_shuffle_epi32(a, 43);
case 44: return _mm_shuffle_epi32(a, 44);
case 45: return _mm_shuffle_epi32(a, 45);
case 46: return _mm_shuffle_epi32(a, 46);
case 47: return _mm_shuffle_epi32(a, 47);
case 48: return _mm_shuffle_epi32(a, 48);
case 49: return _mm_shuffle_epi32(a, 49);
case 50: return _mm_shuffle_epi32(a, 50);
case 51: return _mm_shuffle_epi32(a, 51);
case 52: return _mm_shuffle_epi32(a, 52);
case 53: return _mm_shuffle_epi32(a, 53);
case 54: return _mm_shuffle_epi32(a, 54);
case 55: return _mm_shuffle_epi32(a, 55);
case 56: return _mm_shuffle_epi32(a, 56);
case 57: return _mm_shuffle_epi32(a, 57);
case 58: return _mm_shuffle_epi32(a, 58);
case 59: return _mm_shuffle_epi32(a, 59);
case 60: return _mm_shuffle_epi32(a, 60);
case 61: return _mm_shuffle_epi32(a, 61);
case 62: return _mm_shuffle_epi32(a, 62);
case 63: return _mm_shuffle_epi32(a, 63);
case 64: return _mm_shuffle_epi32(a, 64);
case 65: return _mm_shuffle_epi32(a, 65);
case 66: return _mm_shuffle_epi32(a, 66);
case 67: return _mm_shuffle_epi32(a, 67);
case 68: return _mm_shuffle_epi32(a, 68);
case 69: return _mm_shuffle_epi32(a, 69);
case 70: return _mm_shuffle_epi32(a, 70);
case 71: return _mm_shuffle_epi32(a, 71);
case 72: return _mm_shuffle_epi32(a, 72);
case 73: return _mm_shuffle_epi32(a, 73);
case 74: return _mm_shuffle_epi32(a, 74);
case 75: return _mm_shuffle_epi32(a, 75);
case 76: return _mm_shuffle_epi32(a, 76);
case 77: return _mm_shuffle_epi32(a, 77);
case 78: return _mm_shuffle_epi32(a, 78);
case 79: return _mm_shuffle_epi32(a, 79);
case 80: return _mm_shuffle_epi32(a, 80);
case 81: return _mm_shuffle_epi32(a, 81);
case 82: return _mm_shuffle_epi32(a, 82);
case 83: return _mm_shuffle_epi32(a, 83);
case 84: return _mm_shuffle_epi32(a, 84);
case 85: return _mm_shuffle_epi32(a, 85);
case 86: return _mm_shuffle_epi32(a, 86);
case 87: return _mm_shuffle_epi32(a, 87);
case 88: return _mm_shuffle_epi32(a, 88);
case 89: return _mm_shuffle_epi32(a, 89);
case 90: return _mm_shuffle_epi32(a, 90);
case 91: return _mm_shuffle_epi32(a, 91);
case 92: return _mm_shuffle_epi32(a, 92);
case 93: return _mm_shuffle_epi32(a, 93);
case 94: return _mm_shuffle_epi32(a, 94);
case 95: return _mm_shuffle_epi32(a, 95);
case 96: return _mm_shuffle_epi32(a, 96);
case 97: return _mm_shuffle_epi32(a, 97);
case 98: return _mm_shuffle_epi32(a, 98);
case 99: return _mm_shuffle_epi32(a, 99);
case 100: return _mm_shuffle_epi32(a, 100);
case 101: return _mm_shuffle_epi32(a, 101);
case 102: return _mm_shuffle_epi32(a, 102);
case 103: return _mm_shuffle_epi32(a, 103);
case 104: return _mm_shuffle_epi32(a, 104);
case 105: return _mm_shuffle_epi32(a, 105);
case 106: return _mm_shuffle_epi32(a, 106);
case 107: return _mm_shuffle_epi32(a, 107);
case 108: return _mm_shuffle_epi32(a, 108);
case 109: return _mm_shuffle_epi32(a, 109);
case 110: return _mm_shuffle_epi32(a, 110);
case 111: return _mm_shuffle_epi32(a, 111);
case 112: return _mm_shuffle_epi32(a, 112);
case 113: return _mm_shuffle_epi32(a, 113);
case 114: return _mm_shuffle_epi32(a, 114);
case 115: return _mm_shuffle_epi32(a, 115);
case 116: return _mm_shuffle_epi32(a, 116);
case 117: return _mm_shuffle_epi32(a, 117);
case 118: return _mm_shuffle_epi32(a, 118);
case 119: return _mm_shuffle_epi32(a, 119);
case 120: return _mm_shuffle_epi32(a, 120);
case 121: return _mm_shuffle_epi32(a, 121);
case 122: return _mm_shuffle_epi32(a, 122);
case 123: return _mm_shuffle_epi32(a, 123);
case 124: return _mm_shuffle_epi32(a, 124);
case 125: return _mm_shuffle_epi32(a, 125);
case 126: return _mm_shuffle_epi32(a, 126);
case 127: return _mm_shuffle_epi32(a, 127);
case 128: return _mm_shuffle_epi32(a, 128);
case 129: return _mm_shuffle_epi32(a, 129);
case 130: return _mm_shuffle_epi32(a, 130);
case 131: return _mm_shuffle_epi32(a, 131);
case 132: return _mm_shuffle_epi32(a, 132);
case 133: return _mm_shuffle_epi32(a, 133);
case 134: return _mm_shuffle_epi32(a, 134);
case 135: return _mm_shuffle_epi32(a, 135);
case 136: return _mm_shuffle_epi32(a, 136);
case 137: return _mm_shuffle_epi32(a, 137);
case 138: return _mm_shuffle_epi32(a, 138);
case 139: return _mm_shuffle_epi32(a, 139);
case 140: return _mm_shuffle_epi32(a, 140);
case 141: return _mm_shuffle_epi32(a, 141);
case 142: return _mm_shuffle_epi32(a, 142);
case 143: return _mm_shuffle_epi32(a, 143);
case 144: return _mm_shuffle_epi32(a, 144);
case 145: return _mm_shuffle_epi32(a, 145);
case 146: return _mm_shuffle_epi32(a, 146);
case 147: return _mm_shuffle_epi32(a, 147);
case 148: return _mm_shuffle_epi32(a, 148);
case 149: return _mm_shuffle_epi32(a, 149);
case 150: return _mm_shuffle_epi32(a, 150);
case 151: return _mm_shuffle_epi32(a, 151);
case 152: return _mm_shuffle_epi32(a, 152);
case 153: return _mm_shuffle_epi32(a, 153);
case 154: return _mm_shuffle_epi32(a, 154);
case 155: return _mm_shuffle_epi32(a, 155);
case 156: return _mm_shuffle_epi32(a, 156);
case 157: return _mm_shuffle_epi32(a, 157);
case 158: return _mm_shuffle_epi32(a, 158);
case 159: return _mm_shuffle_epi32(a, 159);
case 160: return _mm_shuffle_epi32(a, 160);
case 161: return _mm_shuffle_epi32(a, 161);
case 162: return _mm_shuffle_epi32(a, 162);
case 163: return _mm_shuffle_epi32(a, 163);
case 164: return _mm_shuffle_epi32(a, 164);
case 165: return _mm_shuffle_epi32(a, 165);
case 166: return _mm_shuffle_epi32(a, 166);
case 167: return _mm_shuffle_epi32(a, 167);
case 168: return _mm_shuffle_epi32(a, 168);
case 169: return _mm_shuffle_epi32(a, 169);
case 170: return _mm_shuffle_epi32(a, 170);
case 171: return _mm_shuffle_epi32(a, 171);
case 172: return _mm_shuffle_epi32(a, 172);
case 173: return _mm_shuffle_epi32(a, 173);
case 174: return _mm_shuffle_epi32(a, 174);
case 175: return _mm_shuffle_epi32(a, 175);
case 176: return _mm_shuffle_epi32(a, 176);
case 177: return _mm_shuffle_epi32(a, 177);
case 178: return _mm_shuffle_epi32(a, 178);
case 179: return _mm_shuffle_epi32(a, 179);
case 180: return _mm_shuffle_epi32(a, 180);
case 181: return _mm_shuffle_epi32(a, 181);
case 182: return _mm_shuffle_epi32(a, 182);
case 183: return _mm_shuffle_epi32(a, 183);
case 184: return _mm_shuffle_epi32(a, 184);
case 185: return _mm_shuffle_epi32(a, 185);
case 186: return _mm_shuffle_epi32(a, 186);
case 187: return _mm_shuffle_epi32(a, 187);
case 188: return _mm_shuffle_epi32(a, 188);
case 189: return _mm_shuffle_epi32(a, 189);
case 190: return _mm_shuffle_epi32(a, 190);
case 191: return _mm_shuffle_epi32(a, 191);
case 192: return _mm_shuffle_epi32(a, 192);
case 193: return _mm_shuffle_epi32(a, 193);
case 194: return _mm_shuffle_epi32(a, 194);
case 195: return _mm_shuffle_epi32(a, 195);
case 196: return _mm_shuffle_epi32(a, 196);
case 197: return _mm_shuffle_epi32(a, 197);
case 198: return _mm_shuffle_epi32(a, 198);
case 199: return _mm_shuffle_epi32(a, 199);
case 200: return _mm_shuffle_epi32(a, 200);
case 201: return _mm_shuffle_epi32(a, 201);
case 202: return _mm_shuffle_epi32(a, 202);
case 203: return _mm_shuffle_epi32(a, 203);
case 204: return _mm_shuffle_epi32(a, 204);
case 205: return _mm_shuffle_epi32(a, 205);
case 206: return _mm_shuffle_epi32(a, 206);
case 207: return _mm_shuffle_epi32(a, 207);
case 208: return _mm_shuffle_epi32(a, 208);
case 209: return _mm_shuffle_epi32(a, 209);
case 210: return _mm_shuffle_epi32(a, 210);
case 211: return _mm_shuffle_epi32(a, 211);
case 212: return _mm_shuffle_epi32(a, 212);
case 213: return _mm_shuffle_epi32(a, 213);
case 214: return _mm_shuffle_epi32(a, 214);
case 215: return _mm_shuffle_epi32(a, 215);
case 216: return _mm_shuffle_epi32(a, 216);
case 217: return _mm_shuffle_epi32(a, 217);
case 218: return _mm_shuffle_epi32(a, 218);
case 219: return _mm_shuffle_epi32(a, 219);
case 220: return _mm_shuffle_epi32(a, 220);
case 221: return _mm_shuffle_epi32(a, 221);
case 222: return _mm_shuffle_epi32(a, 222);
case 223: return _mm_shuffle_epi32(a, 223);
case 224: return _mm_shuffle_epi32(a, 224);
case 225: return _mm_shuffle_epi32(a, 225);
case 226: return _mm_shuffle_epi32(a, 226);
case 227: return _mm_shuffle_epi32(a, 227);
case 228: return a; case 229: return _mm_shuffle_epi32(a, 229);
case 230: return _mm_shuffle_epi32(a, 230);
case 231: return _mm_shuffle_epi32(a, 231);
case 232: return _mm_shuffle_epi32(a, 232);
case 233: return _mm_shuffle_epi32(a, 233);
case 234: return _mm_shuffle_epi32(a, 234);
case 235: return _mm_shuffle_epi32(a, 235);
case 236: return _mm_shuffle_epi32(a, 236);
case 237: return _mm_shuffle_epi32(a, 237);
case 238: return _mm_shuffle_epi32(a, 238);
case 239: return _mm_shuffle_epi32(a, 239);
case 240: return _mm_shuffle_epi32(a, 240);
case 241: return _mm_shuffle_epi32(a, 241);
case 242: return _mm_shuffle_epi32(a, 242);
case 243: return _mm_shuffle_epi32(a, 243);
case 244: return _mm_shuffle_epi32(a, 244);
case 245: return _mm_shuffle_epi32(a, 245);
case 246: return _mm_shuffle_epi32(a, 246);
case 247: return _mm_shuffle_epi32(a, 247);
case 248: return _mm_shuffle_epi32(a, 248);
case 249: return _mm_shuffle_epi32(a, 249);
case 250: return _mm_shuffle_epi32(a, 250);
case 251: return _mm_shuffle_epi32(a, 251);
case 252: return _mm_shuffle_epi32(a, 252);
case 253: return _mm_shuffle_epi32(a, 253);
case 254: return _mm_shuffle_epi32(a, 254);
case 255: return _mm_shuffle_epi32(a, 255);
default: return a; }
    #elif defined SIMD_GRANODI_NEON
int32x4_t t0;
switch (imm8_compile_time_constant & 0xff) {
case 0: return vdupq_laneq_s32(a,0);
case 1: return vcopyq_laneq_s32(vdupq_laneq_s32(a,0),0,a,1);
case 2: return vcopyq_laneq_s32(vdupq_laneq_s32(a,0),0,a,2);
case 3: return vcopyq_laneq_s32(vdupq_laneq_s32(a,0),0,a,3);
case 4: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),3,a,0);
case 5: t0 = vcopyq_laneq_s32(a,3,a,0);
return vtrn2q_s32(t0,t0);
case 6: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),3,a,0),0,a,2);
case 7: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),3,a,0),0,a,3);
case 8: return vuzp1q_s32(a,vcopyq_laneq_s32(a,2,a,0));
case 9: return vextq_s32(vcopyq_laneq_s32(a,3,a,0),a,1);
case 10: t0 = vcopyq_laneq_s32(a,3,a,0);
return vzip2q_s32(t0,t0);
case 11: return vextq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),1,a,3),a,1);
case 12: return vcopyq_laneq_s32(vdupq_laneq_s32(a,0),1,a,3);
case 13: return vuzp2q_s32(a,vdupq_laneq_s32(a,0));
case 14: return vextq_s32(a,vcopyq_laneq_s32(a,1,a,0),2);
case 15: t0 = vextq_s32(a,a,3);
return vzip1q_s32(t0,t0);
case 16: return vzip1q_s32(a,vcopyq_laneq_s32(a,1,a,0));
case 17: t0 = vcopyq_laneq_s32(a,3,a,0);
return vuzp2q_s32(t0,t0);
case 18: return vextq_s32(vcopyq_laneq_s32(a,3,a,2),vcopyq_laneq_s32(a,2,a,0),3);
case 19: return vextq_s32(a,vcopyq_laneq_s32(a,2,a,0),3);
case 20: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),2,a,1);
case 21: return vcopyq_laneq_s32(vdupq_laneq_s32(a,1),3,a,0);
case 22: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),2,a,1),0,a,2);
case 23: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),2,a,1),0,a,3);
case 24: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),2,a,1),1,a,2);
case 25: return vextq_s32(vcopyq_laneq_s32(a,3,a,1),a,1);
case 26: return vextq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,1),1,a,2),a,1);
case 27: return vrev64q_s32(vextq_s32(a,a,2));
case 28: return vzip1q_s32(a,vextq_s32(a,a,3));
case 29: return vuzp2q_s32(a,vcopyq_laneq_s32(a,3,a,0));
case 30: return vextq_s32(a,vrev64q_s32(a),2);
case 31: return vextq_s32(a,vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),0,a,3),3);
case 32: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),3,a,0);
case 33: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),3,a,0),0,a,1);
case 34: return vextq_s32(vuzp1q_s32(a,a),a,1);
case 35: return vextq_s32(a,vuzp1q_s32(a,a),3);
case 36: return vcopyq_laneq_s32(a,3,a,0);
case 37: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),0,a,1);
case 38: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),0,a,2);
case 39: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),0,a,3);
case 40: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),1,a,2);
case 41: return vextq_s32(vcopyq_laneq_s32(a,3,a,2),a,1);
case 42: return vcopyq_laneq_s32(vdupq_laneq_s32(a,2),3,a,0);
case 43: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),1,a,2),0,a,3);
case 44: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),1,a,3);
case 45: return vuzp2q_s32(a,vextq_s32(a,a,1));
case 46: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),0,a,2),1,a,3);
case 47: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),0,a,3),1,a,3);
case 48: return vcopyq_laneq_s32(vdupq_laneq_s32(a,0),2,a,3);
case 49: return vextq_s32(vcopyq_laneq_s32(a,2,a,0),a,1);
case 50: return vzip2q_s32(a,vdupq_laneq_s32(a,0));
case 51: return vextq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),1,a,3),a,1);
case 52: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),2,a,3);
case 53: return vextq_s32(vcopyq_laneq_s32(a,2,a,1),a,1);
case 54: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),0,a,2),2,a,3);
case 55: return vextq_s32(vuzp2q_s32(a,a),a,1);
case 56: return vextq_s32(vcopyq_laneq_s32(a,1,a,0),a,1);
case 57: return vextq_s32(a,a,1);
case 58: return vextq_s32(vcopyq_laneq_s32(a,1,a,2),a,1);
case 59: return vextq_s32(vcopyq_laneq_s32(a,1,a,3),a,1);
case 60: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,0),1,a,3),2,a,3);
case 61: return vextq_s32(vcopyq_laneq_s32(a,2,a,3),a,1);
case 62: return vextq_s32(a,vextq_s32(a,a,3),2);
case 63: return vcopyq_laneq_s32(vdupq_laneq_s32(a,3),3,a,0);
case 64: return vzip1q_s32(vcopyq_laneq_s32(a,1,a,0),a);
case 65: return vzip1q_s32(vrev64q_s32(a),a);
case 66: return vextq_s32(vcopyq_laneq_s32(a,3,a,0),a,2);
case 67: return vextq_s32(vextq_s32(a,a,1),a,2);
case 68: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),3,a,1);
case 69: return vcopyq_laneq_s32(vdupq_laneq_s32(a,1),2,a,0);
case 70: return vextq_s32(vcopyq_laneq_s32(a,3,a,1),a,2);
case 71: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),3,a,1),0,a,3);
case 72: return vuzp1q_s32(a,vcopyq_laneq_s32(a,2,a,1));
case 73: return vextq_s32(vextq_s32(a,a,3),a,2);
case 74: return vextq_s32(vcopyq_laneq_s32(a,3,a,2),a,2);
case 75: return vextq_s32(vrev64q_s32(a),a,2);
case 76: return vextq_s32(vcopyq_laneq_s32(a,2,a,0),a,2);
case 77: return vextq_s32(vcopyq_laneq_s32(a,2,a,1),a,2);
case 78: return vextq_s32(a,a,2);
case 79: return vextq_s32(vcopyq_laneq_s32(a,2,a,3),a,2);
case 80: return vzip1q_s32(a,a);
case 81: return vzip1q_s32(vcopyq_laneq_s32(a,0,a,1),a);
case 82: return vzip1q_s32(vcopyq_laneq_s32(a,0,a,2),a);
case 83: return vextq_s32(a,vcopyq_laneq_s32(a,2,a,1),3);
case 84: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,1),3,a,1);
case 85: return vdupq_laneq_s32(a,1);
case 86: return vcopyq_laneq_s32(vdupq_laneq_s32(a,1),0,a,2);
case 87: return vcopyq_laneq_s32(vdupq_laneq_s32(a,1),0,a,3);
case 88: return vzip1q_s32(a,vcopyq_laneq_s32(a,0,a,2));
case 89: return vcopyq_laneq_s32(vdupq_laneq_s32(a,1),1,a,2);
case 90: t0 = vcopyq_laneq_s32(a,0,a,2);
return vzip1q_s32(t0,t0);
case 91: return vextq_s32(a,vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,1),0,a,2),3);
case 92: return vzip1q_s32(a,vcopyq_laneq_s32(a,0,a,3));
case 93: return vuzp2q_s32(a,vcopyq_laneq_s32(a,3,a,1));
case 94: return vextq_s32(a,vcopyq_laneq_s32(a,0,a,1),2);
case 95: t0 = vcopyq_laneq_s32(a,0,a,3);
return vzip1q_s32(t0,t0);
case 96: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),3,a,1);
case 97: return vzip1q_s32(vextq_s32(a,a,1),a);
case 98: return vzip1q_s32(vdupq_laneq_s32(a,2),a);
case 99: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),3,a,1),0,a,3);
case 100: return vcopyq_laneq_s32(a,3,a,1);
case 101: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,1),3,a,1);
case 102: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,1),0,a,2);
case 103: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,1),0,a,3);
case 104: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,1),1,a,2);
case 105: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,1),3,a,1),1,a,2);
case 106: return vcopyq_laneq_s32(vdupq_laneq_s32(a,2),3,a,1);
case 107: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,1),1,a,2),0,a,3);
case 108: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,1),1,a,3);
case 109: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,1),3,a,1),1,a,3);
case 110: return vextq_s32(a,vcopyq_laneq_s32(a,0,a,2),2);
case 111: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,1),0,a,3),1,a,3);
case 112: return vzip1q_s32(vcopyq_laneq_s32(a,1,a,3),a);
case 113: return vrev64q_s32(vcopyq_laneq_s32(a,2,a,1));
case 114: return vzip1q_s32(vextq_s32(a,a,2),a);
case 115: return vzip1q_s32(vdupq_laneq_s32(a,3),a);
case 116: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,1),2,a,3);
case 117: return vtrn2q_s32(a,vcopyq_laneq_s32(a,3,a,1));
case 118: return vzip2q_s32(a,vdupq_laneq_s32(a,1));
case 119: return vextq_s32(a,vuzp2q_s32(a,a),3);
case 120: return vuzp1q_s32(a,vextq_s32(a,a,3));
case 121: return vextq_s32(a,vcopyq_laneq_s32(a,0,a,1),1);
case 122: return vzip2q_s32(a,vcopyq_laneq_s32(a,3,a,1));
case 123: return vextq_s32(vcopyq_laneq_s32(a,1,a,3),vcopyq_laneq_s32(a,0,a,1),1);
case 124: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,1),1,a,3),2,a,3);
case 125: return vuzp2q_s32(a,vextq_s32(a,a,2));
case 126: return vextq_s32(a,vcopyq_laneq_s32(a,0,a,3),2);
case 127: return vcopyq_laneq_s32(vdupq_laneq_s32(a,3),3,a,1);
case 128: return vuzp1q_s32(vcopyq_laneq_s32(a,2,a,0),a);
case 129: return vrev64q_s32(vcopyq_laneq_s32(a,3,a,0));
case 130: return vuzp1q_s32(vextq_s32(a,a,2),a);
case 131: return vextq_s32(a,vcopyq_laneq_s32(a,1,a,0),3);
case 132: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),3,a,2);
case 133: return vuzp1q_s32(vdupq_laneq_s32(a,1),a);
case 134: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),0,a,2),3,a,2);
case 135: return vuzp1q_s32(vextq_s32(a,a,3),a);
case 136: return vuzp1q_s32(a,a);
case 137: return vuzp1q_s32(vcopyq_laneq_s32(a,0,a,1),a);
case 138: return vuzp1q_s32(vcopyq_laneq_s32(a,0,a,2),a);
case 139: return vuzp1q_s32(vcopyq_laneq_s32(a,0,a,3),a);
case 140: return vuzp1q_s32(vcopyq_laneq_s32(a,2,a,3),a);
case 141: return vuzp1q_s32(vextq_s32(a,a,1),a);
case 142: return vextq_s32(a,vcopyq_laneq_s32(a,1,a,2),2);
case 143: return vuzp1q_s32(vdupq_laneq_s32(a,3),a);
case 144: return vextq_s32(vcopyq_laneq_s32(a,3,a,0),a,3);
case 145: return vextq_s32(vcopyq_laneq_s32(a,3,a,1),a,3);
case 146: return vextq_s32(vcopyq_laneq_s32(a,3,a,2),a,3);
case 147: return vextq_s32(a,a,3);
case 148: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,1),3,a,2);
case 149: return vcopyq_laneq_s32(vdupq_laneq_s32(a,1),3,a,2);
case 150: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,1),0,a,2),3,a,2);
case 151: return vextq_s32(a,vcopyq_laneq_s32(a,0,a,1),3);
case 152: return vuzp1q_s32(a,vcopyq_laneq_s32(a,0,a,1));
case 153: t0 = vcopyq_laneq_s32(a,0,a,1);
return vuzp1q_s32(t0,t0);
case 154: return vcopyq_laneq_s32(vdupq_laneq_s32(a,2),2,a,1);
case 155: return vextq_s32(a,vcopyq_laneq_s32(a,0,a,2),3);
case 156: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,1),3,a,2),1,a,3);
case 157: return vuzp2q_s32(a,vcopyq_laneq_s32(a,3,a,2));
case 158: return vextq_s32(a,vextq_s32(a,a,1),2);
case 159: return vextq_s32(a,vcopyq_laneq_s32(a,0,a,3),3);
case 160: return vtrn1q_s32(a,a);
case 161: return vrev64q_s32(vcopyq_laneq_s32(a,3,a,2));
case 162: return vtrn1q_s32(vcopyq_laneq_s32(a,0,a,2),a);
case 163: return vextq_s32(a,vcopyq_laneq_s32(a,1,a,2),3);
case 164: return vcopyq_laneq_s32(a,3,a,2);
case 165: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,1),3,a,2);
case 166: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,2),3,a,2);
case 167: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,2),0,a,3);
case 168: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,2),3,a,2);
case 169: return vcopyq_laneq_s32(vdupq_laneq_s32(a,2),0,a,1);
case 170: return vdupq_laneq_s32(a,2);
case 171: return vcopyq_laneq_s32(vdupq_laneq_s32(a,2),0,a,3);
case 172: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,2),1,a,3);
case 173: return vuzp2q_s32(a,vdupq_laneq_s32(a,2));
case 174: return vcopyq_laneq_s32(vdupq_laneq_s32(a,2),1,a,3);
case 175: t0 = vcopyq_laneq_s32(a,0,a,3);
return vtrn1q_s32(t0,t0);
case 176: return vrev64q_s32(vcopyq_laneq_s32(a,1,a,0));
case 177: return vrev64q_s32(a);
case 178: return vrev64q_s32(vcopyq_laneq_s32(a,1,a,2));
case 179: return vextq_s32(a,vcopyq_laneq_s32(a,1,a,3),3);
case 180: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,2),2,a,3);
case 181: return vrev64q_s32(vcopyq_laneq_s32(a,0,a,1));
case 182: return vzip2q_s32(a,vextq_s32(a,a,3));
case 183: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,2),0,a,3),2,a,3);
case 184: return vuzp1q_s32(a,vcopyq_laneq_s32(a,0,a,3));
case 185: return vextq_s32(a,vcopyq_laneq_s32(a,0,a,2),1);
case 186: return vzip2q_s32(a,vcopyq_laneq_s32(a,3,a,2));
case 187: t0 = vcopyq_laneq_s32(a,0,a,3);
return vuzp1q_s32(t0,t0);
case 188: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,3,a,2),1,a,3),2,a,3);
case 189: return vrev64q_s32(vcopyq_laneq_s32(a,0,a,3));
case 190: return vzip2q_s32(a,vrev64q_s32(a));
case 191: return vcopyq_laneq_s32(vdupq_laneq_s32(a,3),3,a,2);
case 192: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),2,a,0);
case 193: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),2,a,0),0,a,1);
case 194: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),2,a,0),0,a,2);
case 195: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),2,a,0),0,a,3);
case 196: return vcopyq_laneq_s32(a,2,a,0);
case 197: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),0,a,1);
case 198: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),0,a,2);
case 199: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),0,a,3);
case 200: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),1,a,2);
case 201: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),0,a,1),1,a,2);
case 202: return vzip2q_s32(vcopyq_laneq_s32(a,3,a,0),a);
case 203: return vzip2q_s32(vextq_s32(a,a,1),a);
case 204: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,0),1,a,3);
case 205: return vuzp2q_s32(a,vcopyq_laneq_s32(a,1,a,0));
case 206: return vextq_s32(a,vcopyq_laneq_s32(a,1,a,3),2);
case 207: return vcopyq_laneq_s32(vdupq_laneq_s32(a,3),2,a,0);
case 208: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),2,a,1);
case 209: return vuzp2q_s32(vcopyq_laneq_s32(a,3,a,0),a);
case 210: return vuzp2q_s32(vextq_s32(a,a,1),a);
case 211: return vextq_s32(a,vcopyq_laneq_s32(a,2,a,3),3);
case 212: return vcopyq_laneq_s32(a,2,a,1);
case 213: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,1),2,a,1);
case 214: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,1),0,a,2);
case 215: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,1),0,a,3);
case 216: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,1),1,a,2);
case 217: return vuzp2q_s32(vcopyq_laneq_s32(a,3,a,2),a);
case 218: return vzip2q_s32(vcopyq_laneq_s32(a,3,a,1),a);
case 219: return vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,1),1,a,2),0,a,3);
case 220: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,2,a,1),1,a,3);
case 221: return vuzp2q_s32(a,a);
case 222: return vuzp2q_s32(vcopyq_laneq_s32(a,1,a,2),a);
case 223: return vuzp2q_s32(vcopyq_laneq_s32(a,1,a,3),a);
case 224: return vcopyq_laneq_s32(a,1,a,0);
case 225: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),0,a,1);
case 226: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),0,a,2);
case 227: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),0,a,3);
case 228: return a;
case 229: return vcopyq_laneq_s32(a,0,a,1);
case 230: return vcopyq_laneq_s32(a,0,a,2);
case 231: return vcopyq_laneq_s32(a,0,a,3);
case 232: return vcopyq_laneq_s32(a,1,a,2);
case 233: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,1),1,a,2);
case 234: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,2),1,a,2);
case 235: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,2),0,a,3);
case 236: return vcopyq_laneq_s32(a,1,a,3);
case 237: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,1),1,a,3);
case 238: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,2),1,a,3);
case 239: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,3),1,a,3);
case 240: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,0),2,a,3);
case 241: return vrev64q_s32(vcopyq_laneq_s32(a,2,a,3));
case 242: return vzip2q_s32(a,vcopyq_laneq_s32(a,2,a,0));
case 243: return vcopyq_laneq_s32(vdupq_laneq_s32(a,3),1,a,0);
case 244: return vcopyq_laneq_s32(a,2,a,3);
case 245: return vtrn2q_s32(a,a);
case 246: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,2),2,a,3);
case 247: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,0,a,3),2,a,3);
case 248: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,2),2,a,3);
case 249: return vextq_s32(a,vcopyq_laneq_s32(a,0,a,3),1);
case 250: return vzip2q_s32(a,a);
case 251: return vzip2q_s32(vcopyq_laneq_s32(a,2,a,3),a);
case 252: return vcopyq_laneq_s32(vcopyq_laneq_s32(a,1,a,3),2,a,3);
case 253: return vuzp2q_s32(a,vcopyq_laneq_s32(a,1,a,3));
case 254: return vzip2q_s32(a,vcopyq_laneq_s32(a,2,a,3));
case 255: return vdupq_laneq_s32(a,3);
default: return a; }
    #endif
}

#define sg_shuffle_pi64(a, src1_compile_time_constant, \
    src0_compile_time_constant) \
    sg_shuffle_pi64_switch_(a, sg_sse2_shuffle64_imm( \
        src1_compile_time_constant, src0_compile_time_constant))
static inline sg_pi64 sg_shuffle_pi64_switch_(const sg_pi64 a,
    const int imm8_compile_time_constant) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    const int64_t array[] = { a.l0, a.l1 };
    sg_pi64 result;
    result.l0 = array[imm8_compile_time_constant & 1];
    result.l1 = array[(imm8_compile_time_constant >> 1) & 1];
    return result;
    #elif defined SIMD_GRANODI_SSE2
    switch (imm8_compile_time_constant & 3)
    {
        case 0: return _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(1, 0, 1, 0));
        case 1: return _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(1, 0, 3, 2));
        case 2: return a;
        case 3: return _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(3, 2, 3, 2));
        default: return a;
    }
    #elif defined SIMD_GRANODI_NEON
    switch (imm8_compile_time_constant & 3)
    {
        case 0: return vcopyq_laneq_s64(a, 1, a, 0);
        case 1: return vextq_s64(a, a, 1);
        case 2: return a;
        case 3: return vcopyq_laneq_s64(a, 0, a, 1);
        default: return a;
    }
    #endif
}

#define sg_shuffle_ps(a, src3_compile_time_constant, \
    src2_compile_time_constant, \
    src1_compile_time_constant, \
    src0_compile_time_constant) \
    sg_shuffle_ps_switch_(a, sg_sse2_shuffle32_imm( \
        src3_compile_time_constant, \
        src2_compile_time_constant, \
        src1_compile_time_constant, \
        src0_compile_time_constant))
static inline sg_ps sg_shuffle_ps_switch_(const sg_ps a,
    const int imm8_compile_time_constant)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    const float array[] = { a.f0, a.f1, a.f2, a.f3 };
    sg_ps result;
    result.f0 = array[imm8_compile_time_constant & 3];
    result.f1 = array[(imm8_compile_time_constant >> 2) & 3];
    result.f2 = array[(imm8_compile_time_constant >> 4) & 3];
    result.f3 = array[(imm8_compile_time_constant >> 6) & 3];
    return result;
    #elif defined SIMD_GRANODI_SSE2
switch (imm8_compile_time_constant & 0xff) {
case 0: return _mm_shuffle_ps(a, a, 0); case 1: return _mm_shuffle_ps(a, a, 1);
case 2: return _mm_shuffle_ps(a, a, 2); case 3: return _mm_shuffle_ps(a, a, 3);
case 4: return _mm_shuffle_ps(a, a, 4); case 5: return _mm_shuffle_ps(a, a, 5);
case 6: return _mm_shuffle_ps(a, a, 6); case 7: return _mm_shuffle_ps(a, a, 7);
case 8: return _mm_shuffle_ps(a, a, 8); case 9: return _mm_shuffle_ps(a, a, 9);
case 10: return _mm_shuffle_ps(a, a, 10);
case 11: return _mm_shuffle_ps(a, a, 11);
case 12: return _mm_shuffle_ps(a, a, 12);
case 13: return _mm_shuffle_ps(a, a, 13);
case 14: return _mm_shuffle_ps(a, a, 14);
case 15: return _mm_shuffle_ps(a, a, 15);
case 16: return _mm_shuffle_ps(a, a, 16);
case 17: return _mm_shuffle_ps(a, a, 17);
case 18: return _mm_shuffle_ps(a, a, 18);
case 19: return _mm_shuffle_ps(a, a, 19);
case 20: return _mm_shuffle_ps(a, a, 20);
case 21: return _mm_shuffle_ps(a, a, 21);
case 22: return _mm_shuffle_ps(a, a, 22);
case 23: return _mm_shuffle_ps(a, a, 23);
case 24: return _mm_shuffle_ps(a, a, 24);
case 25: return _mm_shuffle_ps(a, a, 25);
case 26: return _mm_shuffle_ps(a, a, 26);
case 27: return _mm_shuffle_ps(a, a, 27);
case 28: return _mm_shuffle_ps(a, a, 28);
case 29: return _mm_shuffle_ps(a, a, 29);
case 30: return _mm_shuffle_ps(a, a, 30);
case 31: return _mm_shuffle_ps(a, a, 31);
case 32: return _mm_shuffle_ps(a, a, 32);
case 33: return _mm_shuffle_ps(a, a, 33);
case 34: return _mm_shuffle_ps(a, a, 34);
case 35: return _mm_shuffle_ps(a, a, 35);
case 36: return _mm_shuffle_ps(a, a, 36);
case 37: return _mm_shuffle_ps(a, a, 37);
case 38: return _mm_shuffle_ps(a, a, 38);
case 39: return _mm_shuffle_ps(a, a, 39);
case 40: return _mm_shuffle_ps(a, a, 40);
case 41: return _mm_shuffle_ps(a, a, 41);
case 42: return _mm_shuffle_ps(a, a, 42);
case 43: return _mm_shuffle_ps(a, a, 43);
case 44: return _mm_shuffle_ps(a, a, 44);
case 45: return _mm_shuffle_ps(a, a, 45);
case 46: return _mm_shuffle_ps(a, a, 46);
case 47: return _mm_shuffle_ps(a, a, 47);
case 48: return _mm_shuffle_ps(a, a, 48);
case 49: return _mm_shuffle_ps(a, a, 49);
case 50: return _mm_shuffle_ps(a, a, 50);
case 51: return _mm_shuffle_ps(a, a, 51);
case 52: return _mm_shuffle_ps(a, a, 52);
case 53: return _mm_shuffle_ps(a, a, 53);
case 54: return _mm_shuffle_ps(a, a, 54);
case 55: return _mm_shuffle_ps(a, a, 55);
case 56: return _mm_shuffle_ps(a, a, 56);
case 57: return _mm_shuffle_ps(a, a, 57);
case 58: return _mm_shuffle_ps(a, a, 58);
case 59: return _mm_shuffle_ps(a, a, 59);
case 60: return _mm_shuffle_ps(a, a, 60);
case 61: return _mm_shuffle_ps(a, a, 61);
case 62: return _mm_shuffle_ps(a, a, 62);
case 63: return _mm_shuffle_ps(a, a, 63);
case 64: return _mm_shuffle_ps(a, a, 64);
case 65: return _mm_shuffle_ps(a, a, 65);
case 66: return _mm_shuffle_ps(a, a, 66);
case 67: return _mm_shuffle_ps(a, a, 67);
case 68: return _mm_shuffle_ps(a, a, 68);
case 69: return _mm_shuffle_ps(a, a, 69);
case 70: return _mm_shuffle_ps(a, a, 70);
case 71: return _mm_shuffle_ps(a, a, 71);
case 72: return _mm_shuffle_ps(a, a, 72);
case 73: return _mm_shuffle_ps(a, a, 73);
case 74: return _mm_shuffle_ps(a, a, 74);
case 75: return _mm_shuffle_ps(a, a, 75);
case 76: return _mm_shuffle_ps(a, a, 76);
case 77: return _mm_shuffle_ps(a, a, 77);
case 78: return _mm_shuffle_ps(a, a, 78);
case 79: return _mm_shuffle_ps(a, a, 79);
case 80: return _mm_shuffle_ps(a, a, 80);
case 81: return _mm_shuffle_ps(a, a, 81);
case 82: return _mm_shuffle_ps(a, a, 82);
case 83: return _mm_shuffle_ps(a, a, 83);
case 84: return _mm_shuffle_ps(a, a, 84);
case 85: return _mm_shuffle_ps(a, a, 85);
case 86: return _mm_shuffle_ps(a, a, 86);
case 87: return _mm_shuffle_ps(a, a, 87);
case 88: return _mm_shuffle_ps(a, a, 88);
case 89: return _mm_shuffle_ps(a, a, 89);
case 90: return _mm_shuffle_ps(a, a, 90);
case 91: return _mm_shuffle_ps(a, a, 91);
case 92: return _mm_shuffle_ps(a, a, 92);
case 93: return _mm_shuffle_ps(a, a, 93);
case 94: return _mm_shuffle_ps(a, a, 94);
case 95: return _mm_shuffle_ps(a, a, 95);
case 96: return _mm_shuffle_ps(a, a, 96);
case 97: return _mm_shuffle_ps(a, a, 97);
case 98: return _mm_shuffle_ps(a, a, 98);
case 99: return _mm_shuffle_ps(a, a, 99);
case 100: return _mm_shuffle_ps(a, a, 100);
case 101: return _mm_shuffle_ps(a, a, 101);
case 102: return _mm_shuffle_ps(a, a, 102);
case 103: return _mm_shuffle_ps(a, a, 103);
case 104: return _mm_shuffle_ps(a, a, 104);
case 105: return _mm_shuffle_ps(a, a, 105);
case 106: return _mm_shuffle_ps(a, a, 106);
case 107: return _mm_shuffle_ps(a, a, 107);
case 108: return _mm_shuffle_ps(a, a, 108);
case 109: return _mm_shuffle_ps(a, a, 109);
case 110: return _mm_shuffle_ps(a, a, 110);
case 111: return _mm_shuffle_ps(a, a, 111);
case 112: return _mm_shuffle_ps(a, a, 112);
case 113: return _mm_shuffle_ps(a, a, 113);
case 114: return _mm_shuffle_ps(a, a, 114);
case 115: return _mm_shuffle_ps(a, a, 115);
case 116: return _mm_shuffle_ps(a, a, 116);
case 117: return _mm_shuffle_ps(a, a, 117);
case 118: return _mm_shuffle_ps(a, a, 118);
case 119: return _mm_shuffle_ps(a, a, 119);
case 120: return _mm_shuffle_ps(a, a, 120);
case 121: return _mm_shuffle_ps(a, a, 121);
case 122: return _mm_shuffle_ps(a, a, 122);
case 123: return _mm_shuffle_ps(a, a, 123);
case 124: return _mm_shuffle_ps(a, a, 124);
case 125: return _mm_shuffle_ps(a, a, 125);
case 126: return _mm_shuffle_ps(a, a, 126);
case 127: return _mm_shuffle_ps(a, a, 127);
case 128: return _mm_shuffle_ps(a, a, 128);
case 129: return _mm_shuffle_ps(a, a, 129);
case 130: return _mm_shuffle_ps(a, a, 130);
case 131: return _mm_shuffle_ps(a, a, 131);
case 132: return _mm_shuffle_ps(a, a, 132);
case 133: return _mm_shuffle_ps(a, a, 133);
case 134: return _mm_shuffle_ps(a, a, 134);
case 135: return _mm_shuffle_ps(a, a, 135);
case 136: return _mm_shuffle_ps(a, a, 136);
case 137: return _mm_shuffle_ps(a, a, 137);
case 138: return _mm_shuffle_ps(a, a, 138);
case 139: return _mm_shuffle_ps(a, a, 139);
case 140: return _mm_shuffle_ps(a, a, 140);
case 141: return _mm_shuffle_ps(a, a, 141);
case 142: return _mm_shuffle_ps(a, a, 142);
case 143: return _mm_shuffle_ps(a, a, 143);
case 144: return _mm_shuffle_ps(a, a, 144);
case 145: return _mm_shuffle_ps(a, a, 145);
case 146: return _mm_shuffle_ps(a, a, 146);
case 147: return _mm_shuffle_ps(a, a, 147);
case 148: return _mm_shuffle_ps(a, a, 148);
case 149: return _mm_shuffle_ps(a, a, 149);
case 150: return _mm_shuffle_ps(a, a, 150);
case 151: return _mm_shuffle_ps(a, a, 151);
case 152: return _mm_shuffle_ps(a, a, 152);
case 153: return _mm_shuffle_ps(a, a, 153);
case 154: return _mm_shuffle_ps(a, a, 154);
case 155: return _mm_shuffle_ps(a, a, 155);
case 156: return _mm_shuffle_ps(a, a, 156);
case 157: return _mm_shuffle_ps(a, a, 157);
case 158: return _mm_shuffle_ps(a, a, 158);
case 159: return _mm_shuffle_ps(a, a, 159);
case 160: return _mm_shuffle_ps(a, a, 160);
case 161: return _mm_shuffle_ps(a, a, 161);
case 162: return _mm_shuffle_ps(a, a, 162);
case 163: return _mm_shuffle_ps(a, a, 163);
case 164: return _mm_shuffle_ps(a, a, 164);
case 165: return _mm_shuffle_ps(a, a, 165);
case 166: return _mm_shuffle_ps(a, a, 166);
case 167: return _mm_shuffle_ps(a, a, 167);
case 168: return _mm_shuffle_ps(a, a, 168);
case 169: return _mm_shuffle_ps(a, a, 169);
case 170: return _mm_shuffle_ps(a, a, 170);
case 171: return _mm_shuffle_ps(a, a, 171);
case 172: return _mm_shuffle_ps(a, a, 172);
case 173: return _mm_shuffle_ps(a, a, 173);
case 174: return _mm_shuffle_ps(a, a, 174);
case 175: return _mm_shuffle_ps(a, a, 175);
case 176: return _mm_shuffle_ps(a, a, 176);
case 177: return _mm_shuffle_ps(a, a, 177);
case 178: return _mm_shuffle_ps(a, a, 178);
case 179: return _mm_shuffle_ps(a, a, 179);
case 180: return _mm_shuffle_ps(a, a, 180);
case 181: return _mm_shuffle_ps(a, a, 181);
case 182: return _mm_shuffle_ps(a, a, 182);
case 183: return _mm_shuffle_ps(a, a, 183);
case 184: return _mm_shuffle_ps(a, a, 184);
case 185: return _mm_shuffle_ps(a, a, 185);
case 186: return _mm_shuffle_ps(a, a, 186);
case 187: return _mm_shuffle_ps(a, a, 187);
case 188: return _mm_shuffle_ps(a, a, 188);
case 189: return _mm_shuffle_ps(a, a, 189);
case 190: return _mm_shuffle_ps(a, a, 190);
case 191: return _mm_shuffle_ps(a, a, 191);
case 192: return _mm_shuffle_ps(a, a, 192);
case 193: return _mm_shuffle_ps(a, a, 193);
case 194: return _mm_shuffle_ps(a, a, 194);
case 195: return _mm_shuffle_ps(a, a, 195);
case 196: return _mm_shuffle_ps(a, a, 196);
case 197: return _mm_shuffle_ps(a, a, 197);
case 198: return _mm_shuffle_ps(a, a, 198);
case 199: return _mm_shuffle_ps(a, a, 199);
case 200: return _mm_shuffle_ps(a, a, 200);
case 201: return _mm_shuffle_ps(a, a, 201);
case 202: return _mm_shuffle_ps(a, a, 202);
case 203: return _mm_shuffle_ps(a, a, 203);
case 204: return _mm_shuffle_ps(a, a, 204);
case 205: return _mm_shuffle_ps(a, a, 205);
case 206: return _mm_shuffle_ps(a, a, 206);
case 207: return _mm_shuffle_ps(a, a, 207);
case 208: return _mm_shuffle_ps(a, a, 208);
case 209: return _mm_shuffle_ps(a, a, 209);
case 210: return _mm_shuffle_ps(a, a, 210);
case 211: return _mm_shuffle_ps(a, a, 211);
case 212: return _mm_shuffle_ps(a, a, 212);
case 213: return _mm_shuffle_ps(a, a, 213);
case 214: return _mm_shuffle_ps(a, a, 214);
case 215: return _mm_shuffle_ps(a, a, 215);
case 216: return _mm_shuffle_ps(a, a, 216);
case 217: return _mm_shuffle_ps(a, a, 217);
case 218: return _mm_shuffle_ps(a, a, 218);
case 219: return _mm_shuffle_ps(a, a, 219);
case 220: return _mm_shuffle_ps(a, a, 220);
case 221: return _mm_shuffle_ps(a, a, 221);
case 222: return _mm_shuffle_ps(a, a, 222);
case 223: return _mm_shuffle_ps(a, a, 223);
case 224: return _mm_shuffle_ps(a, a, 224);
case 225: return _mm_shuffle_ps(a, a, 225);
case 226: return _mm_shuffle_ps(a, a, 226);
case 227: return _mm_shuffle_ps(a, a, 227);
case 228: return a; case 229: return _mm_shuffle_ps(a, a, 229);
case 230: return _mm_shuffle_ps(a, a, 230);
case 231: return _mm_shuffle_ps(a, a, 231);
case 232: return _mm_shuffle_ps(a, a, 232);
case 233: return _mm_shuffle_ps(a, a, 233);
case 234: return _mm_shuffle_ps(a, a, 234);
case 235: return _mm_shuffle_ps(a, a, 235);
case 236: return _mm_shuffle_ps(a, a, 236);
case 237: return _mm_shuffle_ps(a, a, 237);
case 238: return _mm_shuffle_ps(a, a, 238);
case 239: return _mm_shuffle_ps(a, a, 239);
case 240: return _mm_shuffle_ps(a, a, 240);
case 241: return _mm_shuffle_ps(a, a, 241);
case 242: return _mm_shuffle_ps(a, a, 242);
case 243: return _mm_shuffle_ps(a, a, 243);
case 244: return _mm_shuffle_ps(a, a, 244);
case 245: return _mm_shuffle_ps(a, a, 245);
case 246: return _mm_shuffle_ps(a, a, 246);
case 247: return _mm_shuffle_ps(a, a, 247);
case 248: return _mm_shuffle_ps(a, a, 248);
case 249: return _mm_shuffle_ps(a, a, 249);
case 250: return _mm_shuffle_ps(a, a, 250);
case 251: return _mm_shuffle_ps(a, a, 251);
case 252: return _mm_shuffle_ps(a, a, 252);
case 253: return _mm_shuffle_ps(a, a, 253);
case 254: return _mm_shuffle_ps(a, a, 254);
case 255: return _mm_shuffle_ps(a, a, 255);
default: return a; }
    #elif defined SIMD_GRANODI_NEON
float32x4_t t0;
switch (imm8_compile_time_constant & 0xff) {
case 0: return vdupq_laneq_f32(a,0);
case 1: return vcopyq_laneq_f32(vdupq_laneq_f32(a,0),0,a,1);
case 2: return vcopyq_laneq_f32(vdupq_laneq_f32(a,0),0,a,2);
case 3: return vcopyq_laneq_f32(vdupq_laneq_f32(a,0),0,a,3);
case 4: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),3,a,0);
case 5: t0 = vcopyq_laneq_f32(a,3,a,0);
return vtrn2q_f32(t0,t0);
case 6: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),3,a,0),0,a,2);
case 7: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),3,a,0),0,a,3);
case 8: return vuzp1q_f32(a,vcopyq_laneq_f32(a,2,a,0));
case 9: return vextq_f32(vcopyq_laneq_f32(a,3,a,0),a,1);
case 10: t0 = vcopyq_laneq_f32(a,3,a,0);
return vzip2q_f32(t0,t0);
case 11: return vextq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),1,a,3),a,1);
case 12: return vcopyq_laneq_f32(vdupq_laneq_f32(a,0),1,a,3);
case 13: return vuzp2q_f32(a,vdupq_laneq_f32(a,0));
case 14: return vextq_f32(a,vcopyq_laneq_f32(a,1,a,0),2);
case 15: t0 = vextq_f32(a,a,3);
return vzip1q_f32(t0,t0);
case 16: return vzip1q_f32(a,vcopyq_laneq_f32(a,1,a,0));
case 17: t0 = vcopyq_laneq_f32(a,3,a,0);
return vuzp2q_f32(t0,t0);
case 18: return vextq_f32(vcopyq_laneq_f32(a,3,a,2),vcopyq_laneq_f32(a,2,a,0),3);
case 19: return vextq_f32(a,vcopyq_laneq_f32(a,2,a,0),3);
case 20: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),2,a,1);
case 21: return vcopyq_laneq_f32(vdupq_laneq_f32(a,1),3,a,0);
case 22: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),2,a,1),0,a,2);
case 23: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),2,a,1),0,a,3);
case 24: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),2,a,1),1,a,2);
case 25: return vextq_f32(vcopyq_laneq_f32(a,3,a,1),a,1);
case 26: return vextq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,1),1,a,2),a,1);
case 27: return vrev64q_f32(vextq_f32(a,a,2));
case 28: return vzip1q_f32(a,vextq_f32(a,a,3));
case 29: return vuzp2q_f32(a,vcopyq_laneq_f32(a,3,a,0));
case 30: return vextq_f32(a,vrev64q_f32(a),2);
case 31: return vextq_f32(a,vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),0,a,3),3);
case 32: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),3,a,0);
case 33: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),3,a,0),0,a,1);
case 34: return vextq_f32(vuzp1q_f32(a,a),a,1);
case 35: return vextq_f32(a,vuzp1q_f32(a,a),3);
case 36: return vcopyq_laneq_f32(a,3,a,0);
case 37: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),0,a,1);
case 38: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),0,a,2);
case 39: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),0,a,3);
case 40: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),1,a,2);
case 41: return vextq_f32(vcopyq_laneq_f32(a,3,a,2),a,1);
case 42: return vcopyq_laneq_f32(vdupq_laneq_f32(a,2),3,a,0);
case 43: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),1,a,2),0,a,3);
case 44: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),1,a,3);
case 45: return vuzp2q_f32(a,vextq_f32(a,a,1));
case 46: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),0,a,2),1,a,3);
case 47: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),0,a,3),1,a,3);
case 48: return vcopyq_laneq_f32(vdupq_laneq_f32(a,0),2,a,3);
case 49: return vextq_f32(vcopyq_laneq_f32(a,2,a,0),a,1);
case 50: return vzip2q_f32(a,vdupq_laneq_f32(a,0));
case 51: return vextq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),1,a,3),a,1);
case 52: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),2,a,3);
case 53: return vextq_f32(vcopyq_laneq_f32(a,2,a,1),a,1);
case 54: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),0,a,2),2,a,3);
case 55: return vextq_f32(vuzp2q_f32(a,a),a,1);
case 56: return vextq_f32(vcopyq_laneq_f32(a,1,a,0),a,1);
case 57: return vextq_f32(a,a,1);
case 58: return vextq_f32(vcopyq_laneq_f32(a,1,a,2),a,1);
case 59: return vextq_f32(vcopyq_laneq_f32(a,1,a,3),a,1);
case 60: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,0),1,a,3),2,a,3);
case 61: return vextq_f32(vcopyq_laneq_f32(a,2,a,3),a,1);
case 62: return vextq_f32(a,vextq_f32(a,a,3),2);
case 63: return vcopyq_laneq_f32(vdupq_laneq_f32(a,3),3,a,0);
case 64: return vzip1q_f32(vcopyq_laneq_f32(a,1,a,0),a);
case 65: return vzip1q_f32(vrev64q_f32(a),a);
case 66: return vextq_f32(vcopyq_laneq_f32(a,3,a,0),a,2);
case 67: return vextq_f32(vextq_f32(a,a,1),a,2);
case 68: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),3,a,1);
case 69: return vcopyq_laneq_f32(vdupq_laneq_f32(a,1),2,a,0);
case 70: return vextq_f32(vcopyq_laneq_f32(a,3,a,1),a,2);
case 71: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),3,a,1),0,a,3);
case 72: return vuzp1q_f32(a,vcopyq_laneq_f32(a,2,a,1));
case 73: return vextq_f32(vextq_f32(a,a,3),a,2);
case 74: return vextq_f32(vcopyq_laneq_f32(a,3,a,2),a,2);
case 75: return vextq_f32(vrev64q_f32(a),a,2);
case 76: return vextq_f32(vcopyq_laneq_f32(a,2,a,0),a,2);
case 77: return vextq_f32(vcopyq_laneq_f32(a,2,a,1),a,2);
case 78: return vextq_f32(a,a,2);
case 79: return vextq_f32(vcopyq_laneq_f32(a,2,a,3),a,2);
case 80: return vzip1q_f32(a,a);
case 81: return vzip1q_f32(vcopyq_laneq_f32(a,0,a,1),a);
case 82: return vzip1q_f32(vcopyq_laneq_f32(a,0,a,2),a);
case 83: return vextq_f32(a,vcopyq_laneq_f32(a,2,a,1),3);
case 84: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,1),3,a,1);
case 85: return vdupq_laneq_f32(a,1);
case 86: return vcopyq_laneq_f32(vdupq_laneq_f32(a,1),0,a,2);
case 87: return vcopyq_laneq_f32(vdupq_laneq_f32(a,1),0,a,3);
case 88: return vzip1q_f32(a,vcopyq_laneq_f32(a,0,a,2));
case 89: return vcopyq_laneq_f32(vdupq_laneq_f32(a,1),1,a,2);
case 90: t0 = vcopyq_laneq_f32(a,0,a,2);
return vzip1q_f32(t0,t0);
case 91: return vextq_f32(a,vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,1),0,a,2),3);
case 92: return vzip1q_f32(a,vcopyq_laneq_f32(a,0,a,3));
case 93: return vuzp2q_f32(a,vcopyq_laneq_f32(a,3,a,1));
case 94: return vextq_f32(a,vcopyq_laneq_f32(a,0,a,1),2);
case 95: t0 = vcopyq_laneq_f32(a,0,a,3);
return vzip1q_f32(t0,t0);
case 96: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),3,a,1);
case 97: return vzip1q_f32(vextq_f32(a,a,1),a);
case 98: return vzip1q_f32(vdupq_laneq_f32(a,2),a);
case 99: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),3,a,1),0,a,3);
case 100: return vcopyq_laneq_f32(a,3,a,1);
case 101: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,1),3,a,1);
case 102: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,1),0,a,2);
case 103: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,1),0,a,3);
case 104: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,1),1,a,2);
case 105: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,1),3,a,1),1,a,2);
case 106: return vcopyq_laneq_f32(vdupq_laneq_f32(a,2),3,a,1);
case 107: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,1),1,a,2),0,a,3);
case 108: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,1),1,a,3);
case 109: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,1),3,a,1),1,a,3);
case 110: return vextq_f32(a,vcopyq_laneq_f32(a,0,a,2),2);
case 111: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,1),0,a,3),1,a,3);
case 112: return vzip1q_f32(vcopyq_laneq_f32(a,1,a,3),a);
case 113: return vrev64q_f32(vcopyq_laneq_f32(a,2,a,1));
case 114: return vzip1q_f32(vextq_f32(a,a,2),a);
case 115: return vzip1q_f32(vdupq_laneq_f32(a,3),a);
case 116: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,1),2,a,3);
case 117: return vtrn2q_f32(a,vcopyq_laneq_f32(a,3,a,1));
case 118: return vzip2q_f32(a,vdupq_laneq_f32(a,1));
case 119: return vextq_f32(a,vuzp2q_f32(a,a),3);
case 120: return vuzp1q_f32(a,vextq_f32(a,a,3));
case 121: return vextq_f32(a,vcopyq_laneq_f32(a,0,a,1),1);
case 122: return vzip2q_f32(a,vcopyq_laneq_f32(a,3,a,1));
case 123: return vextq_f32(vcopyq_laneq_f32(a,1,a,3),vcopyq_laneq_f32(a,0,a,1),1);
case 124: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,1),1,a,3),2,a,3);
case 125: return vuzp2q_f32(a,vextq_f32(a,a,2));
case 126: return vextq_f32(a,vcopyq_laneq_f32(a,0,a,3),2);
case 127: return vcopyq_laneq_f32(vdupq_laneq_f32(a,3),3,a,1);
case 128: return vuzp1q_f32(vcopyq_laneq_f32(a,2,a,0),a);
case 129: return vrev64q_f32(vcopyq_laneq_f32(a,3,a,0));
case 130: return vuzp1q_f32(vextq_f32(a,a,2),a);
case 131: return vextq_f32(a,vcopyq_laneq_f32(a,1,a,0),3);
case 132: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),3,a,2);
case 133: return vuzp1q_f32(vdupq_laneq_f32(a,1),a);
case 134: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),0,a,2),3,a,2);
case 135: return vuzp1q_f32(vextq_f32(a,a,3),a);
case 136: return vuzp1q_f32(a,a);
case 137: return vuzp1q_f32(vcopyq_laneq_f32(a,0,a,1),a);
case 138: return vuzp1q_f32(vcopyq_laneq_f32(a,0,a,2),a);
case 139: return vuzp1q_f32(vcopyq_laneq_f32(a,0,a,3),a);
case 140: return vuzp1q_f32(vcopyq_laneq_f32(a,2,a,3),a);
case 141: return vuzp1q_f32(vextq_f32(a,a,1),a);
case 142: return vextq_f32(a,vcopyq_laneq_f32(a,1,a,2),2);
case 143: return vuzp1q_f32(vdupq_laneq_f32(a,3),a);
case 144: return vextq_f32(vcopyq_laneq_f32(a,3,a,0),a,3);
case 145: return vextq_f32(vcopyq_laneq_f32(a,3,a,1),a,3);
case 146: return vextq_f32(vcopyq_laneq_f32(a,3,a,2),a,3);
case 147: return vextq_f32(a,a,3);
case 148: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,1),3,a,2);
case 149: return vcopyq_laneq_f32(vdupq_laneq_f32(a,1),3,a,2);
case 150: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,1),0,a,2),3,a,2);
case 151: return vextq_f32(a,vcopyq_laneq_f32(a,0,a,1),3);
case 152: return vuzp1q_f32(a,vcopyq_laneq_f32(a,0,a,1));
case 153: t0 = vcopyq_laneq_f32(a,0,a,1);
return vuzp1q_f32(t0,t0);
case 154: return vcopyq_laneq_f32(vdupq_laneq_f32(a,2),2,a,1);
case 155: return vextq_f32(a,vcopyq_laneq_f32(a,0,a,2),3);
case 156: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,1),3,a,2),1,a,3);
case 157: return vuzp2q_f32(a,vcopyq_laneq_f32(a,3,a,2));
case 158: return vextq_f32(a,vextq_f32(a,a,1),2);
case 159: return vextq_f32(a,vcopyq_laneq_f32(a,0,a,3),3);
case 160: return vtrn1q_f32(a,a);
case 161: return vrev64q_f32(vcopyq_laneq_f32(a,3,a,2));
case 162: return vtrn1q_f32(vcopyq_laneq_f32(a,0,a,2),a);
case 163: return vextq_f32(a,vcopyq_laneq_f32(a,1,a,2),3);
case 164: return vcopyq_laneq_f32(a,3,a,2);
case 165: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,1),3,a,2);
case 166: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,2),3,a,2);
case 167: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,2),0,a,3);
case 168: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,2),3,a,2);
case 169: return vcopyq_laneq_f32(vdupq_laneq_f32(a,2),0,a,1);
case 170: return vdupq_laneq_f32(a,2);
case 171: return vcopyq_laneq_f32(vdupq_laneq_f32(a,2),0,a,3);
case 172: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,2),1,a,3);
case 173: return vuzp2q_f32(a,vdupq_laneq_f32(a,2));
case 174: return vcopyq_laneq_f32(vdupq_laneq_f32(a,2),1,a,3);
case 175: t0 = vcopyq_laneq_f32(a,0,a,3);
return vtrn1q_f32(t0,t0);
case 176: return vrev64q_f32(vcopyq_laneq_f32(a,1,a,0));
case 177: return vrev64q_f32(a);
case 178: return vrev64q_f32(vcopyq_laneq_f32(a,1,a,2));
case 179: return vextq_f32(a,vcopyq_laneq_f32(a,1,a,3),3);
case 180: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,2),2,a,3);
case 181: return vrev64q_f32(vcopyq_laneq_f32(a,0,a,1));
case 182: return vzip2q_f32(a,vextq_f32(a,a,3));
case 183: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,2),0,a,3),2,a,3);
case 184: return vuzp1q_f32(a,vcopyq_laneq_f32(a,0,a,3));
case 185: return vextq_f32(a,vcopyq_laneq_f32(a,0,a,2),1);
case 186: return vzip2q_f32(a,vcopyq_laneq_f32(a,3,a,2));
case 187: t0 = vcopyq_laneq_f32(a,0,a,3);
return vuzp1q_f32(t0,t0);
case 188: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,3,a,2),1,a,3),2,a,3);
case 189: return vrev64q_f32(vcopyq_laneq_f32(a,0,a,3));
case 190: return vzip2q_f32(a,vrev64q_f32(a));
case 191: return vcopyq_laneq_f32(vdupq_laneq_f32(a,3),3,a,2);
case 192: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),2,a,0);
case 193: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),2,a,0),0,a,1);
case 194: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),2,a,0),0,a,2);
case 195: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),2,a,0),0,a,3);
case 196: return vcopyq_laneq_f32(a,2,a,0);
case 197: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),0,a,1);
case 198: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),0,a,2);
case 199: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),0,a,3);
case 200: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),1,a,2);
case 201: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),0,a,1),1,a,2);
case 202: return vzip2q_f32(vcopyq_laneq_f32(a,3,a,0),a);
case 203: return vzip2q_f32(vextq_f32(a,a,1),a);
case 204: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,0),1,a,3);
case 205: return vuzp2q_f32(a,vcopyq_laneq_f32(a,1,a,0));
case 206: return vextq_f32(a,vcopyq_laneq_f32(a,1,a,3),2);
case 207: return vcopyq_laneq_f32(vdupq_laneq_f32(a,3),2,a,0);
case 208: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),2,a,1);
case 209: return vuzp2q_f32(vcopyq_laneq_f32(a,3,a,0),a);
case 210: return vuzp2q_f32(vextq_f32(a,a,1),a);
case 211: return vextq_f32(a,vcopyq_laneq_f32(a,2,a,3),3);
case 212: return vcopyq_laneq_f32(a,2,a,1);
case 213: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,1),2,a,1);
case 214: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,1),0,a,2);
case 215: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,1),0,a,3);
case 216: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,1),1,a,2);
case 217: return vuzp2q_f32(vcopyq_laneq_f32(a,3,a,2),a);
case 218: return vzip2q_f32(vcopyq_laneq_f32(a,3,a,1),a);
case 219: return vcopyq_laneq_f32(vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,1),1,a,2),0,a,3);
case 220: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,2,a,1),1,a,3);
case 221: return vuzp2q_f32(a,a);
case 222: return vuzp2q_f32(vcopyq_laneq_f32(a,1,a,2),a);
case 223: return vuzp2q_f32(vcopyq_laneq_f32(a,1,a,3),a);
case 224: return vcopyq_laneq_f32(a,1,a,0);
case 225: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),0,a,1);
case 226: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),0,a,2);
case 227: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),0,a,3);
case 228: return a;
case 229: return vcopyq_laneq_f32(a,0,a,1);
case 230: return vcopyq_laneq_f32(a,0,a,2);
case 231: return vcopyq_laneq_f32(a,0,a,3);
case 232: return vcopyq_laneq_f32(a,1,a,2);
case 233: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,1),1,a,2);
case 234: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,2),1,a,2);
case 235: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,2),0,a,3);
case 236: return vcopyq_laneq_f32(a,1,a,3);
case 237: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,1),1,a,3);
case 238: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,2),1,a,3);
case 239: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,3),1,a,3);
case 240: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,0),2,a,3);
case 241: return vrev64q_f32(vcopyq_laneq_f32(a,2,a,3));
case 242: return vzip2q_f32(a,vcopyq_laneq_f32(a,2,a,0));
case 243: return vcopyq_laneq_f32(vdupq_laneq_f32(a,3),1,a,0);
case 244: return vcopyq_laneq_f32(a,2,a,3);
case 245: return vtrn2q_f32(a,a);
case 246: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,2),2,a,3);
case 247: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,0,a,3),2,a,3);
case 248: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,2),2,a,3);
case 249: return vextq_f32(a,vcopyq_laneq_f32(a,0,a,3),1);
case 250: return vzip2q_f32(a,a);
case 251: return vzip2q_f32(vcopyq_laneq_f32(a,2,a,3),a);
case 252: return vcopyq_laneq_f32(vcopyq_laneq_f32(a,1,a,3),2,a,3);
case 253: return vuzp2q_f32(a,vcopyq_laneq_f32(a,1,a,3));
case 254: return vzip2q_f32(a,vcopyq_laneq_f32(a,2,a,3));
case 255: return vdupq_laneq_f32(a,3);
default: return a; }
    #endif
}

#define sg_shuffle_pd(a, src1_compile_time_constant, \
    src0_compile_time_constant) \
    sg_shuffle_pd_switch_(a, sg_sse2_shuffle64_imm( \
        src1_compile_time_constant, \
        src0_compile_time_constant))
static inline sg_pd sg_shuffle_pd_switch_(const sg_pd a,
    const int imm8_compile_time_constant)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    const double array[] = { a.d0, a.d1 };
    sg_pd result;
    result.d0 = array[imm8_compile_time_constant & 1];
    result.d1 = array[(imm8_compile_time_constant >> 1) & 1];
    return result;
    #elif defined SIMD_GRANODI_SSE2
    switch(imm8_compile_time_constant & 3)
    {
        case 0: return _mm_shuffle_pd(a, a, 0);
        case 1: return _mm_shuffle_pd(a, a, 1);
        case 2: return a;
        case 3: return _mm_shuffle_pd(a, a, 3);
        default: return a;
    }
    #elif defined SIMD_GRANODI_NEON
    switch(imm8_compile_time_constant & 3)
    {
        case 0: return vcopyq_laneq_f64(a, 1, a, 0);
        case 1: return vextq_f64(a, a, 1);
        case 2: return a;
        case 3: return vcopyq_laneq_f64(a, 0, a, 1);
        default: return a;
    }
    #endif
}

// Set

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_setzero_pi32() {
    sg_pi32 result;
    result.i0 = 0; result.i1 = 0; result.i2 = 0; result.i3 = 0;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setzero_pi32() _mm_setzero_si128()
#elif defined SIMD_GRANODI_NEON
#define sg_setzero_pi32() vdupq_n_s32(0)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_set1_pi32(const int32_t si) {
    sg_pi32 result;
    result.i0 = si; result.i1 = si; result.i2 = si; result.i3 = si;
    return result;
}
static inline sg_pi32 sg_set1_from_u32_pi32(const uint32_t i) {
    sg_pi32 result;
    result.i0 = sg_bitcast_u32x1_s32x1(i); result.i1 = sg_bitcast_u32x1_s32x1(i);
    result.i2 = sg_bitcast_u32x1_s32x1(i); result.i3 = sg_bitcast_u32x1_s32x1(i);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set1_pi32(si) _mm_set1_epi32(si)
#define sg_set1_from_u32_pi32(i) _mm_set1_epi32(sg_bitcast_u32x1_s32x1(i))
#elif defined SIMD_GRANODI_NEON
#define sg_set1_pi32(si) vdupq_n_s32(si)
#define sg_set1_from_u32_pi32(i) vreinterpretq_s32_u32(vdupq_n_u32(i))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_set_pi32(const int32_t si3, const int32_t si2,
    const int32_t si1, const int32_t si0)
{
    sg_pi32 result;
    result.i0 = si0; result.i1 = si1; result.i2 = si2; result.i3 = si3;
    return result;
}
static inline sg_pi32 sg_set_from_u32_pi32(const uint32_t i3,
    const uint32_t i2, const uint32_t i1, const uint32_t i0)
{
    sg_pi32 result;
    result.i0 = sg_bitcast_u32x1_s32x1(i0); result.i1 = sg_bitcast_u32x1_s32x1(i1);
    result.i2 = sg_bitcast_u32x1_s32x1(i2); result.i3 = sg_bitcast_u32x1_s32x1(i3);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set_pi32(si3, si2, si1, si0) _mm_set_epi32(si3, si2, si1, si0)
#define sg_set_from_u32_pi32(i3, i2, i1, i0) \
    _mm_set_epi32(sg_bitcast_u32x1_s32x1(i3), sg_bitcast_u32x1_s32x1(i2), \
        sg_bitcast_u32x1_s32x1(i1), sg_bitcast_u32x1_s32x1(i0))
#elif defined SIMD_GRANODI_NEON
#define sg_set_pi32(si3, si2, si1, si0) \
    vsetq_lane_s32(si3, vsetq_lane_s32(si2, vsetq_lane_s32(si1, \
        vsetq_lane_s32(si0, vdupq_n_s32(0), 0), 1), 2), 3)
#define sg_set_from_u32_pi32(i3, i2, i1, i0) \
    vreinterpretq_s32_u32(vsetq_lane_u32(i3, vsetq_lane_u32(i2, \
        vsetq_lane_u32(i1, vsetq_lane_u32(i0, \
            vdupq_n_u32(0), 0), 1), 2), 3))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_setzero_pi64() {
    sg_pi64 result;
    result.l0 = 0; result.l1 = 0;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setzero_pi64() _mm_setzero_si128()
#elif defined SIMD_GRANODI_NEON
#define sg_setzero_pi64() vdupq_n_s64(0)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_set1_pi64(const int64_t si) {
    sg_pi64 result;
    result.l0 = si; result.l1 = si;
    return result;
}
static inline sg_pi64 sg_set1_from_u64_pi64(const uint64_t i) {
    sg_pi64 result;
    result.l0 = sg_bitcast_u64x1_s64x1(i); result.l1 = sg_bitcast_u64x1_s64x1(i);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set1_pi64(si) _mm_set1_epi64x(si)
#define sg_set1_from_u64_pi64(i) _mm_set1_epi64x(sg_bitcast_u64x1_s64x1(i))
#elif defined SIMD_GRANODI_NEON
#define sg_set1_pi64(si) vdupq_n_s64(si)
#define sg_set1_from_u64_pi64(i) vreinterpretq_s64_u64(vdupq_n_u64(i))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_set_pi64(const int64_t si1, const int64_t si0) {
    sg_pi64 result;
    result.l0 = si0; result.l1 = si1;
    return result;
}
static inline sg_pi64 sg_set_from_u64_pi64(const uint64_t i1,
    const uint64_t i0)
{
    sg_pi64 result;
    result.l0 = sg_bitcast_u64x1_s64x1(i0); result.l1 = sg_bitcast_u64x1_s64x1(i1);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set_pi64(si1, si0) _mm_set_epi64x(si1, si0)
#define sg_set_from_u64_pi64(i1, i0) \
    _mm_set_epi64x(sg_bitcast_u64x1_s64x1(i1), sg_bitcast_u64x1_s64x1(i0))
#elif defined SIMD_GRANODI_NEON
#define sg_set_pi64(si1, si0) vsetq_lane_s64(si1, vsetq_lane_s64(si0, \
    vdupq_n_s64(0), 0), 1)
#define sg_set_from_u64_pi64(i1, i0) vreinterpretq_s64_u64(vsetq_lane_u64(i1, \
    vsetq_lane_u64(i0, vdupq_n_u64(0), 0), 1))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_setzero_ps() {
    sg_ps result;
    result.f0 = 0.0f; result.f1 = 0.0f; result.f2 = 0.0f; result.f3 = 0.0f;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setzero_ps() _mm_setzero_ps()
#elif defined SIMD_GRANODI_NEON
#define sg_setzero_ps() vdupq_n_f32(0.0f)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_set1_ps(const float f) {
    sg_ps result;
    result.f0 = f; result.f1 = f; result.f2 = f; result.f3 = f;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set1_ps _mm_set1_ps
#elif defined SIMD_GRANODI_NEON
#define sg_set1_ps vdupq_n_f32
#endif
#define sg_set1_from_u32_ps(i) sg_bitcast_pi32_ps(sg_set1_from_u32_pi32(i))

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_set_ps(const float f3, const float f2, const float f1,
    const float f0)
{
    sg_ps result;
    result.f0 = f0; result.f1 = f1; result.f2 = f2; result.f3 = f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set_ps(f3, f2, f1, f0) _mm_set_ps(f3, f2, f1, f0)
#elif defined SIMD_GRANODI_NEON
#define sg_set_ps(f3, f2, f1, f0) vsetq_lane_f32(f3, \
    vsetq_lane_f32(f2, vsetq_lane_f32(f1, \
        vsetq_lane_f32(f0, vdupq_n_f32(0.0f), 0), 1), 2), 3)
#endif
#define sg_set_from_u32_ps(i3, i2, i1, i0) \
    sg_bitcast_pi32_ps(sg_set_from_u32_pi32(i3, i2, i1, i0))

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_setzero_pd() {
    sg_pd result;
    result.d0 = 0.0; result.d1 = 0.0;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setzero_pd() _mm_setzero_pd()
#elif defined SIMD_GRANODI_NEON
#define sg_setzero_pd() vdupq_n_f64(0.0)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_set1_pd(const double d) {
    sg_pd result;
    result.d0 = d; result.d1 = d;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set1_pd _mm_set1_pd
#elif defined SIMD_GRANODI_NEON
#define sg_set1_pd vdupq_n_f64
#endif
#define sg_set1_from_u64_pd(l) sg_bitcast_pi64_pd(sg_set1_from_u64_pi64(l))

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_set_pd(const double d1, const double d0) {
    sg_pd result;
    result.d0 = d0, result.d1 = d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set_pd _mm_set_pd
#elif defined SIMD_GRANODI_NEON
#define sg_set_pd(d1, d0) \
    vsetq_lane_f64(d1, vsetq_lane_f64(d0, vdupq_n_f64(0.0), 0), 1)
#endif
#define sg_set_from_u64_pd(l1, l0) sg_bitcast_pi64_pd(sg_set_from_u64_pi64(l1, l0))

// Set from generic

static inline sg_pi32 sg_set_fromg_pi32(const sg_generic_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set_epi32(a.i3, a.i2, a.i1, a.i0);
    #elif defined SIMD_GRANODI_NEON
    return sg_set_pi32(a.i3, a.i2, a.i1, a.i0);
    #endif
}

static inline sg_pi64 sg_set_fromg_pi64(const sg_generic_pi64 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set_epi64x(a.l1, a.l0);
    #elif defined SIMD_GRANODI_NEON
    return sg_set_pi64(a.l1, a.l0);
    #endif
}

static inline sg_ps sg_set_fromg_ps(const sg_generic_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set_ps(a.f3, a.f2, a.f1, a.f0);
    #elif defined SIMD_GRANODI_NEON
    return sg_set_ps(a.f3, a.f2, a.f1, a.f0);
    #endif
}

static inline sg_pd sg_set_fromg_pd(const sg_generic_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set_pd(a.d1, a.d0);
    #elif defined SIMD_GRANODI_NEON
    return sg_set_pd(a.d1, a.d0);
    #endif
}

// Get

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline int32_t sg_get0_pi32(const sg_pi32 a) { return a.i0; }
#elif defined SIMD_GRANODI_SSE2
#define sg_get0_pi32 _mm_cvtsi128_si32
#elif defined SIMD_GRANODI_NEON
#define sg_get0_pi32(a) vgetq_lane_s32(a, 0)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline int32_t sg_get1_pi32(const sg_pi32 a) { return a.i1; }
#elif defined SIMD_GRANODI_SSE2
#define sg_get1_pi32(a) _mm_cvtsi128_si32( \
    _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(3, 2, 1, 1)))
#elif defined SIMD_GRANODI_NEON
#define sg_get1_pi32(a) vgetq_lane_s32(a, 1)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline int32_t sg_get2_pi32(const sg_pi32 a) { return a.i2; }
#elif defined SIMD_GRANODI_SSE2
#define sg_get2_pi32(a) _mm_cvtsi128_si32( \
    _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(3, 2, 1, 2)))
#elif defined SIMD_GRANODI_NEON
#define sg_get2_pi32(a) vgetq_lane_s32(a, 2)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline int32_t sg_get3_pi32(const sg_pi32 a) { return a.i3; }
#elif defined SIMD_GRANODI_SSE2
#define sg_get3_pi32(a) _mm_cvtsi128_si32( \
    _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(3, 2, 1, 3)))
#elif defined SIMD_GRANODI_NEON
#define sg_get3_pi32(a) vgetq_lane_s32(a, 3)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline int64_t sg_get0_pi64(const sg_pi64 a) { return a.l0; }
#elif defined SIMD_GRANODI_SSE2
#define sg_get0_pi64 _mm_cvtsi128_si64
#elif defined SIMD_GRANODI_NEON
#define sg_get0_pi64(a) vgetq_lane_s64(a, 0)
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline int64_t sg_get1_pi64(const sg_pi64 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.l1;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtsi128_si64(_mm_unpackhi_epi64(a, a));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_get1_pi64(a) vgetq_lane_s64(a, 1)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline float sg_get0_ps(const sg_ps a) { return a.f0; }
#elif defined SIMD_GRANODI_SSE2
#define sg_get0_ps _mm_cvtss_f32
#elif defined SIMD_GRANODI_NEON
#define sg_get0_ps(a) vgetq_lane_f32(a, 0)
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline float sg_get1_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.f1;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtss_f32(
        _mm_shuffle_ps(a, a, sg_sse2_shuffle32_imm(3, 2, 1, 1)));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_get1_ps(a) vgetq_lane_f32(a, 1)
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline float sg_get2_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.f2;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtss_f32(
        _mm_shuffle_ps(a, a, sg_sse2_shuffle32_imm(3, 2, 1, 2)));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_get2_ps(a) vgetq_lane_f32(a, 2)
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline float sg_get3_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.f3;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtss_f32(
        _mm_shuffle_ps(a, a, sg_sse2_shuffle32_imm(3, 2, 1, 3)));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_get3_ps(a) vgetq_lane_f32(a, 3)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline double sg_get0_pd(const sg_pd a) { return a.d0; }
#elif defined SIMD_GRANODI_SSE2
#define sg_get0_pd _mm_cvtsd_f64
#elif defined SIMD_GRANODI_NEON
#define sg_get0_pd(a) vgetq_lane_f64(a, 0)
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline double sg_get1_pd(const sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.d1;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtsd_f64(_mm_unpackhi_pd(a, a));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_get1_pd(a) vgetq_lane_f64(a, 1)
#endif

static inline sg_generic_pi32 sg_getg_pi32(const sg_pi32 a) {
    sg_generic_pi32 result;
    result.i0 = sg_get0_pi32(a); result.i1 = sg_get1_pi32(a);
    result.i2 = sg_get2_pi32(a); result.i3 = sg_get3_pi32(a);
    return result;
}

static inline sg_generic_pi64 sg_getg_pi64(const sg_pi64 a) {
    sg_generic_pi64 result;
    result.l0 = sg_get0_pi64(a); result.l1 = sg_get1_pi64(a);
    return result;
}

static inline sg_generic_ps sg_getg_ps(const sg_ps a) {
    sg_generic_ps result;
    result.f0 = sg_get0_ps(a); result.f1 = sg_get1_ps(a);
    result.f2 = sg_get2_ps(a); result.f3 = sg_get3_ps(a);
    return result;
}

static inline sg_generic_pd sg_getg_pd(const sg_pd a) {
    sg_generic_pd result;
    result.d0 = sg_get0_pd(a); result.d1 = sg_get1_pd(a);
    return result;
}

static inline bool sg_debug_eq_pi32(const sg_pi32 a, const int32_t i3,
    const int32_t i2, const int32_t i1, const int32_t i0)
{
    const sg_generic_pi32 ag = sg_getg_pi32(a);
    return ag.i3 == i3 && ag.i2 == i2 && ag.i1 == i1 && ag.i0 == i0;
}

static inline bool sg_debug_eq_pi64(const sg_pi64 a, const int64_t l1,
    const int64_t l0)
{
    const sg_generic_pi64 ag = sg_getg_pi64(a);
    return ag.l1 == l1 && ag.l0 == l0;
}

// The debug_eq functions for floating point use bitwise equality test
// to catch signed zero etc
static inline bool sg_debug_eq_ps(const sg_ps a, const float f3, const float f2,
    const float f1, const float f0)
{
    return sg_debug_eq_pi32(sg_bitcast_ps_pi32(a), sg_bitcast_f32x1_s32x1(f3),
        sg_bitcast_f32x1_s32x1(f2), sg_bitcast_f32x1_s32x1(f1),
        sg_bitcast_f32x1_s32x1(f0));
}

static inline bool sg_debug_eq_pd(const sg_pd a, const double d1,
    const double d0)
{
    return sg_debug_eq_pi64(sg_bitcast_pd_pi64(a),
        sg_bitcast_f64x1_s64x1(d1), sg_bitcast_f64x1_s64x1(d0));
}

// Conversion

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi64 sg_cvt_pi32_pi64(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi64 result;
    result.l0 = (int64_t) a.i0; result.l1 = (int64_t) a.i1;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    const __m128i a_shuffled = _mm_shuffle_epi32(a,
        sg_sse2_shuffle32_imm(1, 1, 0, 0));
    const __m128i sign_extend = _mm_and_si128(
        _mm_set_epi32(sg_allset_s32, 0, sg_allset_s32, 0),
        _mm_cmplt_epi32(a_shuffled, _mm_setzero_si128()));
    const __m128i result = _mm_and_si128(
        _mm_set_epi32(0, sg_allset_s32, 0, sg_allset_s32),
        a_shuffled);
    return _mm_or_si128(sign_extend, result);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvt_pi32_pi64(a) vshll_n_s32(vget_low_s32(a), 0)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_cvt_pi32_ps(const sg_pi32 a) {
    sg_ps result;
    result.f0 = (float) a.i0; result.f1 = (float) a.i1;
    result.f2 = (float) a.i2; result.f3 = (float) a.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvt_pi32_ps _mm_cvtepi32_ps
#elif defined SIMD_GRANODI_NEON
#define sg_cvt_pi32_ps vcvtq_f32_s32
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_cvt_pi32_pd(const sg_pi32 a) {
    sg_pd result;
    result.d0 = (double) a.i0; result.d1 = (double) a.i1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvt_pi32_pd _mm_cvtepi32_pd
#elif defined SIMD_GRANODI_NEON
#define sg_cvt_pi32_pd(a) vcvtq_f64_s64(vshll_n_s32(vget_low_s32(a), 0))
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
static inline sg_pi32 sg_cvt_pi64_pi32(const sg_pi64 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi32 result;
    result.i0 = (int32_t) (a.l0 & 0xffffffff);
    result.i1 = (int32_t) (a.l1 & 0xffffffff);
    result.i2 = 0; result.i3 = 0;
    return result;
    #elif defined SIMD_GRANODI_NEON
    const int32x4_t cast = vreinterpretq_s32_s64(a);
    return vcombine_s32(vget_low_s32(vcopyq_laneq_s32(cast, 1, cast, 2)),
        vdup_n_s32(0));
    #endif
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvt_pi64_pi32(a) _mm_and_si128(_mm_set_epi64x(0, sg_allset_s64), \
    _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(3, 2, 2, 0)))
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_ps sg_cvt_pi64_ps(const sg_pi64 a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    sg_ps result;
    result.f0 = (float) a.l0; result.f1 = (float) a.l1;
    result.f2 = 0.0f; result.f3 = 0.0f;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    const int64_t si0 = _mm_cvtsi128_si64(a),
        si1 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(a, a));
    __m128 result = _mm_cvtsi64_ss(_mm_setzero_ps(), si1);
    result = _mm_shuffle_ps(result, result, sg_sse2_shuffle32_imm(3, 2, 0, 0));
    return _mm_cvtsi64_ss(result, si0);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvt_pi64_ps(a) \
    vcombine_f32(vcvt_f32_f64(vcvtq_f64_s64(a)), vdup_n_f32(0.0f))
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pd sg_cvt_pi64_pd(const sg_pi64 a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    sg_pd result;
    result.d0 = (double) a.l0; result.d1 = (double) a.l1;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    const int64_t si0 = _mm_cvtsi128_si64(a),
        si1 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(a, a));
    __m128d result = _mm_cvtsi64_sd(_mm_setzero_pd(), si1);
    result = _mm_shuffle_pd(result, result, sg_sse2_shuffle64_imm(0, 0));
    return _mm_cvtsi64_sd(result, si0);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvt_pi64_pd vcvtq_f64_s64
#endif

// Use current rounding mode (default round-to-nearest with 0.5 rounding down)
#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_cvt_ps_pi32(const sg_ps a) {
    sg_pi32 result;
    result.i0 = (int32_t) rintf(a.f0); result.i1 = (int32_t) rintf(a.f1);
    result.i2 = (int32_t) rintf(a.f2); result.i3 = (int32_t) rintf(a.f3);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvt_ps_pi32 _mm_cvtps_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_cvt_ps_pi32 vcvtnq_s32_f32
#endif

// cvtt (extra t) methods use truncation instead of rounding
#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_cvtt_ps_pi32(const sg_ps a) {
    sg_pi32 result;
    result.i0 = (int32_t) a.f0; result.i1 = (int32_t) a.f1;
    result.i2 = (int32_t) a.f2; result.i3 = (int32_t) a.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtt_ps_pi32 _mm_cvttps_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_cvtt_ps_pi32 vcvtq_s32_f32
#endif

// cvtf methods round towards minus infinity (floor)
#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi32 sg_cvtf_ps_pi32(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi32 result;
    result.i0 = (int32_t) floorf(a.f0); result.i1 = (int32_t) floorf(a.f1);
    result.i2 = (int32_t) floorf(a.f2); result.i3 = (int32_t) floorf(a.f3);
    return result;
    #elif defined SIMD_GRANODI_SSE2
    const __m128i trunc = _mm_cvtps_epi32(a);
    const __m128 trunc_ps = _mm_cvtepi32_ps(trunc);
    return _mm_sub_epi32(trunc, _mm_and_si128(
        _mm_castps_si128(_mm_cmpgt_ps(trunc_ps, a)), _mm_set1_epi32(1)));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvtf_ps_pi32 vcvtmq_s32_f32
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi64 sg_cvt_ps_pi64(sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi64 result;
    result.l0 = (int64_t) rintf(a.f0);
    result.l1 = (int64_t) rintf(a.f1);
    return result;
    #elif defined SIMD_GRANODI_SSE2
    int64_t si0 = (int64_t) rintf(_mm_cvtss_f32(a)),
        si1 = (int64_t) rintf(_mm_cvtss_f32(
            _mm_shuffle_ps(a, a, sg_sse2_shuffle32_imm(3, 2, 1, 1))));
    return _mm_set_epi64x(si1, si0);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvt_ps_pi64(a) vcvtnq_s64_f64(vcvt_f64_f32(vget_low_f32(a)))
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi64 sg_cvtt_ps_pi64(sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi64 result;
    result.l0 = (int64_t) a.f0; result.l1 = (int64_t) a.f1;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    int64_t si0 = (int64_t) _mm_cvtss_f32(a),
        si1 = (int64_t) _mm_cvtss_f32(
            _mm_shuffle_ps(a, a, sg_sse2_shuffle32_imm(3, 2, 1, 1)));
    return _mm_set_epi64x(si1, si0);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvtt_ps_pi64(a) vcvtq_s64_f64(vcvt_f64_f32(vget_low_f32(a)))
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi64 sg_cvtf_ps_pi64(sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi64 result;
    result.l0 = (int64_t) floorf(a.f0); result.l1 = (int64_t) floorf(a.f1);
    return result;
    #elif defined SIMD_GRANODI_SSE2
    int64_t si0 = (int64_t) floorf(_mm_cvtss_f32(a)),
        si1 = (int64_t) floorf(_mm_cvtss_f32(
            _mm_shuffle_ps(a, a, sg_sse2_shuffle32_imm(3, 2, 1, 1))));
    return _mm_set_epi64x(si1, si0);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvtf_ps_pi64(a) vcvtmq_s64_f64(vcvt_f64_f32(vget_low_f32(a)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_cvt_ps_pd(sg_ps a) {
    sg_pd result;
    result.d0 = (double) a.f0; result.d1 = (double) a.f1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvt_ps_pd _mm_cvtps_pd
#elif defined SIMD_GRANODI_NEON
#define sg_cvt_ps_pd(a) vcvt_f64_f32(vget_low_f32(a))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_cvt_pd_pi32(sg_pd a) {
    sg_pi32 result;
    result.i0 = (int32_t) rint(a.d0); result.i1 = (int32_t) rint(a.d1);
    result.i2 = 0; result.i3 = 0;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvt_pd_pi32 _mm_cvtpd_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_cvt_pd_pi32(a) sg_cvt_pi64_pi32(vcvtnq_s64_f64(a))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_cvtt_pd_pi32(sg_pd a) {
    sg_pi32 result;
    result.i0 = (int32_t) a.d0; result.i1 = (int32_t) a.d1;
    result.i2 = 0; result.i3 = 0;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtt_pd_pi32 _mm_cvttpd_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_cvtt_pd_pi32(a) sg_cvt_pi64_pi32(vcvtq_s64_f64(a))
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi32 sg_cvtf_pd_pi32(sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi32 result;
    result.i0 = (int32_t) floor(a.d0); result.i1 = (int32_t) floor(a.d1);
    result.i2 = 0; result.i3 = 0;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    const __m128i trunc = _mm_cvtpd_epi32(a);
    const __m128d trunc_pd = _mm_cvtepi32_pd(trunc);
    __m128i cmp_epi32 = _mm_castpd_si128(_mm_cmpgt_pd(trunc_pd, a));
    cmp_epi32 = _mm_shuffle_epi32(cmp_epi32, sg_sse2_shuffle32_imm(3, 2, 2, 0));
    cmp_epi32 = _mm_and_si128(cmp_epi32, _mm_set_epi32(0, 0, -1, -1));
    return _mm_sub_epi32(trunc, _mm_and_si128(cmp_epi32, _mm_set1_epi32(1)));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvtf_pd_pi32(a) sg_cvt_pi64_pi32(vcvtmq_s64_f64(a))
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi64 sg_cvt_pd_pi64(sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi64 result;
    result.l0 = (int64_t) rint(a.d0); result.l1 = (int64_t) rint(a.d1);
    return result;
    #elif defined SIMD_GRANODI_SSE2
    const int64_t si0 = _mm_cvtsd_si64(a),
        si1 = _mm_cvtsd_si64(_mm_unpackhi_pd(a, a));
    return _mm_set_epi64x(si1, si0);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvt_pd_pi64 vcvtnq_s64_f64
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi64 sg_cvtt_pd_pi64(sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi64 result;
    result.l0 = (int64_t) a.d0; result.l1 = (int64_t) a.d1;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    const int64_t si0 = _mm_cvttsd_si64(a),
        si1 = _mm_cvttsd_si64(_mm_unpackhi_pd(a, a));
    return _mm_set_epi64x(si1, si0);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvtt_pd_pi64 vcvtq_s64_f64
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi64 sg_cvtf_pd_pi64(sg_pd a) {
    const sg_generic_pd ag = sg_getg_pd(a); sg_generic_pi64 result;
    result.l0 = (int64_t) floor(ag.d0); result.l1 = (int64_t) floor(ag.d1);
    return sg_set_fromg_pi64(result);
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvtf_pd_pi64 vcvtmq_s64_f64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_cvt_pd_ps(sg_pd a) {
    sg_ps result;
    result.f0 = (float) a.d0; result.f1 = (float) a.d1;
    result.f2 = 0.0f; result.f3 = 0.0f;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvt_pd_ps _mm_cvtpd_ps
#elif defined SIMD_GRANODI_NEON
#define sg_cvt_pd_ps(a) vcombine_f32(vcvt_f32_f64(a), vdup_n_f32(0.0f))
#endif

// Arithmetic

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_add_pi32(const sg_pi32 a, const sg_pi32 b) {
    sg_pi32 result;
    result.i0 = a.i0 + b.i0; result.i1 = a.i1 + b.i1;
    result.i2 = a.i2 + b.i2; result.i3 = a.i3 + b.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_add_pi32 _mm_add_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_add_pi32 vaddq_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_add_pi64(const sg_pi64 a, const sg_pi64 b) {
    sg_pi64 result;
    result.l0 = a.l0 + b.l0; result.l1 = a.l1 + b.l1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_add_pi64 _mm_add_epi64
#elif defined SIMD_GRANODI_NEON
#define sg_add_pi64 vaddq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_add_ps(const sg_ps a, const sg_ps b) {
    sg_ps result;
    result.f0 = a.f0 + b.f0; result.f1 = a.f1 + b.f1;
    result.f2 = a.f2 + b.f2; result.f3 = a.f3 + b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_add_ps _mm_add_ps
#elif defined SIMD_GRANODI_NEON
#define sg_add_ps vaddq_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_add_pd(const sg_pd a, const sg_pd b) {
    sg_pd result;
    result.d0 = a.d0 + b.d0; result.d1 = a.d1 + b.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_add_pd _mm_add_pd
#elif defined SIMD_GRANODI_NEON
#define sg_add_pd vaddq_f64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_sub_pi32(const sg_pi32 a, const sg_pi32 b) {
    sg_pi32 result;
    result.i0 = a.i0 - b.i0; result.i1 = a.i1 - b.i1;
    result.i2 = a.i2 - b.i2; result.i3 = a.i3 - b.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_sub_pi32 _mm_sub_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_sub_pi32 vsubq_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_sub_pi64(const sg_pi64 a, const sg_pi64 b) {
    sg_pi64 result;
    result.l0 = a.l0 - b.l0; result.l1 = a.l1 - b.l1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_sub_pi64 _mm_sub_epi64
#elif defined SIMD_GRANODI_NEON
#define sg_sub_pi64 vsubq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_sub_ps(const sg_ps a, const sg_ps b) {
    sg_ps result;
    result.f0 = a.f0 - b.f0; result.f1 = a.f1 - b.f1;
    result.f2 = a.f2 - b.f2; result.f3 = a.f3 - b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_sub_ps _mm_sub_ps
#elif defined SIMD_GRANODI_NEON
#define sg_sub_ps vsubq_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_sub_pd(const sg_pd a, const sg_pd b) {
    sg_pd result;
    result.d0 = a.d0 - b.d0; result.d1 = a.d1 - b.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_sub_pd _mm_sub_pd
#elif defined SIMD_GRANODI_NEON
#define sg_sub_pd vsubq_f64
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi32 sg_mul_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi32 result;
    result.i0 = a.i0 * b.i0; result.i1 = a.i1 * b.i1;
    result.i2 = a.i2 * b.i2; result.i3 = a.i3 * b.i3;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    // Generate ltz masks
    const __m128i a_ltz = _mm_cmplt_epi32(a, _mm_setzero_si128()),
        b_ltz = _mm_cmplt_epi32(b, _mm_setzero_si128());
    // Absolute value of args
    const __m128i abs_a = sg_choose_pi32(
            a_ltz, _mm_sub_epi32(_mm_setzero_si128(), a), a),
        abs_b = sg_choose_pi32(b_ltz, _mm_sub_epi32(_mm_setzero_si128(), b), b);
    // Rearrange args for multiply
    const __m128i low_a = _mm_and_si128(
        _mm_set_epi32(0, sg_allset_s32, 0, sg_allset_s32),
        _mm_shuffle_epi32(abs_a, sg_sse2_shuffle32_imm(3, 1, 1, 0)));
    const __m128i low_b = _mm_and_si128(
        _mm_set_epi32(0, sg_allset_s32, 0, sg_allset_s32),
        _mm_shuffle_epi32(abs_b, sg_sse2_shuffle32_imm(3, 1, 1, 0)));
    const __m128i high_a = _mm_and_si128(
        _mm_set_epi32(0, sg_allset_s32, 0, sg_allset_s32),
        _mm_shuffle_epi32(abs_a, sg_sse2_shuffle32_imm(3, 3, 1, 2)));
    const __m128i high_b = _mm_and_si128(
        _mm_set_epi32(0, sg_allset_s32, 0, sg_allset_s32),
        _mm_shuffle_epi32(abs_b, sg_sse2_shuffle32_imm(3, 3, 1, 2)));
    // Do the multiplications
    __m128i low_result = _mm_mul_epu32(low_a, low_b),
        high_result = _mm_mul_epu32(high_a, high_b);
    // Combine the unsigned multiplies
    low_result = sg_cvt_pi64_pi32(low_result);
    high_result = _mm_and_si128(
        _mm_set_epi64x(sg_allset_s64, 0),
        _mm_shuffle_epi32(high_result, sg_sse2_shuffle32_imm(2, 0, 1, 0)));
    __m128i result = _mm_or_si128(low_result, high_result);
    const __m128i neg_result = _mm_xor_si128(a_ltz, b_ltz);
    return sg_choose_pi32(neg_result,
        _mm_sub_epi32(_mm_setzero_si128(), result), result);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_mul_pi32 vmulq_s32
#endif

static inline sg_pi64 sg_mul_pi64(const sg_pi64 a, const sg_pi64 b) {
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    sg_generic_pi64 result;
    result.l0 = ag.l0 * bg.l0; result.l1 = ag.l1 * bg.l1;
    return sg_set_fromg_pi64(result);
}

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_mul_ps(const sg_ps a, const sg_ps b) {
    sg_ps result;
    result.f0 = a.f0 * b.f0; result.f1 = a.f1 * b.f1;
    result.f2 = a.f2 * b.f2; result.f3 = a.f3 * b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_mul_ps _mm_mul_ps
#elif defined SIMD_GRANODI_NEON
#define sg_mul_ps vmulq_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_mul_pd(const sg_pd a, const sg_pd b) {
    sg_pd result;
    result.d0 = a.d0 * b.d0; result.d1 = a.d1 * b.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_mul_pd _mm_mul_pd
#elif defined SIMD_GRANODI_NEON
#define sg_mul_pd vmulq_f64
#endif

static inline sg_pi32 sg_div_pi32(const sg_pi32 a, const sg_pi32 b) {
    const sg_generic_pi32 ag = sg_getg_pi32(a), bg = sg_getg_pi32(b);
    sg_generic_pi32 result;
    result.i0 = ag.i0 / bg.i0; result.i1 = ag.i1 / bg.i1;
    result.i2 = ag.i2 / bg.i2; result.i3 = ag.i3 / bg.i3 ;
    return sg_set_fromg_pi32(result);
}

static inline sg_pi64 sg_div_pi64(const sg_pi64 a, const sg_pi64 b) {
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    sg_generic_pi64 result;
    result.l0 = ag.l0 / bg.l0; result.l1 = ag.l1 / bg.l1;
    return sg_set_fromg_pi64(result);
}

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_div_ps(const sg_ps a, const sg_ps b) {
    sg_ps result;
    result.f0 = a.f0 / b.f0; result.f1 = a.f1 / b.f1;
    result.f2 = a.f2 / b.f2; result.f3 = a.f3 / b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_div_ps _mm_div_ps
#elif defined SIMD_GRANODI_NEON
#define sg_div_ps vdivq_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_div_pd(const sg_pd a, const sg_pd b) {
    sg_pd result;
    result.d0 = a.d0 / b.d0; result.d1 = a.d1 / b.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_div_pd _mm_div_pd
#elif defined SIMD_GRANODI_NEON
#define sg_div_pd vdivq_f64
#endif

// a * b + c
// Optionally fused on architectures that have it. May add the AVX version in
// future.
// In C++ the class method, when chained together, is syntactically nicer for
// evaluating polynomials
// On NEON, GCC will (by default) optimize separate mul and add intrinsics into
// a single fma anyway. NEON + Clang will not.

#ifdef FP_FAST_FMAF
#define sg_mul_add_f32x1 fmaf
#else
#define sg_mul_add_f32x1(a, b, c) (((a)*(b))+(c))
#endif

#ifdef FP_FAST_FMA
#define sg_mul_add_f64x1 fma
#else
#define sg_mul_add_f64x1(a, b, c) (((a)*(b))+(c))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_mul_add_ps(const sg_ps a, const sg_ps b, const sg_ps c) {
    sg_ps result;
    result.f0 = sg_mul_add_f32x1(a.f0, b.f0, c.f0);
    result.f1 = sg_mul_add_f32x1(a.f1, b.f1, c.f1);
    result.f2 = sg_mul_add_f32x1(a.f2, b.f2, c.f2);
    result.f3 = sg_mul_add_f32x1(a.f3, b.f3, c.f3);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_mul_add_ps(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
#elif defined SIMD_GRANODI_NEON
#define sg_mul_add_ps(a, b, c) vfmaq_f32(c, a, b)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_mul_add_pd(const sg_pd a, const sg_pd b, const sg_pd c) {
    sg_pd result;
    result.d0 = sg_mul_add_f64x1(a.d0, b.d0, c.d0);
    result.d1 = sg_mul_add_f64x1(a.d1, b.d1, c.d1);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_mul_add_pd(a, b, c) _mm_add_pd(_mm_mul_pd(a, b), c)
#elif defined SIMD_GRANODI_NEON
#define sg_mul_add_pd(a, b, c) vfmaq_f64(c, a, b)
#endif

// Bitwise logic

#ifdef SIMD_GRANODI_SSE2
#define sg_sse2_not_si128(a) _mm_andnot_si128(a, sg_sse2_allset_si128)
#define sg_sse2_not_ps(a) _mm_andnot_ps(a, sg_sse2_allset_ps)
#define sg_sse2_not_pd(a) _mm_andnot_pd(a, sg_sse2_allset_pd)

#elif defined SIMD_GRANODI_NEON
#define sg_neon_not_u64(a) \
    vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(a)))
#define sg_neon_not_pi64(a) \
    vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(a)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_and_pi32(const sg_pi32 a, const sg_pi32 b) {
    sg_pi32 result;
    result.i0 = a.i0 & b.i0; result.i1 = a.i1 & b.i1;
    result.i2 = a.i2 & b.i2; result.i3 = a.i3 & b.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_and_pi32 _mm_and_si128
#elif defined SIMD_GRANODI_NEON
#define sg_and_pi32 vandq_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_and_pi64(const sg_pi64 a, const sg_pi64 b) {
    sg_pi64 result;
    result.l0 = a.l0 & b.l0; result.l1 = a.l1 & b.l1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_and_pi64 _mm_and_si128
#elif defined SIMD_GRANODI_NEON
#define sg_and_pi64 vandq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_and_ps(const sg_ps a, const sg_ps b) {
    return sg_bitcast_pi32_ps(sg_and_pi32(sg_bitcast_ps_pi32(a), sg_bitcast_ps_pi32(b)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_and_ps _mm_and_ps
#elif defined SIMD_GRANODI_NEON
#define sg_and_ps(a, b) vreinterpretq_f32_s32(vandq_s32( \
    vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_and_pd(const sg_pd a, const sg_pd b) {
    return sg_bitcast_pi64_pd(sg_and_pi64(sg_bitcast_pd_pi64(a), sg_bitcast_pd_pi64(b)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_and_pd _mm_and_pd
#elif defined SIMD_GRANODI_NEON
#define sg_and_pd(a, b) vreinterpretq_f64_s64(vandq_s64( \
    vreinterpretq_s64_f64(a), vreinterpretq_s64_f64(b)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_andnot_pi32(const sg_pi32 a, const sg_pi32 b) {
    sg_pi32 result;
    result.i0 = ~a.i0 & b.i0; result.i1 = ~a.i1 & b.i1;
    result.i2 = ~a.i2 & b.i2; result.i3 = ~a.i3 & b.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_andnot_pi32 _mm_andnot_si128
#elif defined SIMD_GRANODI_NEON
#define sg_andnot_pi32(a, b) vbicq_s32(b, a)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_andnot_pi64(const sg_pi64 a, const sg_pi64 b) {
    sg_pi64 result;
    result.l0 = ~a.l0 & b.l0; result.l1 = ~a.l1 & b.l1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_andnot_pi64 _mm_andnot_si128
#elif defined SIMD_GRANODI_NEON
#define sg_andnot_pi64(a, b) vbicq_s64(b, a)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_andnot_ps(const sg_ps a, const sg_ps b) {
    return sg_bitcast_pi32_ps(
        sg_andnot_pi32(sg_bitcast_ps_pi32(a), sg_bitcast_ps_pi32(b)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_andnot_ps _mm_andnot_ps
#elif defined SIMD_GRANODI_NEON
#define sg_andnot_ps(a, b) vreinterpretq_f32_s32(vbicq_s32( \
    vreinterpretq_s32_f32(b), vreinterpretq_s32_f32(a)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_andnot_pd(const sg_pd a, const sg_pd b) {
    return sg_bitcast_pi64_pd(
        sg_andnot_pi64(sg_bitcast_pd_pi64(a), sg_bitcast_pd_pi64(b)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_andnot_pd _mm_andnot_pd
#elif defined SIMD_GRANODI_NEON
#define sg_andnot_pd(a, b) vreinterpretq_f64_s64(vbicq_s64( \
    vreinterpretq_s64_f64(b), vreinterpretq_s64_f64(a)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_not_pi32(const sg_pi32 a) {
    sg_pi32 result;
    result.i0 = ~a.i0; result.i1 = ~a.i1; result.i2 = ~a.i2; result.i3 = ~a.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_not_pi32 sg_sse2_not_si128
#elif defined SIMD_GRANODI_NEON
#define sg_not_pi32 vmvnq_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_not_pi64(const sg_pi64 a) {
    sg_pi64 result;
    result.l0 = ~a.l0; result.l1 = ~a.l1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_not_pi64 sg_sse2_not_si128
#elif defined SIMD_GRANODI_NEON
#define sg_not_pi64 sg_neon_not_pi64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_not_ps(const sg_ps a) {
    return sg_bitcast_pi32_ps(sg_not_pi32(sg_bitcast_ps_pi32(a)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_not_ps sg_sse2_not_ps
#elif defined SIMD_GRANODI_NEON
#define sg_not_ps(a) vreinterpretq_f32_s32(vmvnq_s32(vreinterpretq_s32_f32(a)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_not_pd(const sg_pd a) {
    return sg_bitcast_pi64_pd(sg_not_pi64(sg_bitcast_pd_pi64(a)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_not_pd sg_sse2_not_pd
#elif defined SIMD_GRANODI_NEON
#define sg_not_pd(a) vreinterpretq_f64_s32(vmvnq_s32(vreinterpretq_s32_f64(a)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_or_pi32(const sg_pi32 a, const sg_pi32 b) {
    sg_pi32 result;
    result.i0 = a.i0 | b.i0; result.i1 = a.i1 | b.i1;
    result.i2 = a.i2 | b.i2; result.i3 = a.i3 | b.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_or_pi32 _mm_or_si128
#elif defined SIMD_GRANODI_NEON
#define sg_or_pi32 vorrq_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_or_pi64(const sg_pi64 a, const sg_pi64 b) {
    sg_pi64 result;
    result.l0 = a.l0 | b.l0; result.l1 = a.l1 | b.l1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_or_pi64 _mm_or_si128
#elif defined SIMD_GRANODI_NEON
#define sg_or_pi64 vorrq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_or_ps(const sg_ps a, const sg_ps b) {
    return sg_bitcast_pi32_ps(sg_or_pi32(sg_bitcast_ps_pi32(a), sg_bitcast_ps_pi32(b)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_or_ps _mm_or_ps
#elif defined SIMD_GRANODI_NEON
#define sg_or_ps(a, b) vreinterpretq_f32_s32(vorrq_s32( \
    vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_or_pd(const sg_pd a, const sg_pd b) {
    return sg_bitcast_pi64_pd(sg_or_pi64(sg_bitcast_pd_pi64(a), sg_bitcast_pd_pi64(b)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_or_pd _mm_or_pd
#elif defined SIMD_GRANODI_NEON
#define sg_or_pd(a, b) vreinterpretq_f64_s64(vorrq_s64( \
    vreinterpretq_s64_f64(a), vreinterpretq_s64_f64(b)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_xor_pi32(const sg_pi32 a, const sg_pi32 b) {
    sg_pi32 result;
    result.i0 = a.i0 ^ b.i0; result.i1 = a.i1 ^ b.i1;
    result.i2 = a.i2 ^ b.i2; result.i3 = a.i3 ^ b.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_xor_pi32 _mm_xor_si128
#elif defined SIMD_GRANODI_NEON
#define sg_xor_pi32 veorq_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_xor_pi64(const sg_pi64 a, const sg_pi64 b) {
    sg_pi64 result;
    result.l0 = a.l0 ^ b.l0; result.l1 = a.l1 ^ b.l1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_xor_pi64 _mm_xor_si128
#elif defined SIMD_GRANODI_NEON
#define sg_xor_pi64 veorq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_xor_ps(const sg_ps a, const sg_ps b) {
    return sg_bitcast_pi32_ps(sg_xor_pi32(sg_bitcast_ps_pi32(a), sg_bitcast_ps_pi32(b)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_xor_ps _mm_xor_ps
#elif defined SIMD_GRANODI_NEON
#define sg_xor_ps(a, b) vreinterpretq_f32_s32(veorq_s32( \
    vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_xor_pd(const sg_pd a, const sg_pd b) {
    return sg_bitcast_pi64_pd(sg_xor_pi64(sg_bitcast_pd_pi64(a), sg_bitcast_pd_pi64(b)));
}
#elif defined SIMD_GRANODI_SSE2
#define sg_xor_pd _mm_xor_pd
#elif defined SIMD_GRANODI_NEON
#define sg_xor_pd(a, b) vreinterpretq_f64_s64(veorq_s64( \
    vreinterpretq_s64_f64(a), vreinterpretq_s64_f64(b)))
#endif

// Shift

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi32 sg_sl_pi32(const sg_pi32 a, const sg_pi32 shift) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi32 result;
    result.i0 = a.i0 << shift.i0; result.i1 = a.i1 << shift.i1;
    result.i2 = a.i2 << shift.i2; result.i3 = a.i3 << shift.i3;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    __m128i result = _mm_and_si128(_mm_set_epi32(0,0,0,-1),
        _mm_sll_epi32(a, _mm_and_si128(shift, _mm_set_epi32(0,0,0,-1))));
    result = _mm_or_si128(result, _mm_and_si128(_mm_set_epi32(0,0,-1,0),
        _mm_sll_epi32(a, _mm_and_si128(_mm_set_epi32(0,0,0,-1),
            _mm_shuffle_epi32(shift, sg_sse2_shuffle32_imm(3,2,1,1))))));
    result = _mm_or_si128(result, _mm_and_si128(_mm_set_epi32(0,-1,0,0),
        _mm_sll_epi32(a, _mm_and_si128(_mm_set_epi32(0,0,0,-1),
            _mm_shuffle_epi32(shift, sg_sse2_shuffle32_imm(3,2,1,2))))));
    result = _mm_or_si128(result, _mm_and_si128(_mm_set_epi32(-1,0,0,0),
        _mm_sll_epi32(a, _mm_and_si128(_mm_set_epi32(0,0,0,-1),
            _mm_shuffle_epi32(shift, sg_sse2_shuffle32_imm(3,2,1,3))))));
    return result;
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_sl_pi32 vshlq_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_sl_imm_pi32(const sg_pi32 a, const int32_t shift) {
    sg_pi32 result;
    result.i0 = a.i0 << shift; result.i1 = a.i1 << shift;
    result.i2 = a.i2 << shift; result.i3 = a.i3 << shift;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_sl_imm_pi32 _mm_slli_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_sl_imm_pi32 vshlq_n_s32
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi64 sg_sl_pi64(const sg_pi64 a, const sg_pi64 shift) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi64 result;
    result.l0 = a.l0 << shift.l0; result.l1 = a.l1 << shift.l1;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    __m128i result = _mm_and_si128(_mm_set_epi64x(0,-1),
        _mm_sll_epi64(a, shift));
    result = _mm_or_si128(result, _mm_and_si128(_mm_set_epi64x(-1,0),
        _mm_sll_epi64(a,
            _mm_shuffle_epi32(shift, sg_sse2_shuffle32_imm(3,2,3,2)))));
    return result;
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_sl_pi64 vshlq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_sl_imm_pi64(const sg_pi64 a, const int32_t shift) {
    sg_pi64 result;
    result.l0 = a.l0 << (int64_t) shift; result.l1 = a.l1 << (int64_t) shift;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_sl_imm_pi64 _mm_slli_epi64
#elif defined SIMD_GRANODI_NEON
#define sg_sl_imm_pi64 vshlq_n_s64
#endif

#define sg_srl_s32x1(a, shift) sg_bitcast_u32x1_s32x1( \
    sg_bitcast_s32x1_u32x1(a) >> sg_bitcast_s32x1_u32x1(shift))
#define sg_srl_s64x1(a, shift) sg_bitcast_u64x1_s64x1( \
    sg_bitcast_s64x1_u64x1(a) >> sg_bitcast_s64x1_u64x1(shift))

static inline sg_pi32 sg_srl_pi32(const sg_pi32 a, const sg_pi32 shift) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
    sg_generic_pi32 ag = sg_getg_pi32(a), shiftg = sg_getg_pi32(shift), result;
    result.i0 = sg_srl_s32x1(ag.i0, shiftg.i0);
    result.i1 = sg_srl_s32x1(ag.i1, shiftg.i1);
    result.i2 = sg_srl_s32x1(ag.i2, shiftg.i2);
    result.i3 = sg_srl_s32x1(ag.i3, shiftg.i3);
    return sg_set_fromg_pi32(result);
    #elif defined SIMD_GRANODI_SSE2
    __m128i result = _mm_and_si128(_mm_set_epi32(0,0,0,-1),
        _mm_srl_epi32(a, _mm_and_si128(shift, _mm_set_epi32(0,0,0,-1))));
    result = _mm_or_si128(result, _mm_and_si128(_mm_set_epi32(0,0,-1,0),
        _mm_srl_epi32(a, _mm_and_si128(_mm_set_epi32(0,0,0,-1),
            _mm_shuffle_epi32(shift, sg_sse2_shuffle32_imm(3,2,1,1))))));
    result = _mm_or_si128(result, _mm_and_si128(_mm_set_epi32(0,-1,0,0),
        _mm_srl_epi32(a, _mm_and_si128(_mm_set_epi32(0,0,0,-1),
            _mm_shuffle_epi32(shift, sg_sse2_shuffle32_imm(3,2,1,2))))));
    result = _mm_or_si128(result, _mm_and_si128(_mm_set_epi32(-1,0,0,0),
        _mm_srl_epi32(a, _mm_and_si128(_mm_set_epi32(0,0,0,-1),
            _mm_shuffle_epi32(shift, sg_sse2_shuffle32_imm(3,2,1,3))))));
    return result;
    #endif
}

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_srl_imm_pi32(const sg_pi32 a, const int32_t shift) {
    sg_pi32 result;
    result.i0 = sg_srl_s32x1(a.i0, shift);
    result.i1 = sg_srl_s32x1(a.i1, shift);
    result.i2 = sg_srl_s32x1(a.i2, shift);
    result.i3 = sg_srl_s32x1(a.i3, shift);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_srl_imm_pi32 _mm_srli_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_srl_imm_pi32(a, shift) vreinterpretq_s32_u32( \
    vshrq_n_u32(vreinterpretq_u32_s32(a), (shift)))
#endif

static inline sg_pi64 sg_srl_pi64(const sg_pi64 a, const sg_pi64 shift) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
    sg_generic_pi64 ag = sg_getg_pi64(a), shiftg = sg_getg_pi64(shift), result;
    result.l0 = sg_srl_s64x1(ag.l0, shiftg.l0);
    result.l1 = sg_srl_s64x1(ag.l1, shiftg.l1);
    return sg_set_fromg_pi64(result);
    #elif defined SIMD_GRANODI_SSE2
    __m128i result = _mm_and_si128(_mm_set_epi64x(0,-1),
        _mm_srl_epi64(a, shift));
    result = _mm_or_si128(result, _mm_and_si128(_mm_set_epi64x(-1,0),
        _mm_srl_epi64(a,
            _mm_shuffle_epi32(shift, sg_sse2_shuffle32_imm(3,2,3,2)))));
    return result;
    #endif
}

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_srl_imm_pi64(const sg_pi64 a, const int32_t shift) {
    sg_pi64 result;
    result.l0 = sg_srl_s64x1(a.l0, shift);
    result.l1 = sg_srl_s64x1(a.l1, shift);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_srl_imm_pi64 _mm_srli_epi64
#elif defined SIMD_GRANODI_NEON
#define sg_srl_imm_pi64(a, shift) vreinterpretq_s64_u64( \
    vshrq_n_u64(vreinterpretq_u64_s64(a), (shift)))
#endif

static inline sg_pi32 sg_sra_pi32(const sg_pi32 a, const sg_pi32 shift) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
    sg_generic_pi32 ag = sg_getg_pi32(a), shiftg = sg_getg_pi32(shift), result;
    result.i0 = ag.i0 >> shiftg.i0; result.i1 = ag.i1 >> shiftg.i1;
    result.i2 = ag.i2 >> shiftg.i2; result.i3 = ag.i3 >> shiftg.i3;
    return sg_set_fromg_pi32(result);
    #elif defined SIMD_GRANODI_SSE2
    __m128i result = _mm_and_si128(_mm_set_epi32(0,0,0,-1),
        _mm_sra_epi32(a, _mm_and_si128(shift, _mm_set_epi32(0,0,0,-1))));
    result = _mm_or_si128(result, _mm_and_si128(_mm_set_epi32(0,0,-1,0),
        _mm_sra_epi32(a, _mm_and_si128(_mm_set_epi32(0,0,0,-1),
            _mm_shuffle_epi32(shift, sg_sse2_shuffle32_imm(3,2,1,1))))));
    result = _mm_or_si128(result, _mm_and_si128(_mm_set_epi32(0,-1,0,0),
        _mm_sra_epi32(a, _mm_and_si128(_mm_set_epi32(0,0,0,-1),
            _mm_shuffle_epi32(shift, sg_sse2_shuffle32_imm(3,2,1,2))))));
    result = _mm_or_si128(result, _mm_and_si128(_mm_set_epi32(-1,0,0,0),
        _mm_sra_epi32(a, _mm_and_si128(_mm_set_epi32(0,0,0,-1),
            _mm_shuffle_epi32(shift, sg_sse2_shuffle32_imm(3,2,1,3))))));
    return result;
    #endif
}

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_sra_imm_pi32(const sg_pi32 a, const int32_t shift) {
    sg_pi32 result;
    result.i0 = a.i0 >> shift; result.i1 = a.i1 >> shift;
    result.i2 = a.i2 >> shift; result.i3 = a.i3 >> shift;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_sra_imm_pi32 _mm_srai_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_sra_imm_pi32 vshrq_n_s32
#endif

static inline sg_pi64 sg_sra_pi64(const sg_pi64 a, const sg_pi64 shift) {
    sg_generic_pi64 ag = sg_getg_pi64(a), shiftg = sg_getg_pi64(shift), result;
    result.l0 = ag.l0 >> shiftg.l0; result.l1 = ag.l1 >> shiftg.l1;
    return sg_set_fromg_pi64(result);
}

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi64 sg_sra_imm_pi64(const sg_pi64 a,
    const int32_t shift_compile_time_constant)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi64 result;
    result.l0 = a.l0 >> (int64_t) shift_compile_time_constant;
    result.l1 = a.l1 >> (int64_t) shift_compile_time_constant;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    const __m128i signed_shift_mask = _mm_cmplt_epi32(
        _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(3, 3, 1, 1)),
        _mm_setzero_si128());
    switch (shift_compile_time_constant & 0xff) { case 0: return a;
    case 1: return _mm_or_si128(_mm_srli_epi64(a, 1),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(0x8000000000000000)));
    case 2: return _mm_or_si128(_mm_srli_epi64(a, 2),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(0xc000000000000000)));
    case 3: return _mm_or_si128(_mm_srli_epi64(a, 3),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(0xe000000000000000)));
    case 4: return _mm_or_si128(_mm_srli_epi64(a, 4),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(0xf000000000000000)));
    case 5: return _mm_or_si128(_mm_srli_epi64(a, 5),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(0xf800000000000000)));
    case 6: return _mm_or_si128(_mm_srli_epi64(a, 6),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(0xfc00000000000000)));
    case 7: return _mm_or_si128(_mm_srli_epi64(a, 7),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(0xfe00000000000000)));
    case 8: return _mm_or_si128(_mm_srli_epi64(a, 8),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(0xff00000000000000)));
    case 9: return _mm_or_si128(_mm_srli_epi64(a, 9),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(0xff80000000000000)));
    case 10: return _mm_or_si128(_mm_srli_epi64(a, 10),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(0xffc0000000000000)));
    case 11: return _mm_or_si128(_mm_srli_epi64(a, 11),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-9007199254740992)));
    case 12: return _mm_or_si128(_mm_srli_epi64(a, 12),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-4503599627370496)));
    case 13: return _mm_or_si128(_mm_srli_epi64(a, 13),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-2251799813685248)));
    case 14: return _mm_or_si128(_mm_srli_epi64(a, 14),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-1125899906842624)));
    case 15: return _mm_or_si128(_mm_srli_epi64(a, 15),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-562949953421312)));
    case 16: return _mm_or_si128(_mm_srli_epi64(a, 16),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-281474976710656)));
    case 17: return _mm_or_si128(_mm_srli_epi64(a, 17),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-140737488355328)));
    case 18: return _mm_or_si128(_mm_srli_epi64(a, 18),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-70368744177664)));
    case 19: return _mm_or_si128(_mm_srli_epi64(a, 19),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-35184372088832)));
    case 20: return _mm_or_si128(_mm_srli_epi64(a, 20),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-17592186044416)));
    case 21: return _mm_or_si128(_mm_srli_epi64(a, 21),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-8796093022208)));
    case 22: return _mm_or_si128(_mm_srli_epi64(a, 22),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-4398046511104)));
    case 23: return _mm_or_si128(_mm_srli_epi64(a, 23),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-2199023255552)));
    case 24: return _mm_or_si128(_mm_srli_epi64(a, 24),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-1099511627776)));
    case 25: return _mm_or_si128(_mm_srli_epi64(a, 25),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-549755813888)));
    case 26: return _mm_or_si128(_mm_srli_epi64(a, 26),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-274877906944)));
    case 27: return _mm_or_si128(_mm_srli_epi64(a, 27),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-137438953472)));
    case 28: return _mm_or_si128(_mm_srli_epi64(a, 28),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-68719476736)));
    case 29: return _mm_or_si128(_mm_srli_epi64(a, 29),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-34359738368)));
    case 30: return _mm_or_si128(_mm_srli_epi64(a, 30),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-17179869184)));
    case 31: return _mm_or_si128(_mm_srli_epi64(a, 31),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-8589934592)));
    case 32: return _mm_or_si128(_mm_srli_epi64(a, 32),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-4294967296)));
    case 33: return _mm_or_si128(_mm_srli_epi64(a, 33),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-2147483648)));
    case 34: return _mm_or_si128(_mm_srli_epi64(a, 34),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-1073741824)));
    case 35: return _mm_or_si128(_mm_srli_epi64(a, 35),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-536870912)));
    case 36: return _mm_or_si128(_mm_srli_epi64(a, 36),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-268435456)));
    case 37: return _mm_or_si128(_mm_srli_epi64(a, 37),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-134217728)));
    case 38: return _mm_or_si128(_mm_srli_epi64(a, 38),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-67108864)));
    case 39: return _mm_or_si128(_mm_srli_epi64(a, 39),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-33554432)));
    case 40: return _mm_or_si128(_mm_srli_epi64(a, 40),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-16777216)));
    case 41: return _mm_or_si128(_mm_srli_epi64(a, 41),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-8388608)));
    case 42: return _mm_or_si128(_mm_srli_epi64(a, 42),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-4194304)));
    case 43: return _mm_or_si128(_mm_srli_epi64(a, 43),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-2097152)));
    case 44: return _mm_or_si128(_mm_srli_epi64(a, 44),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-1048576)));
    case 45: return _mm_or_si128(_mm_srli_epi64(a, 45),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-524288)));
    case 46: return _mm_or_si128(_mm_srli_epi64(a, 46),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-262144)));
    case 47: return _mm_or_si128(_mm_srli_epi64(a, 47),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-131072)));
    case 48: return _mm_or_si128(_mm_srli_epi64(a, 48),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-65536)));
    case 49: return _mm_or_si128(_mm_srli_epi64(a, 49),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-32768)));
    case 50: return _mm_or_si128(_mm_srli_epi64(a, 50),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-16384)));
    case 51: return _mm_or_si128(_mm_srli_epi64(a, 51),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-8192)));
    case 52: return _mm_or_si128(_mm_srli_epi64(a, 52),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-4096)));
    case 53: return _mm_or_si128(_mm_srli_epi64(a, 53),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-2048)));
    case 54: return _mm_or_si128(_mm_srli_epi64(a, 54),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-1024)));
    case 55: return _mm_or_si128(_mm_srli_epi64(a, 55),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-512)));
    case 56: return _mm_or_si128(_mm_srli_epi64(a, 56),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-256)));
    case 57: return _mm_or_si128(_mm_srli_epi64(a, 57),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-128)));
    case 58: return _mm_or_si128(_mm_srli_epi64(a, 58),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-64)));
    case 59: return _mm_or_si128(_mm_srli_epi64(a, 59),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-32)));
    case 60: return _mm_or_si128(_mm_srli_epi64(a, 60),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-16)));
    case 61: return _mm_or_si128(_mm_srli_epi64(a, 61),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-8)));
    case 62: return _mm_or_si128(_mm_srli_epi64(a, 62),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-4)));
    case 63: return _mm_or_si128(_mm_srli_epi64(a, 63),
        _mm_and_si128(signed_shift_mask, _mm_set1_epi64x(-2)));
    default: return signed_shift_mask; // 64 and over
    }
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_sra_imm_pi64 vshrq_n_s64
#endif

//
//
//
// Compare

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_setzero_cmp_pi32() {
    sg_cmp_pi32 result;
    result.b0 = false; result.b1 = false; result.b2 = false; result.b3 = false;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setzero_cmp_pi32() _mm_setzero_si128()
#elif defined SIMD_GRANODI_NEON
#define sg_setzero_cmp_pi32() vdupq_n_u32(0)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi64 sg_setzero_cmp_pi64() {
    sg_cmp_pi64 result;
    result.b0 = false; result.b1 = false;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setzero_cmp_pi64() _mm_setzero_si128()
#elif defined SIMD_GRANODI_NEON
#define sg_setzero_cmp_pi64() vdupq_n_u64(0)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_setzero_cmp_ps() {
    return sg_setzero_cmp_pi32();
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setzero_cmp_ps() _mm_setzero_ps()
#elif defined SIMD_GRANODI_NEON
#define sg_setzero_cmp_ps() vdupq_n_u32(0)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_setzero_cmp_pd() {
    return sg_setzero_cmp_pi64();
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setzero_cmp_pd() _mm_setzero_pd()
#elif defined SIMD_GRANODI_NEON
#define sg_setzero_cmp_pd() vdupq_n_u64(0)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_set1cmp_pi32(const bool b) {
    sg_cmp_pi32 result;
    result.b0 = b; result.b1 = b; result.b2 = b; result.b3 = b;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set1cmp_pi32(b) ((b) ? sg_sse2_allset_si128 : _mm_setzero_si128())
#elif defined SIMD_GRANODI_NEON
#define sg_set1cmp_pi32(b) vdupq_n_u32((b) ? sg_allset_u32 : 0)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi64 sg_set1cmp_pi64(const bool b) {
    sg_cmp_pi64 result;
    result.b0 = b; result.b1 = b;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set1cmp_pi64(b) ((b) ? sg_sse2_allset_si128 : _mm_setzero_si128())
#elif defined SIMD_GRANODI_NEON
#define sg_set1cmp_pi64(b) vdupq_n_u64((b) ? sg_allset_u64 : 0)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_set1cmp_ps(const bool b) {
    return sg_set1cmp_pi32(b);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set1cmp_ps(b) ((b) ? sg_sse2_allset_ps : _mm_setzero_ps())
#elif defined SIMD_GRANODI_NEON
#define sg_set1cmp_ps sg_set1cmp_pi32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_set1cmp_pd(const bool b) {
    return sg_set1cmp_pi64(b);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_set1cmp_pd(b) ((b) ? sg_sse2_allset_pd : _mm_setzero_pd())
#elif defined SIMD_GRANODI_NEON
#define sg_set1cmp_pd sg_set1cmp_pi64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_setcmp_pi32(const bool b3, const bool b2,
    const bool b1, const bool b0)
{
    sg_cmp_pi32 result;
    result.b0 = b0; result.b1 = b1; result.b2 = b2; result.b3 = b3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setcmp_pi32(b3, b2, b1, b0) \
    _mm_set_epi32((b3) ? sg_allset_s32 : 0, \
        (b2) ? sg_allset_s32 : 0, (b1) ? sg_allset_s32 : 0, \
        (b0) ? sg_allset_s32 : 0)
#elif defined SIMD_GRANODI_NEON
#define sg_setcmp_pi32(b3, b2, b1, b0) vsetq_lane_u32( \
    (b3) ? sg_allset_u32 : 0, vsetq_lane_u32((b2) ? sg_allset_u32 : 0, \
        vsetq_lane_u32((b1) ? sg_allset_u32 : 0, \
            vsetq_lane_u32((b0) ? sg_allset_u32 : 0, \
                vdupq_n_u32(0), 0), 1), 2), 3)
#endif

static inline sg_cmp_pi32 sg_setcmp_fromg_pi32(const sg_generic_cmp4 cmpg) {
    return sg_setcmp_pi32(cmpg.b3, cmpg.b2, cmpg.b1, cmpg.b0);
}

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi64 sg_setcmp_pi64(const bool b1, const bool b0) {
    sg_cmp_pi64 result;
    result.b0 = b0; result.b1 = b1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setcmp_pi64(b1, b0) \
    _mm_set_epi64x((b1) ? sg_allset_s64 : 0, (b0) ? sg_allset_s64 : 0)
#elif defined SIMD_GRANODI_NEON
#define sg_setcmp_pi64(b1, b0) \
    vsetq_lane_u64((b1) ? sg_allset_u64 : 0, \
        vsetq_lane_u64((b0) ? sg_allset_u64 : 0, vdupq_n_u64(0), 0), 1)
#endif

static inline sg_cmp_pi64 sg_setcmp_fromg_pi64(const sg_generic_cmp2 cmpg) {
    return sg_setcmp_pi64(cmpg.b1, cmpg.b0);
}

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_setcmp_ps(const bool b3, const bool b2,
    const bool b1, const bool b0)
{
    return sg_setcmp_pi32(b3, b2, b1, b0);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setcmp_ps(b3, b2, b1, b0) \
    _mm_castsi128_ps(sg_setcmp_pi32(b3, b2, b1, b0))
#elif defined SIMD_GRANODI_NEON
#define sg_setcmp_ps sg_setcmp_pi32
#endif

static inline sg_cmp_ps sg_setcmp_fromg_ps(const sg_generic_cmp4 cmpg) {
    return sg_setcmp_ps(cmpg.b3, cmpg.b2, cmpg.b1, cmpg.b0);
}

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_setcmp_pd(const bool b1, const bool b0) {
    return sg_setcmp_pi64(b1, b0);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_setcmp_pd(b1, b0) _mm_castsi128_pd(sg_setcmp_pi64(b1, b0))
#elif defined SIMD_GRANODI_NEON
#define sg_setcmp_pd sg_setcmp_pi64
#endif

static inline sg_cmp_pd sg_setcmp_fromg_pd(const sg_generic_cmp2 cmpg) {
    return sg_setcmp_pd(cmpg.b1, cmpg.b0);
}

#ifndef SIMD_GRANODI_FORCE_GENERIC
static inline bool sg_debug_mask_valid_eq_u32(const uint32_t mask, const bool b)
{
    return (mask == sg_allset_u32 && b) || (mask == 0 && !b);
}
static inline bool sg_debug_mask_valid_eq_u64(const uint64_t mask, const bool b)
{
    return (mask == sg_allset_u64 && b) || (mask == 0 && !b);
}
#endif // #ifdef SIMD_GRANODI_FORCE_GENERIC

static inline bool sg_debug_cmp_valid_eq_pi32(const sg_cmp_pi32 cmp,
    const bool b3, const bool b2, const bool b1, const bool b0)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return cmp.b3 == b3 && cmp.b2 == b2 && cmp.b1 == b1 && cmp.b0 == b0;
    #elif defined SIMD_GRANODI_SSE2
    return sg_debug_mask_valid_eq_u32(
        sg_bitcast_s32x1_u32x1(sg_get3_pi32(cmp)), b3) &&
        sg_debug_mask_valid_eq_u32(sg_bitcast_s32x1_u32x1(sg_get2_pi32(cmp)), b2) &&
        sg_debug_mask_valid_eq_u32(sg_bitcast_s32x1_u32x1(sg_get1_pi32(cmp)), b1) &&
        sg_debug_mask_valid_eq_u32(sg_bitcast_s32x1_u32x1(sg_get0_pi32(cmp)), b0);
    #elif defined SIMD_GRANODI_NEON
    return sg_debug_mask_valid_eq_u32(vgetq_lane_u32(cmp, 3), b3) &&
        sg_debug_mask_valid_eq_u32(vgetq_lane_u32(cmp, 2), b2) &&
        sg_debug_mask_valid_eq_u32(vgetq_lane_u32(cmp, 1), b1) &&
        sg_debug_mask_valid_eq_u32(vgetq_lane_u32(cmp, 0), b0);
    #endif
}

static inline bool sg_debug_cmp_valid_eq_pi64(const sg_cmp_pi64 cmp,
    const bool b1, const bool b0)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return cmp.b1 == b1 && cmp.b0 == b0;
    #elif defined SIMD_GRANODI_SSE2
    return sg_debug_mask_valid_eq_u64(
        sg_bitcast_s64x1_u64x1(sg_get1_pi64(cmp)), b1) &&
        sg_debug_mask_valid_eq_u64(sg_bitcast_s64x1_u64x1(sg_get0_pi64(cmp)), b0);
    #elif defined SIMD_GRANODI_NEON
    return sg_debug_mask_valid_eq_u64(vgetq_lane_u64(cmp, 1), b1) &&
        sg_debug_mask_valid_eq_u64(vgetq_lane_u64(cmp, 0), b0);
    #endif
}

static inline bool sg_debug_cmp_valid_eq_ps(const sg_cmp_ps cmp,
    const bool b3, const bool b2, const bool b1, const bool b0)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return cmp.b3 == b3 && cmp.b2 == b2 && cmp.b1 == b1 && cmp.b0 == b0;
    #elif defined SIMD_GRANODI_SSE2
    return sg_debug_mask_valid_eq_u32(sg_bitcast_f32x1_u32x1(sg_get3_ps(cmp)), b3) &&
        sg_debug_mask_valid_eq_u32(sg_bitcast_f32x1_u32x1(sg_get2_ps(cmp)), b2) &&
        sg_debug_mask_valid_eq_u32(sg_bitcast_f32x1_u32x1(sg_get1_ps(cmp)), b1) &&
        sg_debug_mask_valid_eq_u32(sg_bitcast_f32x1_u32x1(sg_get0_ps(cmp)), b0);
    #elif defined SIMD_GRANODI_NEON
    return sg_debug_mask_valid_eq_u32(vgetq_lane_u32(cmp, 3), b3) &&
        sg_debug_mask_valid_eq_u32(vgetq_lane_u32(cmp, 2), b2) &&
        sg_debug_mask_valid_eq_u32(vgetq_lane_u32(cmp, 1), b1) &&
        sg_debug_mask_valid_eq_u32(vgetq_lane_u32(cmp, 0), b0);
    #endif
}

static inline bool sg_debug_cmp_valid_eq_pd(const sg_cmp_pd cmp,
    const bool b1, const bool b0)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return cmp.b1 == b1 && cmp.b0 == b0;
    #elif defined SIMD_GRANODI_SSE2
    return sg_debug_mask_valid_eq_u64(sg_bitcast_f64x1_u64x1(sg_get1_pd(cmp)), b1) &&
        sg_debug_mask_valid_eq_u64(sg_bitcast_f64x1_u64x1(sg_get0_pd(cmp)), b0);
    #elif defined SIMD_GRANODI_NEON
    return sg_debug_mask_valid_eq_u64(vgetq_lane_u64(cmp, 1), b1) &&
        sg_debug_mask_valid_eq_u64(vgetq_lane_u64(cmp, 0), b0);
    #endif
}

// Comparisons

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_cmplt_pi32(const sg_pi32 a, const sg_pi32 b) {
    sg_cmp_pi32 result;
    result.b0 = a.i0 < b.i0; result.b1 = a.i1 < b.i1;
    result.b2 = a.i2 < b.i2; result.b3 = a.i3 < b.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmplt_pi32 _mm_cmplt_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_cmplt_pi32 vcltq_s32
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_cmp_pi64 sg_cmplt_pi64(const sg_pi64 a, const sg_pi64 b) {
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    sg_generic_cmp2 result;
    result.b0 = ag.l0 < bg.l0; result.b1 = ag.l1 < bg.l1;
    return sg_setcmp_fromg_pi64(result);
}
#elif defined SIMD_GRANODI_NEON
#define sg_cmplt_pi64 vcltq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_cmplt_ps(const sg_ps a, const sg_ps b) {
    sg_cmp_ps result;
    result.b0 = a.f0 < b.f0; result.b1 = a.f1 < b.f1;
    result.b2 = a.f2 < b.f2; result.b3 = a.f3 < b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmplt_ps _mm_cmplt_ps
#elif defined SIMD_GRANODI_NEON
#define sg_cmplt_ps vcltq_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_cmplt_pd(const sg_pd a, const sg_pd b) {
    sg_cmp_pd result;
    result.b0 = a.d0 < b.d0; result.b1 = a.d1 < b.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmplt_pd _mm_cmplt_pd
#elif defined SIMD_GRANODI_NEON
#define sg_cmplt_pd vcltq_f64
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_cmp_pi32 sg_cmplte_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_cmp_pi32 result;
    result.b0 = a.i0 <= b.i0; result.b1 = a.i1 <= b.i1;
    result.b2 = a.i2 <= b.i2; result.b3 = a.i3 <= b.i3;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, b));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cmplte_pi32 vcleq_s32
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_cmp_pi64 sg_cmplte_pi64(const sg_pi64 a, const sg_pi64 b) {
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    sg_generic_cmp2 result;
    result.b0 = ag.l0 <= bg.l0; result.b1 = ag.l1 <= bg.l1;
    return sg_setcmp_fromg_pi64(result);
}
#elif defined SIMD_GRANODI_NEON
#define sg_cmplte_pi64 vcleq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_cmplte_ps(const sg_ps a, const sg_ps b) {
    sg_cmp_ps result;
    result.b0 = a.f0 <= b.f0; result.b1 = a.f1 <= b.f1;
    result.b2 = a.f2 <= b.f2; result.b3 = a.f3 <= b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmplte_ps _mm_cmple_ps
#elif defined SIMD_GRANODI_NEON
#define sg_cmplte_ps vcleq_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_cmplte_pd(const sg_pd a, const sg_pd b) {
    sg_cmp_pd result;
    result.b0 = a.d0 <= b.d0; result.b1 = a.d1 <= b.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmplte_pd _mm_cmple_pd
#elif defined SIMD_GRANODI_NEON
#define sg_cmplte_pd vcleq_f64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_cmpeq_pi32(const sg_pi32 a, const sg_pi32 b) {
    sg_cmp_pi32 result;
    result.b0 = a.i0 == b.i0; result.b1 = a.i1 == b.i1;
    result.b2 = a.i2 == b.i2; result.b3 = a.i3 == b.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpeq_pi32 _mm_cmpeq_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_cmpeq_pi32 vceqq_s32
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_cmp_pi64 sg_cmpeq_pi64(const sg_pi64 a, const sg_pi64 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_cmp_pi64 result;
    result.b0 = a.l0 == b.l0; result.b1 = a.l1 == b.l1;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    const __m128i eq_epi32 = _mm_cmpeq_epi32(a, b);
    return _mm_and_si128(eq_epi32,
        _mm_shuffle_epi32(eq_epi32, sg_sse2_shuffle32_imm(2, 3, 0, 1)));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cmpeq_pi64 vceqq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_cmpeq_ps(const sg_ps a, const sg_ps b) {
    sg_cmp_ps result;
    result.b0 = a.f0 == b.f0; result.b1 = a.f1 == b.f1;
    result.b2 = a.f2 == b.f2; result.b3 = a.f3 == b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpeq_ps _mm_cmpeq_ps
#elif defined SIMD_GRANODI_NEON
#define sg_cmpeq_ps vceqq_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_cmpeq_pd(const sg_pd a, const sg_pd b) {
    sg_cmp_pd result;
    result.b0 = a.d0 == b.d0; result.b1 = a.d1 == b.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpeq_pd _mm_cmpeq_pd
#elif defined SIMD_GRANODI_NEON
#define sg_cmpeq_pd vceqq_f64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_cmpneq_pi32(const sg_pi32 a, const sg_pi32 b) {
    sg_cmp_pi32 result;
    result.b0 = a.i0 != b.i0; result.b1 = a.i1 != b.i1;
    result.b2 = a.i2 != b.i2; result.b3 = a.i3 != b.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpneq_pi32(a, b) sg_sse2_not_si128(_mm_cmpeq_epi32(a, b))
#elif defined SIMD_GRANODI_NEON
#define sg_cmpneq_pi32(a, b) vmvnq_u32(vceqq_s32(a, b))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi64 sg_cmpneq_pi64(const sg_pi64 a, const sg_pi64 b) {
    sg_cmp_pi64 result;
    result.b0 = a.l0 != b.l0; result.b1 = a.l1 != b.l1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpneq_pi64(a, b) sg_sse2_not_si128(sg_cmpeq_pi64(a, b))
#elif defined SIMD_GRANODI_NEON
#define sg_cmpneq_pi64(a, b) sg_neon_not_u64(vceqq_s64(a, b))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_cmpneq_ps(const sg_ps a, const sg_ps b) {
    sg_cmp_ps result;
    result.b0 = a.f0 != b.f0; result.b1 = a.f1 != b.f1;
    result.b2 = a.f2 != b.f2; result.b3 = a.f3 != b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpneq_ps _mm_cmpneq_ps
#elif defined SIMD_GRANODI_NEON
#define sg_cmpneq_ps(a, b) vmvnq_u32(vceqq_f32(a, b))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_cmpneq_pd(const sg_pd a, const sg_pd b) {
    sg_cmp_pd result;
    result.b0 = a.d0 != b.d0; result.b1 = a.d1 != b.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpneq_pd _mm_cmpneq_pd
#elif defined SIMD_GRANODI_NEON
#define sg_cmpneq_pd(a, b) sg_neon_not_u64(vceqq_f64(a, b))
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_cmp_pi32 sg_cmpgte_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_cmp_pi32 result;
    result.b0 = a.i0 >= b.i0; result.b1 = a.i1 >= b.i1;
    result.b2 = a.i2 >= b.i2; result.b3 = a.i3 >= b.i3;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, b));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cmpgte_pi32 vcgeq_s32
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_cmp_pi64 sg_cmpgte_pi64(const sg_pi64 a, const sg_pi64 b) {
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    sg_generic_cmp2 result;
    result.b0 = ag.l0 >= bg.l0; result.b1 = ag.l1 >= bg.l1;
    return sg_setcmp_fromg_pi64(result);

}
#elif defined SIMD_GRANODI_NEON
#define sg_cmpgte_pi64 vcgeq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_cmpgte_ps(const sg_ps a, const sg_ps b) {
    sg_cmp_ps result;
    result.b0 = a.f0 >= b.f0; result.b1 = a.f1 >= b.f1;
    result.b2 = a.f2 >= b.f2; result.b3 = a.f3 >= b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpgte_ps _mm_cmpge_ps
#elif defined SIMD_GRANODI_NEON
#define sg_cmpgte_ps vcgeq_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_cmpgte_pd(const sg_pd a, const sg_pd b) {
    sg_cmp_pd result;
    result.b0 = a.d0 >= b.d0; result.b1 = a.d1 >= b.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpgte_pd _mm_cmpge_pd
#elif defined SIMD_GRANODI_NEON
#define sg_cmpgte_pd vcgeq_f64
#endif

static inline sg_cmp_pi32 sg_cmpgt_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_cmp_pi32 result;
    result.b0 = a.i0 > b.i0; result.b1 = a.i1 > b.i1;
    result.b2 = a.i2 > b.i2; result.b3 = a.i3 > b.i3;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpgt_epi32(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcgtq_s32(a, b);
    #endif
}

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_cmp_pi64 sg_cmpgt_pi64(const sg_pi64 a, const sg_pi64 b) {
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    sg_generic_cmp2 result;
    result.b0 = ag.l0 > bg.l0; result.b1 = ag.l1 > bg.l1;
    return sg_setcmp_fromg_pi64(result);
}
#elif defined SIMD_GRANODI_NEON
#define sg_cmpgt_pi64 vcgtq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_cmpgt_ps(const sg_ps a, const sg_ps b) {
    sg_cmp_ps result;
    result.b0 = a.f0 > b.f0; result.b1 = a.f1 > b.f1;
    result.b2 = a.f2 > b.f2; result.b3 = a.f3 > b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpgt_ps _mm_cmpgt_ps
#elif defined SIMD_GRANODI_NEON
#define sg_cmpgt_ps vcgtq_f32
#endif

static inline sg_cmp_pd sg_cmpgt_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_cmp_pd result;
    result.b0 = a.d0 > b.d0; result.b1 = a.d1 > b.d1;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpgt_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcgtq_f64(a, b);
    #endif
}

// Cast comparisons (same layout, no conversion)

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_cvtcmp_pi32_ps(const sg_cmp_pi32 cmp) {
    return cmp;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtcmp_pi32_ps _mm_castsi128_ps
#elif defined SIMD_GRANODI_NEON
#define sg_cvtcmp_pi32_ps(cmp) (cmp)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_cvtcmp_ps_pi32(const sg_cmp_ps cmp) {
    return cmp;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtcmp_ps_pi32 _mm_castps_si128
#elif defined SIMD_GRANODI_NEON
#define sg_cvtcmp_ps_pi32(cmp) (cmp)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_cvtcmp_pi64_pd(const sg_cmp_pi64 cmp) {
    return cmp;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtcmp_pi64_pd _mm_castsi128_pd
#elif defined SIMD_GRANODI_NEON
#define sg_cvtcmp_pi64_pd(cmp) (cmp)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi64 sg_cvtcmp_pd_pi64(const sg_cmp_pd cmp) {
    return cmp;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtcmp_pd_pi64 _mm_castpd_si128
#elif defined SIMD_GRANODI_NEON
#define sg_cvtcmp_pd_pi64(cmp) (cmp)
#endif

// Convert comparison results (shuffling)
#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
static inline sg_cmp_pi64 sg_cvtcmp_pi32_pi64(const sg_cmp_pi32 cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_generic_cmp2 result;
    result.b0 = cmp.b0; result.b1 = cmp.b1;
    return result;
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_u64_u32(vzip1q_u32(cmp, cmp));
    #endif
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtcmp_pi32_pi64(cmp) \
    _mm_shuffle_epi32(cmp, sg_sse2_shuffle32_imm(1, 1, 0, 0))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_cvtcmp_pi32_pd(const sg_cmp_pi32 cmp) {
    return sg_cvtcmp_pi32_pi64(cmp);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtcmp_pi32_pd(cmp) _mm_castsi128_pd(sg_cvtcmp_pi32_pi64(cmp))
#elif defined SIMD_GRANODI_NEON
#define sg_cvtcmp_pi32_pd sg_cvtcmp_pi32_pi64
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_cmp_pd sg_cvtcmp_ps_pd(const sg_cmp_ps cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cvtcmp_pi32_pd(cmp);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castps_pd(
        _mm_shuffle_ps(cmp, cmp, sg_sse2_shuffle32_imm(1, 1, 0, 0)));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvtcmp_ps_pd sg_cvtcmp_pi32_pi64
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_cmp_pi64 sg_cvtcmp_ps_pi64(const sg_cmp_ps cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cvtcmp_pi32_pi64(cmp);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castps_si128(
        _mm_shuffle_ps(cmp, cmp, sg_sse2_shuffle32_imm(1, 1, 0, 0)));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_cvtcmp_ps_pi64 sg_cvtcmp_pi32_pi64
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_cvtcmp_pi64_pi32(const sg_cmp_pi64 cmp) {
    sg_cmp_pi32 result;
    result.b0 = cmp.b0; result.b1 = cmp.b1;
    result.b2 = false; result.b3 = false;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtcmp_pi64_pi32(cmp) \
    _mm_and_si128(_mm_set_epi32(0, 0, sg_allset_s32, sg_allset_s32), \
        _mm_shuffle_epi32(cmp, sg_sse2_shuffle32_imm(3, 2, 3, 0)))
#elif defined SIMD_GRANODI_NEON
#define sg_cvtcmp_pi64_pi32(cmp) \
    vuzp1q_u32(vreinterpretq_u32_u64(cmp), vdupq_n_u32(0))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_cvtcmp_pi64_ps(const sg_cmp_pi64 cmp) {
    return sg_cvtcmp_pi64_pi32(cmp);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtcmp_pi64_ps(cmp) _mm_castsi128_ps(sg_cvtcmp_pi64_pi32(cmp))
#elif defined SIMD_GRANODI_NEON
#define sg_cvtcmp_pi64_ps sg_cvtcmp_pi64_pi32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_cvtcmp_pd_ps(const sg_cmp_pd cmp) {
    return sg_cvtcmp_pi64_pi32(cmp);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtcmp_pd_ps(cmp) _mm_shuffle_ps(_mm_castpd_ps(cmp), \
    _mm_setzero_ps(), sg_sse2_shuffle32_imm(3, 2, 3, 0))
#elif defined SIMD_GRANODI_NEON
#define sg_cvtcmp_pd_ps sg_cvtcmp_pi64_pi32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_cvtcmp_pd_pi32(const sg_cmp_pd cmp) {
    return sg_cvtcmp_pi64_pi32(cmp);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cvtcmp_pd_pi32(cmp) _mm_castps_si128(sg_cvtcmp_pd_ps(cmp))
#elif defined SIMD_GRANODI_NEON
#define sg_cvtcmp_pd_pi32 sg_cvtcmp_pi64_pi32
#endif

// Combine comparisons (bitwise operations repeated if masks)

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_and_cmp_pi32(const sg_cmp_pi32 cmpa,
    const sg_cmp_pi32 cmpb)
{
    sg_cmp_pi32 result;
    result.b0 = cmpa.b0 && cmpb.b0; result.b1 = cmpa.b1 && cmpb.b1;
    result.b2 = cmpa.b2 && cmpb.b2; result.b3 = cmpa.b3 && cmpb.b3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_and_cmp_pi32 _mm_and_si128
#elif defined SIMD_GRANODI_NEON
#define sg_and_cmp_pi32 vandq_u32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi64 sg_and_cmp_pi64(const sg_cmp_pi64 cmpa,
    const sg_cmp_pi64 cmpb)
{
    sg_cmp_pi64 result;
    result.b0 = cmpa.b0 && cmpb.b0; result.b1 = cmpa.b1 && cmpb.b1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_and_cmp_pi64 _mm_and_si128
#elif defined SIMD_GRANODI_NEON
#define sg_and_cmp_pi64 vandq_u64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_and_cmp_ps(const sg_cmp_ps cmpa,
    const sg_cmp_ps cmpb)
{
    return sg_and_cmp_pi32(cmpa, cmpb);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_and_cmp_ps _mm_and_ps
#elif defined SIMD_GRANODI_NEON
#define sg_and_cmp_ps vandq_u32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_and_cmp_pd(const sg_cmp_pd cmpa,
    const sg_cmp_pd cmpb)
{
    return sg_and_cmp_pi64(cmpa, cmpb);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_and_cmp_pd _mm_and_pd
#elif defined SIMD_GRANODI_NEON
#define sg_and_cmp_pd vandq_u64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_andnot_cmp_pi32(const sg_cmp_pi32 cmpa,
    const sg_cmp_pi32 cmpb)
{
    sg_cmp_pi32 result;
    result.b0 = !cmpa.b0 && cmpb.b0; result.b1 = !cmpa.b1 && cmpb.b1;
    result.b2 = !cmpa.b2 && cmpb.b2; result.b3 = !cmpa.b3 && cmpb.b3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_andnot_cmp_pi32 _mm_andnot_si128
#elif defined SIMD_GRANODI_NEON
#define sg_andnot_cmp_pi32(cmpa, cmpb) vbicq_u32(cmpb, cmpa)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi64 sg_andnot_cmp_pi64(const sg_cmp_pi64 cmpa,
    const sg_cmp_pi64 cmpb)
{
    sg_cmp_pi64 result;
    result.b0 = !cmpa.b0 && cmpb.b0; result.b1 = !cmpa.b1 && cmpb.b1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_andnot_cmp_pi64 _mm_andnot_si128
#elif defined SIMD_GRANODI_NEON
#define sg_andnot_cmp_pi64(cmpa, cmpb) vbicq_u64(cmpb, cmpa)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_andnot_cmp_ps(const sg_cmp_ps cmpa, sg_cmp_ps cmpb) {
    return sg_andnot_cmp_pi32(cmpa, cmpb);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_andnot_cmp_ps _mm_andnot_ps
#elif defined SIMD_GRANODI_NEON
#define sg_andnot_cmp_ps(cmpa, cmpb) vbicq_u32(cmpb, cmpa)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_andnot_cmp_pd(const sg_cmp_pd cmpa, sg_cmp_pd cmpb) {
    return sg_andnot_cmp_pi64(cmpa, cmpb);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_andnot_cmp_pd _mm_andnot_pd
#elif defined SIMD_GRANODI_NEON
#define sg_andnot_cmp_pd(cmpa, cmpb) vbicq_u64(cmpb, cmpa)
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_not_cmp_pi32(const sg_cmp_pi32 cmp) {
    sg_cmp_pi32 result;
    result.b0 = !cmp.b0; result.b1 = !cmp.b1;
    result.b2 = !cmp.b2; result.b3 = !cmp.b3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_not_cmp_pi32 sg_sse2_not_si128
#elif defined SIMD_GRANODI_NEON
#define sg_not_cmp_pi32 vmvnq_u32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi64 sg_not_cmp_pi64(const sg_cmp_pi64 cmp) {
    sg_cmp_pi64 result;
    result.b0 = !cmp.b0; result.b1 = !cmp.b1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_not_cmp_pi64 sg_sse2_not_si128
#elif defined SIMD_GRANODI_NEON
#define sg_not_cmp_pi64 sg_neon_not_u64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_not_cmp_ps(const sg_cmp_ps cmp) {
    return sg_not_cmp_pi32(cmp);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_not_cmp_ps sg_sse2_not_ps
#elif defined SIMD_GRANODI_NEON
#define sg_not_cmp_ps vmvnq_u32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_not_cmp_pd(const sg_cmp_pd cmp) {
    return sg_not_cmp_pi64(cmp);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_not_cmp_pd sg_sse2_not_pd
#elif defined SIMD_GRANODI_NEON
#define sg_not_cmp_pd sg_neon_not_u64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_or_cmp_pi32(const sg_cmp_pi32 cmpa,
    const sg_cmp_pi32 cmpb)
{
    sg_cmp_pi32 result;
    result.b0 = cmpa.b0 || cmpb.b0; result.b1 = cmpa.b1 || cmpb.b1;
    result.b2 = cmpa.b2 || cmpb.b2; result.b3 = cmpa.b3 || cmpb.b3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_or_cmp_pi32 _mm_or_si128
#elif defined SIMD_GRANODI_NEON
#define sg_or_cmp_pi32 vorrq_u32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi64 sg_or_cmp_pi64(const sg_cmp_pi64 cmpa,
    const sg_cmp_pi64 cmpb)
{
    sg_cmp_pi64 result;
    result.b0 = cmpa.b0 || cmpb.b0; result.b1 = cmpa.b1 || cmpb.b1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_or_cmp_pi64 _mm_or_si128
#elif defined SIMD_GRANODI_NEON
#define sg_or_cmp_pi64 vorrq_u64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_or_cmp_ps(const sg_cmp_ps cmpa,
    const sg_cmp_ps cmpb)
{
    return sg_or_cmp_pi32(cmpa, cmpb);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_or_cmp_ps _mm_or_ps
#elif defined SIMD_GRANODI_NEON
#define sg_or_cmp_ps vorrq_u32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_or_cmp_pd(const sg_cmp_pd cmpa,
    const sg_cmp_pd cmpb)
{
    return sg_or_cmp_pi64(cmpa, cmpb);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_or_cmp_pd _mm_or_pd
#elif defined SIMD_GRANODI_NEON
#define sg_or_cmp_pd vorrq_u64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_xor_cmp_pi32(const sg_cmp_pi32 cmpa,
    const sg_cmp_pi32 cmpb)
{
    sg_cmp_pi32 result;
    result.b0 = cmpa.b0 != cmpb.b0; result.b1 = cmpa.b1 != cmpb.b1;
    result.b2 = cmpa.b2 != cmpb.b2; result.b3 = cmpa.b3 != cmpb.b3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_xor_cmp_pi32 _mm_xor_si128
#elif defined SIMD_GRANODI_NEON
#define sg_xor_cmp_pi32 veorq_u32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi64 sg_xor_cmp_pi64(const sg_cmp_pi64 cmpa,
    const sg_cmp_pi64 cmpb)
{
    sg_cmp_pi64 result;
    result.b0 = cmpa.b0 != cmpb.b0; result.b1 = cmpa.b1 != cmpb.b1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_xor_cmp_pi64 _mm_xor_si128
#elif defined SIMD_GRANODI_NEON
#define sg_xor_cmp_pi64 veorq_u64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_xor_cmp_ps(const sg_cmp_ps cmpa,
    const sg_cmp_ps cmpb)
{
    return sg_xor_cmp_pi32(cmpa, cmpb);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_xor_cmp_ps _mm_xor_ps
#elif defined SIMD_GRANODI_NEON
#define sg_xor_cmp_ps veorq_u32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_xor_cmp_pd(const sg_cmp_pd cmpa,
    const sg_cmp_pd cmpb)
{
    return sg_xor_cmp_pi64(cmpa, cmpb);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_xor_cmp_pd _mm_xor_pd
#elif defined SIMD_GRANODI_NEON
#define sg_xor_cmp_pd veorq_u64
#endif

// We put "cmp" in the name twice to avoid subtle bugs

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi32 sg_cmpeq_cmp_pi32(const sg_cmp_pi32 cmpa,
    const sg_cmp_pi32 cmpb)
{
    sg_cmp_pi32 result;
    result.b0 = cmpa.b0 == cmpb.b0; result.b1 = cmpa.b1 == cmpb.b1;
    result.b2 = cmpa.b2 == cmpb.b2; result.b3 = cmpa.b3 == cmpb.b3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpeq_cmp_pi32 _mm_cmpeq_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_cmpeq_cmp_pi32 vceqq_u32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pi64 sg_cmpeq_cmp_pi64(const sg_cmp_pi64 cmpa,
    const sg_cmp_pi64 cmpb)
{
    sg_cmp_pi64 result;
    result.b0 = cmpa.b0 == cmpb.b0; result.b1 = cmpa.b1 == cmpb.b1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpeq_cmp_pi64 _mm_cmpeq_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_cmpeq_cmp_pi64 vceqq_u64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_ps sg_cmpeq_cmp_ps(const sg_cmp_ps cmpa,
    const sg_cmp_ps cmpb)
{
    return sg_cmpeq_cmp_pi32(cmpa, cmpb);
}
#elif defined SIMD_GRANODI_SSE2
// NaN is NOT equal to NaN!
#define sg_cmpeq_cmp_ps(a, b) sg_sse2_not_ps(_mm_xor_ps(a, b))
#elif defined SIMD_GRANODI_NEON
#define sg_cmpeq_cmp_ps vceqq_u32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_cmp_pd sg_cmpeq_cmp_pd(const sg_cmp_pd cmpa,
    const sg_cmp_pd cmpb)
{
    return sg_cmpeq_cmp_pi64(cmpa, cmpb);
}
#elif defined SIMD_GRANODI_SSE2
#define sg_cmpeq_cmp_pd(a, b) sg_sse2_not_pd(_mm_xor_pd(a, b))
#elif defined SIMD_GRANODI_NEON
#define sg_cmpeq_cmp_pd vceqq_u64
#endif

#define sg_cmpneq_cmp_pi32 sg_xor_cmp_pi32
#define sg_cmpneq_cmp_pi64 sg_xor_cmp_pi64
#define sg_cmpneq_cmp_ps sg_xor_cmp_ps
#define sg_cmpneq_cmp_pd sg_xor_cmp_pd

// Choose / blend

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi32 sg_choose_pi32(const sg_cmp_pi32 cmp,
    const sg_pi32 if_true, const sg_pi32 if_false)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi32 result;
    result.i0 = cmp.b0 ? if_true.i0 : if_false.i0;
    result.i1 = cmp.b1 ? if_true.i1 : if_false.i1;
    result.i2 = cmp.b2 ? if_true.i2 : if_false.i2;
    result.i3 = cmp.b3 ? if_true.i3 : if_false.i3;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    // There is a single instruction in later versions of SSE for blending,
    // but the throughput is the same as these combined instructions
    return _mm_or_si128(_mm_andnot_si128(cmp, if_false),
        _mm_and_si128(cmp, if_true));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_choose_pi32 vbslq_s32
#endif

// More efficient special case
#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_choose_else_zero_pi32(const sg_cmp_pi32 cmp,
    const sg_pi32 if_true)
{
    sg_pi32 result;
    result.i0 = cmp.b0 ? if_true.i0 : 0; result.i1 = cmp.b1 ? if_true.i1 : 0;
    result.i2 = cmp.b2 ? if_true.i2 : 0; result.i3 = cmp.b3 ? if_true.i3 : 0;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_choose_else_zero_pi32 _mm_and_si128
#elif defined SIMD_GRANODI_NEON
#define sg_choose_else_zero_pi32(cmp, if_true) \
    vreinterpretq_s32_u32(vandq_u32(cmp, vreinterpretq_u32_s32(if_true)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_choose_pi64(const sg_cmp_pi64 cmp,
    const sg_pi64 if_true, const sg_pi64 if_false)
{
    sg_pi64 result;
    result.l0 = cmp.b0 ? if_true.l0 : if_false.l0;
    result.l1 = cmp.b1 ? if_true.l1 : if_false.l1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_choose_pi64 sg_choose_pi32
#elif defined SIMD_GRANODI_NEON
#define sg_choose_pi64 vbslq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_choose_else_zero_pi64(const sg_cmp_pi64 cmp,
    const sg_pi64 if_true)
{
    sg_pi64 result;
    result.l0 = cmp.b0 ? if_true.l0 : 0; result.l1 = cmp.b1 ? if_true.l1 : 0;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_choose_else_zero_pi64 _mm_and_si128
#elif defined SIMD_GRANODI_NEON
#define sg_choose_else_zero_pi64(cmp, if_true) \
    vreinterpretq_s64_u64(vandq_u64(cmp, vreinterpretq_u64_s64(if_true)))
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_ps sg_choose_ps(const sg_cmp_ps cmp,
    const sg_ps if_true, const sg_ps if_false)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_ps result;
    result.f0 = cmp.b0 ? if_true.f0 : if_false.f0;
    result.f1 = cmp.b1 ? if_true.f1 : if_false.f1;
    result.f2 = cmp.b2 ? if_true.f2 : if_false.f2;
    result.f3 = cmp.b3 ? if_true.f3 : if_false.f3;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_ps(_mm_andnot_ps(cmp, if_false),
        _mm_and_ps(cmp, if_true));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_choose_ps vbslq_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_choose_else_zero_ps(const sg_cmp_ps cmp,
    const sg_ps if_true)
{
    sg_ps result;
    result.f0 = cmp.b0 ? if_true.f0 : 0.0f;
    result.f1 = cmp.b1 ? if_true.f1 : 0.0f;
    result.f2 = cmp.b2 ? if_true.f2 : 0.0f;
    result.f3 = cmp.b3 ? if_true.f3 : 0.0f;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_choose_else_zero_ps _mm_and_ps
#elif defined SIMD_GRANODI_NEON
#define sg_choose_else_zero_ps(cmp, if_true) \
    vreinterpretq_f32_u32(vandq_u32(cmp, vreinterpretq_u32_f32(if_true)))
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pd sg_choose_pd(const sg_cmp_pd cmp,
    const sg_pd if_true, const sg_pd if_false)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pd result;
    result.d0 = cmp.b0 ? if_true.d0 : if_false.d0;
    result.d1 = cmp.b1 ? if_true.d1 : if_false.d1;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_pd(_mm_andnot_pd(cmp, if_false),
        _mm_and_pd(cmp, if_true));
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_choose_pd vbslq_f64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_choose_else_zero_pd(const sg_cmp_pd cmp,
    const sg_pd if_true)
{
    sg_pd result;
    result.d0 = cmp.b0 ? if_true.d0 : 0.0;
    result.d1 = cmp.b1 ? if_true.d1 : 0.0;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_choose_else_zero_pd _mm_and_pd
#elif defined SIMD_GRANODI_NEON
#define sg_choose_else_zero_pd(cmp, if_true) \
    vreinterpretq_f64_u64(vandq_u64(cmp, vreinterpretq_u64_f64(if_true)))
#endif

/*
Safe division: Divide by 1 if the denominator is 0.
Sometimes we want to divide by a number that could be zero, and then
discard (via masking) the result of the division later when it turns out
the denominator was zero. But before then, we want the division to happen
safely: even if hardware exceptions are ignored, a slowdown could still
happen.
Note that
- with float / double, and a large numerator & small denom, may
  not be "safe".
- with integers, this does NOT check for dividing INT_MIN by -1
*/
static inline sg_pi32 sg_safediv_pi32(const sg_pi32 a, const sg_pi32 b) {
    const sg_generic_pi32 ag = sg_getg_pi32(a), bg = sg_getg_pi32(b);
    sg_generic_pi32 result;
    result.i0 = bg.i0 == 0 ? ag.i0 : ag.i0 / bg.i0;
    result.i1 = bg.i1 == 0 ? ag.i1 : ag.i1 / bg.i1;
    result.i2 = bg.i2 == 0 ? ag.i2 : ag.i2 / bg.i2;
    result.i3 = bg.i3 == 0 ? ag.i3 : ag.i3 / bg.i3;
    return sg_set_fromg_pi32(result);
}

static inline sg_pi64 sg_safediv_pi64(const sg_pi64 a, const sg_pi64 b) {
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    sg_generic_pi64 result;
    result.l0 = bg.l0 == 0 ? ag.l0 : ag.l0 / bg.l0;
    result.l1 = bg.l1 == 0 ? ag.l1 : ag.l1 / bg.l1;
    return sg_set_fromg_pi64(result);
}

static inline sg_ps sg_safediv_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_ps result;
    result.f0 = b.f0 == 0.0f ? a.f0 : a.f0 / b.f0;
    result.f1 = b.f1 == 0.0f ? a.f1 : a.f1 / b.f1;
    result.f2 = b.f2 == 0.0f ? a.f2 : a.f2 / b.f2;
    result.f3 = b.f3 == 0.0f ? a.f3 : a.f3 / b.f3;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    const __m128 safe_mask = _mm_cmpneq_ps(_mm_setzero_ps(), b);
    const __m128 safe_b = sg_choose_ps(safe_mask, b, _mm_set1_ps(1.0f));
    return _mm_div_ps(a, safe_b);
    #elif defined SIMD_GRANODI_NEON
    const uint32x4_t unsafe_mask = vceqzq_f32(b);
    const float32x4_t safe_b = sg_choose_ps(unsafe_mask, vdupq_n_f32(1.0f), b);
    return vdivq_f32(a, safe_b);
    #endif
}

static inline sg_pd sg_safediv_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pd result;
    result.d0 = b.d0 == 0.0 ? a.d0 : a.d0 / b.d0;
    result.d1 = b.d1 == 0.0 ? a.d1 : a.d1 / b.d1;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    const __m128d safe_mask = _mm_cmpneq_pd(_mm_setzero_pd(), b);
    const __m128d safe_b = sg_choose_pd(safe_mask, b, _mm_set1_pd(1.0));
    return _mm_div_pd(a, safe_b);
    #elif defined SIMD_GRANODI_NEON
    const uint64x2_t unsafe_mask = vceqzq_f64(b);
    const float64x2_t safe_b = sg_choose_pd(unsafe_mask, vdupq_n_f64(1.0), b);
    return vdivq_f64(a, safe_b);
    #endif
}

// Basic maths (abs, min, max)

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi32 sg_abs_pi32(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi32 result;
    result.i0 = abs(a.i0); result.i1 = abs(a.i1);
    result.i2 = abs(a.i2); result.i3 = abs(a.i3);
    return result;
    #elif defined SIMD_GRANODI_SSE2
    return sg_choose_pi32(_mm_cmplt_epi32(a, _mm_setzero_si128()),
        _mm_sub_epi32(_mm_setzero_si128(), a), a);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_abs_pi32 vabsq_s32
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi64 sg_abs_pi64(const sg_pi64 a) {
    const sg_generic_pi64 ag = sg_getg_pi64(a);
    sg_generic_pi64 result;
    result.l0 = ag.l0 < 0 ? -ag.l0 : ag.l0;
    result.l1 = ag.l1 < 0 ? -ag.l1 : ag.l1;
    return sg_set_fromg_pi64(result);
}
#elif defined SIMD_GRANODI_NEON
#define sg_abs_pi64 vabsq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_abs_ps(const sg_ps a) {
    sg_ps result;
    result.f0 = fabsf(a.f0); result.f1 = fabsf(a.f1);
    result.f2 = fabsf(a.f2); result.f3 = fabsf(a.f3);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_abs_ps(a) _mm_and_ps(a, sg_sse2_signmask_ps)
#elif defined SIMD_GRANODI_NEON
#define sg_abs_ps vabsq_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_abs_pd(const sg_pd a) {
    sg_pd result;
    result.d0 = fabs(a.d0); result.d1 = fabs(a.d1);
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_abs_pd(a) _mm_and_pd(a, sg_sse2_signmask_pd)
#elif defined SIMD_GRANODI_NEON
#define sg_abs_pd vabsq_f64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_neg_pi32(const sg_pi32 a) {
    sg_pi32 result;
    result.i0 = -a.i0; result.i1 = -a.i1; result.i2 = -a.i2; result.i3 = -a.i3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_neg_pi32(a) _mm_sub_epi32(_mm_setzero_si128(), a)
#elif defined SIMD_GRANODI_NEON
#define sg_neg_pi32 vnegq_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_neg_pi64(const sg_pi64 a) {
    sg_pi64 result;
    result.l0 = -a.l0; result.l1 = -a.l1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_neg_pi64(a) _mm_sub_epi64(_mm_setzero_si128(), a)
#elif defined SIMD_GRANODI_NEON
#define sg_neg_pi64 vnegq_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_neg_ps(const sg_ps a) {
    sg_ps result;
    result.f0 = -a.f0; result.f1 = -a.f1;
    result.f2 = -a.f2; result.f3 = -a.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_neg_ps(a) _mm_xor_ps(a, sg_sse2_signbit_ps)
#elif defined SIMD_GRANODI_NEON
#define sg_neg_ps vnegq_f32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_neg_pd(const sg_pd a) {
    sg_pd result;
    result.d0 = -a.d0; result.d1 = -a.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_neg_pd(a) _mm_xor_pd(a, sg_sse2_signbit_pd)
#elif defined SIMD_GRANODI_NEON
#define sg_neg_pd vnegq_f64
#endif

// Infinity and minus infinity constants
#define sg_minus_infinity_f32x1 sg_bitcast_u32x1_f32x1(0xff800000)
#define sg_infinity_f32x1 sg_bitcast_u32x1_f32x1(0x7f800000)
#define sg_minus_infinity_f64x1 sg_bitcast_u64x1_f64x1(0xfff0000000000000)
#define sg_infinity_f64x1 sg_bitcast_u64x1_f64x1(0x7ff0000000000000)

#define sg_minus_infinity_ps sg_set1_ps(sg_minus_infinity_f32x1)
#define sg_infinity_ps sg_set1_ps(sg_infinity_f32x1)
#define sg_minus_infinity_pd sg_set1_pd(sg_minus_infinity_f64x1)
#define sg_infinity_pd sg_set1_pd(sg_infinity_f64x1)

// remove signed zero (only floats/doubles can have signed zero),
// but leave intact if any other value

static inline sg_ps sg_remove_signed_zero_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_ps result;
    result.f0 = a.f0 != 0.0f ? a.f0 : 0.0f;
    result.f1 = a.f1 != 0.0f ? a.f1 : 0.0f;
    result.f2 = a.f2 != 0.0f ? a.f2 : 0.0f;
    result.f3 = a.f3 != 0.0f ? a.f3 : 0.0f;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_ps(a, _mm_cmpneq_ps(a, _mm_setzero_ps()));
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a),
        vmvnq_u32(vceqzq_f32(a))));
    #endif
}

static inline sg_pd sg_remove_signed_zero_pd(const sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pd result;
    result.d0 = a.d0 != 0.0 ? a.d0 : 0.0;
    result.d1 = a.d1 != 0.0 ? a.d1 : 0.0;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_pd(a, _mm_cmpneq_pd(a, _mm_setzero_pd()));
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f64_u32(vandq_u32(vreinterpretq_u32_f64(a),
        vmvnq_u32(vreinterpretq_u32_u64(vceqzq_f64(a)))));
    #endif
}

// min

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi32 sg_min_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi32 result;
    result.i0 = a.i0 < b.i0 ? a.i0 : b.i0;
    result.i1 = a.i1 < b.i1 ? a.i1 : b.i1;
    result.i2 = a.i2 < b.i2 ? a.i2 : b.i2;
    result.i3 = a.i3 < b.i3 ? a.i3 : b.i3;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    return sg_choose_pi32(_mm_cmplt_epi32(a, b), a, b);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_min_pi32 vminq_s32
#endif

static inline sg_pi64 sg_min_pi64(const sg_pi64 a, const sg_pi64 b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    sg_generic_pi64 result;
    result.l0 = ag.l0 < bg.l0 ? ag.l0 : bg.l0;
    result.l1 = ag.l1 < bg.l1 ? ag.l1 : bg.l1;
    return sg_set_fromg_pi64(result);
    #elif defined SIMD_GRANODI_NEON
    return sg_choose_pi64(vcltq_s64(a, b), a, b);
    #endif
}

// The floating point functions are called "max_fast" and "min_fast" as they
// do not behave identically across platforms with regard to signed zero.
// For consistent behaviour, recommend combining with sg_remove_signed_zero()
#if defined SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_min_fast_ps(const sg_ps a, const sg_ps b) {
    sg_ps result;
    result.f0 = a.f0 < b.f0 ? a.f0 : b.f0;
    result.f1 = a.f1 < b.f1 ? a.f1 : b.f1;
    result.f2 = a.f2 < b.f2 ? a.f2 : b.f2;
    result.f3 = a.f3 < b.f3 ? a.f3 : b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_min_fast_ps _mm_min_ps
#elif defined SIMD_GRANODI_NEON
#define sg_min_fast_ps vminq_f32
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_min_fast_pd(const sg_pd a, const sg_pd b) {
    sg_pd result;
    result.d0 = a.d0 < b.d0 ? a.d0 : b.d0;
    result.d1 = a.d1 < b.d1 ? a.d1 : b.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_min_fast_pd _mm_min_pd
#elif defined SIMD_GRANODI_NEON
#define sg_min_fast_pd vminq_f64
#endif

// max

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi32 sg_max_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    sg_pi32 result;
    result.i0 = a.i0 > b.i0 ? a.i0 : b.i0;
    result.i1 = a.i1 > b.i1 ? a.i1 : b.i1;
    result.i2 = a.i2 > b.i2 ? a.i2 : b.i2;
    result.i3 = a.i3 > b.i3 ? a.i3 : b.i3;
    return result;
    #elif defined SIMD_GRANODI_SSE2
    return sg_choose_pi32(_mm_cmpgt_epi32(a, b), a, b);
    #endif
}
#elif defined SIMD_GRANODI_NEON
#define sg_max_pi32 vmaxq_s32
#endif

static inline sg_pi64 sg_max_pi64(const sg_pi64 a, const sg_pi64 b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    sg_generic_pi64 result;
    result.l0 = ag.l0 > bg.l0 ? ag.l0 : bg.l0;
    result.l1 = ag.l1 > bg.l1 ? ag.l1 : bg.l1;
    return sg_set_fromg_pi64(result);
    #elif defined SIMD_GRANODI_NEON
    return sg_choose_pi64(vcgtq_s64(a, b), a, b);
    #endif
}

#if defined SIMD_GRANODI_FORCE_GENERIC
static inline sg_ps sg_max_fast_ps(const sg_ps a, const sg_ps b) {
    sg_ps result;
    result.f0 = a.f0 > b.f0 ? a.f0 : b.f0;
    result.f1 = a.f1 > b.f1 ? a.f1 : b.f1;
    result.f2 = a.f2 > b.f2 ? a.f2 : b.f2;
    result.f3 = a.f3 > b.f3 ? a.f3 : b.f3;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_max_fast_ps _mm_max_ps
#elif defined SIMD_GRANODI_NEON
#define sg_max_fast_ps vmaxq_f32
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC
static inline sg_pd sg_max_fast_pd(const sg_pd a, const sg_pd b) {
    sg_pd result;
    result.d0 = a.d0 > b.d0 ? a.d0 : b.d0;
    result.d1 = a.d1 > b.d1 ? a.d1 : b.d1;
    return result;
}
#elif defined SIMD_GRANODI_SSE2
#define sg_max_fast_pd _mm_max_pd
#elif defined SIMD_GRANODI_NEON
#define sg_max_fast_pd vmaxq_f64
#endif

// Constrain

static inline sg_pi32 sg_constrain_pi32(const sg_pi32 lowerb,
    const sg_pi32 upperb, const sg_pi32 a)
{
    return sg_min_pi32(sg_max_pi32(lowerb, a), upperb);
}

static inline sg_pi64 sg_constrain_pi64(const sg_pi64 lowerb,
    const sg_pi64 upperb, const sg_pi64 a)
{
    return sg_min_pi64(sg_max_pi64(lowerb, a), upperb);
}

static inline sg_ps sg_constrain_ps(const sg_ps lowerb,
    const sg_ps upperb, const sg_ps a)
{
    return sg_min_fast_ps(sg_max_fast_ps(a, lowerb), upperb);
}

static inline sg_pd sg_constrain_pd(const sg_pd lowerb,
    const sg_pd upperb, const sg_pd a)
{
    return sg_min_fast_pd(sg_max_fast_pd(a, lowerb), upperb);
}

// Disable denormals

#ifdef SIMD_GRANODI_ARCH_SSE
typedef uint32_t sg_fp_status;
static inline sg_fp_status sg_disable_denormals() {
    // flush_to_zero:
    //     "sets denormal results from floating-point calculations to zero"
    // denormals_are_zero:
    //     "treats denormal values used as input to floating-point instructions
    //     as zero"
    const sg_fp_status flush_to_zero = 0x8000,
        denormals_are_zero = 0x40,
        previous_status = _mm_getcsr();
    _mm_setcsr(previous_status | flush_to_zero | denormals_are_zero);
    return previous_status;
}
static inline void sg_restore_fp_status_after_denormals_disabled(
    const sg_fp_status previous_status)
{
    _mm_setcsr(previous_status);
}

#elif defined SIMD_GRANODI_ARCH_ARM32
typedef uint32_t sg_fp_status;
static inline void sg_set_fp_status_arm_(const sg_fp_status fp_status) {
    __asm__ volatile("vmsr fpscr, %0" : : "ri"(fp_status));
}

static inline sg_fp_status sg_get_fp_status_arm_() {
    sg_fp_status fp_status = 0;
    __asm__ volatile("vmrs &0, fpscr" : "=r"(fpsr));
}

#elif defined SIMD_GRANODI_ARCH_ARM64
typedef uint64_t sg_fp_status;
static inline void sg_set_fp_status_arm_(const sg_fp_status fp_status) {
    __asm__ volatile("msr fpcr, %0" : : "ri"(fp_status));
}

static inline sg_fp_status sg_get_fp_status_arm_() {
    sg_fp_status fp_status = 0;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fp_status));
    return fp_status;
}

#endif

#if defined SIMD_GRANODI_ARCH_ARM64 || SIMD_GRANODI_ARCH_ARM32
static inline sg_fp_status sg_disable_denormals() {
    const sg_fp_status old_fp_status = sg_get_fp_status_arm_();
    sg_set_fp_status_arm_(old_fp_status | 0x1000000);
    return old_fp_status;
}

static inline void sg_restore_fp_status_after_denormals_disabled(
    const sg_fp_status previous_status)
{
    sg_set_fp_status_arm_(previous_status);
}
#endif

#ifdef __cplusplus

namespace simd_granodi {

// C++ classes for operator overloading:
// - Have no setters, but can be changed via += etc (semi-immutable)
// - Are default constructed to zero
// - All type casts or type conversions must be explicit
// - Should never use any SSE2 or NEON types or intrinsics directly

#if defined SIMD_GRANODI_ARCH_SSE || \
    defined SIMD_GRANODI_ARCH_ARM64 || \
    defined SIMD_GRANODI_ARCH_ARM32
class ScopedDenormalsDisable {
    const sg_fp_status fp_status_;
public:
    ScopedDenormalsDisable() : fp_status_{sg_disable_denormals()} {}
    ~ScopedDenormalsDisable() {
        sg_restore_fp_status_after_denormals_disabled(fp_status_);
    }
};
#endif

class Compare_pi32; class Compare_pi64; class Compare_ps; class Compare_pd;
class Vec_pi32; class Vec_pi64; class Vec_ps; class Vec_pd;

class Compare_pi32 {
    sg_cmp_pi32 data_;
public:
    Compare_pi32() : data_{sg_setzero_cmp_pi32()} {}
    Compare_pi32(const bool b) : data_{sg_set1cmp_pi32(b)} {}
    Compare_pi32(const bool b3, const bool b2, const bool b1, const bool b0)
        : data_{sg_setcmp_pi32(b3, b2, b1, b0)} {}
    Compare_pi32(const sg_cmp_pi32& cmp) : data_{cmp} {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Compare_pi32(const sg_generic_cmp4& cmp)
        : data_{sg_setcmp_fromg_pi32(cmp)} {}
    #endif

    sg_cmp_pi32 data() const { return data_; }

    Compare_pi32 operator&&(const Compare_pi32& rhs) const {
        return sg_and_cmp_pi32(data_, rhs.data());
    }

    Compare_pi32 operator||(const Compare_pi32& rhs) const {
        return sg_or_cmp_pi32(data_, rhs.data());
    }

    Compare_pi32 operator!() const { return sg_not_cmp_pi32(data_); }

    bool debug_valid_eq(const bool b3, const bool b2,
        const bool b1, const bool b0) const
    {
        return sg_debug_cmp_valid_eq_pi32(data_, b3, b2, b1, b0);
    }
    bool debug_valid_eq(const bool b) const {
        return debug_valid_eq(b, b, b, b);
    }

    Compare_pi32 convert_to_cmp_s32() const { return *this; }
    inline Compare_pi64 convert_to_cmp_s64() const;
    inline Compare_ps convert_to_cmp_f32() const;
    inline Compare_pd convert_to_cmp_f64() const;
    inline Vec_pi32 choose_else_zero(const Vec_pi32& if_true) const;
    inline Vec_pi32 choose(const Vec_pi32& if_true,
        const Vec_pi32& if_false) const;
};

inline Compare_pi32 operator==(const Compare_pi32& lhs,
    const Compare_pi32& rhs)
{
    return sg_cmpeq_cmp_pi32(lhs.data(), rhs.data());
}
inline Compare_pi32 operator!=(const Compare_pi32& lhs, const Compare_pi32& rhs)
{
    return sg_cmpneq_cmp_pi32(lhs.data(), rhs.data());
}

class Compare_pi64 {
    sg_cmp_pi64 data_;
public:
    Compare_pi64() : data_{sg_setzero_cmp_pi64()} {}
    Compare_pi64(const bool b) : data_{sg_set1cmp_pi64(b)} {}
    Compare_pi64(const bool b1, const bool b0)
        : data_{sg_setcmp_pi64(b1, b0)} {}
    Compare_pi64(const sg_cmp_pi64& cmp) : data_{cmp} {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Compare_pi64(const sg_generic_cmp2& cmp)
        : data_{sg_setcmp_fromg_pi64(cmp)} {}
    #endif

    sg_cmp_pi64 data() const { return data_; }

    Compare_pi64 operator&&(const Compare_pi64& rhs) const {
        return sg_and_cmp_pi64(data_, rhs.data());
    }

    Compare_pi64 operator||(const Compare_pi64& rhs) const {
        return sg_or_cmp_pi64(data_, rhs.data());
    }

    Compare_pi64 operator!() const { return sg_not_cmp_pi64(data_); }

    bool debug_valid_eq(const bool b1, const bool b0) const {
        return sg_debug_cmp_valid_eq_pi64(data_, b1, b0);
    }
    bool debug_valid_eq(const bool b) const { return debug_valid_eq(b, b); }

    inline Compare_pi32 convert_to_cmp_s32() const;
    Compare_pi64 convert_to_cmp_s64() const { return *this; }
    inline Compare_ps convert_to_cmp_f32() const;
    inline Compare_pd convert_to_cmp_f64() const;

    inline Vec_pi64 choose_else_zero(const Vec_pi64& if_true) const;
    inline Vec_pi64 choose(const Vec_pi64& if_true,
        const Vec_pi64& if_false) const;
};

inline Compare_pi64 operator==(const Compare_pi64& lhs,
    const Compare_pi64& rhs)
{
    return sg_cmpeq_cmp_pi64(lhs.data(), rhs.data());
}
inline Compare_pi64 operator!=(const Compare_pi64& lhs, const Compare_pi64& rhs)
{
    return sg_cmpneq_cmp_pi64(lhs.data(), rhs.data());
}

class Compare_ps {
    sg_cmp_ps data_;
public:
    Compare_ps() : data_{sg_setzero_cmp_ps()} {}
    Compare_ps(const bool b) : data_{sg_set1cmp_ps(b)} {}
    Compare_ps(const bool b3, const bool b2, const bool b1, const bool b0)
        : data_{sg_setcmp_ps(b3, b2, b1, b0)} {}
    Compare_ps(const sg_cmp_ps& cmp) : data_{cmp} {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Compare_ps(const sg_generic_cmp4& cmp) : data_{sg_setcmp_fromg_ps(cmp)} {}
    #endif

    sg_cmp_ps data() const { return data_; }

    Compare_ps operator&&(const Compare_ps& rhs) const {
        return sg_and_cmp_ps(data_, rhs.data());
    }

    Compare_ps operator||(const Compare_ps& rhs) const {
        return sg_or_cmp_ps(data_, rhs.data());
    }

    Compare_ps operator!() const { return sg_not_cmp_ps(data_); }

    bool debug_valid_eq(const bool b3, const bool b2,
        const bool b1, const bool b0) const
    {
        return sg_debug_cmp_valid_eq_ps(data_, b3, b2, b1, b0);
    }
    bool debug_valid_eq(const bool b) { return debug_valid_eq(b, b, b, b); }

    inline Compare_pi32 convert_to_cmp_s32() const;
    inline Compare_pi64 convert_to_cmp_s64() const;
    Compare_ps convert_to_cmp_f32() const { return *this; }
    inline Compare_pd convert_to_cmp_f64() const;

    inline Vec_ps choose_else_zero(const Vec_ps& if_true) const;
    inline Vec_ps choose(const Vec_ps& if_true, const Vec_ps& if_false) const;
};

inline Compare_ps operator==(const Compare_ps& lhs,
    const Compare_ps& rhs)
{
    return sg_cmpeq_cmp_ps(lhs.data(), rhs.data());
}
inline Compare_ps operator!=(const Compare_ps& lhs, const Compare_ps& rhs)
{
    return sg_cmpneq_cmp_ps(lhs.data(), rhs.data());
}

class Compare_pd {
    sg_cmp_pd data_;
public:
    Compare_pd() : data_{sg_setzero_cmp_pd()} {}
    Compare_pd(const bool b) : data_{sg_set1cmp_pd(b)} {}
    Compare_pd(const bool b1, const bool b0)
        : data_{sg_setcmp_pd(b1, b0)} {}
    Compare_pd(const sg_cmp_pd& cmp) : data_(cmp) {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Compare_pd(const sg_generic_cmp2& cmp) : data_{sg_setcmp_fromg_pd(cmp)} {}
    #endif

    sg_cmp_pd data() const { return data_; }

    Compare_pd operator&&(const Compare_pd& rhs) const {
        return sg_and_cmp_pd(data_, rhs.data());
    }

    Compare_pd operator||(const Compare_pd& rhs) const {
        return sg_or_cmp_pd(data_, rhs.data());
    }

    Compare_pd operator!() const { return sg_not_cmp_pd(data_); }

    bool debug_valid_eq(const bool b1, const bool b0) const {
        return sg_debug_cmp_valid_eq_pd(data_, b1, b0);
    }
    bool debug_valid_eq(const bool b) const { return debug_valid_eq(b, b); }

    inline Compare_pi32 convert_to_cmp_s32() const;
    inline Compare_pi64 convert_to_cmp_s64() const;
    inline Compare_ps convert_to_cmp_f32() const;
    Compare_pd convert_to_cmp_f64() const { return *this; }

    inline Vec_pd choose_else_zero(const Vec_pd& if_true) const;
    inline Vec_pd choose(const Vec_pd& if_true, const Vec_pd& if_false) const;
};

inline Compare_pd operator==(const Compare_pd& lhs,
    const Compare_pd& rhs)
{
    return sg_cmpeq_cmp_pd(lhs.data(), rhs.data());
}
inline Compare_pd operator!=(const Compare_pd& lhs, const Compare_pd& rhs)
{
    return sg_cmpneq_cmp_pd(lhs.data(), rhs.data());
}

class Vec_pi32 {
    sg_pi32 data_;
public:
    Vec_pi32() : data_{sg_setzero_pi32()} {}
    Vec_pi32(const int32_t i) : data_{sg_set1_pi32(i)} {}
    Vec_pi32(const int32_t i3, const int32_t i2, const int32_t i1,
        const int32_t i0) : data_{sg_set_pi32(i3, i2, i1, i0)} {}
    Vec_pi32(const sg_pi32& pi32) : data_{pi32} {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    // Otherwise, we are defining two identical ctors & won't compile...
    Vec_pi32(const sg_generic_pi32& g_pi32)
        : data_{sg_set_fromg_pi32(g_pi32)} {}
    #endif

    static int32_t elem_t(const int32_t s32) { return s32; }
    static int32_t elem_t(const int64_t s64) {
        return static_cast<int32_t>(s64);
    }
    static int32_t elem_t(const float f32) { return static_cast<int32_t>(f32); }
    static int32_t elem_t(const double f64) {
        return static_cast<int32_t>(f64);
    }

    static Compare_pi32 compare_t(const Compare_pi32& cmp) { return cmp; }
    static Compare_pi32 compare_t(const Compare_pi64& cmp) {
        return sg_cvtcmp_pi64_pi32(cmp.data());
    }
    static Compare_pi32 compare_t(const Compare_ps& cmp) {
        return sg_cvtcmp_ps_pi32(cmp.data());
    }
    static Compare_pi32 compare_t(const Compare_pd& cmp) {
        return sg_cvtcmp_pd_pi32(cmp.data());
    }

    static Vec_pi32 bitcast_from_u32(const uint32_t i) {
        return sg_set1_from_u32_pi32(i);
    }
    static Vec_pi32 bitcast_from_u32(const uint32_t i3, const uint32_t i2,
        const uint32_t i1, const uint32_t i0)
    {
        return sg_set_from_u32_pi32(i3, i2, i1, i0);
    }

    sg_pi32 data() const { return data_; }
    sg_generic_pi32 generic() const { return sg_getg_pi32(data_); }
    int32_t i0() const { return sg_get0_pi32(data_); }
    int32_t i1() const { return sg_get1_pi32(data_); }
    int32_t i2() const { return sg_get2_pi32(data_); }
    int32_t i3() const { return sg_get3_pi32(data_); }

    Vec_pi32& operator++() {
        data_ = sg_add_pi32(data_, sg_set1_pi32(1));
        return *this;
    }
    Vec_pi32 operator++(int) {
        Vec_pi32 old = *this;
        operator++();
        return old;
    }

    Vec_pi32& operator--() {
        data_ = sg_sub_pi32(data_, sg_set1_pi32(1));
        return *this;
    }
    Vec_pi32 operator--(int) {
        Vec_pi32 old = *this;
        operator--();
        return old;
    }

    Vec_pi32& operator+=(const Vec_pi32& rhs) {
        data_ = sg_add_pi32(data_, rhs.data());
        return *this;
    }
    friend Vec_pi32 operator+(Vec_pi32 lhs, const Vec_pi32& rhs) {
        lhs += rhs;
        return lhs;
    }
    Vec_pi32 operator+() const { return *this; }

    Vec_pi32& operator-=(const Vec_pi32& rhs) {
        data_ = sg_sub_pi32(data_, rhs.data());
        return *this;
    }
    friend Vec_pi32 operator-(Vec_pi32 lhs, const Vec_pi32& rhs) {
        lhs -= rhs;
        return lhs;
    }
    Vec_pi32 operator-() const { return sg_neg_pi32(data_); }

    Vec_pi32& operator*=(const Vec_pi32& rhs) {
        data_ = sg_mul_pi32(data_, rhs.data());
        return *this;
    }
    friend Vec_pi32 operator*(Vec_pi32 lhs, const Vec_pi32& rhs) {
        lhs *= rhs;
        return lhs;
    }

    Vec_pi32& operator/=(const Vec_pi32& rhs) {
        data_ = sg_div_pi32(data_, rhs.data());
        return *this;
    }
    friend Vec_pi32 operator/(Vec_pi32 lhs, const Vec_pi32& rhs) {
        lhs /= rhs;
        return lhs;
    }

    Vec_pi32& operator&=(const Vec_pi32& rhs) {
        data_ = sg_and_pi32(data_, rhs.data());
        return *this;
    }
    friend Vec_pi32 operator&(Vec_pi32 lhs, const Vec_pi32& rhs) {
        lhs &= rhs;
        return lhs;
    }

    Vec_pi32& operator|=(const Vec_pi32& rhs) {
        data_ = sg_or_pi32(data_, rhs.data());
        return *this;
    }
    friend Vec_pi32 operator|(Vec_pi32 lhs, const Vec_pi32& rhs) {
        lhs |= rhs;
        return lhs;
    }

    Vec_pi32& operator^=(const Vec_pi32& rhs) {
        data_ = sg_xor_pi32(data_, rhs.data());
        return *this;
    }
    friend Vec_pi32 operator^(Vec_pi32 lhs, const Vec_pi32& rhs) {
        lhs ^= rhs;
        return lhs;
    }

    Vec_pi32 operator~() const { return sg_not_pi32(data_); }

    Compare_pi32 operator<(const Vec_pi32& rhs) const {
        return sg_cmplt_pi32(data_, rhs.data());
    }
    Compare_pi32 operator<=(const Vec_pi32& rhs) const {
        return sg_cmplte_pi32(data_, rhs.data());
    }
    Compare_pi32 operator==(const Vec_pi32& rhs) const {
        return sg_cmpeq_pi32(data_, rhs.data());
    }
    Compare_pi32 operator!=(const Vec_pi32& rhs) const {
        return sg_cmpneq_pi32(data_, rhs.data());
    }
    Compare_pi32 operator>=(const Vec_pi32& rhs) const {
        return sg_cmpgte_pi32(data_, rhs.data());
    }
    Compare_pi32 operator>(const Vec_pi32& rhs) const {
        return sg_cmpgt_pi32(data_, rhs.data());
    }

    template <int shift>
    Vec_pi32 shift_l_imm() const { return sg_sl_imm_pi32(data_, shift); }
    template <int shift>
    Vec_pi32 shift_rl_imm() const { return sg_srl_imm_pi32(data_, shift); }
    template <int shift>
    Vec_pi32 shift_ra_imm() const { return sg_sra_imm_pi32(data_, shift); }

    template <int src3, int src2, int src1, int src0>
    Vec_pi32 shuffle() const {
        return sg_shuffle_pi32(data_, src3, src2, src1, src0);
    }

    Vec_pi32 safe_divide_by(const Vec_pi32& rhs) const {
        return sg_safediv_pi32(data_, rhs.data());
    }
    Vec_pi32 abs() const { return sg_abs_pi32(data_); }
    Vec_pi32 constrain(const Vec_pi32& lowerb, const Vec_pi32& upperb) const {
        return sg_constrain_pi32(lowerb.data(), upperb.data(), data_);
    }

    static Vec_pi32 min(const Vec_pi32& a, const Vec_pi32& b) {
        return sg_min_pi32(a.data(), b.data());
    }
    static Vec_pi32 max(const Vec_pi32& a, const Vec_pi32& b) {
        return sg_max_pi32(a.data(), b.data());
    }

    bool debug_eq(const int32_t i3, const int32_t i2,
        const int32_t i1, const int32_t i0) const
    {
        return sg_debug_eq_pi32(data_, i3, i2, i1, i0);
    }
    bool debug_eq(const int32_t i) const { return debug_eq(i, i, i, i); }

    Vec_pi32 bitcast_to_s32() const { return *this; }
    inline Vec_pi64 bitcast_to_s64() const;
    inline Vec_ps bitcast_to_f32() const;
    inline Vec_pd bitcast_to_f64() const;

    Vec_pi32 convert_to_s32() const { return *this; }
    inline Vec_pi64 convert_to_s64() const;
    inline Vec_ps convert_to_f32() const;
    inline Vec_pd convert_to_f64() const;

    static Vec_pi32 from(const Vec_pi32& pi32) { return pi32; }
    static inline Vec_pi32 from(const Vec_pi64& pi64);
};

class Vec_pi64 {
    sg_pi64 data_;
public:
    Vec_pi64() : data_{sg_setzero_pi64()} {}
    Vec_pi64(const int64_t l) : data_{sg_set1_pi64(l)} {}
    Vec_pi64(const int64_t l1, const int64_t l0) : data_{sg_set_pi64(l1, l0)} {}
    Vec_pi64(const sg_pi64& pi64) : data_{pi64} {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Vec_pi64(const sg_generic_pi64& g_pi64)
        : data_{sg_set_fromg_pi64(g_pi64)} {}
    #endif

    static int64_t elem_t(const int32_t s32) {
        return static_cast<int64_t>(s32);
    }
    static int64_t elem_t(const int64_t s64) { return s64; }
    static int64_t elem_t(const float f32) { return static_cast<int64_t>(f32); }
    static int64_t elem_t(const double f64) {
        return static_cast<int64_t>(f64);
    }

    static Compare_pi64 compare_t(const Compare_pi32& cmp) {
        return sg_cvtcmp_pi32_pi64(cmp.data());
    }
    static Compare_pi64 compare_t(const Compare_pi64& cmp) { return cmp; }
    static Compare_pi64 compare_t(const Compare_ps& cmp) {
        return sg_cvtcmp_ps_pi64(cmp.data());
    }
    static Compare_pi64 compare_t(const Compare_pd& cmp) {
        return sg_cvtcmp_pd_pi64(cmp.data());
    }

    static Vec_pi64 bitcast_from_u64(const uint64_t l) {
        return sg_set1_from_u64_pi64(l);
    }
    static Vec_pi64 bitcast_from_u64(const uint64_t l1, const uint64_t l0) {
        return sg_set_from_u64_pi64(l1, l0);
    }

    sg_pi64 data() const { return data_; }
    sg_generic_pi64 generic() const { return sg_getg_pi64(data_); }
    int64_t l0() const { return sg_get0_pi64(data_); }
    int64_t l1() const { return sg_get1_pi64(data_); }

    Vec_pi64& operator++() {
        data_ = sg_add_pi64(data_, sg_set1_pi64(1));
        return *this;
    }
    Vec_pi64 operator++(int) {
        Vec_pi64 old = *this;
        operator++();
        return old;
    }

    Vec_pi64& operator--() {
        data_ = sg_sub_pi64(data_, sg_set1_pi64(1));
        return *this;
    }
    Vec_pi64 operator--(int) {
        Vec_pi64 old = *this;
        operator--();
        return old;
    }

    Vec_pi64& operator+=(const Vec_pi64& rhs) {
        data_ = sg_add_pi64(data_, rhs.data());
        return *this;
    }
    friend Vec_pi64 operator+(Vec_pi64 lhs, const Vec_pi64& rhs) {
        lhs += rhs;
        return lhs;
    }
    Vec_pi64 operator+() const { return *this; }

    Vec_pi64& operator-=(const Vec_pi64& rhs) {
        data_ = sg_sub_pi64(data_, rhs.data());
        return *this;
    }
    friend Vec_pi64 operator-(Vec_pi64 lhs, const Vec_pi64& rhs) {
        lhs -= rhs;
        return lhs;
    }
    Vec_pi64 operator-() const { return sg_neg_pi64(data_); }

    Vec_pi64& operator*=(const Vec_pi64& rhs) {
        data_ = sg_mul_pi64(data_, rhs.data());
        return *this;
    }
    friend Vec_pi64 operator*(Vec_pi64 lhs, const Vec_pi64& rhs) {
        lhs *= rhs;
        return lhs;
    }

    Vec_pi64& operator/=(const Vec_pi64& rhs) {
        data_ = sg_div_pi64(data_, rhs.data());
        return *this;
    }
    friend Vec_pi64 operator/(Vec_pi64 lhs, const Vec_pi64& rhs) {
        lhs /= rhs;
        return lhs;
    }

    Vec_pi64& operator&=(const Vec_pi64& rhs) {
        data_ = sg_and_pi64(data_, rhs.data());
        return *this;
    }
    friend Vec_pi64 operator&(Vec_pi64 lhs, const Vec_pi64& rhs) {
        lhs &= rhs;
        return lhs;
    }

    Vec_pi64& operator|=(const Vec_pi64& rhs) {
        data_ = sg_or_pi64(data_, rhs.data());
        return *this;
    }
    friend Vec_pi64 operator|(Vec_pi64 lhs, const Vec_pi64& rhs) {
        lhs |= rhs;
        return lhs;
    }

    Vec_pi64& operator^=(const Vec_pi64& rhs) {
        data_ = sg_xor_pi64(data_, rhs.data());
        return *this;
    }
    friend Vec_pi64 operator^(Vec_pi64 lhs, const Vec_pi64& rhs) {
        lhs ^= rhs;
        return lhs;
    }

    Vec_pi64 operator~() const { return sg_not_pi64(data_); }

    Compare_pi64 operator<(const Vec_pi64& rhs) const {
        return sg_cmplt_pi64(data_, rhs.data());
    }
    Compare_pi64 operator<=(const Vec_pi64& rhs) const {
        return sg_cmplte_pi64(data_, rhs.data());
    }
    Compare_pi64 operator==(const Vec_pi64& rhs) const {
        return sg_cmpeq_pi64(data_, rhs.data());
    }
    Compare_pi64 operator!=(const Vec_pi64& rhs) const {
        return sg_cmpneq_pi64(data_, rhs.data());
    }
    Compare_pi64 operator>=(const Vec_pi64& rhs) const {
        return sg_cmpgte_pi64(data_, rhs.data());
    }
    Compare_pi64 operator>(const Vec_pi64& rhs) const {
        return sg_cmpgt_pi64(data_, rhs.data());
    }

    template <int shift>
    Vec_pi64 shift_l_imm() const { return sg_sl_imm_pi64(data_, shift); }
    template <int shift>
    Vec_pi64 shift_rl_imm() const { return sg_srl_imm_pi64(data_, shift); }
    template <int shift>
    Vec_pi64 shift_ra_imm() const { return sg_sra_imm_pi64(data_, shift); }

    template <int src1, int src0>
    Vec_pi64 shuffle() const { return sg_shuffle_pi64(data_, src1, src0); }

    Vec_pi64 safe_divide_by(const Vec_pi64& rhs) const {
        return sg_safediv_pi64(data_, rhs.data());
    }
    Vec_pi64 abs() const { return sg_abs_pi64(data_); }
    Vec_pi64 constrain(const Vec_pi64& lowerb, const Vec_pi64& upperb) const {
        return sg_constrain_pi64(lowerb.data(), upperb.data(), data_);
    }

    static Vec_pi64 min(const Vec_pi64& a, const Vec_pi64& b) {
        return sg_min_pi64(a.data(), b.data());
    }
    static Vec_pi64 max(const Vec_pi64& a, const Vec_pi64& b) {
        return sg_max_pi64(a.data(), b.data());
    }

    bool debug_eq(const int64_t l1, const int64_t l0) const {
        return sg_debug_eq_pi64(data_, l1, l0);
    }
    bool debug_eq(const int64_t l) const { return debug_eq(l, l); }

    inline Vec_pi32 bitcast_to_s32() const;
    Vec_pi64 bitcast_to_s64() const { return *this; }
    inline Vec_ps bitcast_to_f32() const;
    inline Vec_pd bitcast_to_f64() const;

    inline Vec_pi32 convert_to_s32() const;
    Vec_pi64 convert_to_s64() const { return *this; }
    inline Vec_ps convert_to_f32() const;
    inline Vec_pd convert_to_f64() const;

    static Vec_pi64 from(const Vec_pi32& pi32) {
        return sg_cvt_pi32_pi64(pi32.data());
    }
    static inline Vec_pi64 from(const Vec_pi64& pi64) { return pi64; }
};

class Vec_ps {
    sg_ps data_;
public:
    Vec_ps() : data_{sg_setzero_ps()} {}
    Vec_ps(const float f) : data_{sg_set1_ps(f)} {}
    Vec_ps(const float f3, const float f2, const float f1, const float f0)
        : data_{sg_set_ps(f3, f2, f1, f0)} {}
    Vec_ps(const sg_ps& ps) : data_{ps} {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Vec_ps(const sg_generic_ps& g_ps) : data_{sg_set_fromg_ps(g_ps)} {}
    #endif

    static Vec_ps minus_infinity() { return sg_minus_infinity_ps; }
    static Vec_ps infinity() { return sg_infinity_ps; }

    // Useful for writing templated code that might be of the form:
    //     var = vec_ps_arg * (0.312 * double_function_arg);
    // into:
    //     var = vec_ps_arg * (Vec_ps::scalar(0.312) *
    //         Vec_ps::scalar(double_function_arg));
    static float elem_t(const int32_t s32) { return static_cast<float>(s32); }
    static float elem_t(const int64_t s64) { return static_cast<float>(s64); }
    static float elem_t(const float f32) { return f32; }
    static float elem_t(const double f64) { return static_cast<float>(f64); }

    static Compare_ps compare_t(const Compare_pi32& cmp) {
        return sg_cvtcmp_pi32_ps(cmp.data());
    }
    static Compare_ps compare_t(const Compare_pi64& cmp) {
        return sg_cvtcmp_pi64_ps(cmp.data());
    }
    static Compare_ps compare_t(const Compare_ps& cmp) { return cmp; }
    static Compare_ps compare_t(const Compare_pd& cmp) {
        return sg_cvtcmp_pd_ps(cmp.data());
    }

    static Vec_ps bitcast_from_u32(const uint32_t i) {
        return sg_set1_from_u32_ps(i);
    }
    static Vec_ps bitcast_from_u32(const uint32_t i3, const uint32_t i2,
        const uint32_t i1, const uint32_t i0)
    {
        return sg_set_from_u32_ps(i3, i2, i1, i0);
    }

    sg_ps data() const { return data_; }
    sg_generic_ps generic() const { return sg_getg_ps(data_); }
    float f0() const { return sg_get0_ps(data_); }
    float f1() const { return sg_get1_ps(data_); }
    float f2() const { return sg_get2_ps(data_); }
    float f3() const { return sg_get3_ps(data_); }

    Vec_ps& operator+=(const Vec_ps& rhs) {
        data_ = sg_add_ps(data_, rhs.data());
        return *this;
    }
    friend Vec_ps operator+(Vec_ps lhs, const Vec_ps& rhs) {
        lhs += rhs;
        return lhs;
    }
    Vec_ps operator+() const { return *this; }

    Vec_ps& operator-=(const Vec_ps& rhs) {
        data_ = sg_sub_ps(data_, rhs.data());
        return *this;
    }
    friend Vec_ps operator-(Vec_ps lhs, const Vec_ps& rhs) {
        lhs -= rhs;
        return lhs;
    }
    Vec_ps operator-() const { return sg_neg_ps(data_); }

    Vec_ps& operator*=(const Vec_ps& rhs) {
        data_ = sg_mul_ps(data_, rhs.data());
        return *this;
    }
    friend Vec_ps operator*(Vec_ps lhs, const Vec_ps& rhs) {
        lhs *= rhs;
        return lhs;
    }

    Vec_ps& operator/=(const Vec_ps& rhs) {
        data_ = sg_div_ps(data_, rhs.data());
        return *this;
    }
    friend Vec_ps operator/(Vec_ps lhs, const Vec_ps& rhs) {
        lhs /= rhs;
        return lhs;
    }

    Vec_ps mul_add(const Vec_ps& mul, const Vec_ps& add) const {
        return sg_mul_add_ps(data_, mul.data(), add.data());
    }

    Vec_ps& operator&=(const Vec_ps& rhs) {
        data_ = sg_and_ps(data_, rhs.data());
        return *this;
    }
    friend Vec_ps operator&(Vec_ps lhs, const Vec_ps& rhs) {
        lhs &= rhs;
        return lhs;
    }

    Vec_ps& operator|=(const Vec_ps& rhs) {
        data_ = sg_or_ps(data_, rhs.data());
        return *this;
    }
    friend Vec_ps operator|(Vec_ps lhs, const Vec_ps& rhs) {
        lhs |= rhs;
        return lhs;
    }

    Vec_ps& operator^=(const Vec_ps& rhs) {
        data_ = sg_xor_ps(data_, rhs.data());
        return *this;
    }
    friend Vec_ps operator^(Vec_ps lhs, const Vec_ps& rhs) {
        lhs ^= rhs;
        return lhs;
    }

    Vec_ps operator~() const { return sg_not_ps(data_); }

    Compare_ps operator<(const Vec_ps& rhs) const {
        return sg_cmplt_ps(data_, rhs.data());
    }
    Compare_ps operator<=(const Vec_ps& rhs) const {
        return sg_cmplte_ps(data_, rhs.data());
    }
    Compare_ps operator==(const Vec_ps& rhs) const {
        return sg_cmpeq_ps(data_, rhs.data());
    }
    Compare_ps operator!=(const Vec_ps& rhs) const {
        return sg_cmpneq_ps(data_, rhs.data());
    }
    Compare_ps operator>=(const Vec_ps& rhs) const {
        return sg_cmpgte_ps(data_, rhs.data());
    }
    Compare_ps operator>(const Vec_ps& rhs) const {
        return sg_cmpgt_ps(data_, rhs.data());
    }

    template <int src3, int src2, int src1, int src0>
    Vec_ps shuffle() const {
        return sg_shuffle_ps(data_, src3, src2, src1, src0);
    }

    Vec_ps safe_divide_by(const Vec_ps& rhs) const {
        return sg_safediv_ps(data_, rhs.data());
    }
    Vec_ps abs() const { return sg_abs_ps(data_); }
    Vec_ps remove_signed_zero() const {
        return sg_remove_signed_zero_ps(data_);
    }
    Vec_ps constrain(const Vec_ps& lowerb, const Vec_ps& upperb) const {
        return sg_constrain_ps(lowerb.data(), upperb.data(), data_);
    }

    static Vec_ps min_fast(const Vec_ps& a, const Vec_ps& b) {
        return sg_min_fast_ps(a.data(), b.data());
    }
    static Vec_ps max_fast(const Vec_ps& a, const Vec_ps& b) {
        return sg_max_fast_ps(a.data(), b.data());
    }

    bool debug_eq(const float f3, const float f2, const float f1,
        const float f0) const
    {
        return sg_debug_eq_ps(data_, f3, f2, f1, f0);
    }
    bool debug_eq(const float f) const { return debug_eq(f, f, f, f); }

    inline Vec_pi32 bitcast_to_s32() const;
    inline Vec_pi64 bitcast_to_s64() const;
    Vec_ps bitcast_to_f32() const { return *this; }
    inline Vec_pd bitcast_to_f64() const;

    inline Vec_pi32 convert_to_nearest_s32() const;
    inline Vec_pi32 truncate_to_s32() const;
    inline Vec_pi32 floor_to_s32() const;
    inline Vec_pi64 convert_to_nearest_s64() const;
    inline Vec_pi64 truncate_to_s64() const;
    inline Vec_pi64 floor_to_s64() const;
    Vec_ps convert_to_f32() const { return *this; }
    inline Vec_pd convert_to_f64() const;

    static Vec_ps from(const Vec_pi32& pi32) {
        return sg_cvt_pi32_ps(pi32.data());
    }
    static Vec_ps from(const Vec_pi64& pi64) {
        return sg_cvt_pi64_ps(pi64.data());
    }
    static Vec_ps from(const Vec_ps& ps) { return ps; }
    static inline Vec_ps from(const Vec_pd& pd);

    // exponent_frexp() version is equivalent to C standard lib, and computes
    // exponent + 1 for some reason
    Vec_pi32 exponent_frexp() const {
        return sg_sub_pi32(sg_and_pi32(
            sg_srl_imm_pi32(sg_bitcast_ps_pi32(data_), 23),
                sg_set1_pi32(0xff)), sg_set1_pi32(126));
    }
    Vec_pi32 exponent_frexp_s32() const { return exponent_frexp(); }
    Vec_pi32 exponent() const {
        return sg_sub_pi32(sg_and_pi32(
            sg_srl_imm_pi32(sg_bitcast_ps_pi32(data_), 23),
                sg_set1_pi32(0xff)), sg_set1_pi32(127));
    }
    Vec_pi32 exponent_s32() const { return exponent(); }
    // mantissa_frexp() computes half the mantissa
    Vec_ps mantissa_frexp() const {
        return sg_bitcast_pi32_ps(sg_or_pi32(sg_and_pi32(
            sg_bitcast_ps_pi32(data_), sg_set1_pi32(0x807fffff)),
            sg_set1_pi32(0x3f000000)));
    }
    Vec_ps mantissa() const {
        return sg_bitcast_pi32_ps(sg_or_pi32(sg_and_pi32(
            sg_bitcast_ps_pi32(data_), sg_set1_pi32(0x807fffff)),
            sg_set1_pi32(0x3f800000)));
    }

    // Adds e to the existing exponent
    Vec_ps ldexp(const Vec_pi32& e) const {
        return sg_bitcast_pi32_ps(sg_add_pi32(sg_bitcast_ps_pi32(data_),
            sg_sl_imm_pi32(e.data(), 23)));
    }

    Vec_ps std_log() const {
        return Vec_ps { std::log(sg_get3_ps(data_)),
            std::log(sg_get2_ps(data_)), std::log(sg_get1_ps(data_)),
            std::log(sg_get0_ps(data_)) };
    }
    Vec_ps std_exp() const {
        return Vec_ps { std::exp(sg_get3_ps(data_)),
            std::exp(sg_get2_ps(data_)), std::exp(sg_get1_ps(data_)),
            std::exp(sg_get0_ps(data_)) };
    }
    Vec_ps std_sin() const {
        return Vec_ps { std::sin(sg_get3_ps(data_)),
            std::sin(sg_get2_ps(data_)), std::sin(sg_get1_ps(data_)),
            std::sin(sg_get0_ps(data_)) };
    }
    Vec_ps std_cos() const {
        return Vec_ps { std::cos(sg_get3_ps(data_)),
            std::cos(sg_get2_ps(data_)), std::cos(sg_get1_ps(data_)),
            std::cos(sg_get0_ps(data_)) };
    }
};

class Vec_pd {
    sg_pd data_;
public:
    Vec_pd() : data_{sg_setzero_pd()} {}
    Vec_pd(const double d) : data_{sg_set1_pd(d)} {}
    Vec_pd(const double d1, const double d0) : data_{sg_set_pd(d1, d0)} {}
    Vec_pd(const sg_pd& pd) : data_{pd} {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Vec_pd(const sg_generic_pd& g_pd) : data_{sg_set_fromg_pd(g_pd)} {}
    #endif

    static Vec_pd minus_infinity() { return sg_minus_infinity_pd; }
    static Vec_pd infinity() { return sg_infinity_pd; }

    static double elem_t(const int32_t s32) { return static_cast<double>(s32); }
    static double elem_t(const int64_t s64) { return static_cast<double>(s64); }
    static double elem_t(const float f32) { return static_cast<double>(f32); }
    static double elem_t(const double f64) { return f64; }

    static Compare_pd compare_t(const Compare_pi32& cmp) {
        return sg_cvtcmp_pi32_pd(cmp.data());
    }
    static Compare_pd compare_t(const Compare_pi64& cmp) {
        return sg_cvtcmp_pi64_pd(cmp.data());
    }
    static Compare_pd compare_t(const Compare_ps& cmp) {
        return sg_cvtcmp_ps_pd(cmp.data());
    }
    static Compare_pd compare_t(const Compare_pd& cmp) { return cmp; }

    static Vec_pd bitcast_from_u64(const uint64_t l) {
        return sg_set1_from_u64_pd(l);
    }
    static Vec_pd bitcast_from_u64(const uint64_t l1, const uint64_t l0) {
        return sg_set_from_u64_pd(l1, l0);
    }

    sg_pd data() const { return data_; }
    sg_generic_pd generic() const { return sg_getg_pd(data_); }
    double d0() const { return sg_get0_pd(data_); }
    double d1() const { return sg_get1_pd(data_); }

    Vec_pd& operator+=(const Vec_pd& rhs) {
        data_ = sg_add_pd(data_, rhs.data());
        return *this;
    }
    friend Vec_pd operator+(Vec_pd lhs, const Vec_pd& rhs) {
        lhs += rhs;
        return lhs;
    }
    Vec_pd operator+() const { return *this; }

    Vec_pd& operator-=(const Vec_pd& rhs) {
        data_ = sg_sub_pd(data_, rhs.data());
        return *this;
    }
    friend Vec_pd operator-(Vec_pd lhs, const Vec_pd& rhs) {
        lhs -= rhs;
        return lhs;
    }
    Vec_pd operator-() const { return sg_neg_pd(data_); }

    Vec_pd& operator*=(const Vec_pd& rhs) {
        data_ = sg_mul_pd(data_, rhs.data());
        return *this;
    }
    friend Vec_pd operator*(Vec_pd lhs, const Vec_pd& rhs) {
        lhs *= rhs;
        return lhs;
    }

    Vec_pd& operator/=(const Vec_pd& rhs) {
        data_ = sg_div_pd(data_, rhs.data());
        return *this;
    }
    friend Vec_pd operator/(Vec_pd lhs, const Vec_pd& rhs) {
        lhs /= rhs;
        return lhs;
    }

    Vec_pd mul_add(const Vec_pd& mul, const Vec_pd& add) const {
        return sg_mul_add_pd(data_, mul.data(), add.data());
    }

    Vec_pd& operator&=(const Vec_pd& rhs) {
        data_ = sg_and_pd(data_, rhs.data());
        return *this;
    }
    friend Vec_pd operator&(Vec_pd lhs, const Vec_pd& rhs) {
        lhs &= rhs;
        return lhs;
    }

    Vec_pd& operator|=(const Vec_pd& rhs) {
        data_ = sg_or_pd(data_, rhs.data());
        return *this;
    }
    friend Vec_pd operator|(Vec_pd lhs, const Vec_pd& rhs) {
        lhs |= rhs;
        return lhs;
    }

    Vec_pd& operator^=(const Vec_pd& rhs) {
        data_ = sg_xor_pd(data_, rhs.data());
        return *this;
    }
    friend Vec_pd operator^(Vec_pd lhs, const Vec_pd& rhs) {
        lhs ^= rhs;
        return lhs;
    }

    Vec_pd operator~() const { return sg_not_pd(data_); }

    Compare_pd operator<(const Vec_pd& rhs) const {
        return sg_cmplt_pd(data_, rhs.data());
    }
    Compare_pd operator<=(const Vec_pd& rhs) const {
        return sg_cmplte_pd(data_, rhs.data());
    }
    Compare_pd operator==(const Vec_pd& rhs) const {
        return sg_cmpeq_pd(data_, rhs.data());
    }
    Compare_pd operator!=(const Vec_pd& rhs) const {
        return sg_cmpneq_pd(data_, rhs.data());
    }
    Compare_pd operator>=(const Vec_pd& rhs) const {
        return sg_cmpgte_pd(data_, rhs.data());
    }
    Compare_pd operator>(const Vec_pd& rhs) const {
        return sg_cmpgt_pd(data_, rhs.data());
    }

    template <int src1, int src0>
    Vec_pd shuffle() const { return sg_shuffle_pd(data_, src1, src0); }

    Vec_pd safe_divide_by(const Vec_pd& rhs) const {
        return sg_safediv_pd(data_, rhs.data());
    }
    Vec_pd abs() const { return sg_abs_pd(data_); }
    Vec_pd remove_signed_zero() const {
        return sg_remove_signed_zero_pd(data_);
    }
    Vec_pd constrain(const Vec_pd& lowerb, const Vec_pd& upperb) const {
        return sg_constrain_pd(lowerb.data(), upperb.data(), data_);
    }

    static Vec_pd min_fast(const Vec_pd& a, const Vec_pd& b) {
        return sg_min_fast_pd(a.data(), b.data());
    }
    static Vec_pd max_fast(const Vec_pd& a, const Vec_pd& b) {
        return sg_max_fast_pd(a.data(), b.data());
    }

    bool debug_eq(const double d1, const double d0) const {
        return sg_debug_eq_pd(data_, d1, d0);
    }
    bool debug_eq(const double d) const { return debug_eq(d, d); }

    inline Vec_pi32 bitcast_to_s32() const;
    inline Vec_pi64 bitcast_to_s64() const;
    inline Vec_ps bitcast_to_f32() const;
    Vec_pd bitcast_to_f64() const { return *this; }

    inline Vec_pi32 convert_to_nearest_s32() const;
    inline Vec_pi32 truncate_to_s32() const;
    inline Vec_pi32 floor_to_s32() const;
    inline Vec_pi64 convert_to_nearest_s64() const;
    inline Vec_pi64 truncate_to_s64() const;
    inline Vec_pi64 floor_to_s64() const;
    inline Vec_ps convert_to_f32() const;
    Vec_pd convert_to_f64() const { return *this; }

    static Vec_pd from(const Vec_pi32& pi32) {
        return sg_cvt_pi32_pd(pi32.data());
    }
    static Vec_pd from(const Vec_pi64& pi64) {
        return sg_cvt_pi64_pd(pi64.data());
    }
    static Vec_pd from(const Vec_ps& ps) {
        return sg_cvt_ps_pd(ps.data());
    }
    static Vec_pd from(const Vec_pd& pd) { return pd; }

    Vec_pi64 exponent_frexp() const {
        return sg_sub_pi64(sg_and_pi64(
            sg_srl_imm_pi64(sg_bitcast_pd_pi64(data_), 52),
                sg_set1_pi64(0x7ff)), sg_set1_pi64(1022));
    }
    Vec_pi32 exponent_frexp_s32() const {
        return exponent_frexp().convert_to_s32();
    }
    Vec_pi64 exponent() const {
        return sg_sub_pi64(sg_and_pi64(
            sg_srl_imm_pi64(sg_bitcast_pd_pi64(data_), 52),
                sg_set1_pi64(0x7ff)), sg_set1_pi64(1023));
    }
    Vec_pi32 exponent_s32() const { return exponent().convert_to_s32(); }
    Vec_pd mantissa_frexp() const {
        return sg_bitcast_pi64_pd(sg_or_pi64(sg_and_pi64(
            sg_bitcast_pd_pi64(data_), sg_set1_pi64(0x800fffffffffffff)),
            sg_set1_pi64(0x3fe0000000000000)));
    }
    Vec_pd mantissa() const {
        return sg_bitcast_pi64_pd(sg_or_pi64(sg_and_pi64(
            sg_bitcast_pd_pi64(data_), sg_set1_pi64(0x800fffffffffffff)),
            sg_set1_pi64(0x3ff0000000000000)));
    }

    Vec_pd ldexp(const Vec_pi64& e) const {
        return sg_bitcast_pi64_pd(sg_add_pi64(sg_bitcast_pd_pi64(data_),
            sg_sl_imm_pi64(e.data(), 52)));
    }
    Vec_pd ldexp(const Vec_pi32& e) const { return ldexp(e.convert_to_s64()); }

    Vec_pd std_log() const {
        return Vec_pd { std::log(sg_get1_pd(data_)),
            std::log(sg_get0_pd(data_)) };
    }
    Vec_pd std_exp() const {
        return Vec_pd { std::exp(sg_get1_pd(data_)),
            std::exp(sg_get0_pd(data_)) };
    }
    Vec_pd std_sin() const {
        return Vec_pd { std::sin(sg_get1_pd(data_)),
            std::sin(sg_get0_pd(data_)) };
    }
    Vec_pd std_cos() const {
        return Vec_pd { std::cos(sg_get1_pd(data_)),
            std::cos(sg_get0_pd(data_)) };
    }
};

inline Compare_pi64 Compare_pi32::convert_to_cmp_s64() const {
    return sg_cvtcmp_pi32_pi64(data_);
}
inline Compare_ps Compare_pi32::convert_to_cmp_f32() const {
    return sg_cvtcmp_pi32_ps(data_);
}
inline Compare_pd Compare_pi32::convert_to_cmp_f64() const {
    return sg_cvtcmp_pi32_pd(data_);
}

inline Vec_pi32 Compare_pi32::choose_else_zero(const Vec_pi32& if_true) const {
    return sg_choose_else_zero_pi32(data_, if_true.data());
}
inline Vec_pi32 Compare_pi32::choose(const Vec_pi32& if_true,
    const Vec_pi32& if_false) const
{
    return sg_choose_pi32(data_, if_true.data(), if_false.data());
}

inline Compare_pi32 Compare_pi64::convert_to_cmp_s32() const {
    return sg_cvtcmp_pi64_pi32(data_);
}
inline Compare_ps Compare_pi64::convert_to_cmp_f32() const {
    return sg_cvtcmp_pi64_ps(data_);
}
inline Compare_pd Compare_pi64::convert_to_cmp_f64() const {
    return sg_cvtcmp_pi64_pd(data_);
}

inline Vec_pi64 Compare_pi64::choose_else_zero(const Vec_pi64& if_true) const {
    return sg_choose_else_zero_pi64(data_, if_true.data());
}
inline Vec_pi64 Compare_pi64::choose(const Vec_pi64& if_true,
    const Vec_pi64& if_false) const
{
    return sg_choose_pi64(data_, if_true.data(), if_false.data());
}

inline Compare_pi32 Compare_ps::convert_to_cmp_s32() const {
    return sg_cvtcmp_ps_pi32(data_);
}
inline Compare_pi64 Compare_ps::convert_to_cmp_s64() const {
    return sg_cvtcmp_ps_pi64(data_);
}
inline Compare_pd Compare_ps::convert_to_cmp_f64() const {
    return sg_cvtcmp_ps_pd(data_);
}

inline Vec_ps Compare_ps::choose_else_zero(const Vec_ps& if_true) const {
    return sg_choose_else_zero_ps(data_, if_true.data());
}
inline Vec_ps Compare_ps::choose(const Vec_ps& if_true,
    const Vec_ps& if_false) const
{
    return sg_choose_ps(data_, if_true.data(), if_false.data());
}

inline Compare_pi32 Compare_pd::convert_to_cmp_s32() const {
    return sg_cvtcmp_pd_pi32(data_);
}
inline Compare_pi64 Compare_pd::convert_to_cmp_s64() const {
    return sg_cvtcmp_pd_pi64(data_);
}
inline Compare_ps Compare_pd::convert_to_cmp_f32() const {
    return sg_cvtcmp_pd_ps(data_);
}

inline Vec_pd Compare_pd::choose_else_zero(const Vec_pd& if_true) const {
    return sg_choose_else_zero_pd(data_, if_true.data());
}
inline Vec_pd Compare_pd::choose(const Vec_pd& if_true,
    const Vec_pd& if_false) const
{
    return sg_choose_pd(data_, if_true.data(), if_false.data());
}

inline Vec_pi64 Vec_pi32::bitcast_to_s64() const {
    return sg_bitcast_pi32_pi64(data_);
}
inline Vec_ps Vec_pi32::bitcast_to_f32() const {
    return sg_bitcast_pi32_ps(data_);
}
inline Vec_pd Vec_pi32::bitcast_to_f64() const {
    return sg_bitcast_pi32_pd(data_);
}
inline Vec_pi64 Vec_pi32::convert_to_s64() const {
    return sg_cvt_pi32_pi64(data_);
}
inline Vec_ps Vec_pi32::convert_to_f32() const {
    return sg_cvt_pi32_ps(data_);
}
inline Vec_pd Vec_pi32::convert_to_f64() const {
    return sg_cvt_pi32_pd(data_);
}

inline Vec_pi32 Vec_pi64::bitcast_to_s32() const {
    return sg_bitcast_pi64_pi32(data_);
}
inline Vec_ps Vec_pi64::bitcast_to_f32() const {
    return sg_bitcast_pi64_ps(data_);
}
inline Vec_pd Vec_pi64::bitcast_to_f64() const {
    return sg_bitcast_pi64_pd(data_);
}
inline Vec_pi32 Vec_pi64::convert_to_s32() const {
    return sg_cvt_pi64_pi32(data_);
}
inline Vec_ps Vec_pi64::convert_to_f32() const {
    return sg_cvt_pi64_ps(data_);
}
inline Vec_pd Vec_pi64::convert_to_f64() const {
    return sg_cvt_pi64_pd(data_);
}

inline Vec_pi32 Vec_ps::bitcast_to_s32() const {
    return sg_bitcast_ps_pi32(data_);
}
inline Vec_pi64 Vec_ps::bitcast_to_s64() const {
    return sg_bitcast_ps_pi64(data_);
}
inline Vec_pd Vec_ps::bitcast_to_f64() const {
    return sg_bitcast_ps_pd(data_);
}
inline Vec_pi32 Vec_ps::convert_to_nearest_s32() const {
    return sg_cvt_ps_pi32(data_);
}
inline Vec_pi32 Vec_ps::truncate_to_s32() const {
    return sg_cvtt_ps_pi32(data_);
}
inline Vec_pi32 Vec_ps::floor_to_s32() const {
    return sg_cvtf_ps_pi32(data_);
}
inline Vec_pi64 Vec_ps::convert_to_nearest_s64() const {
    return sg_cvt_ps_pi64(data_);
}
inline Vec_pi64 Vec_ps::truncate_to_s64() const {
    return sg_cvtt_ps_pi64(data_);
}
inline Vec_pi64 Vec_ps::floor_to_s64() const {
    return sg_cvtf_ps_pi64(data_);
}
inline Vec_pd Vec_ps::convert_to_f64() const {
    return sg_cvt_ps_pd(data_);
}

inline Vec_pi32 Vec_pd::bitcast_to_s32() const {
    return sg_bitcast_pd_pi32(data_);
}
inline Vec_pi64 Vec_pd::bitcast_to_s64() const {
    return sg_bitcast_pd_pi64(data_);
}
inline Vec_ps Vec_pd::bitcast_to_f32() const {
    return sg_bitcast_pd_ps(data_);
}
inline Vec_pi32 Vec_pd::convert_to_nearest_s32() const {
    return sg_cvt_pd_pi32(data_);
}
inline Vec_pi32 Vec_pd::truncate_to_s32() const {
    return sg_cvtt_pd_pi32(data_);
}
inline Vec_pi32 Vec_pd::floor_to_s32() const {
    return sg_cvtf_pd_pi32(data_);
}
inline Vec_pi64 Vec_pd::convert_to_nearest_s64() const {
    return sg_cvt_pd_pi64(data_);
}
inline Vec_pi64 Vec_pd::truncate_to_s64() const {
    return sg_cvtt_pd_pi64(data_);
}
inline Vec_pi64 Vec_pd::floor_to_s64() const {
    return sg_cvtf_pd_pi64(data_);
}
inline Vec_ps Vec_pd::convert_to_f32() const {
    return sg_cvt_pd_ps(data_);
}

inline Vec_pi32 Vec_pi32::from(const Vec_pi64& pi64) {
    return sg_cvt_pi64_pi32(pi64.data());
}

inline Vec_ps Vec_ps::from(const Vec_pd& pd) {
    return sg_cvt_pd_ps(pd.data());
}

// Shim types - for using double / float etc in template code that expects
// a vector type

class Vec_s32x1; class Vec_s64x1; class Vec_f32x1; class Vec_f64x1;
class Compare_s32x1; class Compare_s64x1;
class Compare_f32x1; class Compare_f64x1;

class Compare_s32x1 {
private:
    bool data_;
public:
    Compare_s32x1() : data_{false} {}
    Compare_s32x1(const bool b) : data_{b} {}

    bool data() const { return data_; }
    bool debug_valid_eq(const bool b) { return data_ == b; }

    Compare_s32x1 operator&&(const Compare_s32x1& rhs) const {
        return data_ && rhs.data(); }
    Compare_s32x1 operator||(const Compare_s32x1& rhs) const {
        return data_ || rhs.data(); }
    Compare_s32x1 operator!() const { return !data_; }

    Compare_s32x1 convert_to_cmp_s32() const { return *this; }
    Compare_s64x1 convert_to_cmp_s64() const;
    Compare_f32x1 convert_to_cmp_f32() const;
    Compare_f64x1 convert_to_cmp_f64() const;

    Vec_s32x1 choose_else_zero(const Vec_s32x1& if_true) const;
    Vec_s32x1 choose(const Vec_s32x1& if_true, const Vec_s32x1& if_false) const;
};

inline Compare_s32x1 operator==(const Compare_s32x1& lhs,
    const Compare_s32x1& rhs)
{
    return lhs.data() == rhs.data();
}
inline Compare_s32x1 operator!=(const Compare_s32x1& lhs,
    const Compare_s32x1& rhs)
{
    return lhs.data() != rhs.data();
}

class Compare_s64x1 {
private:
    bool data_;
public:
    Compare_s64x1() : data_{false} {}
    Compare_s64x1(const bool b) : data_{b} {}

    bool data() const { return data_; }
    bool debug_valid_eq(const bool b) { return data_ == b; }

    Compare_s64x1 operator&&(const Compare_s64x1& rhs) const {
        return data_ && rhs.data(); }
    Compare_s64x1 operator||(const Compare_s64x1& rhs) const {
        return data_ || rhs.data(); }
    Compare_s64x1 operator!() const { return !data_; }

    Compare_s32x1 convert_to_cmp_s32() const { return data_; }
    Compare_s64x1 convert_to_cmp_s64() const { return *this; }
    Compare_f32x1 convert_to_cmp_f32() const;
    Compare_f64x1 convert_to_cmp_f64() const;

    Vec_s64x1 choose_else_zero(const Vec_s64x1& if_true) const;
    Vec_s64x1 choose(const Vec_s64x1& if_true, const Vec_s64x1& if_false) const;
};

inline Compare_s64x1 operator==(const Compare_s64x1& lhs,
    const Compare_s64x1& rhs)
{
    return lhs.data() == rhs.data();
}
inline Compare_s64x1 operator!=(const Compare_s64x1& lhs,
    const Compare_s64x1& rhs)
{
    return lhs.data() != rhs.data();
}

class Compare_f32x1 {
private:
    bool data_;
public:
    Compare_f32x1() : data_{false} {}
    Compare_f32x1(const bool b) : data_{b} {}

    bool data() const { return data_; }
    bool debug_valid_eq(const bool b) { return data_ == b; }

    Compare_f32x1 operator&&(const Compare_f32x1& rhs) const {
        return data_ && rhs.data(); }
    Compare_f32x1 operator||(const Compare_f32x1& rhs) const {
        return data_ || rhs.data(); }
    Compare_f32x1 operator!() const { return !data_; }

    Compare_s32x1 convert_to_cmp_s32() const { return data_; }
    Compare_s64x1 convert_to_cmp_s64() const { return data_; }
    Compare_f32x1 convert_to_cmp_f32() const { return *this; }
    Compare_f64x1 convert_to_cmp_f64() const;

    Vec_f32x1 choose_else_zero(const Vec_f32x1& if_true) const;
    Vec_f32x1 choose(const Vec_f32x1& if_true, const Vec_f32x1& if_false) const;
};

inline Compare_f32x1 operator==(const Compare_f32x1& lhs,
    const Compare_f32x1& rhs)
{
    return lhs.data() == rhs.data();
}
inline Compare_f32x1 operator!=(const Compare_f32x1& lhs,
    const Compare_f32x1& rhs)
{
    return lhs.data() != rhs.data();
}

class Compare_f64x1 {
private:
    bool data_;
public:
    Compare_f64x1() : data_{false} {}
    Compare_f64x1(const bool b) : data_{b} {}

    bool data() const { return data_; }
    bool debug_valid_eq(const bool b) { return data_ == b; }

    Compare_f64x1 operator&&(const Compare_f64x1& rhs) const {
        return data_ && rhs.data(); }
    Compare_f64x1 operator||(const Compare_f64x1& rhs) const {
        return data_ || rhs.data(); }
    Compare_f64x1 operator!() const { return !data_; }

    Compare_s32x1 convert_to_cmp_s32() const { return data_; }
    Compare_s64x1 convert_to_cmp_s64() const { return data_; }
    Compare_f32x1 convert_to_cmp_f32() const { return data_; }
    Compare_f64x1 convert_to_cmp_f64() const { return *this; }

    Vec_f64x1 choose_else_zero(const Vec_f64x1& if_true) const;
    Vec_f64x1 choose(const Vec_f64x1& if_true, const Vec_f64x1& if_false) const;
};

inline Compare_f64x1 operator==(const Compare_f64x1& lhs,
    const Compare_f64x1& rhs)
{
    return lhs.data() == rhs.data();
}
inline Compare_f64x1 operator!=(const Compare_f64x1& lhs,
    const Compare_f64x1& rhs)
{
    return lhs.data() != rhs.data();
}

typedef Compare_f32x1 Compare_ss;
typedef Compare_f64x1 Compare_sd;

class Vec_s32x1 {
    int32_t data_;
public:
    Vec_s32x1() : data_{0} {}
    Vec_s32x1(const int32_t s32) : data_{s32} {}

    static int32_t elem_t(const int32_t s32) { return s32; }
    static int32_t elem_t(const int64_t s64) {
        return static_cast<int32_t>(s64);
    }
    static int32_t elem_t(const float f32) {
        return static_cast<int32_t>(f32);
    }
    static int32_t elem_t(const double f64) {
        return static_cast<int32_t>(f64);
    }

    static Compare_s32x1 compare_t(const Compare_s32x1& cmp) { return cmp; }
    static Compare_s32x1 compare_t(const Compare_s64x1& cmp) {
        return cmp.data();
    }
    static Compare_s32x1 compare_t(const Compare_f32x1& cmp) {
        return cmp.data();
    }
    static Compare_s32x1 compare_t(const Compare_f64x1& cmp) {
        return cmp.data();
    }

    static Vec_s32x1 bitcast_from_u32(const uint32_t i) {
        return sg_bitcast_u32x1_s32x1(i);
    }

    int32_t data() const { return data_; }
    explicit operator int32_t() const { return data_; }

    Vec_s32x1& operator++() {
        ++data_;
        return *this;
    }
    Vec_s32x1 operator++(int) {
        Vec_s32x1 old = *this;
        operator++();
        return old;
    }

    Vec_s32x1& operator--() {
        --data_;
        return *this;
    }
    Vec_s32x1 operator--(int) {
        Vec_s32x1 old = *this;
        operator--();
        return old;
    }

    Vec_s32x1& operator+=(const Vec_s32x1& rhs) {
        data_ += rhs.data();
        return *this;
    }
    friend Vec_s32x1 operator+(Vec_s32x1 lhs, const Vec_s32x1& rhs) {
        lhs += rhs;
        return lhs;
    }
    Vec_s32x1 operator+() const { return *this; }

    Vec_s32x1& operator-=(const Vec_s32x1& rhs) {
        data_ -= rhs.data();
        return *this;
    }
    friend Vec_s32x1 operator-(Vec_s32x1 lhs, const Vec_s32x1& rhs) {
        lhs -= rhs;
        return lhs;
    }
    Vec_s32x1 operator-() const { return -data_; }

    Vec_s32x1& operator*=(const Vec_s32x1& rhs) {
        data_ *= rhs.data();
        return *this;
    }
    friend Vec_s32x1 operator*(Vec_s32x1 lhs, const Vec_s32x1& rhs) {
        lhs *= rhs;
        return lhs;
    }

    Vec_s32x1& operator/=(const Vec_s32x1& rhs) {
        data_ /= rhs.data();
        return *this;
    }
    friend Vec_s32x1 operator/(Vec_s32x1 lhs, const Vec_s32x1& rhs) {
        lhs /= rhs;
        return lhs;
    }

    Vec_s32x1& operator&=(const Vec_s32x1& rhs) {
        data_ &= rhs.data();
        return *this;
    }
    friend Vec_s32x1 operator&(Vec_s32x1 lhs, const Vec_s32x1& rhs) {
        lhs &= rhs.data();
        return lhs;
    }

    Vec_s32x1& operator|=(const Vec_s32x1& rhs) {
        data_ |= rhs.data();
        return *this;
    }
    friend Vec_s32x1 operator|(Vec_s32x1 lhs, const Vec_s32x1& rhs) {
        lhs |= rhs.data();
        return lhs;
    }

    Vec_s32x1& operator^=(const Vec_s32x1& rhs) {
        data_ ^= rhs.data();
        return *this;
    }
    friend Vec_s32x1 operator^(Vec_s32x1 lhs, const Vec_s32x1& rhs) {
        lhs ^= rhs.data();
        return lhs;
    }

    Vec_s32x1 operator~() const { return ~data_; }

    Compare_s32x1 operator<(const Vec_s32x1& rhs) const {
        return data_ < rhs.data();
    }
    Compare_s32x1 operator<=(const Vec_s32x1& rhs) const {
        return data_ <= rhs.data();
    }
    Compare_s32x1 operator==(const Vec_s32x1& rhs) const {
        return data_ == rhs.data();
    }
    Compare_s32x1 operator>=(const Vec_s32x1& rhs) const {
        return data_ >= rhs.data();
    }
    Compare_s32x1 operator>(const Vec_s32x1& rhs) const {
        return data_ > rhs.data();
    }

    template<int shift>
    Vec_s32x1 shift_l_imm() const { return data_ << shift; }
    template<int shift>
    Vec_s32x1 shift_rl_imm() const { return sg_srl_s32x1(data_, shift); }
    template<int shift>
    Vec_s32x1 shift_ra_imm() const { return data_ >> shift; }

    Vec_s32x1 safe_divide_by(const Vec_s32x1& rhs) const {
        return rhs.data() == 0 ? data_ : data_ / rhs.data();
    }
    Vec_s32x1 abs() const { return std::abs(data_); }
    static Vec_s32x1 min(const Vec_s32x1& a, const Vec_s32x1& b) {
        return std::min(a.data(), b.data());
    }
    static Vec_s32x1 max(const Vec_s32x1& a, const Vec_s32x1& b) {
        return std::max(a.data(), b.data());
    }
    Vec_s32x1 constrain(const Vec_s32x1& lowerb, const Vec_s32x1& upperb) const {
        return Vec_s32x1::min(Vec_s32x1::max(lowerb, data_), upperb);
    }

    bool debug_eq(int32_t i) const { return data_ == i; }

    Vec_s32x1 bitcast_to_s32() const { return *this; }
    inline Vec_s64x1 bitcast_to_s64() const;
    inline Vec_f32x1 bitcast_to_f32() const;
    inline Vec_f64x1 bitcast_to_f64() const;

    Vec_s32x1 convert_to_s32() const { return *this; }
    inline Vec_s64x1 convert_to_s64() const;
    inline Vec_f32x1 convert_to_f32() const;
    inline Vec_f64x1 convert_to_f64() const;

    static Vec_s32x1 from(const Vec_s32x1& s32) { return s32; }
    static inline Vec_s32x1 from(const Vec_s64x1& s64);
};

class Vec_s64x1 {
    int64_t data_;
public:
    Vec_s64x1() : data_{0} {}
    Vec_s64x1(const int64_t s64) : data_{s64} {}

    static int64_t elem_t(const int32_t s32) {
        return static_cast<int64_t>(s32);
    }
    static int64_t elem_t(const int64_t s64) { return s64; }
    static int64_t elem_t(const float f32) {
        return static_cast<int64_t>(f32);
    }
    static int64_t elem_t(const double f64) {
        return static_cast<int64_t>(f64);
    }

    static Compare_s64x1 compare_t(const Compare_s32x1& cmp) {
        return cmp.data();
    }
    static Compare_s64x1 compare_t(const Compare_s64x1& cmp) { return cmp; }
    static Compare_s64x1 compare_t(const Compare_f32x1& cmp) {
        return cmp.data();
    }
    static Compare_s64x1 compare_t(const Compare_f64x1& cmp) {
        return cmp.data();
    }

    static Vec_s64x1 bitcast_from_u64(const uint64_t i) {
        return sg_bitcast_u64x1_s64x1(i);
    }

    int64_t data() const { return data_; }
    explicit operator int64_t() const { return data_; }

    Vec_s64x1& operator++() {
        ++data_;
        return *this;
    }
    Vec_s64x1 operator++(int) {
        Vec_s64x1 old = *this;
        operator++();
        return old;
    }

    Vec_s64x1& operator--() {
        --data_;
        return *this;
    }
    Vec_s64x1 operator--(int) {
        Vec_s64x1 old = *this;
        operator--();
        return old;
    }

    Vec_s64x1& operator+=(const Vec_s64x1& rhs) {
        data_ += rhs.data();
        return *this;
    }
    friend Vec_s64x1 operator+(Vec_s64x1 lhs, const Vec_s64x1& rhs) {
        lhs += rhs;
        return lhs;
    }
    Vec_s64x1 operator+() const { return *this; }

    Vec_s64x1& operator-=(const Vec_s64x1& rhs) {
        data_ -= rhs.data();
        return *this;
    }
    friend Vec_s64x1 operator-(Vec_s64x1 lhs, const Vec_s64x1& rhs) {
        lhs -= rhs;
        return lhs;
    }
    Vec_s64x1 operator-() const { return -data_; }

    Vec_s64x1& operator*=(const Vec_s64x1& rhs) {
        data_ *= rhs.data();
        return *this;
    }
    friend Vec_s64x1 operator*(Vec_s64x1 lhs, const Vec_s64x1& rhs) {
        lhs *= rhs;
        return lhs;
    }

    Vec_s64x1& operator/=(const Vec_s64x1& rhs) {
        data_ /= rhs.data();
        return *this;
    }
    friend Vec_s64x1 operator/(Vec_s64x1 lhs, const Vec_s64x1& rhs) {
        lhs /= rhs;
        return lhs;
    }

    Vec_s64x1& operator&=(const Vec_s64x1& rhs) {
        data_ &= rhs.data();
        return *this;
    }
    friend Vec_s64x1 operator&(Vec_s64x1 lhs, const Vec_s64x1& rhs) {
        lhs &= rhs.data();
        return lhs;
    }

    Vec_s64x1& operator|=(const Vec_s64x1& rhs) {
        data_ |= rhs.data();
        return *this;
    }
    friend Vec_s64x1 operator|(Vec_s64x1 lhs, const Vec_s64x1& rhs) {
        lhs |= rhs.data();
        return lhs;
    }

    Vec_s64x1& operator^=(const Vec_s64x1& rhs) {
        data_ ^= rhs.data();
        return *this;
    }
    friend Vec_s64x1 operator^(Vec_s64x1 lhs, const Vec_s64x1& rhs) {
        lhs ^= rhs.data();
        return lhs;
    }

    Vec_s64x1 operator~() const { return ~data_; }

    Compare_s64x1 operator<(const Vec_s64x1& rhs) const {
        return data_ < rhs.data();
    }
    Compare_s64x1 operator<=(const Vec_s64x1& rhs) const {
        return data_ <= rhs.data();
    }
    Compare_s64x1 operator==(const Vec_s64x1& rhs) const {
        return data_ == rhs.data();
    }
    Compare_s64x1 operator>=(const Vec_s64x1& rhs) const {
        return data_ >= rhs.data();
    }
    Compare_s64x1 operator>(const Vec_s64x1& rhs) const {
        return data_ > rhs.data();
    }

    template<int shift>
    Vec_s64x1 shift_l_imm() const { return data_ << shift; }
    template<int shift>
    Vec_s64x1 shift_rl_imm() const { return sg_srl_s64x1(data_, shift); }
    template<int shift>
    Vec_s64x1 shift_ra_imm() const { return data_ >> shift; }

    Vec_s64x1 safe_divide_by(const Vec_s64x1& rhs) const {
        return rhs.data() == 0 ? data_ : data_ / rhs.data();
    }
    Vec_s64x1 abs() const { return std::abs(data_); }
    static Vec_s64x1 min(const Vec_s64x1& a, const Vec_s64x1& b) {
        return std::min(a.data(), b.data());
    }
    static Vec_s64x1 max(const Vec_s64x1& a, const Vec_s64x1& b) {
        return std::max(a.data(), b.data());
    }
    Vec_s64x1 constrain(const Vec_s64x1& lowerb, const Vec_s64x1& upperb) const {
        return Vec_s64x1::min(Vec_s64x1::max(lowerb, data_), upperb);
    }

    bool debug_eq(int64_t i) const { return data_ == i; }

    Vec_s32x1 bitcast_to_s32() const {
        return sg_bitcast_u32x1_s32x1(
            static_cast<uint32_t>(data_ & 0xffffffff));
    }
    Vec_s64x1 bitcast_to_s64() const { return *this; }
    inline Vec_f32x1 bitcast_to_f32() const;
    inline Vec_f64x1 bitcast_to_f64() const;

    Vec_s32x1 convert_to_s32() const {
        return static_cast<int32_t>(data_);
    }
    Vec_s64x1 convert_to_s64() const { return *this; }
    inline Vec_f32x1 convert_to_f32() const;
    inline Vec_f64x1 convert_to_f64() const;

    static Vec_s64x1 from(const Vec_s32x1& s32) {
        return static_cast<int64_t>(s32.data());
    }
    static Vec_s64x1 from(const Vec_s64x1& s64) { return s64; }
};

class Vec_f32x1 {
    float data_;
public:
    Vec_f32x1() : data_{0.0f} {}
    Vec_f32x1(const float f32) : data_{f32} {}

    static Vec_f32x1 minus_infinity() { return sg_minus_infinity_f32x1; }
    static Vec_f32x1 infinity() { return sg_infinity_f32x1; }

    static float elem_t(const int32_t s32) { return static_cast<float>(s32); }
    static float elem_t(const int64_t s64) { return static_cast<float>(s64); }
    static float elem_t(const float f32) { return f32; }
    static float elem_t(const double f64) { return static_cast<float>(f64); }

    static Compare_f32x1 compare_t(const Compare_s32x1& cmp) {
        return cmp.data();
    }
    static Compare_f32x1 compare_t(const Compare_s64x1& cmp) {
        return cmp.data();
    }
    static Compare_f32x1 compare_t(const Compare_f32x1& cmp) { return cmp; }
    static Compare_f32x1 compare_t(const Compare_f64x1& cmp) {
        return cmp.data();
    }

    static Vec_f32x1 bitcast_from_u32(const uint32_t i) {
        return sg_bitcast_u32x1_f32x1(i);
    }

    float data() const { return data_; }
    explicit operator float() const { return data_; }

    Vec_f32x1& operator+=(const Vec_f32x1& rhs) {
        data_ += rhs.data();
        return *this;
    }
    friend Vec_f32x1 operator+(Vec_f32x1 lhs, const Vec_f32x1& rhs) {
        lhs += rhs;
        return lhs;
    }
    Vec_f32x1 operator+() const { return *this; }

    Vec_f32x1& operator-=(const Vec_f32x1& rhs) {
        data_ -= rhs.data();
        return *this;
    }
    friend Vec_f32x1 operator-(Vec_f32x1 lhs, const Vec_f32x1& rhs) {
        lhs -= rhs;
        return lhs;
    }
    Vec_f32x1 operator-() const { return -data_; }

    Vec_f32x1& operator*=(const Vec_f32x1& rhs) {
        data_ *= rhs.data();
        return *this;
    }
    friend Vec_f32x1 operator*(Vec_f32x1 lhs, const Vec_f32x1& rhs) {
        lhs *= rhs;
        return lhs;
    }

    Vec_f32x1& operator/=(const Vec_f32x1& rhs) {
        data_ /= rhs.data();
        return *this;
    }
    friend Vec_f32x1 operator/(Vec_f32x1 lhs, const Vec_f32x1& rhs) {
        lhs /= rhs;
        return lhs;
    }

    Vec_f32x1 mul_add(const Vec_f32x1& mul, const Vec_f32x1& add) const {
        return sg_mul_add_f32x1(data_, mul.data(), add.data());
    }

    Vec_f32x1& operator&=(const Vec_f32x1& rhs) {
        data_ = sg_bitcast_u32x1_f32x1(
            sg_bitcast_f32x1_u32x1(data_) & sg_bitcast_f32x1_u32x1(rhs.data()));
        return *this;
    }
    friend Vec_f32x1 operator&(Vec_f32x1 lhs, const Vec_f32x1& rhs) {
        lhs &= rhs.data();
        return lhs;
    }

    Vec_f32x1& operator|=(const Vec_f32x1& rhs) {
        data_ = sg_bitcast_u32x1_f32x1(
            sg_bitcast_f32x1_u32x1(data_) | sg_bitcast_f32x1_u32x1(rhs.data()));
        return *this;
    }
    friend Vec_f32x1 operator|(Vec_f32x1 lhs, const Vec_f32x1& rhs) {
        lhs |= rhs.data();
        return lhs;
    }

    Vec_f32x1& operator^=(const Vec_f32x1& rhs) {
        data_ = sg_bitcast_u32x1_f32x1(
            sg_bitcast_f32x1_u32x1(data_) ^ sg_bitcast_f32x1_u32x1(rhs.data()));
        return *this;
    }
    friend Vec_f32x1 operator^(Vec_f32x1 lhs, const Vec_f32x1& rhs) {
        lhs ^= rhs.data();
        return lhs;
    }

    Vec_f32x1 operator~() const {
        return sg_bitcast_u32x1_f32x1(~sg_bitcast_f32x1_u32x1(data_));
    }

    Compare_f32x1 operator<(const Vec_f32x1& rhs) const {
        return data_ < rhs.data();
    }
    Compare_f32x1 operator<=(const Vec_f32x1& rhs) const {
        return data_ <= rhs.data();
    }
    Compare_f32x1 operator==(const Vec_f32x1& rhs) const {
        return data_ == rhs.data();
    }
    Compare_f32x1 operator>=(const Vec_f32x1& rhs) const {
        return data_ >= rhs.data();
    }
    Compare_f32x1 operator>(const Vec_f32x1& rhs) const {
        return data_ > rhs.data();
    }

    Vec_f32x1 safe_divide_by(const Vec_f32x1& rhs) const {
        return rhs.data() == 0.0f ? data_ : data_ / rhs.data();
    }
    Vec_f32x1 abs() const { return std::abs(data_); }
    Vec_f32x1 remove_signed_zero() const {
        return data_ == 0.0f ? 0.0f : data_;
    }
    static Vec_f32x1 min_fast(const Vec_f32x1& a, const Vec_f32x1& b) {
        return std::min(a.data(), b.data());
    }
    static Vec_f32x1 max_fast(const Vec_f32x1& a, const Vec_f32x1& b) {
        return std::max(a.data(), b.data());
    }
    Vec_f32x1 constrain(const Vec_f32x1& lowerb, const Vec_f32x1& upperb) const
    {
        return Vec_f32x1::min_fast(Vec_f32x1::max_fast(lowerb, data_), upperb);
    }

    bool debug_eq(float f) const {
        return sg_bitcast_f32x1_u32x1(data_) == sg_bitcast_f32x1_u32x1(f);
    }

    Vec_s32x1 bitcast_to_s32() const {
        return sg_bitcast_f32x1_s32x1(data_);
    }
    Vec_s64x1 bitcast_to_s64() const {
        return static_cast<int64_t>(sg_bitcast_f32x1_u32x1(data_));
    }
    Vec_f32x1 bitcast_to_f32() const { return *this; }
    inline Vec_f64x1 bitcast_to_f64() const;

    Vec_s32x1 convert_to_nearest_s32() const {
        return static_cast<int32_t>(std::round(data_));
    }
    Vec_s32x1 truncate_to_s32() const { return static_cast<int32_t>(data_); }
    Vec_s32x1 floor_to_s32() const {
        return static_cast<int32_t>(std::floor(data_));
    }
    Vec_s64x1 convert_to_nearest_s64() const {
        return static_cast<int64_t>(std::round(data_));
    }
    Vec_s32x1 truncate_to_s64() const {
        return static_cast<int64_t>(data_);
    }
    Vec_s32x1 floor_to_s64() const {
        return static_cast<int64_t>(std::floor(data_));
    }
    Vec_f32x1 convert_to_f32() const { return *this; }
    inline Vec_f64x1 convert_to_f64() const;

    static Vec_f32x1 from(const Vec_s32x1& s32) {
        return static_cast<float>(s32.data());
    }
    static Vec_f32x1 from(const Vec_s64x1& s64) {
        return static_cast<float>(s64.data());
    }
    static Vec_f32x1 from(const Vec_f32x1& f32) { return f32; }
    static inline Vec_f32x1 from(const Vec_f64x1& f64);

    Vec_s32x1 exponent_frexp() const {
        return static_cast<int32_t>(std::ilogb(data_) + 1);
    }
    Vec_s32x1 exponent_frexp_s32() const { return exponent_frexp(); }
    Vec_s32x1 exponent() const {
        return static_cast<int32_t>(std::ilogb(data_));
    }
    Vec_s32x1 exponent_s32() const { return exponent(); }
    Vec_f32x1 mantissa_frexp() const {
        int discard;
        return std::frexp(data_, &discard);
    }
    Vec_f32x1 mantissa() const { return mantissa_frexp() * 2.0f; }

    Vec_f32x1 ldexp(const Vec_s32x1& e) const {
        return std::ldexp(data_, static_cast<int>(e.data()));
    }

    Vec_f32x1 std_log() const { return std::log(data_); }
    Vec_f32x1 std_exp() const { return std::exp(data_); }
    Vec_f32x1 std_sin() const { return std::sin(data_); }
    Vec_f32x1 std_cos() const { return std::cos(data_); }
};

class Vec_f64x1 {
    double data_;
public:
    Vec_f64x1() : data_{0.0} {}
    Vec_f64x1(const double f64) : data_{f64} {}

    static Vec_f64x1 minus_infinity() { return sg_minus_infinity_f64x1; }
    static Vec_f64x1 infinity() { return sg_infinity_f64x1; }

    static double elem_t(const int32_t s32) { return static_cast<double>(s32); }
    static double elem_t(const int64_t s64) { return static_cast<double>(s64); }
    static double elem_t(const float f32) { return static_cast<double>(f32); }
    static double elem_t(const double f64) { return f64; }

    static Compare_f64x1 compare_t(const Compare_s32x1& cmp) {
        return cmp.data();
    }
    static Compare_f64x1 compare_t(const Compare_s64x1& cmp) {
        return cmp.data();
    }
    static Compare_f64x1 compare_t(const Compare_f32x1& cmp) {
        return cmp.data();
    }
    static Compare_f64x1 compare_t(const Compare_f64x1& cmp) { return cmp; }

    static Vec_f64x1 bitcast_from_u64(const uint64_t i) {
        return sg_bitcast_u64x1_f64x1(i);
    }

    double data() const { return data_; }
    explicit operator double() const { return data_; }

    Vec_f64x1& operator+=(const Vec_f64x1& rhs) {
        data_ += rhs.data();
        return *this;
    }
    friend Vec_f64x1 operator+(Vec_f64x1 lhs, const Vec_f64x1& rhs) {
        lhs += rhs;
        return lhs;
    }
    Vec_f64x1 operator+() const { return *this; }

    Vec_f64x1& operator-=(const Vec_f64x1& rhs) {
        data_ -= rhs.data();
        return *this;
    }
    friend Vec_f64x1 operator-(Vec_f64x1 lhs, const Vec_f64x1& rhs) {
        lhs -= rhs;
        return lhs;
    }
    Vec_f64x1 operator-() const { return -data_; }

    Vec_f64x1& operator*=(const Vec_f64x1& rhs) {
        data_ *= rhs.data();
        return *this;
    }
    friend Vec_f64x1 operator*(Vec_f64x1 lhs, const Vec_f64x1& rhs) {
        lhs *= rhs;
        return lhs;
    }

    Vec_f64x1& operator/=(const Vec_f64x1& rhs) {
        data_ /= rhs.data();
        return *this;
    }
    friend Vec_f64x1 operator/(Vec_f64x1 lhs, const Vec_f64x1& rhs) {
        lhs /= rhs;
        return lhs;
    }

    Vec_f64x1 mul_add(const Vec_f64x1& mul, const Vec_f64x1& add) const {
        return sg_mul_add_f64x1(data_, mul.data(), add.data());
    }

    Vec_f64x1& operator&=(const Vec_f64x1& rhs) {
        data_ = sg_bitcast_u64x1_f64x1(
            sg_bitcast_f64x1_u64x1(data_) & sg_bitcast_f64x1_u64x1(rhs.data()));
        return *this;
    }
    friend Vec_f64x1 operator&(Vec_f64x1 lhs, const Vec_f64x1& rhs) {
        lhs &= rhs.data();
        return lhs;
    }

    Vec_f64x1& operator|=(const Vec_f64x1& rhs) {
        data_ = sg_bitcast_u64x1_f64x1(
            sg_bitcast_f64x1_u64x1(data_) | sg_bitcast_f64x1_u64x1(rhs.data()));
        return *this;
    }
    friend Vec_f64x1 operator|(Vec_f64x1 lhs, const Vec_f64x1& rhs) {
        lhs |= rhs.data();
        return lhs;
    }

    Vec_f64x1& operator^=(const Vec_f64x1& rhs) {
        data_ = sg_bitcast_u64x1_f64x1(
            sg_bitcast_f64x1_u64x1(data_) ^ sg_bitcast_f64x1_u64x1(rhs.data()));
        return *this;
    }
    friend Vec_f64x1 operator^(Vec_f64x1 lhs, const Vec_f64x1& rhs) {
        lhs ^= rhs.data();
        return lhs;
    }

    Vec_f64x1 operator~() const {
        return sg_bitcast_u64x1_f64x1(~sg_bitcast_f64x1_u64x1(data_));
    }

    Compare_f64x1 operator<(const Vec_f64x1& rhs) const {
        return data_ < rhs.data();
    }
    Compare_f64x1 operator<=(const Vec_f64x1& rhs) const {
        return data_ <= rhs.data();
    }
    Compare_f64x1 operator==(const Vec_f64x1& rhs) const {
        return data_ == rhs.data();
    }
    Compare_f64x1 operator>=(const Vec_f64x1& rhs) const {
        return data_ >= rhs.data();
    }
    Compare_f64x1 operator>(const Vec_f64x1& rhs) const {
        return data_ > rhs.data();
    }

    Vec_f64x1 safe_divide_by(const Vec_f64x1& rhs) const {
        return rhs.data() == 0.0 ? data_ : data_ / rhs.data();
    }
    Vec_f64x1 abs() const { return std::abs(data_); }
    Vec_f64x1 remove_signed_zero() const {
        return data_ == 0.0 ? 0.0 : data_;
    }
    static Vec_f64x1 min_fast(const Vec_f64x1& a, const Vec_f64x1& b) {
        return std::min(a.data(), b.data());
    }
    static Vec_f64x1 max_fast(const Vec_f64x1& a, const Vec_f64x1& b) {
        return std::max(a.data(), b.data());
    }
    Vec_f64x1 constrain(const Vec_f64x1& lowerb, const Vec_f64x1& upperb) const
    {
        return Vec_f64x1::min_fast(Vec_f64x1::max_fast(lowerb, data_), upperb);
    }

    bool debug_eq(double d) const {
        return sg_bitcast_f64x1_u64x1(data_) == sg_bitcast_f64x1_u64x1(d);
    }

    Vec_s32x1 bitcast_to_s32() const {
        return sg_bitcast_u32x1_s32x1(
            static_cast<uint32_t>(sg_bitcast_f64x1_u64x1(data_) & 0xffffffff));
    }
    Vec_s64x1 bitcast_to_s64() const {
        return sg_bitcast_f64x1_s64x1(data_);
    }
    Vec_f32x1 bitcast_to_f32() const {
        return sg_bitcast_u32x1_f32x1(
            static_cast<uint32_t>(sg_bitcast_f64x1_u64x1(data_) & 0xffffffff));
    }
    Vec_f64x1 bitcast_to_f64() const { return *this; }

    Vec_s32x1 convert_to_nearest_s32() const {
        return static_cast<int32_t>(std::round(data_));
    }
    Vec_s32x1 truncate_to_s32() const { return static_cast<int32_t>(data_); }
    Vec_s32x1 floor_to_s32() const {
        return static_cast<int32_t>(std::floor(data_));
    }
    Vec_s64x1 convert_to_nearest_s64() const {
        return static_cast<int64_t>(std::round(data_));
    }
    Vec_s64x1 truncate_to_s64() const {
        return static_cast<int64_t>(data_);
    }
    Vec_s64x1 floor_to_s64() const {
        return static_cast<int64_t>(std::floor(data_));
    }
    Vec_f32x1 convert_to_f32() const { return static_cast<float>(data_); }
    Vec_f64x1 convert_to_f64() const { return *this; }

    static Vec_f64x1 from(const Vec_s32x1& s32) {
        return static_cast<double>(s32.data());
    }
    static Vec_f64x1 from(const Vec_s64x1& s64) {
        return static_cast<double>(s64.data());
    }
    static Vec_f64x1 from(const Vec_f32x1& f32) {
        return static_cast<double>(f32.data());
    }
    static Vec_f64x1 from(const Vec_f64x1& f64) { return f64; }

    Vec_s64x1 exponent_frexp() const {
        return static_cast<int64_t>(std::ilogb(data_) + 1);
    }
    Vec_s32x1 exponent_frexp_s32() const {
        return exponent_frexp().convert_to_s32();
    }
    Vec_s64x1 exponent() const {
        return static_cast<int64_t>(std::ilogb(data_));
    }
    Vec_s32x1 exponent_s32() const { return exponent().convert_to_s32(); }
    Vec_f64x1 mantissa_frexp() const {
        int discard;
        return std::frexp(data_, &discard);
    }
    Vec_f64x1 mantissa() const { return mantissa_frexp() * 2.0; }

    Vec_f64x1 ldexp(const Vec_s64x1& e) const {
        return std::ldexp(data_, static_cast<int>(e.data()));
    }
    Vec_f64x1 ldexp(const Vec_s32x1& e) const {
        return std::ldexp(data_, static_cast<int>(e.data()));
    }

    Vec_f64x1 std_log() const { return std::log(data_); }
    Vec_f64x1 std_exp() const { return std::exp(data_); }
    Vec_f64x1 std_sin() const { return std::sin(data_); }
    Vec_f64x1 std_cos() const { return std::cos(data_); }
};

typedef Vec_f32x1 Vec_ss;
typedef Vec_f64x1 Vec_sd;

inline Vec_s64x1 Vec_s32x1::bitcast_to_s64() const {
    return static_cast<int64_t>(sg_bitcast_s32x1_u32x1(data_));
}
inline Vec_f32x1 Vec_s32x1::bitcast_to_f32() const {
    return sg_bitcast_s32x1_f32x1(data_);
}
inline Vec_f64x1 Vec_s32x1::bitcast_to_f64() const {
    return sg_bitcast_u64x1_f64x1(
        static_cast<uint64_t>(sg_bitcast_s32x1_u32x1(data_)));
}

inline Vec_s64x1 Vec_s32x1::convert_to_s64() const {
    return static_cast<int64_t>(data_);
}
inline Vec_f32x1 Vec_s32x1::convert_to_f32() const {
    return static_cast<float>(data_);
}
inline Vec_f64x1 Vec_s32x1::convert_to_f64() const {
    return static_cast<double>(data_);
}

inline Vec_f32x1 Vec_s64x1::bitcast_to_f32() const {
    return sg_bitcast_u32x1_f32x1(
        static_cast<uint32_t>((sg_bitcast_s64x1_u64x1(data_) & 0xffffffff)));
}
inline Vec_f64x1 Vec_s64x1::bitcast_to_f64() const {
    return sg_bitcast_s64x1_f64x1(data_);
}

inline Vec_f32x1 Vec_s64x1::convert_to_f32() const {
    return static_cast<float>(data_);
}
inline Vec_f64x1 Vec_s64x1::convert_to_f64() const {
    return static_cast<double>(data_);
}

inline Vec_f64x1 Vec_f32x1::bitcast_to_f64() const {
    return sg_bitcast_u64x1_f64x1(static_cast<uint64_t>(
        sg_bitcast_f32x1_u32x1(data_)));
}
inline Vec_f64x1 Vec_f32x1::convert_to_f64() const {
    return static_cast<double>(data_);
}

inline Vec_s32x1 Vec_s32x1::from(const Vec_s64x1& s64) {
    return static_cast<int32_t>(s64.data());
}

inline Vec_f32x1 Vec_f32x1::from(const Vec_f64x1& f64) {
    return static_cast<float>(f64.data());
}

inline Compare_s64x1 Compare_s32x1::convert_to_cmp_s64() const { return data_; }
inline Compare_f32x1 Compare_s32x1::convert_to_cmp_f32() const { return data_; }
inline Compare_f64x1 Compare_s32x1::convert_to_cmp_f64() const { return data_; }

inline Compare_f32x1 Compare_s64x1::convert_to_cmp_f32() const { return data_; }
inline Compare_f64x1 Compare_s64x1::convert_to_cmp_f64() const { return data_; }

inline Compare_f64x1 Compare_f32x1::convert_to_cmp_f64() const { return data_; }

inline Vec_s32x1 Compare_s32x1::choose_else_zero(const Vec_s32x1& if_true) const
{
    return data_ ? if_true : 0;
}
inline Vec_s64x1 Compare_s64x1::choose_else_zero(const Vec_s64x1& if_true) const
{
    return data_ ? if_true : 0;
}
inline Vec_f32x1 Compare_f32x1::choose_else_zero(const Vec_f32x1& if_true) const
{
    return data_ ? if_true : 0.0f;
}
inline Vec_f64x1 Compare_f64x1::choose_else_zero(const Vec_f64x1& if_true) const
{
    return data_ ? if_true : 0.0;
}

inline Vec_s32x1 Compare_s32x1::choose(const Vec_s32x1& if_true,
    const Vec_s32x1& if_false) const
{
    return data_ ? if_true : if_false;
}

inline Vec_s64x1 Compare_s64x1::choose(const Vec_s64x1& if_true,
    const Vec_s64x1& if_false) const
{
    return data_ ? if_true : if_false;
}
inline Vec_f32x1 Compare_f32x1::choose(const Vec_f32x1& if_true,
    const Vec_f32x1& if_false) const
{
    return data_ ? if_true : if_false;
}
inline Vec_f64x1 Compare_f64x1::choose(const Vec_f64x1& if_true,
    const Vec_f64x1& if_false) const
{
    return data_ ? if_true : if_false;
}

} // namespace simd_granodi

#endif // __cplusplus

#endif // SIMD_GRANODI_H
