#ifndef SIMD_GRANODI_H
#define SIMD_GRANODI_H

/*

SIMD GRANODI
Copyright 2021-2022 Jon Ville / Jon V Audio

OBJECTIVES:

Thin layer to abstract ARM64/NEON and x64/SSE2, mostly focusing on
packed floats and doubles but with support for packed 32 & 64 bit
signed integers.

Minimal use of preprocessor, except where needed to generate an immediate value.

Target C99 and C++11.

The C implementation only uses typedefs, with no new structs (other than for
the generic implementation).

The C++ classes for operator overloading are written using the C
implementation, without knowledge of the hardware.

Separate typedefs / classes for the result of a vector comparison.

Avoid UB (obey strict aliasing rules & use memcpy where necessary). No unions

Generic, SSE, and NEON definitions interleaved for comparison, and education /
documentation / reference etc

Assume high confidence in modern compilers to inline, elide copies,
pre-calculate constants etc (eg no force inline)

Behaviour of corner cases may NOT be identical on separate platforms
(eg min / max of signed floating point zero, potential untested corner cases
involving conversion, special values etc)

Naming conventions are based on SSE2


NON-VECTOR FUNCTIONS:
(ie, might switch to general purpose registers, stall etc)

All NEON shuffles happen in the NEON registers, but build their result in a
scalar way, in order to be as general as intel shuffles:
NEON 32x4 shuffles take 4 instructions, and 64x2 shuffles take 2 instructions.

Non-vector on SSE2 only:
sg_cvt_pi64_ps
sg_cvt_pi64_pd
sg_cvt_ps_pi64
sg_cvtt_ps_pi64
sg_cvt_pd_pi64
sg_cvtt_pd_pi64
sg_cmplt_pi64
sg_cmplte_pi64
sg_cmpgte_pi64
sg_cmpgt_pi64
sg_abs_pi64
sg_min_pi64
sg_max_pi64

SSE2 special mention:
sg_mul_pi32 is in vector registers, but involves shuffling, bitwise ops,
negation, and two _mm_mul_epu32

Non-vector on SSE2 and NEON:
sg_mul_pi64
sg_div_pi32
sg_div_pi64
sg_safediv_pi32
sg_safediv_pi64

PLATFORM DETECTION
32 bit ARM NEON hardware does not have all the intrinsic instructions used here,
but can compile using SIMD_GRANODI_FORCE_GENERIC, and can disable denormals.

32-bit x86 can work, but you cannot disable denormal numbers on x87,
so it is recommended to set your compiler to generate SSE2 code for all floats
if you wish to disable denormal numbers with mixed intrinsic / scalar code.

*/

// TODO:
// - Add truncate / round intrinsics

// Sanity check
#if defined (SIMD_GRANODI_SSE2) || defined (SIMD_GRANODI_NEON) || \
    defined (SIMD_GRANODI_DENORMAL_SSE) || \
    defined (SIMD_GRANODI_DENORMAL_ARM64) || \
    defined (SIMD_GRANODI_DENORMAL_ARM32)
#error "A SIMD_GRANODI macro was defined before it should be"
#endif

#if defined (__GNUC__) || defined (__clang__)
    #if defined (__x86_64__) || (defined (__i386__) && defined (__SSE2__))
        // Warning: on x86 (32-bit), this doesn't guarantee that x87
        // instructions won't also be generated. Check your compiler options!
        #define SIMD_GRANODI_SSE2
        #define SIMD_GRANODI_DENORMAL_SSE
    #elif defined (__aarch64__)
        #define SIMD_GRANODI_NEON
        #define SIMD_GRANODI_DENORMAL_ARM64
    #elif defined (__arm__)
        #define SIMD_GRANODI_FORCE_GENERIC
        #define SIMD_GRANODI_DENORMAL_ARM32
    #else
        #define SIMD_GRANODI_FORCE_GENERIC
    #endif
#elif defined (_MSC_VER)
    #if defined (_M_AMD64) || (defined (_M_IX86) && (_M_IX86_FP == 2))
        #define SIMD_GRANODI_SSE2
        #define SIMD_GRANODI_DENORMAL_SSE
    #else
        #define SIMD_GRANODI_FORCE_GENERIC
    #endif
#endif

// In case the user wants to FORC_GENERIC, the #ifdefs below should prioritize
// this, but just in case there is an error
#ifdef SIMD_GRANODI_FORCE_GENERIC
#undef SIMD_GRANODI_SSE2
#undef SIMD_GRANODI_NEON
#endif

#ifndef __cplusplus
#include <stdbool.h>
#endif
#include <stdint.h>
#include <stdlib.h> // for abs(int32/64)
#include <string.h> // for memcpy

// For rint, rintf, fabs, fabsf
// SSE2 uses rintf() for sg_cvt_ps_pi64()
#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
#include <math.h>
#endif

#ifdef SIMD_GRANODI_DENORMAL_SSE
#include <emmintrin.h>
#elif defined SIMD_GRANODI_NEON
#include <arm_neon.h>
#endif

#ifdef __cplusplus
namespace simd_granodi {
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

#define sg_allset_u32 ((uint32_t) -1)
#define sg_allset_u64 ((uint64_t) -1)
#define sg_fp_signbit_u32 ((uint32_t) 0x80000000)
#define sg_fp_signbit_u64 ((uint64_t) 0x8000000000000000)
#define sg_fp_signmask_u32 (0x7fffffff)
#define sg_fp_signmask_u64 (0x7fffffffffffffff)

#define sg_allset_s32 (-1)
#define sg_allset_s64 (-1)
#define sg_fp_signbit_s32 ((int32_t) 0x80000000)
#define sg_fp_signbit_s64 ((int64_t) 0x8000000000000000)
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

#ifdef SIMD_GRANODI_SSE2
#define sg_sse2_allset_si128 (_mm_set_epi64x(sg_allset_s64, sg_allset_s64))
#define sg_sse2_allset_ps (_mm_castsi128_ps(sg_sse2_allset_si128))
#define sg_sse2_allset_pd (_mm_castsi128_pd(sg_sse2_allset_si128))
#define sg_sse2_signbit_ps (_mm_castsi128_ps(_mm_set1_epi32(sg_fp_signbit_s32)))
#define sg_sse2_signbit_pd (_mm_castsi128_pd( \
    _mm_set1_epi64x(sg_fp_signbit_s64)))
#define sg_sse2_signmask_ps (_mm_castsi128_ps( \
    _mm_set1_epi32(sg_fp_signmask_s32)))
#define sg_sse2_signmask_pd (_mm_castsi128_pd( \
    _mm_set1_epi64x(sg_fp_signmask_s64)))
#endif // SIMD_GRANODI_SSE2

// Declarations for functions used out of order
static inline sg_pi32 sg_choose_pi32(sg_cmp_pi32, sg_pi32, sg_pi32);
static inline sg_pi64 sg_choose_pi64(sg_cmp_pi64, sg_pi64, sg_pi64);

// Bitcasts (reinterpret, no conversion takes place)

// Scalar bitcasts

static inline int32_t sg_scast_u32_s32(const uint32_t a) {
    int32_t result; memcpy(&result, &a, sizeof(int32_t)); return result;
}
static inline float sg_scast_u32_f32(const uint32_t a) {
    float result; memcpy(&result, &a, sizeof(float)); return result;
}
static inline uint32_t sg_scast_s32_u32(const int32_t a) {
    uint32_t result; memcpy(&result, &a, sizeof(uint32_t)); return result;
}
static inline float sg_scast_s32_f32(const int32_t a) {
    float result; memcpy(&result, &a, sizeof(float)); return result;
}
static inline uint32_t sg_scast_f32_u32(const float a) {
    uint32_t result; memcpy(&result, &a, sizeof(uint32_t)); return result;
}
static inline int32_t sg_scast_f32_s32(const float a) {
    int32_t result; memcpy(&result, &a, sizeof(int32_t)); return result;
}

static inline int64_t sg_scast_u64_s64(const uint64_t a) {
    int64_t result; memcpy(&result, &a, sizeof(int64_t)); return result;
}
static inline double sg_scast_u64_f64(const uint64_t a) {
    double result; memcpy(&result, &a, sizeof(double)); return result;
}
static inline uint64_t sg_scast_s64_u64(const int64_t a) {
    uint64_t result; memcpy(&result, &a, sizeof(uint64_t)); return result;
}
static inline double sg_scast_s64_f64(const int64_t a) {
    double result; memcpy(&result, &a, sizeof(double)); return result;
}
static inline uint64_t sg_scast_f64_u64(const double a) {
    uint64_t result; memcpy(&result, &a, sizeof(uint64_t)); return result;
}
static inline int64_t sg_scast_f64_s64(const double a) {
    int64_t result; memcpy(&result, &a, sizeof(int64_t)); return result;
}

// For the generic versions, we directly define casts for:
// - Same size, but different type
// - Different sized integer
// And then write the rest in terms of those

static inline sg_pi64 sg_cast_pi32_pi64(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) {
        .l0 = sg_scast_u64_s64((((uint64_t) sg_scast_s32_u32(a.i1)) << 32)
            | ((uint64_t) sg_scast_s32_u32(a.i0))),
        .l1 = sg_scast_u64_s64((((uint64_t) sg_scast_s32_u32(a.i3)) << 32)
            | ((uint64_t) sg_scast_s32_u32(a.i2))) };
    #elif defined SIMD_GRANODI_SSE2
    return a;
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_s64_s32(a);
    #endif
}

static inline sg_pi32 sg_cast_pi64_pi32(const sg_pi64 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    const uint64_t u0 = sg_scast_s64_u64(a.l0),
        u1 = sg_scast_s64_u64(a.l1);
    return (sg_pi32) {
        .i0 = sg_scast_u32_s32((uint32_t) (u0 & 0xffffffffu)),
        .i1 = sg_scast_u32_s32((uint32_t) (u0 >> 32)),
        .i2 = sg_scast_u32_s32((uint32_t) (u1 & 0xffffffffu)),
        .i3 = sg_scast_u32_s32((uint32_t) (u1 >> 32)) };
    #elif defined SIMD_GRANODI_SSE2
    return a;
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_s32_s64(a);
    #endif
}

static inline sg_ps sg_cast_pi32_ps(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = sg_scast_s32_f32(a.i0),
        .f1 = sg_scast_s32_f32(a.i1), .f2 = sg_scast_s32_f32(a.i2),
        .f3 = sg_scast_s32_f32(a.i3) };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castsi128_ps(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f32_s32(a);
    #endif
}

static inline sg_pi32 sg_cast_ps_pi32(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = sg_scast_f32_s32(a.f0),
        .i1 = sg_scast_f32_s32(a.f1), .i2 = sg_scast_f32_s32(a.f2),
        .i3 = sg_scast_f32_s32(a.f3) };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castps_si128(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_s32_f32(a);
    #endif
}

static inline sg_pd sg_cast_pi64_pd(const sg_pi64 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = sg_scast_s64_f64(a.l0),
        .d1 = sg_scast_s64_f64(a.l1) };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castsi128_pd(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f64_s64(a);
    #endif
}

static inline sg_pi64 sg_cast_pd_pi64(const sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = sg_scast_f64_s64(a.d0),
        .l1 = sg_scast_f64_s64(a.d1) };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castpd_si128(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_s64_f64(a);
    #endif
}

static inline sg_pd sg_cast_pi32_pd(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi64_pd(sg_cast_pi32_pi64(a));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castsi128_pd(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f64_s32(a);
    #endif
}

static inline sg_ps sg_cast_pi64_ps(const sg_pi64 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi32_ps(sg_cast_pi64_pi32(a));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castsi128_ps(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f32_s64(a);
    #endif
}

static inline sg_pi64 sg_cast_ps_pi64(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi32_pi64(sg_cast_ps_pi32(a));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castps_si128(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_s64_f32(a);
    #endif
}

static inline sg_pd sg_cast_ps_pd(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi64_pd(sg_cast_pi32_pi64(sg_cast_ps_pi32(a)));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castps_pd(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f64_f32(a);
    #endif
}

static inline sg_pi32 sg_cast_pd_pi32(const sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi64_pi32(sg_cast_pd_pi64(a));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castpd_si128(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_s32_f64(a);
    #endif
}

static inline sg_ps sg_cast_pd_ps(const sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi32_ps(sg_cast_pi64_pi32(sg_cast_pd_pi64(a)));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castpd_ps(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f32_f64(a);
    #endif
}

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
    return (sg_pi32) { .i0 = array[imm8_compile_time_constant & 3],
        .i1 = array[(imm8_compile_time_constant >> 2) & 3],
        .i2 = array[(imm8_compile_time_constant >> 4) & 3],
        .i3 = array[(imm8_compile_time_constant >> 6) & 3] };
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
    return (sg_pi64) { .l0 = array[imm8_compile_time_constant & 1],
        .l1 = array[(imm8_compile_time_constant >> 1) & 1] };
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
    return (sg_ps) { .f0 = array[imm8_compile_time_constant & 3],
        .f1 = array[(imm8_compile_time_constant >> 2) & 3],
        .f2 = array[(imm8_compile_time_constant >> 4) & 3],
        .f3 = array[(imm8_compile_time_constant >> 6) & 3] };
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
    return (sg_pd) { .d0 = array[imm8_compile_time_constant & 1],
        .d1 = array[(imm8_compile_time_constant >> 1) & 1] };
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

static inline sg_pi32 sg_setzero_pi32() {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = 0, .i1 = 0, .i2 = 0, .i3 = 0 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_setzero_si128();
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_s32(0);
    #endif
}

static inline sg_pi32 sg_set1_pi32(const int32_t si) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = si, .i1 = si, .i2 = si, .i3 = si };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set1_epi32(si);
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_s32(si);
    #endif
}
// Useful for creating bit masks eg getting exponent etc
static inline sg_pi32 sg_set1_from_u32_pi32(const uint32_t i) {
    return sg_set1_pi32(sg_scast_s32_u32(i));
}

static inline sg_pi32 sg_set_pi32(const int32_t si3, const int32_t si2,
    const int32_t si1, const int32_t si0)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = si0, .i1 = si1, .i2 = si2, .i3 = si3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set_epi32(si3, si2, si1, si0);
    #elif defined SIMD_GRANODI_NEON
    int32x4_t result = vdupq_n_s32(0);
    result = vsetq_lane_s32(si0, result, 0);
    result = vsetq_lane_s32(si1, result, 1);
    result = vsetq_lane_s32(si2, result, 2);
    return vsetq_lane_s32(si3, result, 3);
    #endif
}
static inline sg_pi32 sg_set_from_u32_pi32(const uint32_t i3, const uint32_t i2,
    const uint32_t i1, const uint32_t i0)
{
    return sg_set_pi32(sg_scast_s32_u32(i3), sg_scast_s32_u32(i2),
        sg_scast_s32_u32(i1), sg_scast_s32_u32(i0));
}

static inline sg_pi64 sg_setzero_pi64() {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = 0, .l1 = 0 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_setzero_si128();
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_s64(0);
    #endif
}

static inline sg_pi64 sg_set1_pi64(const int64_t si) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = si, .l1 = si };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set1_epi64x(si);
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_s64(si);
    #endif
}
static inline sg_pi64 sg_set1_from_u64_pi64(const uint64_t i) {
    return sg_set1_pi64(sg_scast_u64_s64(i));
}

static inline sg_pi64 sg_set_pi64(const int64_t si1, const int64_t si0) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = si0, .l1 = si1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set_epi64x(si1, si0);
    #elif defined SIMD_GRANODI_NEON
    return vsetq_lane_s64(si1, vsetq_lane_s64(si0, vdupq_n_s64(0), 0), 1);
    #endif
}
static inline sg_pi64 sg_set_from_u64_pi64(const uint64_t i1,
    const uint64_t i0)
{
    return sg_set_pi64(sg_scast_u64_s64(i1), sg_scast_u64_s64(i0));
}

static inline sg_ps sg_setzero_ps() {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = 0.0f, .f1 = 0.0f, .f2 = 0.0f, .f3 = 0.0f };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_setzero_ps();
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_f32(0.0f);
    #endif
}

static inline sg_ps sg_set1_ps(const float f) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = f, .f1 = f, .f2 = f, .f3 = f };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set1_ps(f);
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_f32(f);
    #endif
}
static inline sg_ps sg_set1_from_u32_ps(const uint32_t i) {
    return sg_set1_ps(sg_scast_s32_f32(i));
}

static inline sg_ps sg_set_ps(const float f3, const float f2, const float f1,
    const float f0)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = f0, .f1 = f1, .f2 = f2, .f3 = f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set_ps(f3, f2, f1, f0);
    #elif defined SIMD_GRANODI_NEON
    float32x4_t result = vdupq_n_f32(0.0f);
    result = vsetq_lane_f32(f0, result, 0);
    result = vsetq_lane_f32(f1, result, 1);
    result = vsetq_lane_f32(f2, result, 2);
    return vsetq_lane_f32(f3, result, 3);
    #endif
}
static inline sg_ps sg_set_from_u32_ps(const uint32_t i3, const uint32_t i2,
    const uint32_t i1, const uint32_t i0)
{
    return sg_set_ps(sg_scast_u32_f32(i3), sg_scast_u32_f32(i2),
        sg_scast_u32_f32(i1), sg_scast_u32_f32(i0));
}

static inline sg_pd sg_setzero_pd() {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = 0.0, .d1 = 0.0 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_setzero_pd();
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_f64(0.0);
    #endif
}

static inline sg_pd sg_set1_pd(const double d) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = d, .d1 = d };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set1_pd(d);
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_f64(d);
    #endif
}
static inline sg_pd sg_set1_from_u64_pd(const uint64_t l) {
    return sg_set1_pd(sg_scast_u64_f64(l));
}

static inline sg_pd sg_set_pd(const double d1, const double d0) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = d0, .d1 = d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set_pd(d1, d0);
    #elif defined SIMD_GRANODI_NEON
    float64x2_t result = vdupq_n_f64(0.0);
    result = vsetq_lane_f64(d0, result, 0);
    return vsetq_lane_f64(d1, result, 1);
    #endif
}
static inline sg_pd sg_set_from_u64_pd(const uint64_t l1, const uint64_t l0) {
    return sg_set_pd(sg_scast_u64_f64(l1), sg_scast_u64_f64(l0));
}

// Set generic

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

static inline int32_t sg_get0_pi32(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.i0;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtsi128_si32(a);
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_s32(a, 0);
    #endif
}

static inline int32_t sg_get1_pi32(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.i1;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtsi128_si32(
        _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(3, 2, 1, 1)));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_s32(a, 1);
    #endif
}

static inline int32_t sg_get2_pi32(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.i2;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtsi128_si32(
        _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(3, 2, 1, 2)));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_s32(a, 2);
    #endif
}

static inline int32_t sg_get3_pi32(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.i3;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtsi128_si32(
        _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(3, 2, 1, 3)));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_s32(a, 3);
    #endif
}

static inline int64_t sg_get0_pi64(const sg_pi64 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.l0;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtsi128_si64(a);
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_s64(a, 0);
    #endif
}

static inline int64_t sg_get1_pi64(const sg_pi64 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.l1;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtsi128_si64(_mm_unpackhi_epi64(a, a));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_s64(a, 1);
    #endif
}

static inline float sg_get0_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.f0;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtss_f32(a);
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_f32(a, 0);
    #endif
}

static inline float sg_get1_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.f1;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtss_f32(
        _mm_shuffle_ps(a, a, sg_sse2_shuffle32_imm(3, 2, 1, 1)));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_f32(a, 1);
    #endif
}

static inline float sg_get2_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.f2;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtss_f32(
        _mm_shuffle_ps(a, a, sg_sse2_shuffle32_imm(3, 2, 1, 2)));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_f32(a, 2);
    #endif
}

static inline float sg_get3_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.f3;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtss_f32(
        _mm_shuffle_ps(a, a, sg_sse2_shuffle32_imm(3, 2, 1, 3)));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_f32(a, 3);
    #endif
}

static inline double sg_get0_pd(const sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.d0;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtsd_f64(a);
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_f64(a, 0);
    #endif
}

static inline double sg_get1_pd(const sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return a.d1;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtsd_f64(_mm_unpackhi_pd(a, a));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_f64(a, 1);
    #endif
}

static inline sg_generic_pi32 sg_getg_pi32(const sg_pi32 a) {
    return (sg_generic_pi32) { .i0 = sg_get0_pi32(a), .i1 = sg_get1_pi32(a),
        .i2 = sg_get2_pi32(a), .i3 = sg_get3_pi32(a) };
}

static inline sg_generic_pi64 sg_getg_pi64(const sg_pi64 a) {
    return (sg_generic_pi64) { .l0 = sg_get0_pi64(a), .l1 = sg_get1_pi64(a) };
}

static inline sg_generic_ps sg_getg_ps(const sg_ps a) {
    return (sg_generic_ps) { .f0 = sg_get0_ps(a), .f1 = sg_get1_ps(a),
        .f2 = sg_get2_ps(a), .f3 = sg_get3_ps(a) };
}

static inline sg_generic_pd sg_getg_pd(const sg_pd a) {
    return (sg_generic_pd) { .d0 = sg_get0_pd(a), .d1 = sg_get1_pd(a) };
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
    return sg_debug_eq_pi32(sg_cast_ps_pi32(a), sg_scast_f32_s32(f3),
        sg_scast_f32_s32(f2), sg_scast_f32_s32(f1), sg_scast_f32_s32(f0));
}

static inline bool sg_debug_eq_pd(const sg_pd a, const double d1,
    const double d0)
{
    return sg_debug_eq_pi64(sg_cast_pd_pi64(a),
        sg_scast_f64_s64(d1), sg_scast_f64_s64(d0));
}

// Conversion

static inline sg_pi64 sg_cvt_pi32_pi64(const sg_pi32 a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return sg_set_fromg_pi64((sg_generic_pi64) {
        .l0 = (int64_t) a.i0, .l1 = (int64_t) a.i1 });
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
    #elif defined SIMD_GRANODI_NEON
    return vshll_n_s32(vget_low_s32(a), 0);
    #endif
}

static inline sg_ps sg_cvt_pi32_ps(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = (float) a.i0, .f1 = (float) a.i1,
        .f2 = (float) a.i2, .f3 = (float) a.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtepi32_ps(a);
    #elif defined SIMD_GRANODI_NEON
    return vcvtq_f32_s32(a);
    #endif
}

static inline sg_pd sg_cvt_pi32_pd(const sg_pi32 a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return sg_set_fromg_pd((sg_generic_pd) { .d0 = (double) a.i0,
        .d1 = (double) a.i1 });
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtepi32_pd(a);
    #elif defined SIMD_GRANODI_NEON
    return vcvtq_f64_s64(vshll_n_s32(vget_low_s32(a), 0));
    #endif
}

static inline sg_pi32 sg_cvt_pi64_pi32(const sg_pi64 a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return sg_set_fromg_pi32((sg_generic_pi32) { .i0 = (int32_t) a.l0,
        .i1 = (int32_t) a.l1, .i2 = 0, .i3 = 0 });
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_si128(_mm_set_epi64x(0, sg_allset_s64),
        _mm_shuffle_epi32(a, sg_sse2_shuffle32_imm(3, 2, 2, 0)));
    #elif defined SIMD_GRANODI_NEON
    const int32x4_t cast = vreinterpretq_s32_s64(a);
    return vcombine_s32(vget_low_s32(vcopyq_laneq_s32(cast, 1, cast, 2)),
        vdup_n_s32(0));
    #endif
}

static inline sg_ps sg_cvt_pi64_ps(const sg_pi64 a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_ps) { .f0 = (float) a.l0,
        .f1 = (float) a.l1, .f2 = 0.0f, .f3 = 0.0f };
    #elif defined SIMD_GRANODI_SSE2
    const int64_t si0 = _mm_cvtsi128_si64(a),
        si1 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(a, a));
    __m128 result = _mm_cvtsi64_ss(_mm_setzero_ps(), si1);
    result = _mm_shuffle_ps(result, result, sg_sse2_shuffle32_imm(3, 2, 0, 0));
    return _mm_cvtsi64_ss(result, si0);
    #elif defined SIMD_GRANODI_NEON
    return vcombine_f32(vcvt_f32_f64(vcvtq_f64_s64(a)), vdup_n_f32(0.0f));
    #endif
}

static inline sg_pd sg_cvt_pi64_pd(const sg_pi64 a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_pd) { .d0 = (double) a.l0, .d1 = (double) a.l1 };
    #elif defined SIMD_GRANODI_SSE2
    const int64_t si0 = _mm_cvtsi128_si64(a),
        si1 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(a, a));
    __m128d result = _mm_cvtsi64_sd(_mm_setzero_pd(), si1);
    result = _mm_shuffle_pd(result, result, sg_sse2_shuffle64_imm(0, 0));
    return _mm_cvtsi64_sd(result, si0);
    #elif defined SIMD_GRANODI_NEON
    return vcvtq_f64_s64(a);
    #endif
}

// Use current rounding mode (default round-to-nearest with 0.5 rounding down)
static inline sg_pi32 sg_cvt_ps_pi32(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = (int32_t) rintf(a.f0),
        .i1 = (int32_t) rintf(a.f1), .i2 = (int32_t) rintf(a.f2),
        .i3 = (int32_t) rintf(a.f3) };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtps_epi32(a);
    #elif defined SIMD_GRANODI_NEON
    return vcvtnq_s32_f32(a);
    #endif
}

// cvtt (extra t) methods use truncation instead of rounding
static inline sg_pi32 sg_cvtt_ps_pi32(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = (int32_t) a.f0, .i1 = (int32_t) a.f1,
     .i2 = (int32_t) a.f2, .i3 = (int32_t) a.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvttps_epi32(a);
    #elif defined SIMD_GRANODI_NEON
    return vcvtq_s32_f32(a);
    #endif
}

static inline sg_pi64 sg_cvt_ps_pi64(sg_ps a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return sg_set_fromg_pi64((sg_generic_pi64) { .l0 = (int64_t) rintf(a.f0),
        .l1 = (int64_t) rintf(a.f1) });
    #elif defined SIMD_GRANODI_SSE2
    int64_t si0 = (int64_t) rintf(_mm_cvtss_f32(a)),
        si1 = (int64_t) rintf(_mm_cvtss_f32(
            _mm_shuffle_ps(a, a, sg_sse2_shuffle32_imm(3, 2, 1, 1))));
    return _mm_set_epi64x(si1, si0);
    #elif defined SIMD_GRANODI_NEON
    return vcvtnq_s64_f64(vcvt_f64_f32(vget_low_f32(a)));
    #endif
}

static inline sg_pi64 sg_cvtt_ps_pi64(sg_ps a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_pi64) { .l0 = (int64_t) a.f0, .l1 = (int64_t) a.f1 };
    #elif defined SIMD_GRANODI_SSE2
    int64_t si0 = (int64_t) _mm_cvtss_f32(a),
        si1 = (int64_t) _mm_cvtss_f32(
            _mm_shuffle_ps(a, a, sg_sse2_shuffle32_imm(3, 2, 2, 1)));
    return _mm_set_epi64x(si1, si0);
    #elif defined SIMD_GRANODI_NEON
    return vcvtq_s64_f64(vcvt_f64_f32(vget_low_f32(a)));
    #endif
}

static inline sg_pd sg_cvt_ps_pd(sg_ps a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return sg_set_fromg_pd((sg_generic_pd) { .d0 = (double) a.f0,
        .d1 = (double) a.f1 });
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtps_pd(a);
    #elif defined SIMD_GRANODI_NEON
    return vcvt_f64_f32(vget_low_f32(a));
    #endif
}

static inline sg_pi32 sg_cvt_pd_pi32(sg_pd a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_pi32) { .i0 = (int32_t) rint(a.d0),
        .i1 = (int32_t) rint(a.d1), .i2 = 0, .i3 = 0 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtpd_epi32(a);
    #elif defined SIMD_GRANODI_NEON
    return sg_cvt_pi64_pi32(vcvtnq_s64_f64(a));
    #endif
}

static inline sg_pi32 sg_cvtt_pd_pi32(sg_pd a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
    const sg_generic_pd ag = sg_getg_pd(a);
    return sg_set_fromg_pi32((sg_generic_pi32) {
        .i0 = (int32_t) ag.d0, .i1 = (int32_t) ag.d1, .i2 = 0, .i3 = 0 });
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvttpd_epi32(a);
    #elif defined SIMD_GRANODI_NEON
    return sg_cvt_pi64_pi32(vcvtq_s64_f64(a));
    #endif
}

static inline sg_pi64 sg_cvt_pd_pi64(sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = (int64_t) rint(a.d0),
        .l1 = (int64_t) rint(a.d1) };
    #elif defined SIMD_GRANODI_SSE2
    const int64_t si0 = _mm_cvtsd_si64(a),
        si1 = _mm_cvtsd_si64(_mm_unpackhi_pd(a, a));
    return _mm_set_epi64x(si1, si0);
    #elif defined SIMD_GRANODI_NEON
    return vcvtnq_s64_f64(a);
    #endif
}

static inline sg_pi64 sg_cvtt_pd_pi64(sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = (int64_t) a.d0, .l1 = (int64_t) a.d1 };
    #elif defined SIMD_GRANODI_SSE2
    const int64_t si0 = _mm_cvttsd_si64(a),
        si1 = _mm_cvttsd_si64(_mm_unpackhi_pd(a, a));
    return _mm_set_epi64x(si1, si0);
    #elif defined SIMD_GRANODI_NEON
    return vcvtq_s64_f64(a);
    #endif
}

// Todo: I don't know what rounding mode C/C++ uses for (float) cast of double
static inline sg_ps sg_cvt_pd_ps(sg_pd a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return sg_set_fromg_ps((sg_generic_ps) { .f0 = (float) a.d0,
        .f1 = (float) a.d1, .f2 = 0.0f, .f3 = 0.0f });
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cvtpd_ps(a);
    #elif defined SIMD_GRANODI_NEON
    return vcombine_f32(vcvt_f32_f64(a), vdup_n_f32(0.0f));
    #endif
}

// Arithmetic

static inline sg_pi32 sg_add_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = a.i0 + b.i0, .i1 = a.i1 + b.i1,
        .i2 = a.i2 + b.i2, .i3 = a.i3 + b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_add_epi32(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vaddq_s32(a, b);
    #endif
}

static inline sg_pi64 sg_add_pi64(const sg_pi64 a, const sg_pi64 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = a.l0 + b.l0, .l1 = a.l1 + b.l1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_add_epi64(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vaddq_s64(a, b);
    #endif
}

static inline sg_ps sg_add_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = a.f0 + b.f0, .f1 = a.f1 + b.f1,
        .f2 = a.f2 + b.f2, .f3 = a.f3 + b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_add_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vaddq_f32(a, b);
    #endif
}

static inline sg_pd sg_add_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = a.d0 + b.d0, .d1 = a.d1 + b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_add_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vaddq_f64(a, b);
    #endif
}

static inline sg_pi32 sg_sub_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = a.i0 - b.i0, .i1 = a.i1 - b.i1,
        .i2 = a.i2 - b.i2, .i3 = a.i3 - b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_sub_epi32(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vsubq_s32(a, b);
    #endif
}

static inline sg_pi64 sg_sub_pi64(const sg_pi64 a, const sg_pi64 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = a.l0 - b.l0, .l1 = a.l1 - b.l1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_sub_epi64(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vsubq_s64(a, b);
    #endif
}

static inline sg_ps sg_sub_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = a.f0 - b.f0, .f1 = a.f1 - b.f1,
        .f2 = a.f2 - b.f2, .f3 = a.f3 - b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_sub_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vsubq_f32(a, b);
    #endif
}

static inline sg_pd sg_sub_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = a.d0 - b.d0, .d1 = a.d1 - b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_sub_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vsubq_f64(a, b);
    #endif
}

static inline sg_pi32 sg_mul_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_pi32) { .i0 = a.i0 * b.i0,
        .i1 = a.i1 * b.i1, .i2 = a.i2 * b.i2, .i3 = a.i3 * b.i3 };
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
    #elif defined SIMD_GRANODI_NEON
    return vmulq_s32(a, b);
    #endif
}

static inline sg_pi64 sg_mul_pi64(const sg_pi64 a, const sg_pi64 b) {
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    return sg_set_fromg_pi64((sg_generic_pi64) { .l0 = ag.l0 * bg.l0,
        .l1 = ag.l1 * bg.l1 });
}

static inline sg_ps sg_mul_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = a.f0 * b.f0, .f1 = a.f1 * b.f1,
        .f2 = a.f2 * b.f2, .f3 = a.f3 * b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_mul_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vmulq_f32(a, b);
    #endif
}

static inline sg_pd sg_mul_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = a.d0 * b.d0, .d1 = a.d1 * b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_mul_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vmulq_f64(a, b);
    #endif
}

static inline sg_pi32 sg_div_pi32(const sg_pi32 a, const sg_pi32 b) {
    const sg_generic_pi32 ag = sg_getg_pi32(a), bg = sg_getg_pi32(b);
    return sg_set_fromg_pi32((sg_generic_pi32) { .i0 = ag.i0 / bg.i0,
        .i1 = ag.i1 / bg.i1, .i2 = ag.i2 / bg.i2, .i3 = ag.i3 / bg.i3 });
}

static inline sg_pi64 sg_div_pi64(const sg_pi64 a, const sg_pi64 b) {
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    return sg_set_fromg_pi64((sg_generic_pi64) { .l0 = ag.l0 / bg.l0,
        .l1 = ag.l1 / bg.l1 });
}

static inline sg_ps sg_div_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = a.f0 / b.f0, .f1 = a.f1 / b.f1,
        .f2 = a.f2 / b.f2, .f3 = a.f3 / b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_div_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vdivq_f32(a, b);
    #endif
}

static inline sg_pd sg_div_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = a.d0 / b.d0, .d1 = a.d1 / b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_div_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vdivq_f64(a, b);
    #endif
}

// Bitwise logic

#ifdef SIMD_GRANODI_SSE2
static inline __m128i sg_sse2_not_si128(const __m128i a) {
    return _mm_andnot_si128(a, sg_sse2_allset_si128);
}

static inline __m128 sg_sse2_not_ps(const __m128 a) {
    return _mm_andnot_ps(a, sg_sse2_allset_ps);
}

static inline __m128d sg_sse2_not_pd(const __m128d a) {
    return _mm_andnot_pd(a, sg_sse2_allset_pd);
}
#elif defined SIMD_GRANODI_NEON
static inline uint64x2_t sg_neon_not_u64(const uint64x2_t a) {
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(a)));
}

static inline int64x2_t sg_neon_not_pi64(const int64x2_t a) {
    return vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(a)));
}
#endif

static inline sg_pi32 sg_and_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = a.i0 & b.i0, .i1 = a.i1 & b.i1,
        .i2 = a.i2 & b.i2, .i3 = a.i3 & b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_si128(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vandq_s32(a, b);
    #endif
}

static inline sg_pi64 sg_and_pi64(const sg_pi64 a, const sg_pi64 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = a.l0 & b.l0, .l1 = a.l1 & b.l1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_si128(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vandq_s64(a, b);
    #endif
}

static inline sg_ps sg_and_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi32_ps(sg_and_pi32(sg_cast_ps_pi32(a), sg_cast_ps_pi32(b)));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(a),
        vreinterpretq_s32_f32(b)));
    #endif
}

static inline sg_pd sg_and_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi64_pd(sg_and_pi64(sg_cast_pd_pi64(a), sg_cast_pd_pi64(b)));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f64_s64(vandq_s64(vreinterpretq_s64_f64(a),
        vreinterpretq_s64_f64(b)));
    #endif
}

static inline sg_pi32 sg_andnot_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = ~a.i0 & b.i0, .i1 = ~a.i1 & b.i1,
        .i2 = ~a.i2 & b.i2, .i3 = ~a.i3 & b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_andnot_si128(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vbicq_s32(b, a);
    #endif
}

static inline sg_pi64 sg_andnot_pi64(const sg_pi64 a, const sg_pi64 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = ~a.l0 & b.l0, .l1 = ~a.l1 & b.l1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_andnot_si128(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vbicq_s64(b, a);
    #endif
}

static inline sg_ps sg_andnot_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi32_ps(
        sg_andnot_pi32(sg_cast_ps_pi32(a), sg_cast_ps_pi32(b)));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_andnot_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f32_s32(vbicq_s32(vreinterpretq_s32_f32(b),
        vreinterpretq_s32_f32(a)));
    #endif
}

static inline sg_pd sg_andnot_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi64_pd(
        sg_andnot_pi64(sg_cast_pd_pi64(a), sg_cast_pd_pi64(b)));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_andnot_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f64_s64(vbicq_s64(vreinterpretq_s64_f64(b),
        vreinterpretq_s64_f64(a)));
    #endif
}

static inline sg_pi32 sg_not_pi32(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) {.i0 = ~a.i0, .i1 = ~a.i1,
        .i2 = ~a.i2, .i3 = ~a.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return sg_sse2_not_si128(a);
    #elif defined SIMD_GRANODI_NEON
    return vmvnq_s32(a);
    #endif
}

static inline sg_pi64 sg_not_pi64(const sg_pi64 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = ~a.l0, .l1 = ~a.l1 };
    #elif defined SIMD_GRANODI_SSE2
    return sg_sse2_not_si128(a);
    #elif defined SIMD_GRANODI_NEON
    return sg_neon_not_pi64(a);
    #endif
}

static inline sg_ps sg_not_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi32_ps(sg_not_pi32(sg_cast_ps_pi32(a)));
    #elif defined SIMD_GRANODI_SSE2
    return sg_sse2_not_ps(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f32_s32(vmvnq_s32(vreinterpretq_s32_f32(a)));
    #endif
}

static inline sg_pd sg_not_pd(const sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi64_pd(sg_not_pi64(sg_cast_pd_pi64(a)));
    #elif defined SIMD_GRANODI_SSE2
    return sg_sse2_not_pd(a);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f64_s32(vmvnq_s32(vreinterpretq_s32_f64(a)));
    #endif
}

static inline sg_pi32 sg_or_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = a.i0 | b.i0, .i1 = a.i1 | b.i1,
        .i2 = a.i2 | b.i2, .i3 = a.i3 | b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_si128(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vorrq_s32(a, b);
    #endif
}

static inline sg_pi64 sg_or_pi64(const sg_pi64 a, const sg_pi64 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = a.l0 | b.l0, .l1 = a.l1 | b.l1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_si128(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vorrq_s64(a, b);
    #endif
}

static inline sg_ps sg_or_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi32_ps(sg_or_pi32(sg_cast_ps_pi32(a), sg_cast_ps_pi32(b)));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(a),
        vreinterpretq_s32_f32(b)));
    #endif
}

static inline sg_pd sg_or_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi64_pd(sg_or_pi64(sg_cast_pd_pi64(a), sg_cast_pd_pi64(b)));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f64_s64(vorrq_s64(vreinterpretq_s64_f64(a),
        vreinterpretq_s64_f64(b)));
    #endif
}

static inline sg_pi32 sg_xor_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = a.i0 ^ b.i0, .i1 = a.i1 ^ b.i1,
        .i2 = a.i2 ^ b.i2, .i3 = a.i3 ^ b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_xor_si128(a, b);
    #elif defined SIMD_GRANODI_NEON
    return veorq_s32(a, b);
    #endif
}

static inline sg_pi64 sg_xor_pi64(const sg_pi64 a, const sg_pi64 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = a.l0 ^ b.l0, .l1 = a.l1 ^ b.l1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_xor_si128(a, b);
    #elif defined SIMD_GRANODI_NEON
    return veorq_s64(a, b);
    #endif
}

static inline sg_ps sg_xor_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi32_ps(sg_xor_pi32(sg_cast_ps_pi32(a), sg_cast_ps_pi32(b)));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_xor_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(a),
        vreinterpretq_s32_f32(b)));
    #endif
}

static inline sg_pd sg_xor_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cast_pi64_pd(sg_xor_pi64(sg_cast_pd_pi64(a), sg_cast_pd_pi64(b)));
    #elif defined SIMD_GRANODI_SSE2
    return _mm_xor_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f64_s64(veorq_s64(vreinterpretq_s64_f64(a),
        vreinterpretq_s64_f64(b)));
    #endif
}

// Shift

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_sl_imm_pi32(const sg_pi32 a, const int32_t shift) {
    return (sg_pi32) { .i0 = a.i0 << shift, .i1 = a.i1 << shift,
        .i2 = a.i2 << shift, .i3 = a.i3 << shift };
}
#elif defined SIMD_GRANODI_SSE2
#define sg_sl_imm_pi32 _mm_slli_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_sl_imm_pi32 vshlq_n_s32
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_sl_imm_pi64(const sg_pi64 a, const int32_t shift) {
    return (sg_pi64) { .l0 = a.l0 << (int64_t) shift,
        .l1 = a.l1 << (int64_t) shift };
}
#elif defined SIMD_GRANODI_SSE2
#define sg_sl_imm_pi64 _mm_slli_epi64
#elif defined SIMD_GRANODI_NEON
#define sg_sl_imm_pi64 vshlq_n_s64
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_srl_imm_pi32(const sg_pi32 a, const int32_t shift) {
    return (sg_pi32) {
        .i0 = sg_scast_u32_s32((sg_scast_s32_u32(a.i0) >> shift)),
        .i1 = sg_scast_u32_s32((sg_scast_s32_u32(a.i1) >> shift)),
        .i2 = sg_scast_u32_s32((sg_scast_s32_u32(a.i2) >> shift)),
        .i3 = sg_scast_u32_s32((sg_scast_s32_u32(a.i3) >> shift)) };
}
#elif defined SIMD_GRANODI_SSE2
#define sg_srl_imm_pi32 _mm_srli_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_srl_imm_pi32(a, shift) vreinterpretq_s32_u32( \
    vshrq_n_u32(vreinterpretq_u32_s32(a), (shift)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi64 sg_srl_imm_pi64(const sg_pi64 a, const int32_t shift) {
    return (sg_pi64){
        .l0 = sg_scast_u64_s64(sg_scast_s64_u64(a.l0) >> (uint64_t) shift),
        .l1 = sg_scast_u64_s64(sg_scast_s64_u64(a.l1) >> (uint64_t) shift) };
}
#elif defined SIMD_GRANODI_SSE2
#define sg_srl_imm_pi64 _mm_srli_epi64
#elif defined SIMD_GRANODI_NEON
#define sg_srl_imm_pi64(a, shift) vreinterpretq_s64_u64( \
    vshrq_n_u64(vreinterpretq_u64_s64(a), (shift)))
#endif

#ifdef SIMD_GRANODI_FORCE_GENERIC
static inline sg_pi32 sg_sra_imm_pi32(const sg_pi32 a, const int32_t shift) {
    return (sg_pi32) { .i0 = a.i0 >> shift, .i1 = a.i1 >> shift,
        .i2 = a.i2 >> shift, .i3 = a.i3 >> shift };
}
#elif defined SIMD_GRANODI_SSE2
#define sg_sra_imm_pi32 _mm_srai_epi32
#elif defined SIMD_GRANODI_NEON
#define sg_sra_imm_pi32 vshrq_n_s32
#endif

#if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
static inline sg_pi64 sg_sra_imm_pi64(const sg_pi64 a,
    const int32_t shift_compile_time_constant)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = a.l0 >> (int64_t) shift_compile_time_constant,
        .l1 = a.l1 >> (int64_t) shift_compile_time_constant };
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

static inline sg_cmp_pi32 sg_setzero_cmp_pi32() {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp4) { .b0 = false, .b1 = false,
        .b2 = false, .b3 = false };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_setzero_si128();
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_u32(0);
    #endif
}

static inline sg_cmp_pi64 sg_setzero_cmp_pi64() {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp2) { .b0 = false, .b1 = false };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_setzero_si128();
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_u64(0);
    #endif
}

static inline sg_cmp_ps sg_setzero_cmp_ps() {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp4) { .b0 = false, .b1 = false,
        .b2 = false, .b3 = false };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_setzero_ps();
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_u32(0);
    #endif
}

static inline sg_cmp_pd sg_setzero_cmp_pd() {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp2) { .b0 = false, .b1 = false };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_setzero_pd();
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_u64(0);
    #endif
}

static inline sg_cmp_pi32 sg_set1cmp_pi32(const bool b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp4) { .b0 = b, .b1 = b, .b2 = b, .b3 = b };
    #elif defined SIMD_GRANODI_SSE2
    return b ? sg_sse2_allset_si128 : _mm_setzero_si128();
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_u32(b ? sg_allset_u32 : 0);
    #endif
}

static inline sg_cmp_pi64 sg_set1cmp_pi64(const bool b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp2) { .b0 = b, .b1 = b};
    #elif defined SIMD_GRANODI_SSE2
    return b ? sg_sse2_allset_si128 : _mm_setzero_si128();
    #elif defined SIMD_GRANODI_NEON
    return vdupq_n_u64(b ? sg_allset_u64 : 0);
    #endif
}

static inline sg_cmp_ps sg_set1cmp_ps(const bool b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
    return sg_set1cmp_pi32(b);
    #elif defined SIMD_GRANODI_SSE2
    return b ? sg_sse2_allset_ps : _mm_setzero_ps();
    #endif
}

static inline sg_cmp_pd sg_set1cmp_pd(const bool b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
    return sg_set1cmp_pi64(b);
    #elif defined SIMD_GRANODI_SSE2
    return b ? sg_sse2_allset_pd : _mm_setzero_pd();
    #endif
}

static inline sg_cmp_pi32 sg_setcmp_pi32(const bool b3, const bool b2,
    const bool b1, const bool b0)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp4) { .b0 = b0, .b1 = b1, .b2 = b2, .b3 = b3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set_epi32(b3 ? sg_allset_s32 : 0,
        b2 ? sg_allset_s32 : 0,
        b1 ? sg_allset_s32 : 0,
        b0 ? sg_allset_s32 : 0);
    #elif defined SIMD_GRANODI_NEON
    uint32x4_t result = vdupq_n_u32(0);
    result = vsetq_lane_u32(b0 ? sg_allset_u32 : 0, result, 0);
    result = vsetq_lane_u32(b1 ? sg_allset_u32 : 0, result, 1);
    result = vsetq_lane_u32(b2 ? sg_allset_u32 : 0, result, 2);
    return vsetq_lane_u32(b3 ? sg_allset_u32 : 0, result, 3);
    #endif
}

static inline sg_cmp_pi32 sg_setcmp_fromg_pi32(const sg_generic_cmp4 cmpg) {
    return sg_setcmp_pi32(cmpg.b3, cmpg.b2, cmpg.b1, cmpg.b0);
}

static inline sg_cmp_pi64 sg_setcmp_pi64(const bool b1, const bool b0) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp2) { .b0 = b0, .b1 = b1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_set_epi64x(b1 ? sg_allset_s64 : 0, b0 ? sg_allset_s64 : 0);
    #elif defined SIMD_GRANODI_NEON
    uint64x2_t result = vdupq_n_u64(0);
    result = vsetq_lane_u64(b0 ? sg_allset_u64 : 0, result, 0);
    return vsetq_lane_u64(b1 ? sg_allset_u64 : 0, result, 1);
    #endif
}

static inline sg_cmp_pi64 sg_setcmp_fromg_pi64(const sg_generic_cmp2 cmpg) {
    return sg_setcmp_pi64(cmpg.b1, cmpg.b0);
}

static inline sg_cmp_ps sg_setcmp_ps(const bool b3, const bool b2,
    const bool b1, const bool b0)
{
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
    return sg_setcmp_pi32(b3, b2, b1, b0);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castsi128_ps(sg_setcmp_pi32(b3, b2, b1, b0));
    #endif
}

static inline sg_cmp_ps sg_setcmp_fromg_ps(const sg_generic_cmp4 cmpg) {
    return sg_setcmp_ps(cmpg.b3, cmpg.b2, cmpg.b1, cmpg.b0);
}

static inline sg_cmp_pd sg_setcmp_pd(const bool b1, const bool b0) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp2) { .b0 = b0, .b1 = b1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castsi128_pd(sg_setcmp_pi64(b1, b0));
    #elif defined SIMD_GRANODI_NEON
    return sg_setcmp_pi64(b1, b0);
    #endif
}

static inline sg_cmp_pd sg_setcmp_fromg_pd(const sg_generic_cmp2 cmpg) {
    return sg_setcmp_pd(cmpg.b1, cmpg.b0);
}

#ifndef SIMD_GRANODI_FORCE_GENERIC

static inline uint32_t sg_cmp_getmask0_pi32(const sg_cmp_pi32 cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_s32_u32(sg_get0_pi32(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u32(cmp, 0);
    #endif
}

static inline uint32_t sg_cmp_getmask1_pi32(const sg_cmp_pi32 cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_s32_u32(sg_get1_pi32(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u32(cmp, 1);
    #endif
}

static inline uint32_t sg_cmp_getmask2_pi32(const sg_cmp_pi32 cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_s32_u32(sg_get2_pi32(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u32(cmp, 2);
    #endif
}

static inline uint32_t sg_cmp_getmask3_pi32(const sg_cmp_pi32 cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_s32_u32(sg_get3_pi32(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u32(cmp, 3);
    #endif
}

static inline uint64_t sg_cmp_getmask0_pi64(const sg_cmp_pi64 cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_s64_u64(sg_get0_pi64(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u64(cmp, 0);
    #endif
}

static inline uint64_t sg_cmp_getmask1_pi64(const sg_cmp_pi64 cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_s64_u64(sg_get1_pi64(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u64(cmp, 1);
    #endif
}

static inline uint32_t sg_cmp_getmask0_ps(const sg_cmp_ps cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_f32_u32(sg_get0_ps(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u32(cmp, 0);
    #endif
}

static inline uint32_t sg_cmp_getmask1_ps(const sg_cmp_ps cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_f32_u32(sg_get1_ps(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u32(cmp, 1);
    #endif
}

static inline uint32_t sg_cmp_getmask2_ps(const sg_cmp_ps cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_f32_u32(sg_get2_ps(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u32(cmp, 2);
    #endif
}

static inline uint32_t sg_cmp_getmask3_ps(const sg_cmp_ps cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_f32_u32(sg_get3_ps(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u32(cmp, 3);
    #endif
}

static inline uint64_t sg_cmp_getmask0_pd(const sg_cmp_pd cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_f64_u64(sg_get0_pd(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u64(cmp, 0);
    #endif
}

static inline uint64_t sg_cmp_getmask1_pd(const sg_cmp_pd cmp) {
    #ifdef SIMD_GRANODI_SSE2
    return sg_scast_f64_u64(sg_get1_pd(cmp));
    #elif defined SIMD_GRANODI_NEON
    return vgetq_lane_u64(cmp, 1);
    #endif
}

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
    #elif defined SIMD_GRANODI_SSE2 || defined SIMD_GRANODI_NEON
    return sg_debug_mask_valid_eq_u32(sg_cmp_getmask3_pi32(cmp), b3) &&
        sg_debug_mask_valid_eq_u32(sg_cmp_getmask2_pi32(cmp), b2) &&
        sg_debug_mask_valid_eq_u32(sg_cmp_getmask1_pi32(cmp), b1) &&
        sg_debug_mask_valid_eq_u32(sg_cmp_getmask0_pi32(cmp), b0);
    #endif
}

static inline bool sg_debug_cmp_valid_eq_pi64(const sg_cmp_pi64 cmp,
    const bool b1, const bool b0)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return cmp.b1 == b1 && cmp.b0 == b0;
    #elif defined SIMD_GRANODI_SSE2 || defined SIMD_GRANODI_NEON
    return sg_debug_mask_valid_eq_u64(sg_cmp_getmask1_pi64(cmp), b1) &&
        sg_debug_mask_valid_eq_u64(sg_cmp_getmask0_pi64(cmp), b0);
    #endif
}

static inline bool sg_debug_cmp_valid_eq_ps(const sg_cmp_ps cmp,
    const bool b3, const bool b2, const bool b1, const bool b0)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return cmp.b3 == b3 && cmp.b2 == b2 && cmp.b1 == b1 && cmp.b0 == b0;
    #elif defined SIMD_GRANODI_SSE2 || defined SIMD_GRANODI_NEON
    return sg_debug_mask_valid_eq_u32(sg_cmp_getmask3_ps(cmp), b3) &&
        sg_debug_mask_valid_eq_u32(sg_cmp_getmask2_ps(cmp), b2) &&
        sg_debug_mask_valid_eq_u32(sg_cmp_getmask1_ps(cmp), b1) &&
        sg_debug_mask_valid_eq_u32(sg_cmp_getmask0_ps(cmp), b0);
    #endif
}

static inline bool sg_debug_cmp_valid_eq_pd(const sg_cmp_pd cmp,
    const bool b1, const bool b0)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return cmp.b1 == b1 && cmp.b0 == b0;
    #elif defined SIMD_GRANODI_SSE2 || defined SIMD_GRANODI_NEON
    return sg_debug_mask_valid_eq_u64(sg_cmp_getmask1_pd(cmp), b1) &&
        sg_debug_mask_valid_eq_u64(sg_cmp_getmask0_pd(cmp), b0);
    #endif
}

static inline sg_cmp_pi32 sg_cmplt_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi32) { .b0 = a.i0 < b.i0, .b1 = a.i1 < b.i1,
        .b2 = a.i2 < b.i2, .b3 = a.i3 < b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmplt_epi32(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcltq_s32(a, b);
    #endif
}

static inline sg_cmp_pi64 sg_cmplt_pi64(const sg_pi64 a, const sg_pi64 b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    return sg_setcmp_fromg_pi64((sg_generic_cmp2) { .b0 = ag.l0 < bg.l0,
        .b1 = ag.l1 < bg.l1 });
    #elif defined SIMD_GRANODI_NEON
    return vcltq_s64(a, b);
    #endif
}

static inline sg_cmp_ps sg_cmplt_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_ps) { .b0 = a.f0 < b.f0, .b1 = a.f1 < b.f1,
        .b2 = a.f2 < b.f2, .b3 = a.f3 < b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmplt_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcltq_f32(a, b);
    #endif
}

static inline sg_cmp_pd sg_cmplt_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pd) { .b0 = a.d0 < b.d0, .b1 = a.d1 < b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmplt_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcltq_f64(a, b);
    #endif
}

static inline sg_cmp_pi32 sg_cmplte_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi32) { .b0 = a.i0 <= b.i0, .b1 = a.i1 <= b.i1,
        .b2 = a.i2 <= b.i2, .b3 = a.i3 <= b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, b));
    #elif defined SIMD_GRANODI_NEON
    return vcleq_s32(a, b);
    #endif
}

static inline sg_cmp_pi64 sg_cmplte_pi64(const sg_pi64 a, const sg_pi64 b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    return sg_setcmp_fromg_pi64((sg_generic_cmp2) {
        .b0 = ag.l0 <= bg.l0, .b1 = ag.l1 <= bg.l1 });
    #elif defined SIMD_GRANODI_NEON
    return vcleq_s64(a, b);
    #endif
}

static inline sg_cmp_ps sg_cmplte_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_ps) { .b0 = a.f0 <= b.f0, .b1 = a.f1 <= b.f1,
        .b2 = a.f2 <= b.f2, .b3 = a.f3 <= b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmple_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcleq_f32(a, b);
    #endif
}

static inline sg_cmp_pd sg_cmplte_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pd) { .b0 = a.d0 <= b.d0, .b1 = a.d1 <= b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmple_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcleq_f64(a, b);
    #endif
}

static inline sg_cmp_pi32 sg_cmpeq_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi32) { .b0 = a.i0 == b.i0, .b1 = a.i1 == b.i1,
        .b2 = a.i2 == b.i2, .b3 = a.i3 == b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpeq_epi32(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vceqq_s32(a, b);
    #endif
}

static inline sg_cmp_pi64 sg_cmpeq_pi64(const sg_pi64 a, const sg_pi64 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp2) { .b0 = a.l0 == b.l0, .b1 = a.l1 == b.l1 };
    #elif defined SIMD_GRANODI_SSE2
    const __m128i eq_epi32 = _mm_cmpeq_epi32(a, b);
    return _mm_and_si128(eq_epi32,
        _mm_shuffle_epi32(eq_epi32, sg_sse2_shuffle32_imm(2, 3, 0, 1)));
    #elif defined SIMD_GRANODI_NEON
    return vceqq_s64(a, b);
    #endif
}

static inline sg_cmp_ps sg_cmpeq_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_ps) { .b0 = a.f0 == b.f0, .b1 = a.f1 == b.f1,
        .b2 = a.f2 == b.f2, .b3 = a.f3 == b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpeq_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vceqq_f32(a, b);
    #endif
}

static inline sg_cmp_pd sg_cmpeq_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pd) { .b0 = a.d0 == b.d0, .b1 = a.d1 == b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpeq_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vceqq_f64(a, b);
    #endif
}

static inline sg_cmp_pi32 sg_cmpneq_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi32) { .b0 = a.i0 != b.i0,
        .b1 = a.i1 != b.i1,
        .b2 = a.i2 != b.i2,
        .b3 = a.i3 != b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return sg_sse2_not_si128(_mm_cmpeq_epi32(a, b));
    #elif defined SIMD_GRANODI_NEON
    return vmvnq_u32(vceqq_s32(a, b));
    #endif
}

static inline sg_cmp_pi64 sg_cmpneq_pi64(const sg_pi64 a, const sg_pi64 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp2) { .b0 = a.l0 != b.l0, .b1 = a.l1 != b.l1 };
    #elif defined SIMD_GRANODI_SSE2
    return sg_sse2_not_si128(sg_cmpeq_pi64(a, b));
    #elif defined SIMD_GRANODI_NEON
    return sg_neon_not_u64(vceqq_s64(a, b));
    #endif
}

static inline sg_cmp_ps sg_cmpneq_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_ps) { .b0 = a.f0 != b.f0,
        .b1 = a.f1 != b.f1,
        .b2 = a.f2 != b.f2,
        .b3 = a.f3 != b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpneq_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vmvnq_u32(vceqq_f32(a, b));
    #endif
}

static inline sg_cmp_pd sg_cmpneq_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pd) { .b0 = a.d0 != b.d0,
        .b1 = a.d1 != b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpneq_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return sg_neon_not_u64(vceqq_f64(a, b));
    #endif
}

static inline sg_cmp_pi32 sg_cmpgte_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi32) { .b0 = a.i0 >= b.i0, .b1 = a.i1 >= b.i1,
        .b2 = a.i2 >= b.i2, .b3 = a.i3 >= b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, b));
    #elif defined SIMD_GRANODI_NEON
    return vcgeq_s32(a, b);
    #endif
}

static inline sg_cmp_pi64 sg_cmpgte_pi64(const sg_pi64 a, const sg_pi64 b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    return sg_setcmp_fromg_pi64((sg_generic_cmp2) {
        .b0 = ag.l0 >= bg.l0, .b1 = ag.l1 >= bg.l1 });
    #elif defined SIMD_GRANODI_NEON
    return vcgeq_s64(a, b);
    #endif
}

static inline sg_cmp_ps sg_cmpgte_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_ps) { .b0 = a.f0 >= b.f0, .b1 = a.f1 >= b.f1,
        .b2 = a.f2 >= b.f2, .b3 = a.f3 >= b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpge_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcgeq_f32(a, b);
    #endif
}

static inline sg_cmp_pd sg_cmpgte_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pd) { .b0 = a.d0 >= b.d0, .b1 = a.d1 >= b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpge_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcgeq_f64(a, b);
    #endif
}

static inline sg_cmp_pi32 sg_cmpgt_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi32) { .b0 = a.i0 > b.i0, .b1 = a.i1 > b.i1,
        .b2 = a.i2 > b.i2, .b3 = a.i3 > b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpgt_epi32(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcgtq_s32(a, b);
    #endif
}

static inline sg_cmp_pi64 sg_cmpgt_pi64(const sg_pi64 a, const sg_pi64 b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    return sg_setcmp_fromg_pi64((sg_generic_cmp2) { .b0 = ag.l0 > bg.l0,
        .b1 = ag.l1 > bg.l1 });
    #elif defined SIMD_GRANODI_NEON
    return vcgtq_s64(a, b);
    #endif
}

static inline sg_cmp_ps sg_cmpgt_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_ps) { .b0 = a.f0 > b.f0, .b1 = a.f1 > b.f1,
        .b2 = a.f2 > b.f2, .b3 = a.f3 > b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpgt_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcgtq_f32(a, b);
    #endif
}

static inline sg_cmp_pd sg_cmpgt_pd(const sg_pd a, const sg_pd b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pd) { .b0 = a.d0 > b.d0, .b1 = a.d1 > b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_cmpgt_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vcgtq_f64(a, b);
    #endif
}

// Cast comparisons (same layout, no conversion)

static inline sg_cmp_ps sg_castcmp_pi32_ps(const sg_cmp_pi32 cmp) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
    return cmp;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castsi128_ps(cmp);
    #endif
}

static inline sg_cmp_pi32 sg_castcmp_ps_pi32(const sg_cmp_ps cmp) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
    return cmp;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castps_si128(cmp);
    #endif
}

static inline sg_cmp_pd sg_castcmp_pi64_pd(const sg_cmp_pi64 cmp) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
    return cmp;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castsi128_pd(cmp);
    #endif
}

static inline sg_cmp_pi64 sg_castcmp_pd_pi64(const sg_cmp_pd cmp) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_NEON
    return cmp;
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castpd_si128(cmp);
    #endif
}

// Convert comparison results (shuffling)

static inline sg_cmp_pi64 sg_cvtcmp_pi32_pi64(const sg_cmp_pi32 cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_cmp2) { .b0 = cmp.b0, .b1 = cmp.b1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_shuffle_epi32(cmp, sg_sse2_shuffle32_imm(1, 1, 0, 0));
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_u64_u32(vzip1q_u32(cmp, cmp));
    #endif
}

static inline sg_cmp_pd sg_cvtcmp_pi32_pd(const sg_cmp_pi32 cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cvtcmp_pi32_pi64(cmp);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castsi128_pd(sg_cvtcmp_pi32_pi64(cmp));
    #elif defined SIMD_GRANODI_NEON
    return sg_cvtcmp_pi32_pi64(cmp);
    #endif
}

static inline sg_cmp_pd sg_cvtcmp_ps_pd(const sg_cmp_ps cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cvtcmp_pi32_pd(cmp);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castps_pd(
        _mm_shuffle_ps(cmp, cmp, sg_sse2_shuffle32_imm(1, 1, 0, 0)));
    #elif defined SIMD_GRANODI_NEON
    return sg_cvtcmp_pi32_pi64(cmp);
    #endif
}

static inline sg_cmp_pi64 sg_cvtcmp_ps_pi64(const sg_cmp_ps cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cvtcmp_pi32_pi64(cmp);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castps_si128(
        _mm_shuffle_ps(cmp, cmp, sg_sse2_shuffle32_imm(1, 1, 0, 0)));
    #elif defined SIMD_GRANODI_NEON
    return sg_cvtcmp_pi32_pi64(cmp);
    #endif
}

static inline sg_cmp_pi32 sg_cvtcmp_pi64_pi32(const sg_cmp_pi64 cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi32) { .b0 = cmp.b0, .b1 = cmp.b1, .b2 = 0, .b3 = 0 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_si128(_mm_set_epi32(0, 0, sg_allset_s32, sg_allset_s32),
        _mm_shuffle_epi32(cmp, sg_sse2_shuffle32_imm(3, 2, 3, 0)));
    #elif defined SIMD_GRANODI_NEON
    const uint32x4_t cmp_u32 = vreinterpretq_u32_u64(cmp);
    return vuzp1q_u32(cmp_u32, vdupq_n_u32(0));
    #endif
}

static inline sg_cmp_ps sg_cvtcmp_pi64_ps(const sg_cmp_pi64 cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cvtcmp_pi64_pi32(cmp);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castsi128_ps(sg_cvtcmp_pi64_pi32(cmp));
    #elif defined SIMD_GRANODI_NEON
    return sg_cvtcmp_pi64_pi32(cmp);
    #endif
}

static inline sg_cmp_ps sg_cvtcmp_pd_ps(const sg_cmp_pd cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cvtcmp_pi64_pi32(cmp);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_shuffle_ps(_mm_castpd_ps(cmp),
        _mm_setzero_ps(),
        sg_sse2_shuffle32_imm(3, 2, 3, 0));
    #elif defined SIMD_GRANODI_NEON
    return sg_cvtcmp_pi64_pi32(cmp);
    #endif
}

static inline sg_cmp_pi32 sg_cvtcmp_pd_pi32(const sg_cmp_pd cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_cvtcmp_pi64_pi32(cmp);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_castps_si128(sg_cvtcmp_pd_ps(cmp));
    #elif defined SIMD_GRANODI_NEON
    return sg_cvtcmp_pi64_pi32(cmp);
    #endif
}

// Combine comparisons (bitwise operations repeated if masks)

static inline sg_cmp_pi32 sg_and_cmp_pi32(const sg_cmp_pi32 cmpa,
    const sg_cmp_pi32 cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi32) { .b0 = cmpa.b0 && cmpb.b0,
        .b1 = cmpa.b1 && cmpb.b1, .b2 = cmpa.b2 && cmpb.b2,
        .b3 = cmpa.b3 && cmpb.b3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_si128(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vandq_u32(cmpa, cmpb);
    #endif
}

static inline sg_cmp_pi64 sg_and_cmp_pi64(const sg_cmp_pi64 cmpa,
    const sg_cmp_pi64 cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi64) { .b0 = cmpa.b0 && cmpb.b0,
        .b1 = cmpa.b1 && cmpb.b1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_si128(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vandq_u64(cmpa, cmpb);
    #endif
}

static inline sg_cmp_ps sg_and_cmp_ps(const sg_cmp_ps cmpa,
    const sg_cmp_ps cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_and_cmp_pi32(cmpa, cmpb);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_ps(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vandq_u32(cmpa, cmpb);
    #endif
}

static inline sg_cmp_pd sg_and_cmp_pd(const sg_cmp_pd cmpa,
    const sg_cmp_pd cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pd) { .b0 = cmpa.b0 && cmpb.b0,
        .b1 = cmpa.b1 && cmpb.b1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_pd(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vandq_u64(cmpa, cmpb);
    #endif
}

static inline sg_cmp_pi32 sg_andnot_cmp_pi32(const sg_cmp_pi32 cmpa,
    const sg_cmp_pi32 cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi32) { .b0 = !cmpa.b0 && cmpb.b0,
        .b1 = !cmpa.b1 && cmpb.b1, .b2 = !cmpa.b2 && cmpb.b2,
        .b3 = !cmpa.b3 && cmpb.b3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_andnot_si128(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vbicq_u32(cmpb, cmpa);
    #endif
}

static inline sg_cmp_pi64 sg_andnot_cmp_pi64(const sg_cmp_pi64 cmpa,
    const sg_cmp_pi64 cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi64) { .b0 = !cmpa.b0 && cmpb.b0,
        .b1 = !cmpa.b1 && cmpb.b1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_andnot_si128(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vbicq_u64(cmpb, cmpa);
    #endif
}

static inline sg_cmp_ps sg_andnot_cmp_ps(const sg_cmp_ps cmpa, sg_cmp_ps cmpb) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_andnot_cmp_pi32(cmpa, cmpb);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_andnot_ps(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vbicq_u32(cmpb, cmpa);
    #endif
}

static inline sg_cmp_pd sg_andnot_cmp_pd(const sg_cmp_pd cmpa, sg_cmp_pd cmpb) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pd) { .b0 = !cmpa.b0 && cmpb.b0,
        .b1 = !cmpa.b1 && cmpb.b1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_andnot_pd(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vbicq_u64(cmpb, cmpa);
    #endif
}

static inline sg_cmp_pi32 sg_not_cmp_pi32(const sg_cmp_pi32 cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi32) { .b0 = !cmp.b0, .b1 = !cmp.b1,
        .b2 = !cmp.b2, .b3 = !cmp.b3 };
    #elif defined SIMD_GRANODI_SSE2
    return sg_sse2_not_si128(cmp);
    #elif defined SIMD_GRANODI_NEON
    return vmvnq_u32(cmp);
    #endif
}

static inline sg_cmp_pi64 sg_not_cmp_pi64(const sg_cmp_pi64 cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi64) { .b0 = !cmp.b0, .b1 = !cmp.b1 };
    #elif defined SIMD_GRANODI_SSE2
    return sg_sse2_not_si128(cmp);
    #elif defined SIMD_GRANODI_NEON
    return sg_neon_not_u64(cmp);
    #endif
}

static inline sg_cmp_ps sg_not_cmp_ps(const sg_cmp_ps cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_not_cmp_pi32(cmp);
    #elif defined SIMD_GRANODI_SSE2
    return sg_sse2_not_ps(cmp);
    #elif defined SIMD_GRANODI_NEON
    return vmvnq_u32(cmp);
    #endif
}

static inline sg_cmp_pd sg_not_cmp_pd(const sg_cmp_pd cmp) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pd) { .b0 = !cmp.b0, .b1 = !cmp.b1 };
    #elif defined SIMD_GRANODI_SSE2
    return sg_sse2_not_pd(cmp);
    #elif defined SIMD_GRANODI_NEON
    return sg_neon_not_u64(cmp);
    #endif
}

static inline sg_cmp_pi32 sg_or_cmp_pi32(const sg_cmp_pi32 cmpa,
    const sg_cmp_pi32 cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi32) { .b0 = cmpa.b0 || cmpb.b0,
        .b1 = cmpa.b1 || cmpb.b1, .b2 = cmpa.b2 || cmpb.b2,
        .b3 = cmpa.b3 || cmpb.b3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_si128(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vorrq_u32(cmpa, cmpb);
    #endif
}

static inline sg_cmp_pi64 sg_or_cmp_pi64(const sg_cmp_pi64 cmpa,
    const sg_cmp_pi64 cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi64) { .b0 = cmpa.b0 || cmpb.b0,
        .b1 = cmpa.b1 || cmpb.b1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_si128(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vorrq_u64(cmpa, cmpb);
    #endif
}

static inline sg_cmp_ps sg_or_cmp_ps(const sg_cmp_ps cmpa,
    const sg_cmp_ps cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_or_cmp_pi32(cmpa, cmpb);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_ps(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vorrq_u32(cmpa, cmpb);
    #endif
}

static inline sg_cmp_pd sg_or_cmp_pd(const sg_cmp_pd cmpa,
    const sg_cmp_pd cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pd) { .b0 = cmpa.b0 || cmpb.b0,
        .b1 = cmpa.b1 || cmpb.b1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_pd(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return vorrq_u64(cmpa, cmpb);
    #endif
}

static inline sg_cmp_pi32 sg_xor_cmp_pi32(const sg_cmp_pi32 cmpa,
    const sg_cmp_pi32 cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    // the additional unary ! is unneccessary if comparison obtained correctly,
    // but used out of caution
    return (sg_cmp_pi32) { .b0 = !cmpa.b0 != !cmpb.b0,
        .b1 = !cmpa.b1 != !cmpb.b1, .b2 = !cmpa.b2 != !cmpb.b2,
        .b3 = !cmpa.b3 != !cmpb.b3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_xor_si128(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return veorq_u32(cmpa, cmpb);
    #endif
}

static inline sg_cmp_pi64 sg_xor_cmp_pi64(const sg_cmp_pi64 cmpa,
    const sg_cmp_pi64 cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_cmp_pi64) { .b0 = !cmpa.b0 != !cmpb.b0,
        .b1 = !cmpa.b1 != !cmpb.b1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_xor_si128(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return veorq_u64(cmpa, cmpb);
    #endif
}

static inline sg_cmp_ps sg_xor_cmp_ps(const sg_cmp_ps cmpa,
    const sg_cmp_ps cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_xor_cmp_pi32(cmpa, cmpb);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_xor_ps(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return veorq_u32(cmpa, cmpb);
    #endif
}

static inline sg_cmp_pd sg_xor_cmp_pd(const sg_cmp_pd cmpa,
    const sg_cmp_pd cmpb)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return sg_xor_cmp_pi64(cmpa, cmpb);
    #elif defined SIMD_GRANODI_SSE2
    return _mm_xor_pd(cmpa, cmpb);
    #elif defined SIMD_GRANODI_NEON
    return veorq_u64(cmpa, cmpb);
    #endif
}

// Choose / blend

static inline sg_pi32 sg_choose_pi32(const sg_cmp_pi32 cmp,
    const sg_pi32 if_true, const sg_pi32 if_false)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = cmp.b0 ? if_true.i0 : if_false.i0,
        .i1 = cmp.b1 ? if_true.i1 : if_false.i1,
        .i2 = cmp.b2 ? if_true.i2 : if_false.i2,
        .i3 = cmp.b3 ? if_true.i3 : if_false.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_si128(_mm_andnot_si128(cmp, if_false),
        _mm_and_si128(cmp, if_true));
    #elif defined SIMD_GRANODI_NEON
    return vbslq_s32(cmp, if_true, if_false);
    #endif
}

// More efficient special case
static inline sg_pi32 sg_choose_else_zero_pi32(const sg_cmp_pi32 cmp,
    const sg_pi32 if_true)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = cmp.b0 ? if_true.i0 : 0,
        .i1 = cmp.b1 ? if_true.i1 : 0, .i2 = cmp.b2 ? if_true.i2 : 0,
        .i3 = cmp.b3 ? if_true.i3 : 0 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_si128(cmp, if_true);
    #elif defined SIMD_GRANODI_NEON
    const uint32x4_t if_true_u32 = vreinterpretq_u32_s32(if_true);
    return vreinterpretq_s32_u32(vandq_u32(cmp, if_true_u32));
    #endif
}

static inline sg_pi64 sg_choose_pi64(const sg_cmp_pi64 cmp,
    const sg_pi64 if_true, const sg_pi64 if_false)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = cmp.b0 ? if_true.l0 : if_false.l0,
        .l1 = cmp.b1 ? if_true.l1 : if_false.l1 };
    #elif defined SIMD_GRANODI_SSE2
    return sg_choose_pi32(cmp, if_true, if_false);
    #elif defined SIMD_GRANODI_NEON
    return vbslq_s64(cmp, if_true, if_false);
    #endif
}

static inline sg_pi64 sg_choose_else_zero_pi64(const sg_cmp_pi64 cmp,
    const sg_pi64 if_true)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = cmp.b0 ? if_true.l0 : 0,
        .l1 = cmp.b1 ? if_true.l1 : 0 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_si128(cmp, if_true);
    #elif defined SIMD_GRANODI_NEON
    const uint64x2_t if_true_u64 = vreinterpretq_u64_s64(if_true);
    return vreinterpretq_s64_u64(vandq_u64(cmp, if_true_u64));
    #endif
}

static inline sg_ps sg_choose_ps(const sg_cmp_ps cmp,
    const sg_ps if_true, const sg_ps if_false)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) {. f0 = cmp.b0 ? if_true.f0 : if_false.f0,
        .f1 = cmp.b1 ? if_true.f1 : if_false.f1,
        .f2 = cmp.b2 ? if_true.f2 : if_false.f2,
        .f3 = cmp.b3 ? if_true.f3 : if_false.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_ps(_mm_andnot_ps(cmp, if_false),
        _mm_and_ps(cmp, if_true));
    #elif defined SIMD_GRANODI_NEON
    return vbslq_f32(cmp, if_true, if_false);
    #endif
}

static inline sg_ps sg_choose_else_zero_ps(const sg_cmp_ps cmp,
    const sg_ps if_true)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = cmp.b0 ? if_true.f0 : 0.0f,
        .f1 = cmp.b1 ? if_true.f1 : 0.0f,
        .f2 = cmp.b2 ? if_true.f2 : 0.0f,
        .f3 = cmp.b3 ? if_true.f3 : 0.0f };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_ps(cmp, if_true);
    #elif defined SIMD_GRANODI_NEON
    const uint32x4_t if_true_u32 = vreinterpretq_u32_f32(if_true);
    return vreinterpretq_f32_u32(vandq_u32(cmp, if_true_u32));
    #endif
}

static inline sg_pd sg_choose_pd(const sg_cmp_pd cmp,
    const sg_pd if_true, const sg_pd if_false)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = cmp.b0 ? if_true.d0 : if_false.d0,
        .d1 = cmp.b1 ? if_true.d1 : if_false.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_or_pd(_mm_andnot_pd(cmp, if_false),
        _mm_and_pd(cmp, if_true));
    #elif defined SIMD_GRANODI_NEON
    return vbslq_f64(cmp, if_true, if_false);
    #endif
}

static inline sg_pd sg_choose_else_zero_pd(const sg_cmp_pd cmp,
    const sg_pd if_true)
{
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = cmp.b0 ? if_true.d0 : 0.0,
        .d1 = cmp.b1 ? if_true.d1 : 0.0 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_pd(cmp, if_true);
    #elif defined SIMD_GRANODI_NEON
    const uint64x2_t if_true_u64 = vreinterpretq_u64_f64(if_true);
    return vreinterpretq_f64_u64(vandq_u64(cmp, if_true_u64));
    #endif
}

// Safe division: Divide by 1 if the denominator is 0.
// Quite often we want to divide by a number that could be zero, and then
// discard (via masking) the result of the division later when it turns out
// the denominator was zero. But before then, we want the division to happen
// safely: even if hardware exceptions are ignored, a slowdown could still
// happen.
static inline sg_pi32 sg_safediv_pi32(const sg_pi32 a, const sg_pi32 b) {
    const sg_generic_pi32 ag = sg_getg_pi32(a);
    sg_generic_pi32 b_safe = sg_getg_pi32(b);
    b_safe.i0 = b_safe.i0 != 0 ? b_safe.i0 : 1;
    b_safe.i1 = b_safe.i1 != 0 ? b_safe.i1 : 1;
    b_safe.i2 = b_safe.i2 != 0 ? b_safe.i2 : 1;
    b_safe.i3 = b_safe.i3 != 0 ? b_safe.i3 : 1;
    return sg_set_fromg_pi32((sg_generic_pi32) { .i0 = ag.i0 / b_safe.i0,
        .i1 = ag.i1 / b_safe.i1, .i2 = ag.i2 / b_safe.i2,
        .i3 = ag.i3 / b_safe.i3 });
}

static inline sg_pi64 sg_safediv_pi64(const sg_pi64 a, const sg_pi64 b) {
    const sg_generic_pi64 ag = sg_getg_pi64(a);
    sg_generic_pi64 b_safe = sg_getg_pi64(b);
    b_safe.l0 = b_safe.l0 != 0 ? b_safe.l0 : 1;
    b_safe.l1 = b_safe.l1 != 0 ? b_safe.l1 : 1;
    return sg_set_fromg_pi64((sg_generic_pi64) { .l0 = ag.l0 / b_safe.l0,
        .l1 = ag.l1 / b_safe.l1 });
}

static inline sg_ps sg_safediv_ps(const sg_ps a, const sg_ps b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    const sg_ps b_safe = { .f0 = b.f0 != 0.0f ? b.f0 : 1.0f,
        .f1 = b.f1 != 0.0f ? b.f1 : 1.0f,
        .f2 = b.f2 != 0.0f ? b.f2 : 1.0f,
        .f3 = b.f3 != 0.0f ? b.f3 : 1.0f };
    return (sg_generic_ps) { .f0 = a.f0 / b_safe.f0,
        .f1 = a.f1 / b_safe.f1,
        .f2 = a.f2 / b_safe.f2,
        .f3 = a.f3 / b_safe.f3 };
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
    const sg_pd b_safe = { .d0 = b.d0 != 0.0 ? b.d0 : 1.0,
        .d1 = b.d1 != 0.0 ? b.d1 : 1.0 };
    return (sg_generic_pd) { .d0 = a.d0 / b_safe.d0,
        .d1 = a.d1 / b_safe.d1 };
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

static inline sg_pi32 sg_abs_pi32(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = abs(a.i0), .i1 = abs(a.i1),
        .i2 = abs(a.i2), .i3 = abs(a.i3) };
    #elif defined SIMD_GRANODI_SSE2
    return sg_choose_pi32(_mm_cmplt_epi32(a, _mm_setzero_si128()),
        _mm_sub_epi32(_mm_setzero_si128(), a), a);
    #elif defined SIMD_GRANODI_NEON
    return vabsq_s32(a);
    #endif
}

static inline sg_pi64 sg_abs_pi64(const sg_pi64 a) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
    // Todo: SSE2 version could be implemented by comparing highest order
    // bit, then shuffling
    const sg_generic_pi64 ag = sg_getg_pi64(a);
    return sg_set_fromg_pi64((sg_generic_pi64) {
        .l0 = ag.l0 < 0 ? -ag.l0 : ag.l0,
        .l1 = ag.l1 < 0 ? -ag.l1 : ag.l1 });
    #elif defined SIMD_GRANODI_NEON
    return vabsq_s64(a);
    #endif
}

static inline sg_ps sg_abs_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_generic_ps) { .f0 = fabsf(a.f0), .f1 = fabsf(a.f1),
        .f2 = fabsf(a.f2), .f3 = fabsf(a.f3) };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_ps(a, sg_sse2_signmask_ps);
    #elif defined SIMD_GRANODI_NEON
    return vabsq_f32(a);
    #endif
}

static inline sg_pd sg_abs_pd(const sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = fabs(a.d0), .d1 = fabs(a.d1) };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_pd(a, sg_sse2_signmask_pd);
    #elif defined SIMD_GRANODI_NEON
    return vabsq_f64(a);
    #endif
}

// note: two's complement does not allow negative zero
static inline sg_pi32 sg_neg_pi32(const sg_pi32 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = -a.i0, .i1 = -a.i1,
        .i2 = -a.i2, .i3 = -a.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_sub_epi32(_mm_setzero_si128(), a);
    #elif defined SIMD_GRANODI_NEON
    return vnegq_s32(a);
    #endif
}

static inline sg_pi64 sg_neg_pi64(const sg_pi64 a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi64) { .l0 = -a.l0, .l1 = -a.l1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_sub_epi64(_mm_setzero_si128(), a);
    #elif defined SIMD_GRANODI_NEON
    return vnegq_s64(a);
    #endif
}

static inline sg_ps sg_neg_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = -a.f0, .f1 = -a.f1,
        .f2 = -a.f2, .f3 = -a.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_xor_ps(a, sg_sse2_signbit_ps);
    #elif defined SIMD_GRANODI_NEON
    return vnegq_f32(a);
    #endif
}

static inline sg_pd sg_neg_pd(const sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = -a.d0, .d1 = -a.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_xor_pd(a, sg_sse2_signbit_pd);
    #elif defined SIMD_GRANODI_NEON
    return vnegq_f64(a);
    #endif
}

// remove signed zero (only floats/doubles can have signed zero),
// but leave intact if any other value

static inline sg_ps sg_remove_signed_zero_ps(const sg_ps a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = a.f0 != 0.0f ? a.f0 : 0.0f,
        .f1 = a.f1 != 0.0f ? a.f1 : 0.0f,
        .f2 = a.f2 != 0.0f ? a.f2 : 0.0f,
        .f3 = a.f3 != 0.0f ? a.f3 : 0.0f };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_ps(a, _mm_cmpneq_ps(a, _mm_setzero_ps()));
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a),
        vmvnq_u32(vceqzq_f32(a))));
    #endif
}

static inline sg_pd sg_remove_signed_zero_pd(const sg_pd a) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = a.d0 != 0.0 ? a.d0 : 0.0,
        .d1 = a.d1 != 0.0 ? a.d1 : 0.0 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_and_pd(a, _mm_cmpneq_pd(a, _mm_setzero_pd()));
    #elif defined SIMD_GRANODI_NEON
    return vreinterpretq_f64_u32(vandq_u32(vreinterpretq_u32_f64(a),
        vmvnq_u32(vreinterpretq_u32_u64(vceqzq_f64(a)))));
    #endif
}

// min

static inline sg_pi32 sg_min_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = a.i0 < b.i0 ? a.i0 : b.i0,
        .i1 = a.i1 < b.i1 ? a.i1 : b.i1,
        .i2 = a.i2 < b.i2 ? a.i2 : b.i2,
        .i3 = a.i3 < b.i3 ? a.i3 : b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return sg_choose_pi32(_mm_cmplt_epi32(a, b), a, b);
    #elif defined SIMD_GRANODI_NEON
    return vminq_s32(a, b);
    #endif
}

static inline sg_pi64 sg_min_pi64(const sg_pi64 a, const sg_pi64 b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
        return sg_set_fromg_pi64((sg_generic_pi64) {
            .l0 = ag.l0 < bg.l0 ? ag.l0 : bg.l0,
            .l1 = ag.l1 < bg.l1 ? ag.l1 : bg.l1 });
    #elif defined SIMD_GRANODI_NEON
    return sg_choose_pi64(vcltq_s64(a, b), a, b);
    #endif
}

// The floating point functions are called "max_fast" and "min_fast" as they
// do not behave identically across platforms with regard to signed zero
static inline sg_ps sg_min_fast_ps(const sg_ps a, const sg_ps b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = a.f0 < b.f0 ? a.f0 : b.f0,
        .f1 = a.f1 < b.f1 ? a.f1 : b.f1,
        .f2 = a.f2 < b.f2 ? a.f2 : b.f2,
        .f3 = a.f3 < b.f3 ? a.f3 : b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_min_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vminq_f32(a, b);
    #endif
}

static inline sg_pd sg_min_fast_pd(const sg_pd a, const sg_pd b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = a.d0 < b.d0 ? a.d0 : b.d0,
        .d1 = a.d1 < b.d1 ? a.d1 : b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_min_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vminq_f64(a, b);
    #endif
}

// max

static inline sg_pi32 sg_max_pi32(const sg_pi32 a, const sg_pi32 b) {
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    return (sg_pi32) { .i0 = a.i0 > b.i0 ? a.i0 : b.i0,
        .i1 = a.i1 > b.i1 ? a.i1 : b.i1,
        .i2 = a.i2 > b.i2 ? a.i2 : b.i2,
        .i3 = a.i3 > b.i3 ? a.i3 : b.i3 };
    #elif defined SIMD_GRANODI_SSE2
    return sg_choose_pi32(_mm_cmpgt_epi32(a, b), a, b);
    #elif defined SIMD_GRANODI_NEON
    return vmaxq_s32(a, b);
    #endif
}

static inline sg_pi64 sg_max_pi64(const sg_pi64 a, const sg_pi64 b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC || defined SIMD_GRANODI_SSE2
    const sg_generic_pi64 ag = sg_getg_pi64(a), bg = sg_getg_pi64(b);
    return sg_set_fromg_pi64((sg_generic_pi64) {
        .l0 = ag.l0 > bg.l0 ? ag.l0 : bg.l0,
        .l1 = ag.l1 > bg.l1 ? ag.l1 : bg.l1 });
    #elif defined SIMD_GRANODI_NEON
    return sg_choose_pi64(vcgtq_s64(a, b), a, b);
    #endif
}

static inline sg_ps sg_max_fast_ps(const sg_ps a, const sg_ps b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return (sg_ps) { .f0 = a.f0 > b.f0 ? a.f0 : b.f0,
        .f1 = a.f1 > b.f1 ? a.f1 : b.f1,
        .f2 = a.f2 > b.f2 ? a.f2 : b.f2,
        .f3 = a.f3 > b.f3 ? a.f3 : b.f3 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_max_ps(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vmaxq_f32(a, b);
    #endif
}

static inline sg_pd sg_max_fast_pd(const sg_pd a, const sg_pd b) {
    #if defined SIMD_GRANODI_FORCE_GENERIC
    return (sg_pd) { .d0 = a.d0 > b.d0 ? a.d0 : b.d0,
        .d1 = a.d1 > b.d1 ? a.d1 : b.d1 };
    #elif defined SIMD_GRANODI_SSE2
    return _mm_max_pd(a, b);
    #elif defined SIMD_GRANODI_NEON
    return vmaxq_f64(a, b);
    #endif
}

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

#ifdef SIMD_GRANODI_DENORMAL_SSE
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

#elif defined SIMD_GRANODI_DENORMAL_ARM32
typedef uint32_t sg_fp_status;
static inline void sg_set_fp_status_arm_(const sg_fp_status fp_status) {
    __asm__ volatile("vmsr fpscr, %0" : : "ri"(fp_status));
}

static inline sg_fp_status sg_get_fp_status_arm_() {
    sg_fp_status fp_status = 0;
    __asm__ volatile("vmrs &0, fpscr" : "=r"(fpsr));
}

#elif defined SIMD_GRANODI_DENORMAL_ARM64
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

#if defined SIMD_GRANODI_DENORMAL_ARM64 || SIMD_GRANODI_DENORMAL_ARM32
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

// Very thin conversion layer for compiling old SSE2 code on NEON, used by
// author (in future, can be spun into different header).
// Does NOT include every intrinsic
// Some intrinsics are NOT implemented efficiently
#ifdef SIMD_GRANODI_NEON
typedef int32x4_t __m128i // The most common, then reinterpret for other ints
typedef float32x4_t __m128
typedef float64x2_t __m128d
#define _mm_castsi128_ps vreinterpretq_f32_s32
#define _mm_castps_si128 vreinterpretq_s32_f32
#define _mm_castsi128d_pd vreinterpretq_f64_s32
#define _mm_castpd_si128 vreinterpretq_s32_f64
#define _mm_castps_pd vreinterpretq_f64_f32
#define _mm_castpd_ps vreinterpretq_f32_f64
#define _mm_shuffle_epi32 sg_shuffle_pi32_switch_
static inline float32x4_t _mm_shuffle_ps(const float32x4_t a,
    const float32x4_t b, const int imm8_compile_time_constant) {
    return vcombineq_f32(
        vget_low_f32(sg_shuffle_ps_switch_(a, imm8_compile_time_constant)),
        vget_high_f32(sg_shuffle_ps_switch_(b, imm8_compile_time_constant))); }
static inline float64x2_t _mm_shuffle_pd(const float64x2_t a,
    const float64x4_t b, const int imm8_compile_time_constant) {
    return vcombine_f64(
        vget_low_f64(sg_shuffle_pd_switch_(a, imm8_compile_time_constant)),
        vget_high_f64(sg_shuffle_pd_switch_(b, imm8_compile_time_constant))); }
#define _mm_setzero_si128() vdupq_n_s32(0)
#define _mm_setzero_ps() vdupq_n_f32(0.0f)
#define _mm_setzero_pd() vdupq_n_f64(0.0)
#define _mm_set1_epi32 vdupq_n_s32
#define _mm_set1_epi64x(a) vreinterpretq_s32_s64(vdupq_n_s64(a))
#define _mm_set1_ps vdupq_n_f32
#define _mm_set1_pd vdupq_n_f64
#define _mm_set_epi32 sg_set_pi32
#define _mm_set_epi64x(si1, si0) vreinterpretq_s32_s64(sg_set_pi64(si1, si0))
#define _mm_set_ps sg_set_ps
#define _mm_set_pd sg_set_pd
#define _mm_cvtsi128_si32(a) vgetq_lane_s32(a, 0)
#define _mm_cvtsi128_si64(a) vgetq_lane_s64(vreinterpretq_s64_s32(a))
#define _mm_cvtss_f32(a) vgetq_lane_f32(a, 0)
#define _mm_cvtsd_f64(a) vgetq_lane_f64(a, 0)
#define _mm_cvtsi64_ss(a, b) vsetq_lane_f32((float) (b), a, 0)
#define _mm_cvtsi64_sd(a, b) vsetq_lane_f64((double) (b), a, 0)
#define _mm_cvtsd_si64(a) ((int64_t) vgetq_lane_f64(a))
#define _mm_cvtepi32_ps vcvtq_f32_s32
#define _mm_cvtepi32_pd sg_cvt_pi32_pd
#define _mm_cvtps_epi32 vcvtnq_s32_f32
#define _mm_cvttps_epi32 vcvtq_s32_f32
#define _mm_cvtpd_epi32 sg_cvt_pd_pi32
#define _mm_cvttpd_epi32 sg_cvtt_pd_pi32
#define _mm_cvtpd_ps sg_cvt_pd_ps
#define _mm_cvtps_pd sg_cvt_ps_pd
#define _mm_unpackhi_pd(a, b) vcombine_f64(vget_high_f64(a), vget_high_f64(b))
#define _mm_unpackhi_epi64(a, b) vreinterpretq_s32_s64(vcombine_s64( \
    vget_high_s64(vreinterpretq_s64_s32(a)), \
    vget_high_s64(vreinterpretq_s64_s32(b))))
#define _mm_add_epi32 vaddq_s32
#define _mm_add_epi64(a, b) vreinterpretq_s32_s64( \
    vaddq_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)))
#define _mm_add_ps vaddq_f32
#define _mm_add_pd vaddq_f64
#define _mm_sub_epi32 vsubq_s32
#define _mm_sub_epi64(a, b) vreinterpretq_s32_s64( \
    vsubq_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)))
#define _mm_sub_ps vsubq_f32
#define _mm_sub_pd vsubq_f64
#define _mm_mul_ps vmulq_f32
#define _mm_mul_pd vmulq_f64
#define _mm_div_ps vdivq_f32
#define _mm_div_pd vdivq_f64
#define _mm_andnot_si128(a, b) vbicq_s32(b, a) // Yes, arg swap is correct!
#define _mm_andnot_ps(a, b) vbicq_f32(b, a)
#define _mm_andnot_pd(a, b) vbicq_f64(b, a)
#define _mm_and_si128 vandq_s32
#define _mm_and_ps sg_and_ps
#define _mm_and_pd sg_and_pd
#define _mm_or_si128 vorrq_s32
#define _mm_or_ps sg_or_ps
#define _mm_or_pd sg_or_pd
#define _mm_xor_si128 veorq_s32
#define _mm_xor_ps sg_xor_ps
#define _mm_xor_pd sg_xor_pd
#define _mm_slli_epi32 vshlq_n_s32
#define _mm_slli_epi64(a, imm) vreinterpretq_s32_s64( \
    vshlq_n_s64(vreinterpretq_s64_s32(a), imm))
#define _mm_srli_epi32(a, imm) vreinterpretq_s32_u32( \
    vshrq_n_u32(vreinterpretq_u32_s32(a), imm))
#define _mm_srli_epi64(a, imm) vreinterpretq_s32_u64( \
    vshrq_n_u64(vreinterpretq_u64_s32(a), imm))
#define _mm_sra_epi32 vshrq_n_s32
#define _mm_cmplt_epi32 vcltq_s32
#define _mm_cmplt_ps vcltq_f32
#define _mm_cmplt_pd vcltq_f64
#define _mm_cmple_ps vcleq_f32
#define _mm_cmple_pd vcleq_f64
#define _mm_cmpeq_epi32 vceqq_s32
#define _mm_cmpeq_ps vceqq_f32
#define _mm_cmpeq_pd vceqq_f64
#define _mm_cmpneq_ps sg_cmpneq_ps
#define _mm_cmpneq_pd sg_cmpneq_pd
#define _mm_cmpge_ps vcgeq_f32
#define _mm_cmpge_pd vcgeq_f64
#define _mm_cmpgt_epi32 vcgtq_s32
#define _mm_cmpgt_ps vcgtq_f32
#define _mm_cmpgt_pd vcgtq_f64
#endif

#ifdef __cplusplus

// C++ classes for operator overloading:
// - Have no setters, but can be changed via += etc (semi-immutable)
// - Are default constructed to zero
// - All type casts or type conversions must be explicit
// - Should never use any SSE2 or NEON types or intrinsics directly

#if defined SIMD_GRANODI_DENORMAL_SSE || \
    defined SIMD_GRANODI_DENORMAL_ARM64 || \
    defined SIMD_GRANODI_DENORMAL_ARM32
class ScopedDenormalsDisable {
    const sg_fp_status fp_status_;
public:
    ScopedDenormalsDisable() : fp_status_(sg_disable_denormals()) {}
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
    Compare_pi32() : data_(sg_setzero_cmp_pi32()) {}
    Compare_pi32(const bool b) : data_(sg_set1cmp_pi32(b)) {}
    Compare_pi32(const bool b3, const bool b2, const bool b1, const bool b0)
        : data_(sg_setcmp_pi32(b3, b2, b1, b0)) {}
    Compare_pi32(const sg_cmp_pi32& cmp) : data_(cmp) {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Compare_pi32(const sg_generic_cmp4& cmp)
        : data_(sg_setcmp_fromg_pi32(cmp)) {}
    #endif

    sg_cmp_pi32 data() const { return data_; }

    Compare_pi32& operator&=(const Compare_pi32& rhs) {
        data_ = sg_and_cmp_pi32(data_, rhs.data());
        return *this;
    }
    friend Compare_pi32 operator&(Compare_pi32 lhs, const Compare_pi32& rhs) {
        lhs &= rhs;
        return lhs;
    }
    Compare_pi32 operator&&(const Compare_pi32& rhs) const {
        return *this & rhs;
    }

    Compare_pi32& operator|=(const Compare_pi32& rhs) {
        data_ = sg_or_cmp_pi32(data_, rhs.data());
        return *this;
    }
    friend Compare_pi32 operator|(Compare_pi32 lhs, const Compare_pi32& rhs) {
        lhs |= rhs;
        return lhs;
    }
    Compare_pi32 operator||(const Compare_pi32& rhs) const {
        return *this | rhs;
    }

    Compare_pi32& operator^=(const Compare_pi32& rhs) {
        data_ = sg_xor_cmp_pi32(data_, rhs.data());
        return *this;
    }
    friend Compare_pi32 operator^(Compare_pi32 lhs, const Compare_pi32& rhs) {
        lhs ^= rhs;
        return lhs;
    }

    Compare_pi32 operator~() const { return sg_not_cmp_pi32(data_); }
    Compare_pi32 operator!() const { return ~*this; }

    bool debug_valid_eq(const bool b3, const bool b2,
        const bool b1, const bool b0) const
    {
        return sg_debug_cmp_valid_eq_pi32(data_, b3, b2, b1, b0);
    }
    bool debug_valid_eq(const bool b) const {
        return debug_valid_eq(b, b, b, b);
    }

    inline Compare_pi64 convert_to_cmp_pi64() const;
    inline Compare_ps bitcast_to_cmp_ps() const;
    inline Compare_pd convert_to_cmp_pd() const;
    inline Vec_pi32 choose_else_zero(const Vec_pi32& if_true) const;
    inline Vec_pi32 choose(const Vec_pi32& if_true,
        const Vec_pi32& if_false) const;
};

class Compare_pi64 {
    sg_cmp_pi64 data_;
public:
    Compare_pi64() : data_(sg_setzero_cmp_pi64()) {}
    Compare_pi64(const bool b) : data_(sg_set1cmp_pi64(b)) {}
    Compare_pi64(const bool b1, const bool b0)
        : data_(sg_setcmp_pi64(b1, b0)) {}
    Compare_pi64(const sg_cmp_pi64& cmp) : data_(cmp) {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Compare_pi64(const sg_generic_cmp2& cmp)
        : data_(sg_setcmp_fromg_pi64(cmp)) {}
    #endif

    sg_cmp_pi64 data() const { return data_; }

    Compare_pi64& operator&=(const Compare_pi64& rhs) {
        data_ = sg_and_cmp_pi64(data_, rhs.data());
        return *this;
    }
    friend Compare_pi64 operator&(Compare_pi64 lhs, const Compare_pi64& rhs) {
        lhs &= rhs;
        return lhs;
    }
    Compare_pi64 operator&&(const Compare_pi64& rhs) const {
        return *this & rhs;
    }

    Compare_pi64& operator|=(const Compare_pi64& rhs) {
        data_ = sg_or_cmp_pi64(data_, rhs.data());
        return *this;
    }
    friend Compare_pi64 operator|(Compare_pi64 lhs, const Compare_pi64& rhs) {
        lhs |= rhs;
        return lhs;
    }
    Compare_pi64 operator||(const Compare_pi64& rhs) const {
        return *this | rhs;
    }

    Compare_pi64& operator^=(const Compare_pi64& rhs) {
        data_ = sg_xor_cmp_pi64(data_, rhs.data());
        return *this;
    }
    friend Compare_pi64 operator^(Compare_pi64 lhs, const Compare_pi64& rhs) {
        lhs ^= rhs;
        return lhs;
    }

    Compare_pi64 operator~() const { return sg_not_cmp_pi64(data_); }
    Compare_pi64 operator!() const { return ~*this; }

    bool debug_valid_eq(const bool b1, const bool b0) const {
        return sg_debug_cmp_valid_eq_pi64(data_, b1, b0);
    }
    bool debug_valid_eq(const bool b) const { return debug_valid_eq(b, b); }

    inline Compare_pi32 convert_to_cmp_pi32() const;
    inline Compare_ps convert_to_cmp_ps() const;
    inline Compare_pd bitcast_to_cmp_pd() const;

    inline Vec_pi64 choose_else_zero(const Vec_pi64& if_true) const;
    inline Vec_pi64 choose(const Vec_pi64& if_true,
        const Vec_pi64& if_false) const;
};

class Compare_ps {
    sg_cmp_ps data_;
public:
    Compare_ps() : data_(sg_setzero_cmp_ps()) {}
    Compare_ps(const bool b) : data_(sg_set1cmp_ps(b)) {}
    Compare_ps(const bool b3, const bool b2, const bool b1, const bool b0)
        : data_(sg_setcmp_ps(b3, b2, b1, b0)) {}
    Compare_ps(const sg_cmp_ps& cmp) : data_(cmp) {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Compare_ps(const sg_generic_cmp4& cmp) : data_(sg_setcmp_fromg_ps(cmp)) {}
    #endif

    sg_cmp_ps data() const { return data_; }

    Compare_ps& operator&=(const Compare_ps& rhs) {
        data_ = sg_and_cmp_ps(data_, rhs.data());
        return *this;
    }
    friend Compare_ps operator&(Compare_ps lhs, const Compare_ps& rhs) {
        lhs &= rhs;
        return lhs;
    }
    Compare_ps operator&&(const Compare_ps& rhs) const {
        return *this & rhs;
    }

    Compare_ps& operator|=(const Compare_ps& rhs) {
        data_ = sg_or_cmp_ps(data_, rhs.data());
        return *this;
    }
    friend Compare_ps operator|(Compare_ps lhs, const Compare_ps& rhs) {
        lhs |= rhs;
        return lhs;
    }
    Compare_ps operator||(const Compare_ps& rhs) const {
        return *this | rhs;
    }

    Compare_ps& operator^=(const Compare_ps& rhs) {
        data_ = sg_xor_cmp_ps(data_, rhs.data());
        return *this;
    }
    friend Compare_ps operator^(Compare_ps lhs, const Compare_ps& rhs) {
        lhs ^= rhs;
        return lhs;
    }

    Compare_ps operator~() const { return sg_not_cmp_ps(data_); }
    Compare_ps operator!() const { return ~*this; }

    bool debug_valid_eq(const bool b3, const bool b2,
        const bool b1, const bool b0) const
    {
        return sg_debug_cmp_valid_eq_ps(data_, b3, b2, b1, b0);
    }
    bool debug_valid_eq(const bool b) { return debug_valid_eq(b, b, b, b); }

    inline Compare_pi32 bitcast_to_cmp_pi32() const;
    inline Compare_pi64 convert_to_cmp_pi64() const;
    inline Compare_pd convert_to_cmp_pd() const;

    inline Vec_ps choose_else_zero(const Vec_ps& if_true) const;
    inline Vec_ps choose(const Vec_ps& if_true, const Vec_ps& if_false) const;
};

class Compare_pd {
    sg_cmp_pd data_;
public:
    Compare_pd() : data_(sg_setzero_cmp_pd()) {}
    Compare_pd(const bool b) : data_(sg_set1cmp_pd(b)) {}
    Compare_pd(const bool b1, const bool b0)
        : data_(sg_setcmp_pd(b1, b0)) {}
    Compare_pd(const sg_cmp_pd& cmp) : data_(cmp) {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Compare_pd(const sg_generic_cmp2& cmp) : data_(sg_setcmp_fromg_pd(cmp)) {}
    #endif

    sg_cmp_pd data() const { return data_; }

    Compare_pd& operator&=(const Compare_pd& rhs) {
        data_ = sg_and_cmp_pd(data_, rhs.data());
        return *this;
    }
    friend Compare_pd operator&(Compare_pd lhs, const Compare_pd& rhs) {
        lhs &= rhs;
        return lhs;
    }
    Compare_pd operator&&(const Compare_pd& rhs) const {
        return *this & rhs;
    }

    Compare_pd& operator|=(const Compare_pd& rhs) {
        data_ = sg_or_cmp_pd(data_, rhs.data());
        return *this;
    }
    friend Compare_pd operator|(Compare_pd lhs, const Compare_pd& rhs) {
        lhs |= rhs;
        return lhs;
    }
    Compare_pd operator||(const Compare_pd& rhs) const {
        return *this | rhs;
    }

    Compare_pd& operator^=(const Compare_pd& rhs) {
        data_ = sg_xor_cmp_pd(data_, rhs.data());
        return *this;
    }
    friend Compare_pd operator^(Compare_pd lhs, const Compare_pd& rhs) {
        lhs ^= rhs;
        return lhs;
    }

    Compare_pd operator~() const { return sg_not_cmp_pd(data_); }
    Compare_pd operator!() const { return ~*this; }

    bool debug_valid_eq(const bool b1, const bool b0) const {
        return sg_debug_cmp_valid_eq_pd(data_, b1, b0);
    }
    bool debug_valid_eq(const bool b) const { return debug_valid_eq(b, b); }

    inline Compare_pi32 convert_to_cmp_pi32() const;
    inline Compare_pi64 bitcast_to_cmp_pi64() const;
    inline Compare_ps convert_to_cmp_ps() const;

    inline Vec_pd choose_else_zero(const Vec_pd& if_true) const;
    inline Vec_pd choose(const Vec_pd& if_true, const Vec_pd& if_false) const;
};

class Vec_pi32 {
    sg_pi32 data_;
public:
    Vec_pi32() : data_(sg_setzero_pi32()) {}
    Vec_pi32(const int32_t i) : data_(sg_set1_pi32(i)) {}
    Vec_pi32(const int32_t i3, const int32_t i2, const int32_t i1,
        const int32_t i0) : data_(sg_set_pi32(i3, i2, i1, i0)) {}
    Vec_pi32(const sg_pi32& pi32) : data_(pi32) {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    // Otherwise, we are defining two identical ctors & won't compile...
    Vec_pi32(const sg_generic_pi32& g_pi32)
        : data_(sg_set_fromg_pi32(g_pi32)) {}
    #endif

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

    inline Vec_pi64 bitcast_to_pi64() const;
    inline Vec_ps bitcast_to_ps() const;
    inline Vec_pd bitcast_to_pd() const;

    inline Vec_pi64 convert_to_pi64() const;
    inline Vec_ps convert_to_ps() const;
    inline Vec_pd convert_to_pd() const;
};

class Vec_pi64 {
    sg_pi64 data_;
public:
    Vec_pi64() : data_(sg_setzero_pi64()) {}
    Vec_pi64(const int64_t l) : data_(sg_set1_pi64(l)) {}
    Vec_pi64(const int64_t l1, const int64_t l0) : data_(sg_set_pi64(l1, l0)) {}
    Vec_pi64(const sg_pi64& pi64) : data_(pi64) {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Vec_pi64(const sg_generic_pi64& g_pi64)
        : data_(sg_set_fromg_pi64(g_pi64)) {}
    #endif

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

    inline Vec_pi32 bitcast_to_pi32() const;
    inline Vec_ps bitcast_to_ps() const;
    inline Vec_pd bitcast_to_pd() const;

    inline Vec_pi32 convert_to_pi32() const;
    inline Vec_ps convert_to_ps() const;
    inline Vec_pd convert_to_pd() const;
};

class Vec_ps {
    sg_ps data_;
public:
    Vec_ps() : data_(sg_setzero_ps()) {}
    Vec_ps(const float f) : data_(sg_set1_ps(f)) {}
    Vec_ps(const float f3, const float f2, const float f1, const float f0)
        : data_(sg_set_ps(f3, f2, f1, f0)) {}
    Vec_ps(const sg_ps& ps) : data_(ps) {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Vec_ps(const sg_generic_ps& g_ps) : data_(sg_set_fromg_ps(g_ps)) {}
    #endif

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

    inline Vec_pi32 bitcast_to_pi32() const;
    inline Vec_pi64 bitcast_to_pi64() const;
    inline Vec_pd bitcast_to_pd() const;

    inline Vec_pi32 convert_to_nearest_pi32() const;
    inline Vec_pi32 truncate_to_pi32() const;
    inline Vec_pi64 convert_to_nearest_pi64() const;
    inline Vec_pi64 truncate_to_pi64() const;
    inline Vec_pd convert_to_pd() const;
};

class Vec_pd {
    sg_pd data_;
public:
    Vec_pd() : data_(sg_setzero_pd()) {}
    Vec_pd(const double d) : data_(sg_set1_pd(d)) {}
    Vec_pd(const double d1, const double d0) : data_(sg_set_pd(d1, d0)) {}
    Vec_pd(const sg_pd& pd) : data_(pd) {}
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    Vec_pd(const sg_generic_pd& g_pd) : data_(sg_set_fromg_pd(g_pd)) {}
    #endif

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

    inline Vec_pi32 bitcast_to_pi32() const;
    inline Vec_pi64 bitcast_to_pi64() const;
    inline Vec_ps bitcast_to_ps() const;

    inline Vec_pi32 convert_to_nearest_pi32() const;
    inline Vec_pi32 truncate_to_pi32() const;
    inline Vec_pi64 convert_to_nearest_pi64() const;
    inline Vec_pi64 truncate_to_pi64() const;
    inline Vec_ps convert_to_ps() const;
};

inline Compare_pi64 Compare_pi32::convert_to_cmp_pi64() const {
    return sg_cvtcmp_pi32_pi64(data_);
}
inline Compare_ps Compare_pi32::bitcast_to_cmp_ps() const {
    return sg_castcmp_pi32_ps(data_);
}
inline Compare_pd Compare_pi32::convert_to_cmp_pd() const {
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

inline Compare_pi32 Compare_pi64::convert_to_cmp_pi32() const {
    return sg_cvtcmp_pi64_pi32(data_);
}
inline Compare_ps Compare_pi64::convert_to_cmp_ps() const {
    return sg_cvtcmp_pi64_ps(data_);
}
inline Compare_pd Compare_pi64::bitcast_to_cmp_pd() const {
    return sg_castcmp_pi64_pd(data_);
}

inline Vec_pi64 Compare_pi64::choose_else_zero(const Vec_pi64& if_true) const {
    return sg_choose_else_zero_pi64(data_, if_true.data());
}
inline Vec_pi64 Compare_pi64::choose(const Vec_pi64& if_true,
    const Vec_pi64& if_false) const
{
    return sg_choose_pi64(data_, if_true.data(), if_false.data());
}

inline Compare_pi32 Compare_ps::bitcast_to_cmp_pi32() const {
    return sg_castcmp_ps_pi32(data_);
}
inline Compare_pi64 Compare_ps::convert_to_cmp_pi64() const {
    return sg_cvtcmp_ps_pi64(data_);
}
inline Compare_pd Compare_ps::convert_to_cmp_pd() const {
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

inline Compare_pi32 Compare_pd::convert_to_cmp_pi32() const {
    return sg_cvtcmp_pd_pi32(data_);
}
inline Compare_pi64 Compare_pd::bitcast_to_cmp_pi64() const {
    return sg_castcmp_pd_pi64(data_);
}
inline Compare_ps Compare_pd::convert_to_cmp_ps() const {
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

inline Vec_pi64 Vec_pi32::bitcast_to_pi64() const {
    return sg_cast_pi32_pi64(data_);
}
inline Vec_ps Vec_pi32::bitcast_to_ps() const { return sg_cast_pi32_ps(data_); }
inline Vec_pd Vec_pi32::bitcast_to_pd() const { return sg_cast_pi32_pd(data_); }
inline Vec_pi64 Vec_pi32::convert_to_pi64() const {
    return sg_cvt_pi32_pi64(data_);
}
inline Vec_ps Vec_pi32::convert_to_ps() const {
    return sg_cvt_pi32_ps(data_);
}
inline Vec_pd Vec_pi32::convert_to_pd() const {
    return sg_cvt_pi32_pd(data_);
}

inline Vec_pi32 Vec_pi64::bitcast_to_pi32() const {
    return sg_cast_pi64_pi32(data_);
}
inline Vec_ps Vec_pi64::bitcast_to_ps() const { return sg_cast_pi64_ps(data_); }
inline Vec_pd Vec_pi64::bitcast_to_pd() const { return sg_cast_pi64_pd(data_); }
inline Vec_pi32 Vec_pi64::convert_to_pi32() const {
    return sg_cvt_pi64_pi32(data_);
}
inline Vec_ps Vec_pi64::convert_to_ps() const {
    return sg_cvt_pi64_ps(data_);
}
inline Vec_pd Vec_pi64::convert_to_pd() const {
    return sg_cvt_pi64_pd(data_);
}

inline Vec_pi32 Vec_ps::bitcast_to_pi32() const {
    return sg_cast_ps_pi32(data_);
}
inline Vec_pi64 Vec_ps::bitcast_to_pi64() const {
    return sg_cast_ps_pi64(data_);
}
inline Vec_pd Vec_ps::bitcast_to_pd() const { return sg_cast_ps_pd(data_); }
inline Vec_pi32 Vec_ps::convert_to_nearest_pi32() const {
    return sg_cvt_ps_pi32(data_);
}
inline Vec_pi32 Vec_ps::truncate_to_pi32() const {
    return sg_cvtt_ps_pi32(data_);
}
inline Vec_pi64 Vec_ps::convert_to_nearest_pi64() const {
    return sg_cvt_ps_pi64(data_);
}
inline Vec_pi64 Vec_ps::truncate_to_pi64() const {
    return sg_cvtt_ps_pi64(data_);
}
inline Vec_pd Vec_ps::convert_to_pd() const {
    return sg_cvt_ps_pd(data_);
}

inline Vec_pi32 Vec_pd::bitcast_to_pi32() const {
    return sg_cast_pd_pi32(data_);
}
inline Vec_pi64 Vec_pd::bitcast_to_pi64() const {
    return sg_cast_pd_pi64(data_);
}
inline Vec_ps Vec_pd::bitcast_to_ps() const { return sg_cast_pd_ps(data_); }
inline Vec_pi32 Vec_pd::convert_to_nearest_pi32() const {
    return sg_cvt_pd_pi32(data_);
}
inline Vec_pi32 Vec_pd::truncate_to_pi32() const {
    return sg_cvtt_pd_pi32(data_);
}
inline Vec_pi64 Vec_pd::convert_to_nearest_pi64() const {
    return sg_cvt_pd_pi64(data_);
}
inline Vec_pi64 Vec_pd::truncate_to_pi64() const {
    return sg_cvtt_pd_pi64(data_);
}
inline Vec_ps Vec_pd::convert_to_ps() const {
    return sg_cvt_pd_ps(data_);
}

} // namespace simd_granodi

#endif // __cplusplus

#endif // SIMD_GRANODI_H
