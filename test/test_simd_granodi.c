#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h> // for exit()
#include <inttypes.h> // For printf PRId64 format

#include "../simd_granodi.h"

#ifdef __cplusplus
using namespace simd_granodi;
#endif

#define sg_assert(cond) do { \
    if (!(cond)) { \
        printf("sg_assert() failed on line %d.\n", __LINE__); \
        exit(1); \
    } } while(0)

static void print_platform_start();

#ifdef SG_PRINT
static void print_pi32(const sg_pi32);
static void print_pi64(const sg_pi64);
static void print_ps(const sg_ps);
static void print_pd(const sg_pd);
#endif

static void test_128endian();
static void test_cast();
static void test_shuffle();
static void test_set();
static void test_get();
static void test_convert();
static void test_add_sub();
static void test_mul_div();
static void test_bitwise();
static void test_shift();
static void test_cmp();
static void test_abs_neg();
static void test_min_max();
static void test_constrain();

#ifdef __cplusplus
static void test_opover();
static void test_opover_cmp();
#endif

int main() {
    print_platform_start();

    //sg_assert(false);

    test_128endian();
    test_cast();
    test_shuffle();
    test_set();
    test_get();
    test_convert();
    test_add_sub();
    test_mul_div();
    test_bitwise();
    test_shift();
    test_cmp();
    test_abs_neg();
    test_min_max();
    test_constrain();

    #ifdef __cplusplus
    test_opover();
    test_opover_cmp();
    #endif

    printf("\n");

    return 0;
}

void print_platform_start() {
    printf("Implementation: ");
    #ifdef SIMD_GRANODI_FORCE_GENERIC
    printf("generic\n");
    #elif defined SIMD_GRANODI_SSE2
    printf("SSE2\n");
    #elif defined SIMD_GRANODI_NEON
    printf("NEON\n");
    #else
    sg_assert(false);
    #endif

    printf("Arch: ");
    #ifdef SIMD_GRANODI_ARCH_SSE
    printf("SSE\n");
    #elif defined SIMD_GRANODI_ARCH_ARM64
    printf("ARM64\n");
    #elif defined SIMD_GRANODI_ARCH_ARM32
    printf("ARM32\n");
    #else
    sg_assert(false);
    #endif

    #if defined FP_FAST_FMAF || defined FP_FAST_FMA
    printf("Compiler FMA macros: ");
    #ifdef FP_FAST_FMAF
    printf("FP_FAST_FMAF ");
    #endif
    #ifdef FP_FAST_FMA
    printf("FP_FAST_FMA");
    #endif
    printf("\n");
    #else
    printf("Fast FMA implementation NOT enabled in compiler\n");
    #endif

    #ifdef NDEBUG
    printf("NDEBUG defined (optimized)\n");
    #else
    printf("NDEBUG not defined (debug)\n");
    #endif

    /*print_pi32(sg_set1_pi32(0)); printf(", ");
    print_pi64(sg_set1_pi64(0)); printf(", ");
    print_ps(sg_set1_ps(0.0f)); printf(", ");
    print_pd(sg_set1_pd(0.0)); printf("\n");*/
}

#ifdef SG_PRINT
void print_pi32(const sg_pi32 a) {
    sg_generic_pi32 ag = sg_getg_pi32(a);
    printf("[i3: %d, i2: %d, i1: %d, i0: %d]", ag.i3, ag.i2, ag.i1, ag.i0);
}

void print_pi64(const sg_pi64 a) {
    sg_generic_pi64 ag = sg_getg_pi64(a);
    printf("[l1: %ld, l0: %ld]", ag.l1, ag.l0);
}

void print_ps(const sg_ps a) {
    sg_generic_ps ag = sg_getg_ps(a);
    printf("[f3: %.2f, f2: %.2f, f1: %.2f, f0: %.2f]",
        ag.f3, ag.f2, ag.f1, ag.f0);
}

void print_pd(const sg_pd a) {
    sg_generic_pd ag = sg_getg_pd(a);
    printf("[d1: %.2f, d0: %.2f]", ag.d1, ag.d0);
}
#endif

//
//
//
//
//
//
//
//
//
//
// ASSERTION HELPERS

#define assert_eq_pi32(a, i3, i2, i1, i0) \
    sg_assert(sg_debug_eq_pi32(a, i3, i2, i1, i0))
#define assert_eq_pi64(a, l1, l0) \
    sg_assert(sg_debug_eq_pi64(a, l1, l0))
#define assert_eq_ps(a, f3, f2, f1, f0) \
    sg_assert(sg_debug_eq_ps(a, f3, f2, f1, f0))
#define assert_eq_pd(a, d1, d0) \
    sg_assert(sg_debug_eq_pd(a, d1, d0))

#define assert_eq_cmp_pi32(a, b3, b2, b1, b0) \
    sg_assert(sg_debug_cmp_valid_eq_pi32(a, b3, b2, b1, b0))
#define assert_eq_cmp_pi64(a, b1, b0) \
    sg_assert(sg_debug_cmp_valid_eq_pi64(a, b1, b0))
#define assert_eq_cmp_ps(a, b3, b2, b1, b0) \
    sg_assert(sg_debug_cmp_valid_eq_ps(a, b3, b2, b1, b0))
#define assert_eq_cmp_pd(a, b1, b0) \
    sg_assert(sg_debug_cmp_valid_eq_pd(a, b1, b0))

// WARNING: these macros evaluate g twice, only for testing
#define assert_eqg_cmp_pi32(a, g) \
    sg_assert(sg_debug_cmp_valid_eq_pi32(a, g.b3, g.b2, g.b1, g.b0))
#define assert_eqg_cmp_pi64(a, g) \
    sg_assert(sg_debug_cmp_valid_eq_pi64(a, g.b1, g.b0))
#define assert_eqg_cmp_ps(a, g) \
    sg_assert(sg_debug_cmp_valid_eq_ps(a, g.b3, g.b2, g.b1, g.b0))
#define assert_eqg_cmp_pd(a, g) \
    sg_assert(sg_debug_cmp_valid_eq_pd(a, g.b1, g.b0))

#ifdef __cplusplus
#define assert_eq_cpp_pi32(a, i) sg_assert(a.debug_eq(i))
#define assert_eq_cpp_pi64(a, l) sg_assert(a.debug_eq(l))
#define assert_eq_cpp_ps(a, f) sg_assert(a.debug_eq(f))
#define assert_eq_pcpp_d(a, d) sg_assert(a.debug_eq(d))
#endif

//
//
//
//
//
//
//
//
//
// TEST ROUTINES

void test_128endian() {
    #ifndef SIMD_GRANODI_FORCE_GENERIC
    if (sizeof(sg_generic_pi32) != sizeof(sg_pi32)) {
        printf("WARNING: Cannot test endianness due to sizeof() difference. "
            "Big Problem!\n");
    } else {
        sg_pi32 a = sg_set_pi32(3, 2, 1, 0);
        sg_generic_pi32 ag;
        memcpy(&ag, &a, sizeof(sg_generic_pi32));
        if (ag.i0 == 0) {
            printf("Little endian ordering of 128 bit si32 vector\n");
        }
        else if (ag.i0 == 3) {
            printf("Big endian ordering of 128 bit si32 vector\n");
        }
        else {
            printf("Big problem: memcpy of vector to generic vector "
                "went horribly wrong. Code might be broken!\n");
        }
    }
    #else
    printf("Using generic code, cannot test 128-bit endianness\n");
    #endif
}

void test_cast() {
    sg_pi32 si32; sg_pi64 si64; sg_ps srps; sg_pd srpd;

    si32 = sg_set_pi32(3, 2, 1, 0);
    assert_eq_pi32(sg_bitcast_pi64_pi32(sg_bitcast_pi32_pi64(si32)), 3, 2, 1, 0);
    assert_eq_pi32(sg_bitcast_ps_pi32(sg_bitcast_pi32_ps(si32)), 3, 2, 1, 0);
    assert_eq_pi32(sg_bitcast_pd_pi32(sg_bitcast_pi32_pd(si32)), 3, 2, 1, 0);

    si64 = sg_set_pi64(1, 0);
    assert_eq_pi64(sg_bitcast_ps_pi64(sg_bitcast_pi64_ps(si64)), 1, 0);
    assert_eq_pi64(sg_bitcast_pd_pi64(sg_bitcast_pi64_pd(si64)), 1, 0);

    srps = sg_set_ps(3.0f, 2.0f, 1.0f, 0.0f);
    assert_eq_ps(sg_bitcast_pd_ps(sg_bitcast_ps_pd(srps)), 3.0f, 2.0f, 1.0f, 0.0f);

    srpd = sg_set_pd(1.0, 0.0);
    assert_eq_pd(sg_bitcast_ps_pd(sg_bitcast_pd_ps(srpd)), 1.0, 0.0);

    //printf("Cast test succeeeded\n");
}

void test_shuffle() {
    for (int src3 = 0; src3 < 4; ++src3) {
    for (int src2 = 0; src2 < 4; ++ src2) {
    for (int src1 = 0; src1 < 4; ++src1) {
    for (int src0 = 0; src0 < 4; ++src0) {
        assert_eq_pi32(sg_shuffle_pi32(sg_set_pi32(3, 2, 1, 0),
        src3, src2, src1, src0), src3, src2, src1, src0);
        assert_eq_ps(sg_shuffle_ps(sg_set_ps(3.0f, 2.0f, 1.0f, 0.0f),
        src3, src2, src1, src0), src3, src2, src1, src0);
    } } } }

    for (int src1 = 0; src1 < 2; ++src1) {
    for (int src0 = 0; src0 < 2; ++src0) {
        assert_eq_pi64(sg_shuffle_pi64(sg_set_pi64(1, 0), src1, src0),
            src1, src0);
        assert_eq_pd(sg_shuffle_pd(sg_set_pd(1.0, 0.0), src1, src0),
            src1, src0);
    } }

    //printf("Shuffle test succeeeded\n");
}

void test_set() {
    assert_eq_pi32(sg_set_pi32(3, 2, 1, 0), 3, 2, 1, 0);
    assert_eq_pi32(sg_set_from_u32_pi32(3, 2, 1, 0xffffffff), 3, 2, 1, -1);
    assert_eq_pi32(sg_set1_pi32(1), 1, 1, 1, 1);
    assert_eq_pi32(sg_set1_from_u32_pi32(0xffffffff), -1, -1, -1, -1);
    assert_eq_pi32(sg_setzero_pi32(), 0, 0, 0, 0);

    assert_eq_pi64(sg_set_pi64(1, 0), 1, 0);
    assert_eq_pi64(sg_set_from_u64_pi64(1, 0xffffffffffffffff), 1, -1);
    assert_eq_pi64(sg_set1_pi64(1), 1, 1);
    assert_eq_pi64(sg_set1_from_u64_pi64(0xffffffffffffffff), -1, -1);
    assert_eq_pi64(sg_setzero_pi64(), 0, 0);

    assert_eq_ps(sg_set_ps(3.0f, 2.0f, 1.0f, 0.0f), 3.0f, 2.0f, 1.0f, 0.0f);
    assert_eq_ps(sg_set_from_u32_ps(sg_bitcast_f32x1_u32x1(3.0f),
        sg_bitcast_f32x1_u32x1(2.0f), sg_bitcast_f32x1_u32x1(1.0f),
        sg_bitcast_f32x1_u32x1(0.0f)), 3.0f, 2.0f, 1.0f, 0.0f);
    assert_eq_ps(sg_set1_ps(1.0f), 1.0f, 1.0f, 1.0f, 1.0f);
    assert_eq_ps(sg_set1_from_u32_ps(sg_bitcast_f32x1_u32x1(3.0f)),
        3.0f, 3.0f, 3.0f, 3.0f);
    assert_eq_ps(sg_setzero_ps(), 0.0f, 0.0f, 0.0f, 0.0f);

    assert_eq_pd(sg_set_pd(1.0, 0.0), 1.0, 0.0);
    assert_eq_pd(sg_set_from_u64_pd(sg_bitcast_f64x1_u64x1(1.0),
        sg_bitcast_f64x1_u64x1(0.0)), 1.0, 0.0);
    assert_eq_pd(sg_set1_pd(1.0), 1.0, 1.0);
    assert_eq_pd(sg_set1_from_u64_pd(sg_bitcast_f64x1_u64x1(3.0)), 3.0, 3.0);
    assert_eq_pd(sg_setzero_pd(), 0.0, 0.0);

    //printf("Set test succeeeded\n");
}

void test_get() {
    // The getg() (get generic) functions are covered by the assert_eq
    // functions

    sg_assert(sg_get0_pi32(sg_set_pi32(4,3,2,1)) == 1);
    sg_assert(sg_get1_pi32(sg_set_pi32(4,3,2,1)) == 2);
    sg_assert(sg_get2_pi32(sg_set_pi32(4,3,2,1)) == 3);
    sg_assert(sg_get3_pi32(sg_set_pi32(4,3,2,1)) == 4);

    sg_assert(sg_get0_pi64(sg_set_pi64(2, 1)) == 1);
    sg_assert(sg_get1_pi64(sg_set_pi64(2, 1)) == 2);

    sg_assert(sg_get0_ps(sg_set_ps(4.0f, 3.0f, 2.0f, 1.0f)) == 1.0f);
    sg_assert(sg_get1_ps(sg_set_ps(4.0f, 3.0f, 2.0f, 1.0f)) == 2.0f);
    sg_assert(sg_get2_ps(sg_set_ps(4.0f, 3.0f, 2.0f, 1.0f)) == 3.0f);
    sg_assert(sg_get3_ps(sg_set_ps(4.0f, 3.0f, 2.0f, 1.0f)) == 4.0f);

    sg_assert(sg_get0_pd(sg_set_pd(2.0, 1.0)) == 1.0);
    sg_assert(sg_get1_pd(sg_set_pd(2.0, 1.0)) == 2.0);


    // Check SSE2 rounding
    #if defined SIMD_GRANODI_SSE2 && !defined _MSC_VER
    sg_assert(_mm_cvtsd_si64(_mm_set_pd(0.0, -1.7)) == -2);
    sg_assert(_mm_cvtss_si32(_mm_set_ps(0.0f, 0.0f, 0.0f, -1.7f)) == -2);
    #endif
    //printf("Get test succeeeded\n");
}

void test_convert() {
    sg_pi32 si32 = sg_set_pi32(3, 2, 1, 0);
    assert_eq_pi64(sg_cvt_pi32_pi64(si32), 1, 0);
    assert_eq_ps(sg_cvt_pi32_ps(si32), 3.0f, 2.0f, 1.0f, 0.0f);
    assert_eq_pd(sg_cvt_pi32_pd(si32), 1.0, 0.0);

    sg_pi64 si64 = sg_set_pi64(2, 1);
    assert_eq_pi32(sg_cvt_pi64_pi32(si64), 0, 0, 2, 1);
    assert_eq_ps(sg_cvt_pi64_ps(si64), 0.0f, 0.0f, 2.0f, 1.0f);
    assert_eq_pd(sg_cvt_pi64_pd(si64), 2.0, 1.0);

    sg_ps psp = sg_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
    assert_eq_pd(sg_cvt_ps_pd(psp), 2.0, 1.0);

    psp = sg_set_ps(3.7f, 2.7f, 1.7f, 0.7f);
    sg_ps psn = sg_neg_ps(psp);
    assert_eq_pi32(sg_cvt_ps_pi32(psp), 4, 3, 2, 1);
    assert_eq_pi32(sg_cvtt_ps_pi32(psp), 3, 2, 1, 0);
    assert_eq_pi32(sg_cvtf_ps_pi32(psp), 3, 2, 1, 0);
    assert_eq_pi32(sg_cvt_ps_pi32(psn), -4, -3, -2, -1);
    assert_eq_pi32(sg_cvtt_ps_pi32(psn), -3, -2, -1, 0);
    assert_eq_pi32(sg_cvtf_ps_pi32(psn), -4, -3, -2, -1);
    assert_eq_pi64(sg_cvt_ps_pi64(psp), 2, 1);
    assert_eq_pi64(sg_cvtt_ps_pi64(psp), 1, 0);
    assert_eq_pi64(sg_cvtf_ps_pi64(psp), 1, 0);
    assert_eq_pi64(sg_cvt_ps_pi64(psn), -2, -1);
    assert_eq_pi64(sg_cvtt_ps_pi64(psn), -1, 0);
    assert_eq_pi64(sg_cvtf_ps_pi64(psn), -2, -1);

    sg_pd pdp = sg_set_pd(2.0, 1.0);
    assert_eq_ps(sg_cvt_pd_ps(pdp), 0.0f, 0.0f, 2.0f, 1.0f);

    pdp = sg_set_pd(1.7, 0.7);
    sg_pd pdn = sg_neg_pd(pdp);
    assert_eq_pi32(sg_cvt_pd_pi32(pdp), 0, 0, 2, 1);
    assert_eq_pi32(sg_cvtt_pd_pi32(pdp), 0, 0, 1, 0);
    assert_eq_pi32(sg_cvtf_pd_pi32(pdp), 0, 0, 1, 0);
    assert_eq_pi32(sg_cvt_pd_pi32(pdn), 0, 0, -2, -1);
    assert_eq_pi32(sg_cvtt_pd_pi32(pdn), 0, 0, -1, 0);
    assert_eq_pi32(sg_cvtf_pd_pi32(pdn), 0, 0, -2, -1);
    assert_eq_pi64(sg_cvt_pd_pi64(pdp), 2, 1);
    assert_eq_pi64(sg_cvtt_pd_pi64(pdp), 1, 0);
    assert_eq_pi64(sg_cvtf_pd_pi64(pdp), 1, 0);
    assert_eq_pi64(sg_cvt_pd_pi64(pdn), -2, -1);
    assert_eq_pi64(sg_cvtt_pd_pi64(pdn), -1, 0);
    assert_eq_pi64(sg_cvtf_pd_pi64(pdn), -2, -1);

    // Test half way rounding
    assert_eq_pi32(sg_cvt_ps_pi32(sg_set1_ps(2.5f)), 2, 2, 2, 2);
    assert_eq_pi32(sg_cvt_ps_pi32(sg_set1_ps(-2.5f)), -2, -2, -2, -2);
    assert_eq_pi64(sg_cvt_ps_pi64(sg_set1_ps(2.5f)), 2, 2);
    assert_eq_pi64(sg_cvt_ps_pi64(sg_set1_ps(-2.5f)), -2, -2);

    assert_eq_pi64(sg_cvt_pd_pi64(sg_set1_pd(2.5)), 2, 2);
    assert_eq_pi64(sg_cvt_pd_pi64(sg_set1_pd(-2.5)), -2, -2);
    assert_eq_pi32(sg_cvt_pd_pi32(sg_set1_pd(2.5)), 0, 0, 2, 2);
    assert_eq_pi32(sg_cvt_pd_pi32(sg_set1_pd(-2.5)), 0, 0, -2, -2);

    // Test edge cases when converting pi32 <-> pi64
    assert_eq_pi64(sg_cvt_pi32_pi64(sg_set1_pi32(-5)), -5, -5);
    assert_eq_pi32(sg_cvt_pi64_pi32(sg_set1_pi64(-5)), 0, 0, -5, -5);
    int64_t large = 1e11, large_neg = -large;
    //printf("large: %" PRId64 ", large_neg: %" PRId64 "\n", large, large_neg);
    //printf("large: %i, large_neg: %i\n", (int32_t) large, (int32_t) large_neg);
    assert_eq_pi32(sg_cvt_pi64_pi32(sg_set1_pi64(large)),
        0, 0, (int32_t) large, (int32_t) large);
    assert_eq_pi32(sg_cvt_pi64_pi32(sg_set1_pi64(large_neg)),
        0, 0, (int32_t) large_neg, (int32_t) large_neg);

    //printf("Convert test succeeeded\n");
}

void test_add_sub() {
    sg_pi32 a = sg_set_pi32(144, 24, 6, 1), b = sg_set_pi32(288, 48, 12, 2);
    assert_eq_pi32(sg_add_pi32(a, b), 432, 72, 18, 3);
    assert_eq_pi32(sg_sub_pi32(a, b), -144, -24, -6, -1);

    sg_pi64 c = sg_set_pi64(6, 1), d = sg_set_pi64(12, 2);
    assert_eq_pi64(sg_add_pi64(c, d), 18, 3);
    assert_eq_pi64(sg_sub_pi64(c, d), -6, -1);

    sg_ps e = sg_set_ps(144.0f, 24.0f, 6.0f, 1.0f),
        f = sg_set_ps(288.0f, 48.0f, 12.0f, 2.0f);
    assert_eq_ps(sg_add_ps(e, f), 432.0f, 72.0f, 18.0f, 3.0f);
    assert_eq_ps(sg_sub_ps(e, f), -144.0f, -24.0f, -6.0f, -1.0f);

    sg_pd g = sg_set_pd(6.0, 1.0), h = sg_set_pd(12.0, 2.0);
    assert_eq_pd(sg_add_pd(g, h), 18.0, 3.0);
    assert_eq_pd(sg_sub_pd(g, h), -6.0, -1.0);

    //printf("Add subtract test succeeeded\n");
}

void test_mul_div() {
    // We check safe_div along with div here with non-zero denominators,
    // then test zero denominators for safe_div at the end
    assert_eq_pi32(sg_mul_pi32(sg_set_pi32(17, 11, 5, 1),
        sg_set_pi32(13, 7, 3, 2)), 221, 77, 15, 2);
    assert_eq_pi32(sg_div_pi32(sg_set_pi32(98, 50, 18, 8),
        sg_set_pi32(14, 10, 6, 4)), 7, 5, 3, 2);
    assert_eq_pi32(sg_safediv_pi32(sg_set_pi32(98, 50, 18, 8),
        sg_set_pi32(14, 10, 6, 4)), 7, 5, 3, 2);

    // Extra test cases for our own implementation of mul pi32 on SSE2
    assert_eq_pi32(sg_mul_pi32(sg_set_pi32(-17, -11, -5, -1),
        sg_set_pi32(13, 7, 3, 2)), -221, -77, -15, -2);
    assert_eq_pi32(sg_mul_pi32(sg_set_pi32(17, 11, 5, 1),
        sg_set_pi32(-13, -7, -3, -2)), -221, -77, -15, -2);
    assert_eq_pi32(sg_mul_pi32(sg_set_pi32(-17, -11, -5, -1),
        sg_set_pi32(-13, -7, -3, -2)), 221, 77, 15, 2);

    assert_eq_pi64(sg_mul_pi64(sg_set_pi64(5, 1), sg_set_pi64(3, 2)), 15, 2);
    assert_eq_pi64(sg_div_pi64(sg_set_pi64(18, 8), sg_set_pi64(6, 4)), 3, 2);
    assert_eq_pi64(sg_safediv_pi64(sg_set_pi64(18, 8), sg_set_pi64(6, 4)), 3, 2);

    assert_eq_ps(sg_mul_ps(sg_set_ps(17.0f, 11.0f, 5.0f, 1.0f),
        sg_set_ps(13.0f, 7.0f, 3.0f, 2.0f)), 221.0f, 77.0f, 15.0f, 2.0f);
    assert_eq_ps(sg_div_ps(sg_set_ps(98.0f, 50.0f, 18.0f, 8.0f),
        sg_set_ps(14.0f, 10.0f, 6.0f, 4.0f)), 7.0f, 5.0f, 3.0f, 2.0f);
    assert_eq_ps(sg_safediv_ps(sg_set_ps(98.0f, 50.0f, 18.0f, 8.0f),
        sg_set_ps(14.0f, 10.0f, 6.0f, 4.0f)), 7.0f, 5.0f, 3.0f, 2.0f);

    assert_eq_pd(sg_mul_pd(sg_set_pd(5.0, 1.0), sg_set_pd(3.0, 2.0)),
        15.0, 2.0);
    assert_eq_pd(sg_div_pd(sg_set_pd(18.0, 8.0), sg_set_pd(6.0, 4.0)),
        3.0, 2.0);
    assert_eq_pd(sg_safediv_pd(sg_set_pd(18.0, 8.0), sg_set_pd(6.0, 4.0)),
        3.0, 2.0);

    // Test mul_add
    assert_eq_ps(sg_mul_add_ps(sg_set_ps(1.0f, 2.0f, 3.0f, 4.0f),
        sg_set_ps(5.0f, 6.0f, 7.0f, 8.0f),
        sg_set_ps(9.0f, 10.0f, 11.0f, 12.0f)),
        14.0f, 22.0f, 32.0f, 44.0f);
    assert_eq_pd(sg_mul_add_pd(sg_set_pd(1.0, 2.0), sg_set_pd(5.0, 6.0),
        sg_set_pd(9.0, 10.0)), 14.0, 22.0);

    // Test safediv
    assert_eq_pi32(sg_safediv_pi32(sg_set_pi32(8, 8, 8, 8),
        sg_set_pi32(4, 4, 4, 0)), 2, 2, 2, 8);
    assert_eq_pi32(sg_safediv_pi32(sg_set_pi32(8, 8, 8, 8),
        sg_set_pi32(4, 4, 0, 4)), 2, 2, 8, 2);
    assert_eq_pi32(sg_safediv_pi32(sg_set_pi32(8, 8, 8, 8),
        sg_set_pi32(4, 0, 4, 4)), 2, 8, 2, 2);
    assert_eq_pi32(sg_safediv_pi32(sg_set_pi32(8, 8, 8, 8),
        sg_set_pi32(0, 4, 4, 4)), 8, 2, 2, 2);

    assert_eq_pi64(sg_safediv_pi64(sg_set_pi64(8, 8),
        sg_set_pi64(4, 0)), 2, 8);
    assert_eq_pi64(sg_safediv_pi64(sg_set_pi64(8, 8),
        sg_set_pi64(0, 4)), 8, 2);

    assert_eq_ps(sg_safediv_ps(sg_set_ps(8.0f, 8.0f, 8.0f, 8.0f),
        sg_set_ps(4.0f, 4.0f, 4.0f, 0.0f)), 2.0f, 2.0f, 2.0f, 8.0f);
    assert_eq_ps(sg_safediv_ps(sg_set_ps(8.0f, 8.0f, 8.0f, 8.0f),
        sg_set_ps(4.0f, 4.0f, 4.0f, -0.0f)), 2.0f, 2.0f, 2.0f, 8.0f);

    assert_eq_ps(sg_safediv_ps(sg_set_ps(8.0f, 8.0f, 8.0f, 8.0f),
        sg_set_ps(4.0f, 4.0f, 0.0f, 4.0f)), 2.0f, 2.0f, 8.0f, 2.0f);
    assert_eq_ps(sg_safediv_ps(sg_set_ps(8.0f, 8.0f, 8.0f, 8.0f),
        sg_set_ps(4.0f, 4.0f, -0.0f, 4.0f)), 2.0f, 2.0f, 8.0f, 2.0f);

    assert_eq_ps(sg_safediv_ps(sg_set_ps(8.0f, 8.0f, 8.0f, 8.0f),
        sg_set_ps(4.0f, 0.0f, 4.0f, 4.0f)), 2.0f, 8.0f, 2.0f, 2.0f);
    assert_eq_ps(sg_safediv_ps(sg_set_ps(8.0f, 8.0f, 8.0f, 8.0f),
        sg_set_ps(4.0f, -0.0f, 4.0f, 4.0f)), 2.0f, 8.0f, 2.0f, 2.0f);

    assert_eq_ps(sg_safediv_ps(sg_set_ps(8.0f, 8.0f, 8.0f, 8.0f),
        sg_set_ps(0.0f, 4.0f, 4.0f, 4.0f)), 8.0f, 2.0f, 2.0f, 2.0f);
    assert_eq_ps(sg_safediv_ps(sg_set_ps(8.0f, 8.0f, 8.0f, 8.0f),
        sg_set_ps(-0.0f, 4.0f, 4.0f, 4.0f)), 8.0f, 2.0f, 2.0f, 2.0f);

    assert_eq_pd(sg_safediv_pd(sg_set_pd(8.0, 8.0),
        sg_set_pd(4.0, 0.0)), 2.0, 8.0);
    assert_eq_pd(sg_safediv_pd(sg_set_pd(8.0, 8.0),
        sg_set_pd(4.0, -0.0)), 2.0, 8.0);

    assert_eq_pd(sg_safediv_pd(sg_set_pd(8.0, 8.0),
        sg_set_pd(0.0, 4.0)), 8.0, 2.0);
    assert_eq_pd(sg_safediv_pd(sg_set_pd(8.0, 8.0),
        sg_set_pd(-0.0, 4.0)), 8.0, 2.0);

    //printf("Multiply divide test succeeeded\n");
}

void test_bitwise() {
    // Warning: disabling denormals may cause this test to incorrectly fail
    for (int a3 = 0; a3 < 2; ++a3) {
    for (int a2 = 0; a2 < 2; ++a2) {
    for (int a1 = 0; a1 < 2; ++a1) {
    for (int a0 = 0; a0 < 2; ++a0) {
    for (int b3 = 0; b3 < 2; ++b3) {
    for (int b2 = 0; b2 < 2; ++b2) {
    for (int b1 = 0; b1 < 2; ++b1) {
    for (int b0 = 0; b0 < 2; ++b0) {
        sg_pi32 ai32 = sg_set_pi32(a3, a2, a1, a0),
            bi32 = sg_set_pi32(b3, b2, b1, b0);
        sg_pi64 ai64 = sg_set_pi64(a1, a0),
            bi64 = sg_set_pi64(b1, b0);
        sg_ps aps = sg_bitcast_pi32_ps(ai32), bps = sg_bitcast_pi32_ps(bi32);
        sg_pd apd = sg_bitcast_pi64_pd(ai64), bpd = sg_bitcast_pi64_pd(bi64);

        assert_eq_pi32(sg_and_pi32(ai32, bi32),
            a3 & b3, a2 & b2, a1 & b1, a0 & b0);
        assert_eq_pi32(sg_bitcast_ps_pi32(sg_and_ps(aps, bps)),
            a3 & b3, a2 & b2, a1 & b1, a0 & b0);
        assert_eq_pi32(sg_andnot_pi32(ai32, bi32),
            ~a3 & b3, ~a2 & b2, ~a1 & b1, ~a0 & b0);
        assert_eq_pi32(sg_bitcast_ps_pi32(sg_andnot_ps(aps, bps)),
            ~a3 & b3, ~a2 & b2, ~a1 & b1, ~a0 & b0);
        assert_eq_pi32(sg_not_pi32(ai32), ~a3, ~a2, ~a1, ~a0);
        assert_eq_pi32(sg_bitcast_ps_pi32(sg_not_ps(aps)),  ~a3, ~a2, ~a1, ~a0);
        assert_eq_pi32(sg_or_pi32(ai32, bi32),
            a3 | b3, a2 | b2, a1 | b1, a0 | b0);
        assert_eq_pi32(sg_bitcast_ps_pi32(sg_or_ps(aps, bps)),
            a3 | b3, a2 | b2, a1 | b1, a0 | b0);
        assert_eq_pi32(sg_xor_pi32(ai32, bi32),
            a3 ^ b3, a2 ^ b2, a1 ^ b1, a0 ^ b0);
        assert_eq_pi32(sg_bitcast_ps_pi32(sg_xor_ps(aps, bps)),
            a3 ^ b3, a2 ^ b2, a1 ^ b1, a0 ^ b0);

        assert_eq_pi64(sg_and_pi64(ai64, bi64), a1 & b1, a0 & b0);
        assert_eq_pi64(sg_bitcast_pd_pi64(sg_and_pd(apd, bpd)), a1 & b1, a0 & b0);
        assert_eq_pi64(sg_andnot_pi64(ai64, bi64), ~a1 & b1, ~a0 & b0);
        assert_eq_pi64(sg_bitcast_pd_pi64(sg_andnot_pd(apd, bpd)),
            ~a1 & b1, ~a0 & b0);
        assert_eq_pi64(sg_not_pi64(ai64), ~a1, ~a0);
        assert_eq_pi64(sg_bitcast_pd_pi64(sg_not_pd(apd)), ~a1, ~a0);
        assert_eq_pi64(sg_or_pi64(ai64, bi64), a1 | b1, a0 | b0);
        assert_eq_pi64(sg_bitcast_pd_pi64(sg_or_pd(apd, bpd)), a1 | b1, a0 | b0);
        assert_eq_pi64(sg_xor_pi64(ai64, bi64), a1 ^ b1, a0 ^ b0);
        assert_eq_pi64(sg_bitcast_pd_pi64(sg_xor_pd(apd, bpd)), a1 ^ b1, a0 ^ b0);
    } } } } } } } }

    //printf("Bitwise test succeeeded\n");
}

void test_shift() {
    // Test immediate
    assert_eq_pi32(sg_sl_imm_pi32(sg_set_pi32(64, 16, 4, 1), 1), 128, 32, 8, 2);
    assert_eq_pi32(sg_sra_imm_pi32(sg_set_pi32(-64, -16, -4, -2), 1),
        -32, -8, -2, -1);
    assert_eq_pi32(sg_srl_imm_pi32(sg_set_pi32(-64, -16, -4, -2), 1),
        2147483616, 2147483640, 2147483646, 2147483647);

    assert_eq_pi64(sg_sl_imm_pi64(sg_set_pi64(4, 1), 1), 8, 2);
    assert_eq_pi64(sg_sra_imm_pi64(sg_set_pi64(-4, -2), 1), -2, -1);
    assert_eq_pi64(sg_srl_imm_pi64(sg_set_pi64(-4, -2), 1),
        9223372036854775806, 9223372036854775807);

    // Test in-register
    assert_eq_pi32(sg_sl_pi32(sg_set_pi32(8, 4, 2, 1), sg_set_pi32(4, 3, 2, 1)),
        128, 32, 8, 2);
    assert_eq_pi32(sg_sra_pi32(sg_set_pi32(-64, -16, -4, -2),
        sg_set_pi32(4, 3, 2, 1)), -4, -2, -1, -1);
    assert_eq_pi32(sg_srl_pi32(sg_set_pi32(-64, -16, -4, -2),
        sg_set_pi32(4, 3, 2, 1)),
        268435452, 536870910, 1073741823, 2147483647);

    assert_eq_pi64(sg_sl_pi64(sg_set_pi64(2, 1), sg_set_pi64(2, 1)), 8, 2);
    assert_eq_pi64(sg_sra_pi64(sg_set_pi64(-4, -2), sg_set_pi64(2, 1)), -1, -1);
    assert_eq_pi64(sg_srl_pi64(sg_set_pi64(-4, -2), sg_set_pi64(2, 1)),
        4611686018427387903, 9223372036854775807);
    // Test shift with negative numbers
    //print_pi32(sg_sl_imm_pi32(sg_set1_pi32(2), -1)); printf("\n");

    //printf("Shift test succeeeded\n");
}

void test_cmp() {
    // Test that 0.0 and -0.0 are equal
    const sg_ps z_ps = sg_setzero_ps(), nz_ps = sg_set1_ps(-0.0f);
    const sg_pd z_pd = sg_setzero_pd(), nz_pd = sg_set1_pd(-0.0);
    assert_eq_cmp_ps(sg_cmpeq_ps(z_ps, nz_ps), true, true, true, true);
    assert_eq_cmp_pd(sg_cmpeq_pd(z_pd, nz_pd), true, true);

    // Test comparisons happen in the correct lanes
    for (int a0 = 1; a0 < 4; ++a0) {
    for (int a1 = 4; a1 < 7; ++a1) {
    for (int a2 = 8; a2 < 11; ++a2) {
    for (int a3 = 11; a3 < 14; ++a3) {
    for (int b0 = 1; b0 < 4; ++b0) {
    for (int b1 = 4; b1 < 7; ++b1) {
    for (int b2 = 8; b2 < 11; ++b2) {
    for (int b3 = 11; b3 < 14; ++ b3) {
        const sg_pi32 a_pi32 = sg_set_pi32(a3, a2, a1, a0),
            b_pi32 = sg_set_pi32(b3, b2, b1, b0);
        const sg_pi64 a_pi64 = sg_cvt_pi32_pi64(a_pi32),
            b_pi64 = sg_cvt_pi32_pi64(b_pi32);
        const sg_ps a_ps = sg_cvt_pi32_ps(a_pi32),
            b_ps = sg_cvt_pi32_ps(b_pi32);
        const sg_pd a_pd = sg_cvt_pi32_pd(a_pi32),
            b_pd = sg_cvt_pi32_pd(b_pi32);

        sg_generic_cmp4 cmp4_lt = { a0 < b0, a1 < b1, a2 < b2, a3 < b3 },
            cmp4_lte = { a0 <= b0, a1 <= b1, a2 <= b2, a3 <= b3 },
            cmp4_eq = { a0 == b0, a1 == b1, a2 == b2, a3 == b3 },
            cmp4_neq = { a0 != b0, a1 != b1, a2 != b2, a3 != b3 },
            cmp4_gte = { a0 >= b0, a1 >= b1, a2 >= b2, a3 >= b3},
            cmp4_gt = { a0 > b0, a1 > b1, a2 > b2, a3 > b3 };

        sg_generic_cmp2 cmp2_lt = { cmp4_lt.b0, cmp4_lt.b1 },
            cmp2_lte = { cmp4_lte.b0, cmp4_lte.b1 },
            cmp2_eq = { cmp4_eq.b0, cmp4_eq.b1 },
            cmp2_neq = { cmp4_neq.b0, cmp4_neq.b1 },
            cmp2_gte = { cmp4_gte.b0, cmp4_gte.b1 },
            cmp2_gt = { cmp4_gt.b0, cmp4_gt.b1 };

        assert_eqg_cmp_pi32(sg_cmplt_pi32(a_pi32, b_pi32), cmp4_lt);
        assert_eqg_cmp_ps(sg_cmplt_ps(a_ps, b_ps), cmp4_lt);
        assert_eqg_cmp_pi64(sg_cmplt_pi64(a_pi64, b_pi64), cmp2_lt);
        assert_eqg_cmp_pd(sg_cmplt_pd(a_pd, b_pd), cmp2_lt);

        assert_eqg_cmp_pi32(sg_cmplte_pi32(a_pi32, b_pi32), cmp4_lte);
        assert_eqg_cmp_ps(sg_cmplte_ps(a_ps, b_ps), cmp4_lte);
        assert_eqg_cmp_pi64(sg_cmplte_pi64(a_pi64, b_pi64), cmp2_lte);
        assert_eqg_cmp_pd(sg_cmplte_pd(a_pd, b_pd), cmp2_lte);

        assert_eqg_cmp_pi32(sg_cmpeq_pi32(a_pi32, b_pi32), cmp4_eq);
        assert_eqg_cmp_ps(sg_cmpeq_ps(a_ps, b_ps), cmp4_eq);
        assert_eqg_cmp_pi64(sg_cmpeq_pi64(a_pi64, b_pi64), cmp2_eq);
        assert_eqg_cmp_pd(sg_cmpeq_pd(a_pd, b_pd), cmp2_eq);

        assert_eqg_cmp_pi32(sg_cmpneq_pi32(a_pi32, b_pi32), cmp4_neq);
        assert_eqg_cmp_ps(sg_cmpneq_ps(a_ps, b_ps), cmp4_neq);
        assert_eqg_cmp_pi64(sg_cmpneq_pi64(a_pi64, b_pi64), cmp2_neq);
        assert_eqg_cmp_pd(sg_cmpneq_pd(a_pd, b_pd), cmp2_neq);

        assert_eqg_cmp_pi32(sg_cmpgte_pi32(a_pi32, b_pi32), cmp4_gte);
        assert_eqg_cmp_ps(sg_cmpgte_ps(a_ps, b_ps), cmp4_gte);
        assert_eqg_cmp_pi64(sg_cmpgte_pi64(a_pi64, b_pi64), cmp2_gte);
        assert_eqg_cmp_pd(sg_cmpgte_pd(a_pd, b_pd), cmp2_gte);

        assert_eqg_cmp_pi32(sg_cmpgt_pi32(a_pi32, b_pi32), cmp4_gt);
        assert_eqg_cmp_ps(sg_cmpgt_ps(a_ps, b_ps), cmp4_gt);
        assert_eqg_cmp_pi64(sg_cmpgt_pi64(a_pi64, b_pi64), cmp2_gt);
        assert_eqg_cmp_pd(sg_cmpgt_pd(a_pd, b_pd), cmp2_gt);
    } } } } } } } }

    // Some extra test cases for our own implementation of sg_cmpeq_pi64
    // on SSE2
    assert_eq_cmp_pi64(sg_cmpeq_pi64(
        sg_bitcast_pi32_pi64(sg_set_pi32(8, 8, 7, 7)),
        sg_bitcast_pi32_pi64(sg_set_pi32(8, 7, 7, 8))),
        false, false);
    assert_eq_cmp_pi64(sg_cmpeq_pi64(
        sg_bitcast_pi32_pi64(sg_set_pi32(8, 8, 7, 7)),
        sg_bitcast_pi32_pi64(sg_set_pi32(7, 8, 8, 7))),
        false, false);
    assert_eq_cmp_pi64(sg_cmpeq_pi64(
        sg_bitcast_pi32_pi64(sg_set_pi32(8, 8, 7, 7)),
        sg_bitcast_pi32_pi64(sg_set_pi32(8, 8, 7, 7))),
        true, true);

    // Test cast & convert
    for (int a0 = 0; a0 < 2; ++a0) {
    for (int a1 = 0; a1 < 2; ++a1) {
    for (int a2 = 0; a2 < 2; ++a2) {
    for (int a3 = 0; a3 < 2; ++a3) {
        sg_generic_cmp4 gcmp4 = { (bool) a0, (bool) a1,
            (bool) a2, (bool) a3 };
        sg_generic_cmp4 gcmp4_lower = { (bool) a0, (bool) a1, 0, 0 };
        sg_generic_cmp2 gcmp2 = { (bool) a0, (bool) a1 };
        sg_cmp_pi32 cmp_pi32 = sg_setcmp_fromg_pi32(gcmp4);
        sg_cmp_pi64 cmp_pi64 = sg_setcmp_fromg_pi64(gcmp2);
        sg_cmp_ps cmp_ps = sg_cvtcmp_pi32_ps(cmp_pi32);
        sg_cmp_pd cmp_pd = sg_cvtcmp_pi64_pd(cmp_pi64);

        // Cast
        assert_eqg_cmp_ps(cmp_ps, gcmp4);
        assert_eqg_cmp_pd(cmp_pd, gcmp2);
        assert_eqg_cmp_pi32(sg_cvtcmp_ps_pi32(cmp_ps), gcmp4);
        assert_eqg_cmp_pi64(sg_cvtcmp_pd_pi64(cmp_pd), gcmp2);

        // Convert
        assert_eqg_cmp_pi64(sg_cvtcmp_pi32_pi64(cmp_pi32), gcmp2);
        assert_eqg_cmp_pi32(sg_cvtcmp_pi64_pi32(cmp_pi64), gcmp4_lower);
        assert_eqg_cmp_pd(sg_cvtcmp_pi32_pd(cmp_pi32), gcmp2);
        assert_eqg_cmp_pi32(sg_cvtcmp_pd_pi32(cmp_pd), gcmp4_lower);
        assert_eqg_cmp_ps(sg_cvtcmp_pd_ps(cmp_pd), gcmp4_lower);
        assert_eqg_cmp_pd(sg_cvtcmp_ps_pd(cmp_ps), gcmp2);
        assert_eqg_cmp_pi64(sg_cvtcmp_ps_pi64(cmp_ps), gcmp2);
        assert_eqg_cmp_ps(sg_cvtcmp_pi64_ps(cmp_pi64), gcmp4_lower);
    } } } }

    // Test logic operations on comparison functions
    for (int a0 = 0; a0 < 2; ++a0) {
    for (int a1 = 0; a1 < 2; ++a1) {
    for (int a2 = 0; a2 < 2; ++a2) {
    for (int a3 = 0; a3 < 2; ++a3) {
    for (int b0 = 0; b0 < 2; ++b0) {
    for (int b1 = 0; b1 < 2; ++b1) {
    for (int b2 = 0; b2 < 2; ++b2) {
    for (int b3 = 0; b3 < 2; ++b3) {
        sg_generic_cmp4 a_gcmp4 = { .b0 = (bool) a0, .b1 = (bool) a1,
            .b2 = (bool) a2, .b3 = (bool) a3 },
            b_gcmp4 = { .b0 = (bool) b0, .b1 = (bool) b1,
                .b2 = (bool) b2, .b3 = (bool) b3 },
            a_and_b = { .b0 = (bool) a0 && (bool) b0,
                .b1 = (bool) a1 && (bool) b1,
                .b2 = (bool) a2 && (bool) b2, .b3 = (bool) a3 && (bool) b3 },
            a_andnot_b = { .b0 = !((bool) a0) && (bool) b0,
                .b1 = !((bool) a1) && (bool) b1,
                .b2 = !((bool) a2) && (bool) b2,
                .b3 = !((bool) a3) && (bool) b3 },
            a_not = { .b0 = !(bool) a0, .b1 = !(bool) a1,
                .b2 = !(bool) a2, .b3 = !(bool) a3 },
            a_or_b = { .b0 = (bool) a0 || (bool) b0,
                .b1 = (bool) a1 || (bool) b1,
                .b2 = (bool) a2 || (bool) b2, .b3 = (bool) a3 || (bool) b3 },
            a_xor_b = { .b0 = (bool) a0 != (bool) b0,
                .b1 = (bool) a1 != (bool) b1,
                .b2 = (bool) a2 != (bool) b2, .b3 = (bool) a3 != (bool) b3 },
            a_eq_b = { .b0 = (bool) a0 == (bool) b0,
                .b1 = (bool) a1 == (bool) b1,
                .b2 = (bool) a2 == (bool) b2, .b3 = (bool) a3 == (bool) b3 };
        sg_generic_cmp2 a_gcmp2 = { .b0 = (bool) a0, .b1 = (bool) a1 },
            b_gcmp2 = { .b0 = (bool) b0, .b1 = (bool) b1 },
            a_and_b_2 = { .b0 = a_and_b.b0, .b1 = a_and_b.b1 },
            a_andnot_b_2 = { .b0 = a_andnot_b.b0, .b1 = a_andnot_b.b1 },
            a_not_2 = { .b0 = a_not.b0, .b1 = a_not.b1 },
            a_or_b_2 = { .b0 = a_or_b.b0, .b1 = a_or_b.b1 },
            a_xor_b_2 = { .b0 = a_xor_b.b0, .b1 = a_xor_b.b1 },
            a_eq_b_2 = { .b0 = a_eq_b.b0, .b1 = a_eq_b.b1 };
        sg_cmp_pi32 a_pi32 = sg_setcmp_fromg_pi32(a_gcmp4),
            b_pi32 = sg_setcmp_fromg_pi32(b_gcmp4);
        sg_cmp_pi64 a_pi64 = sg_setcmp_fromg_pi64(a_gcmp2),
            b_pi64 = sg_setcmp_fromg_pi64(b_gcmp2);
        sg_cmp_ps a_ps = sg_cvtcmp_pi32_ps(a_pi32),
            b_ps = sg_cvtcmp_pi32_ps(b_pi32);
        sg_cmp_pd a_pd = sg_cvtcmp_pi64_pd(a_pi64),
            b_pd = sg_cvtcmp_pi64_pd(b_pi64);

        assert_eqg_cmp_pi32(sg_and_cmp_pi32(a_pi32, b_pi32), a_and_b);
        assert_eqg_cmp_pi64(sg_and_cmp_pi64(a_pi64, b_pi64), a_and_b_2);
        assert_eqg_cmp_ps(sg_and_cmp_ps(a_ps, b_ps), a_and_b);
        assert_eqg_cmp_pd(sg_and_cmp_pd(a_pd, b_pd), a_and_b_2);

        assert_eqg_cmp_pi32(sg_andnot_cmp_pi32(a_pi32, b_pi32), a_andnot_b);
        assert_eqg_cmp_pi64(sg_andnot_cmp_pi64(a_pi64, b_pi64), a_andnot_b_2);
        assert_eqg_cmp_ps(sg_andnot_cmp_ps(a_ps, b_ps), a_andnot_b);
        assert_eqg_cmp_pd(sg_andnot_cmp_pd(a_pd, b_pd), a_andnot_b_2);

        assert_eqg_cmp_pi32(sg_not_cmp_pi32(a_pi32), a_not);
        assert_eqg_cmp_pi64(sg_not_cmp_pi64(a_pi64), a_not_2);
        assert_eqg_cmp_ps(sg_not_cmp_ps(a_ps), a_not);
        assert_eqg_cmp_pd(sg_not_cmp_pd(a_pd), a_not_2);

        assert_eqg_cmp_pi32(sg_or_cmp_pi32(a_pi32, b_pi32), a_or_b);
        assert_eqg_cmp_pi64(sg_or_cmp_pi64(a_pi64, b_pi64), a_or_b_2);
        assert_eqg_cmp_ps(sg_or_cmp_ps(a_ps, b_ps), a_or_b);
        assert_eqg_cmp_pd(sg_or_cmp_pd(a_pd, b_pd), a_or_b_2);

        assert_eqg_cmp_pi32(sg_xor_cmp_pi32(a_pi32, b_pi32), a_xor_b);
        assert_eqg_cmp_pi32(sg_cmpneq_cmp_pi32(a_pi32, b_pi32), a_xor_b);
        assert_eqg_cmp_pi64(sg_xor_cmp_pi64(a_pi64, b_pi64), a_xor_b_2);
        assert_eqg_cmp_pi64(sg_cmpneq_cmp_pi64(a_pi64, b_pi64), a_xor_b_2);
        assert_eqg_cmp_ps(sg_xor_cmp_ps(a_ps, b_ps), a_xor_b);
        assert_eqg_cmp_ps(sg_cmpneq_cmp_ps(a_ps, b_ps), a_xor_b);
        assert_eqg_cmp_pd(sg_xor_cmp_pd(a_pd, b_pd), a_xor_b_2);
        assert_eqg_cmp_pd(sg_cmpneq_cmp_pd(a_pd, b_pd), a_xor_b_2);

        assert_eqg_cmp_pi32(sg_cmpeq_cmp_pi32(a_pi32, b_pi32), a_eq_b);
        assert_eqg_cmp_pi64(sg_cmpeq_cmp_pi64(a_pi64, b_pi64), a_eq_b_2);
        assert_eqg_cmp_ps(sg_cmpeq_cmp_ps(a_ps, b_ps), a_eq_b);
        assert_eqg_cmp_pd(sg_cmpeq_cmp_pd(a_pd, b_pd), a_eq_b_2);
    } } } } } } } }

    // Test choosers
    const int32_t true_val = 2, false_val = 3;
    const sg_pi32 if_true_pi32 = sg_set1_pi32(true_val),
        if_false_pi32 = sg_set1_pi32(false_val);
    const sg_pi64 if_true_pi64 = sg_set1_pi64(true_val),
        if_false_pi64 = sg_set1_pi64(false_val);
    const sg_ps if_true_ps = sg_set1_ps(true_val),
        if_false_ps = sg_set1_ps(false_val);
    const sg_pd if_true_pd = sg_set1_pd(true_val),
        if_false_pd = sg_set1_pd(false_val);

    for (int c0 = 0; c0 < 2; ++c0) {
    for (int c1 = 0; c1 < 2; ++c1) {
    for (int c2 = 0; c2 < 2; ++c2) {
    for (int c3 = 0; c3 < 2; ++c3) {
        const sg_generic_cmp2 cmp2 = { .b0 = (bool) c0, .b1 = (bool) c1 };
        const sg_generic_cmp4 cmp4 = { .b0 = (bool) c0, .b1 = (bool) c1,
            .b2 = (bool) c2, .b3 = (bool) c3 };
        const sg_generic_pi32 exp_pi32 = {
            .i0 = c0 ? true_val : false_val,
            .i1 = c1 ? true_val : false_val,
            .i2 = c2 ? true_val : false_val,
            .i3 = c3 ? true_val : false_val };
        const sg_generic_pi32 exp_oz_pi32 = {
            .i0 = c0 ? true_val : 0,
            .i1 = c1 ? true_val : 0,
            .i2 = c2 ? true_val : 0,
            .i3 = c3 ? true_val : 0 };
        const sg_generic_pi64 exp_pi64 = { .l0 = exp_pi32.i0, .l1 = exp_pi32.i1 },
            exp_oz_pi64 = { .l0 = exp_oz_pi32.i0, .l1 = exp_oz_pi32.i1 };
        const sg_generic_ps exp_ps = { .f0 = (float) exp_pi32.i0,
            .f1 = (float) exp_pi32.i1,
            .f2 = (float) exp_pi32.i2, .f3 = (float) exp_pi32.i3 },
            exp_oz_ps = { .f0 = (float) exp_oz_pi32.i0,
                .f1 = (float) exp_oz_pi32.i1,
                .f2 = (float) exp_oz_pi32.i2, .f3 = (float) exp_oz_pi32.i3 };
        const sg_generic_pd exp_pd = { .d0 = (double) exp_pi32.i0,
            .d1 = (double) exp_pi32.i1 },
            exp_oz_pd = { .d0 = (double) exp_oz_pi32.i0,
                .d1 = (double) exp_oz_pi32.i1 };

        const sg_cmp_pi32 cmp_pi32 = sg_setcmp_fromg_pi32(cmp4);
        const sg_cmp_pi64 cmp_pi64 = sg_setcmp_fromg_pi64(cmp2);
        const sg_cmp_ps cmp_ps = sg_setcmp_fromg_ps(cmp4);
        const sg_cmp_pd cmp_pd = sg_setcmp_fromg_pd(cmp2);

        assert_eq_pi32(sg_choose_pi32(cmp_pi32, if_true_pi32, if_false_pi32),
            exp_pi32.i3, exp_pi32.i2, exp_pi32.i1, exp_pi32.i0);
        assert_eq_pi32(sg_choose_else_zero_pi32(cmp_pi32, if_true_pi32),
            exp_oz_pi32.i3, exp_oz_pi32.i2, exp_oz_pi32.i1, exp_oz_pi32.i0);

        assert_eq_pi64(sg_choose_pi64(cmp_pi64, if_true_pi64, if_false_pi64),
            exp_pi64.l1, exp_pi64.l0);
        assert_eq_pi64(sg_choose_else_zero_pi64(cmp_pi64, if_true_pi64),
            exp_oz_pi64.l1, exp_oz_pi64.l0);

        assert_eq_ps(sg_choose_ps(cmp_ps, if_true_ps, if_false_ps),
            exp_ps.f3, exp_ps.f2, exp_ps.f1, exp_ps.f0);
        assert_eq_ps(sg_choose_else_zero_ps(cmp_ps, if_true_ps),
            exp_oz_ps.f3, exp_oz_ps.f2, exp_oz_ps.f1, exp_oz_ps.f0);

        assert_eq_pd(sg_choose_pd(cmp_pd, if_true_pd, if_false_pd),
            exp_pd.d1, exp_pd.d0);
        assert_eq_pd(sg_choose_else_zero_pd(cmp_pd, if_true_pd),
            exp_oz_pd.d1, exp_oz_pd.d0);
    }}}}

    //printf("Comparison test succeeeded\n");
}

void test_abs_neg() {
    assert_eq_pi32(sg_abs_pi32(sg_set_pi32(3, 2, 1, 0)), 3, 2, 1, 0);
    assert_eq_pi32(sg_abs_pi32(sg_set_pi32(-3, -2, -1, 0)), 3, 2, 1, 0);
    assert_eq_pi32(sg_neg_pi32(sg_set_pi32(3, 2, 1, 0)), -3, -2, -1, 0);
    assert_eq_pi32(sg_neg_pi32(sg_set_pi32(-3, -2, -1, 0)), 3, 2, 1, 0);

    assert_eq_pi64(sg_abs_pi64(sg_set_pi64(1, 0)), 1, 0);
    assert_eq_pi64(sg_abs_pi64(sg_set_pi64(-1, 0)), 1, 0);
    assert_eq_pi64(sg_neg_pi64(sg_set_pi64(1, 0)), -1, 0);
    assert_eq_pi64(sg_neg_pi64(sg_set_pi64(-1, 0)), 1, 0);

    assert_eq_ps(sg_abs_ps(sg_set_ps(3.0f, 2.0f, 1.0f, 0.0f)),
        3.0f, 2.0f, 1.0f, 0.0f);
    assert_eq_ps(sg_abs_ps(sg_set_ps(-3.0f, -2.0f, -1.0f, -0.0f)),
        3.0f, 2.0f, 1.0f, 0.0f);
    assert_eq_ps(sg_neg_ps(sg_set_ps(3.0f, 2.0f, 1.0f, 0.0f)),
        -3.0f, -2.0f, -1.0f, -0.0f);
    assert_eq_ps(sg_neg_ps(sg_set_ps(-3.0f, -2.0f, -1.0f, -0.0f)),
        3.0f, 2.0f, 1.0f, 0.0f);

    assert_eq_pd(sg_abs_pd(sg_set_pd(1.0, 0.0)), 1.0, 0.0);
    assert_eq_pd(sg_abs_pd(sg_set_pd(-1.0, -0.0)), 1.0, 0.0);
    assert_eq_pd(sg_neg_pd(sg_set_pd(1.0, 0.0)), -1.0, -0.0);
    assert_eq_pd(sg_neg_pd(sg_set_pd(-1.0, -0.0)), 1.0, 0.0);

    // Test remove signed zero
    assert_eq_ps(sg_remove_signed_zero_ps(sg_set_ps(3.0f, 2.0f, 1.0f, 0.0f)),
        3.0f, 2.0f, 1.0f, 0.0f);
    assert_eq_ps(sg_remove_signed_zero_ps(
        sg_set_ps(-3.0f, -2.0f, -1.0f, -0.0f)), -3.0f, -2.0f, -1.0f, 0.0f);
    assert_eq_pi32(sg_bitcast_ps_pi32(
        sg_remove_signed_zero_ps(sg_set_ps(0.0f, 0.0f, 0.0f, 0.0f))),
        0, 0, 0, 0);
    assert_eq_pi32(sg_bitcast_ps_pi32(
        sg_remove_signed_zero_ps(sg_set_ps(-0.0f, -0.0f, -0.0f, -0.0f))),
        0, 0, 0, 0);

    //printf("Abs and neg test succeeeded\n");
}

void test_min_max() {
    assert_eq_pi32(sg_max_pi32(
        sg_set_pi32(3, 2, 1, 0), sg_set_pi32(3, 2, 1, 0)),
        3, 2, 1, 0);
    assert_eq_pi32(sg_max_pi32(
        sg_set_pi32(8, 6, 4, 2), sg_set_pi32(7, 5, 3, 1)),
        8, 6, 4, 2);
    assert_eq_pi32(sg_max_pi32(
        sg_set_pi32(7, 5, 3, 1), sg_set_pi32(8, 6, 4, 2)),
        8, 6, 4, 2);

    assert_eq_pi32(sg_min_pi32(
        sg_set_pi32(3, 2, 1, 0), sg_set_pi32(3, 2, 1, 0)),
        3, 2, 1, 0);
    assert_eq_pi32(sg_min_pi32(
        sg_set_pi32(8, 6, 4, 2), sg_set_pi32(7, 5, 3, 1)),
        7, 5, 3, 1);
    assert_eq_pi32(sg_min_pi32(
        sg_set_pi32(7, 5, 3, 1), sg_set_pi32(8, 6, 4, 2)),
        7, 5, 3, 1);

    assert_eq_pi64(sg_max_pi64(sg_set_pi64(1, 0), sg_set_pi64(1, 0)),
        1, 0);
    assert_eq_pi64(sg_max_pi64(sg_set_pi64(4, 2), sg_set_pi64(1, 0)),
        4, 2);
    assert_eq_pi64(sg_max_pi64(sg_set_pi64(1, 0), sg_set_pi64(4, 2)),
        4, 2);

    assert_eq_pi64(sg_min_pi64(sg_set_pi64(1, 0), sg_set_pi64(1, 0)),
        1, 0);
    assert_eq_pi64(sg_min_pi64(sg_set_pi64(4, 2), sg_set_pi64(1, 0)),
        1, 0);
    assert_eq_pi64(sg_min_pi64(sg_set_pi64(1, 0), sg_set_pi64(4, 2)),
        1, 0);

    assert_eq_ps(sg_max_ps(
        sg_set_ps(3.0f, 2.0f, 1.0f, 0.0f), sg_set_ps(3.0f, 2.0f, 1.0f, 0.0f)),
        3.0f, 2.0f, 1.0f, 0.0f);
    assert_eq_ps(sg_max_ps(
        sg_set_ps(8.0f, 6.0f, 4.0f, 2.0f), sg_set_ps(7.0f, 5.0f, 3.0f, 1.0f)),
        8.0f, 6.0f, 4.0f, 2.0f);
    assert_eq_ps(sg_max_ps(
        sg_set_ps(7.0f, 5.0f, 3.0f, 1.0f), sg_set_ps(8.0f, 6.0f, 4.0f, 2.0f)),
        8.0f, 6.0f, 4.0f, 2.0f);

    assert_eq_ps(sg_min_ps(
        sg_set_ps(3.0f, 2.0f, 1.0f, 0.0f), sg_set_ps(3.0f, 2.0f, 1.0f, 0.0f)),
        3.0f, 2.0f, 1.0f, 0.0f);
    assert_eq_ps(sg_min_ps(
        sg_set_ps(8.0f, 6.0f, 4.0f, 2.0f), sg_set_ps(7.0f, 5.0f, 3.0f, 1.0f)),
        7.0f, 5.0f, 3.0f, 1.0f);
    assert_eq_ps(sg_min_ps(
        sg_set_ps(7.0f, 5.0f, 3.0f, 1.0f), sg_set_ps(8.0f, 6.0f, 4.0f, 2.0f)),
        7.0f, 5.0f, 3.0f, 1.0f);

    assert_eq_pd(sg_max_pd(
        sg_set_pd(1.0, 0.0), sg_set_pd(1.0, 0.0)),
        1.0, 0.0);
    assert_eq_pd(sg_max_pd(
        sg_set_pd(4.0, 2.0), sg_set_pd(3.0, 1.0)),
        4.0, 2.0);
    assert_eq_pd(sg_max_pd(
        sg_set_pd(3.0, 1.0), sg_set_pd(4.0, 2.0)),
        4.0, 2.0);

    assert_eq_pd(sg_min_pd(
        sg_set_pd(1.0, 0.0), sg_set_pd(1.0, 0.0)),
        1.0, 0.0);
    assert_eq_pd(sg_min_pd(
        sg_set_pd(4.0, 2.0), sg_set_pd(3.0, 1.0)),
        3.0, 1.0);
    assert_eq_pd(sg_min_pd(
        sg_set_pd(3.0, 1.0), sg_set_pd(4.0, 2.0)),
        3.0, 1.0);
    //printf("Min max test succeeded\n");
}

void test_constrain() {
    assert_eq_pi32(sg_constrain_pi32(sg_set1_pi32(1), sg_set1_pi32(3),
        sg_set1_pi32(0)), 1, 1, 1, 1);
    assert_eq_pi32(sg_constrain_pi32(sg_set1_pi32(1), sg_set1_pi32(3),
        sg_set1_pi32(2)), 2, 2, 2, 2);
    assert_eq_pi32(sg_constrain_pi32(sg_set1_pi32(1), sg_set1_pi32(3),
        sg_set1_pi32(4)), 3, 3, 3, 3);

    assert_eq_pi64(sg_constrain_pi64(sg_set1_pi64(1), sg_set1_pi64(3),
        sg_set1_pi64(0)), 1, 1);
    assert_eq_pi64(sg_constrain_pi64(sg_set1_pi64(1), sg_set1_pi64(3),
        sg_set1_pi64(2)), 2, 2);
    assert_eq_pi64(sg_constrain_pi64(sg_set1_pi64(1), sg_set1_pi64(3),
        sg_set1_pi64(4)), 3, 3);

    assert_eq_ps(sg_constrain_ps(sg_set1_ps(1.0f), sg_set1_ps(3.0f),
        sg_set1_ps(0.0f)), 1.0f, 1.0f, 1.0f, 1.0f);
    assert_eq_ps(sg_constrain_ps(sg_set1_ps(1.0f), sg_set1_ps(3.0f),
        sg_set1_ps(2.0f)), 2.0f, 2.0f, 2.0f, 2.0f);
    assert_eq_ps(sg_constrain_ps(sg_set1_ps(1.0f), sg_set1_ps(3.0f),
        sg_set1_ps(4.0f)), 3.0f, 3.0f, 3.0f, 3.0f);

    assert_eq_pd(sg_constrain_pd(sg_set1_pd(1.0), sg_set1_pd(3.0),
        sg_set1_pd(0.0)), 1.0, 1.0);
    assert_eq_pd(sg_constrain_pd(sg_set1_pd(1.0), sg_set1_pd(3.0),
        sg_set1_pd(2.0)), 2.0, 2.0);
    assert_eq_pd(sg_constrain_pd(sg_set1_pd(1.0), sg_set1_pd(3.0),
        sg_set1_pd(4.0)), 3.0, 3.0);

    //printf("Constrain test succeeded\n");
}

#ifdef __cplusplus

static void test_opover() {
    // These tests do not test the implementation, only that the operator
    // overloads call the correct C functions

    sg_assert(Vec_pi32::elem_size == 4);
    sg_assert(Vec_pi32::elem_count == 4);
    sg_assert(Vec_s32x1::elem_size == 4);
    sg_assert(Vec_s32x1::elem_count == 1);

    sg_assert(Vec_pi64::elem_size == 8);
    sg_assert(Vec_pi64::elem_count == 2);
    sg_assert(Vec_s64x1::elem_size == 8);
    sg_assert(Vec_s64x1::elem_count == 1);

    sg_assert(Vec_ps::elem_size == 4);
    sg_assert(Vec_ps::elem_count == 4);
    sg_assert(Vec_f32x1::elem_size == 4);
    sg_assert(Vec_f32x1::elem_count == 1);

    sg_assert(Vec_pd::elem_size == 8);
    sg_assert(Vec_pd::elem_count == 2);
    sg_assert(Vec_f64x1::elem_size == 8);
    sg_assert(Vec_f64x1::elem_count == 1);

    // Constructors and some getters
    sg_assert(Vec_pi32{}.debug_eq(0)); sg_assert(Vec_pi64{}.debug_eq(0));
    sg_assert(Vec_ps{}.debug_eq(0.f)); sg_assert(Vec_pd{}.debug_eq(0.0));
    sg_assert(Vec_s32x1{}.debug_eq(0)); sg_assert(Vec_s64x1{}.debug_eq(0));
    sg_assert(Vec_f32x1{}.debug_eq(0.f)); sg_assert(Vec_f64x1{}.debug_eq(0.0));

    sg_assert(Vec_pi32{5}.debug_eq(5)); sg_assert(Vec_pi64{5}.debug_eq(5));
    sg_assert(Vec_ps{5.0f}.debug_eq(5.0f));
    sg_assert(Vec_pd{5.0}.debug_eq(5.0));
    sg_assert(Vec_s32x1{5}.debug_eq(5)); sg_assert(Vec_s64x1{5}.debug_eq(5));
    sg_assert(Vec_f32x1{5.f}.debug_eq(5.f));
    sg_assert(Vec_f64x1{5.0}.debug_eq(5.0));

    sg_assert((Vec_pi32{5, 4, 3, 2}.debug_eq(5, 4, 3, 2)));
    sg_assert((Vec_pi64{5, 4}.debug_eq(5, 4)));
    sg_assert((Vec_ps{5.0f, 4.0f, 3.0f, 2.0f}
        .debug_eq(5.0f, 4.0f, 3.0f, 2.0f)));
    sg_assert((Vec_pd{5.0, 4.0}.debug_eq(5.0, 4.0)));

    sg_assert(Vec_pi32::bitcast_from_u32(5).debug_eq(5));
    sg_assert(Vec_pi32::bitcast_from_u32(5, 4, 3, 2).debug_eq(5, 4, 3, 2));
    sg_assert(Vec_pi64::bitcast_from_u64(5).debug_eq(5));
    sg_assert(Vec_pi64::bitcast_from_u64(5, 4).debug_eq(5, 4));
    sg_assert(Vec_ps::bitcast_from_u32(sg_bitcast_f32x1_u32x1(5.0f))
        .debug_eq(5.0f));
    sg_assert(Vec_ps::bitcast_from_u32(sg_bitcast_f32x1_u32x1(5.0f),
        sg_bitcast_f32x1_u32x1(4.0f), sg_bitcast_f32x1_u32x1(3.0f),
        sg_bitcast_f32x1_u32x1(2.0f)).debug_eq(5.0f, 4.0f, 3.0f, 2.0f));
    sg_assert(Vec_pd::bitcast_from_u64(sg_bitcast_f64x1_u64x1(5.0)).debug_eq(5));
    sg_assert(Vec_pd::bitcast_from_u64(sg_bitcast_f64x1_u64x1(5.0),
        sg_bitcast_f64x1_u64x1(4.0)).debug_eq(5.0, 4.0));
    sg_assert(Vec_s32x1::bitcast_from_u32(5).debug_eq(5));
    sg_assert(Vec_s64x1::bitcast_from_u64(5).debug_eq(5));
    sg_assert(Vec_f32x1::bitcast_from_u32(5).debug_eq(
        sg_bitcast_u32x1_f32x1(5)));
    sg_assert(Vec_f64x1::bitcast_from_u64(5).debug_eq(
        sg_bitcast_u64x1_f64x1(5)));

    sg_assert((Vec_pi32{5, 4, 3, 2}.i3() == 5));
    sg_assert((Vec_pi32{5, 4, 3, 2}.i2() == 4));
    sg_assert((Vec_pi32{5, 4, 3, 2}.i1() == 3));
    sg_assert((Vec_pi32{5, 4, 3, 2}.i0() == 2));
    sg_assert((Vec_pi32{5, 4, 3, 2}.get<3>() == 5));
    sg_assert((Vec_pi32{5, 4, 3, 2}.get<2>() == 4));
    sg_assert((Vec_pi32{5, 4, 3, 2}.get<1>() == 3));
    sg_assert((Vec_pi32{5, 4, 3, 2}.get<0>() == 2));

    sg_assert((Vec_s32x1{2}.i0() == 2));
    sg_assert((Vec_s32x1{2}.get<0>() == 2));
    sg_assert((Vec_s32x1{2}.data() == 2));

    sg_assert((Vec_pi64{5, 4}.l1() == 5));
    sg_assert((Vec_pi64{5, 4}.l0() == 4));
    sg_assert((Vec_pi64{5, 4}.get<1>() == 5));
    sg_assert((Vec_pi64{5, 4}.get<0>() == 4));

    sg_assert((Vec_s64x1{4}.l0() == 4));
    sg_assert((Vec_s64x1{4}.get<0>() == 4));
    sg_assert((Vec_s64x1{4}.data() == 4));

    sg_assert((Vec_ps{5.0f, 4.0f, 3.0f, 2.0f}.f3() == 5.0f));
    sg_assert((Vec_ps{5.0f, 4.0f, 3.0f, 2.0f}.f2() == 4.0f));
    sg_assert((Vec_ps{5.0f, 4.0f, 3.0f, 2.0f}.f1() == 3.0f));
    sg_assert((Vec_ps{5.0f, 4.0f, 3.0f, 2.0f}.f0() == 2.0f));
    sg_assert((Vec_ps{5.0f, 4.0f, 3.0f, 2.0f}.get<3>() == 5.0f));
    sg_assert((Vec_ps{5.0f, 4.0f, 3.0f, 2.0f}.get<2>() == 4.0f));
    sg_assert((Vec_ps{5.0f, 4.0f, 3.0f, 2.0f}.get<1>() == 3.0f));
    sg_assert((Vec_ps{5.0f, 4.0f, 3.0f, 2.0f}.get<0>() == 2.0f));

    sg_assert((Vec_f32x1{2.0f}.f0() == 2.0f));
    sg_assert((Vec_f32x1{2.0f}.get<0>() == 2.0f));
    sg_assert((Vec_f32x1{2.0f}.data() == 2.0f));

    sg_assert((Vec_pd{5.0, 4.0}.d1() == 5.0));
    sg_assert((Vec_pd{5.0, 4.0}.d0() == 4.0));
    sg_assert((Vec_pd{5.0, 4.0}.get<1>() == 5.0));
    sg_assert((Vec_pd{5.0, 4.0}.get<0>() == 4.0));

    sg_assert((Vec_f64x1{4.0}.d0() == 4.0));
    sg_assert((Vec_f64x1{4.0}.get<0>() == 4.0));
    sg_assert((Vec_f64x1{4.0}.data() == 4.0));

    // Arithmetic
    Vec_pi32 pi32, rv_pi32; Vec_pi64 pi64, rv_pi64;
    Vec_s32x1 s32, rv_s32; Vec_s64x1 s64, rv_s64;

    rv_pi32 = pi32++; rv_pi64 = pi64++;
    rv_s32 = s32++; rv_s64 = s64++;
    sg_assert(rv_pi32.debug_eq(0) && pi32.debug_eq(1));
    sg_assert(rv_pi64.debug_eq(0) && pi64.debug_eq(1));
    sg_assert(rv_s32.debug_eq(0) && s32.debug_eq(1));
    sg_assert(rv_s64.debug_eq(0) && s64.debug_eq(1));

    rv_pi32 = ++pi32; rv_pi64 = ++ pi64;
    rv_s32 = ++s32; rv_s64 = ++s64;
    sg_assert(rv_pi32.debug_eq(2) && pi32.debug_eq(2));
    sg_assert(rv_pi64.debug_eq(2) && pi64.debug_eq(2));
    sg_assert(rv_s32.debug_eq(2) && s32.debug_eq(2));
    sg_assert(rv_s64.debug_eq(2) && s64.debug_eq(2));

    rv_pi32 = pi32--; rv_pi64 = pi64--;
    rv_s32 = s32--; rv_s64 = s64--;
    sg_assert(rv_pi32.debug_eq(2) && pi32.debug_eq(1));
    sg_assert(rv_pi64.debug_eq(2) && pi64.debug_eq(1));
    sg_assert(rv_s32.debug_eq(2) && s32.debug_eq(1));
    sg_assert(rv_s64.debug_eq(2) && s64.debug_eq(1));

    rv_pi32 = --pi32; rv_pi64 = --pi64;
    rv_s32 = --s32; rv_s64 = -- s64;
    sg_assert(rv_pi32.debug_eq(0) && pi32.debug_eq(0));
    sg_assert(rv_pi64.debug_eq(0) && pi64.debug_eq(0));
    sg_assert(rv_s32.debug_eq(0) && s32.debug_eq(0));
    sg_assert(rv_s64.debug_eq(0) && s64.debug_eq(0));

    Vec_ps ps, rv_ps; Vec_pd pd, rv_pd;
    Vec_f32x1 f32, rv_f32; Vec_f64x1 f64, rv_f64;

    rv_pi32 = (pi32 += 1); rv_pi64 = (pi64 += 1);
    rv_ps = (ps += 1.0f); rv_pd = (pd += 1.0);
    rv_s32 = (s32 += 1); rv_s64 = (s64 += 1);
    rv_f32 = (f32 += 1); rv_f64 = (f64 += 1);
    sg_assert(rv_pi32.debug_eq(1) && pi32.debug_eq(1));
    sg_assert(rv_pi64.debug_eq(1) && pi64.debug_eq(1));
    sg_assert(rv_ps.debug_eq(1.0f) && ps.debug_eq(1.0f));
    sg_assert(rv_pd.debug_eq(1.0) && pd.debug_eq(1.0));
    sg_assert(rv_s32.debug_eq(1) && s32.debug_eq(1));
    sg_assert(rv_s64.debug_eq(1) && s64.debug_eq(1));
    sg_assert(rv_f32.debug_eq(1) && f32.debug_eq(1));
    sg_assert(rv_f64.debug_eq(1) && f64.debug_eq(1));

    rv_pi32 = pi32 + 1; rv_pi64 = pi64 + 1;
    rv_ps = ps + 1.0f; rv_pd = pd + 1.0;
    rv_s32 = s32 + 1; rv_s64 = s64 + 1;
    rv_f32 = f32 + 1; rv_f64 = f64 + 1;
    sg_assert(rv_pi32.debug_eq(2) && pi32.debug_eq(1));
    sg_assert(rv_pi64.debug_eq(2) && pi64.debug_eq(1));
    sg_assert(rv_ps.debug_eq(2.0f) && ps.debug_eq(1.0f));
    sg_assert(rv_pd.debug_eq(2.0) && pd.debug_eq(1.0));
    sg_assert(rv_s32.debug_eq(2) && s32.debug_eq(1));
    sg_assert(rv_s64.debug_eq(2) && s64.debug_eq(1));
    sg_assert(rv_f32.debug_eq(2) && f32.debug_eq(1));
    sg_assert(rv_f64.debug_eq(2) && f64.debug_eq(1));

    rv_pi32 = +pi32; rv_pi64 = +pi64;
    rv_ps = +ps; rv_pd = +pd;
    rv_s32 = +s32; rv_s64 = +s64; rv_f32 = +f32; rv_f64 = +f64;
    sg_assert(rv_pi32.debug_eq(1) && pi32.debug_eq(1));
    sg_assert(rv_pi64.debug_eq(1) && pi64.debug_eq(1));
    sg_assert(rv_ps.debug_eq(1.0f) && ps.debug_eq(1.0f));
    sg_assert(rv_pd.debug_eq(1.0) && pd.debug_eq(1.0));
    sg_assert(rv_s32.debug_eq(1) && s32.debug_eq(1));
    sg_assert(rv_s64.debug_eq(1) && s64.debug_eq(1));
    sg_assert(rv_f32.debug_eq(1) && f32.debug_eq(1));
    sg_assert(rv_f64.debug_eq(1) && f64.debug_eq(1));

    rv_pi32 = (pi32 -= 2); rv_pi64 = (pi64 -= 2);
    rv_ps = (ps -= 2.0f); rv_pd = (pd -= 2.0);
    rv_s32 = (s32 -= 2); rv_s64 = (s64 -= 2);
    rv_f32 = (f32 -= 2); rv_f64 = (f64 -= 2);
    sg_assert(rv_pi32.debug_eq(-1) && pi32.debug_eq(-1));
    sg_assert(rv_pi64.debug_eq(-1) && pi64.debug_eq(-1));
    sg_assert(rv_ps.debug_eq(-1.0f) && ps.debug_eq(-1.0f));
    sg_assert(rv_pd.debug_eq(-1.0) && pd.debug_eq(-1.0));
    sg_assert(rv_s32.debug_eq(-1) && s32.debug_eq(-1));
    sg_assert(rv_s64.debug_eq(-1) && s64.debug_eq(-1));
    sg_assert(rv_f32.debug_eq(-1) && f32.debug_eq(-1));
    sg_assert(rv_f64.debug_eq(-1) && f64.debug_eq(-1));

    rv_pi32 = pi32 - 1; rv_pi64 = pi64 - 1;
    rv_ps = ps - 1.0f; rv_pd = pd - 1.0;
    rv_s32 = s32 - 1; rv_s64 = s64 - 1;
    rv_f32 = f32 - 1; rv_f64 = f64 - 1;
    sg_assert(rv_pi32.debug_eq(-2) && pi32.debug_eq(-1));
    sg_assert(rv_pi64.debug_eq(-2) && pi64.debug_eq(-1));
    sg_assert(rv_ps.debug_eq(-2.0f) && ps.debug_eq(-1.0f));
    sg_assert(rv_pd.debug_eq(-2.0) && pd.debug_eq(-1.0));
    sg_assert(rv_s32.debug_eq(-2) && s32.debug_eq(-1));
    sg_assert(rv_s64.debug_eq(-2) && s64.debug_eq(-1));
    sg_assert(rv_f32.debug_eq(-2) && f32.debug_eq(-1));
    sg_assert(rv_f64.debug_eq(-2) && f64.debug_eq(-1));

    rv_pi32 = -pi32; rv_pi64 = -pi64;
    rv_ps = -ps; rv_pd = -pd;
    rv_s32 = -s32; rv_s64 = -s64; rv_f32 = -f32; rv_f64 = -f64;
    sg_assert(rv_pi32.debug_eq(1) && pi32.debug_eq(-1));
    sg_assert(rv_pi64.debug_eq(1) && pi64.debug_eq(-1));
    sg_assert(rv_ps.debug_eq(1.0f) && ps.debug_eq(-1.0f));
    sg_assert(rv_pd.debug_eq(1.0) && pd.debug_eq(-1.0));
    sg_assert(rv_s32.debug_eq(1) && s32.debug_eq(-1));
    sg_assert(rv_s64.debug_eq(1) && s64.debug_eq(-1));
    sg_assert(rv_f32.debug_eq(1) && f32.debug_eq(-1));
    sg_assert(rv_f64.debug_eq(1) && f64.debug_eq(-1));

    rv_pi32 = (pi32 *= -16); rv_pi64 = (pi64 *= -16);
    rv_ps = (ps *= -16.0f); rv_pd = (pd *= -16.0);
    rv_s32 = (s32 *= -16); rv_s64 = (s64 *= -16);
    rv_f32 = (f32 *= -16); rv_f64 = (f64 *= -16);
    sg_assert(rv_pi32.debug_eq(16) && pi32.debug_eq(16));
    sg_assert(rv_pi64.debug_eq(16) && pi64.debug_eq(16));
    sg_assert(rv_ps.debug_eq(16.0f) && ps.debug_eq(16.0f));
    sg_assert(rv_pd.debug_eq(16.0) && pd.debug_eq(16.0));
    sg_assert(rv_s32.debug_eq(16) && s32.debug_eq(16));
    sg_assert(rv_s64.debug_eq(16) && s64.debug_eq(16));
    sg_assert(rv_f32.debug_eq(16) && f32.debug_eq(16));
    sg_assert(rv_f64.debug_eq(16) && f64.debug_eq(16));

    rv_pi32 = pi32 * 2; rv_pi64 = pi64 * 2;
    rv_ps = ps * 2.0f; rv_pd = pd * 2.0;
    rv_s32 = s32 * 2; rv_s64 = s64 * 2; rv_f32 = f32 * 2; rv_f64 = f64 * 2;
    sg_assert(rv_pi32.debug_eq(32) && pi32.debug_eq(16));
    sg_assert(rv_pi64.debug_eq(32) && pi64.debug_eq(16));
    sg_assert(rv_ps.debug_eq(32.0f) && ps.debug_eq(16.0f));
    sg_assert(rv_pd.debug_eq(32.0) && pd.debug_eq(16.0));
    sg_assert(rv_s32.debug_eq(32) && s32.debug_eq(16));
    sg_assert(rv_s64.debug_eq(32) && s64.debug_eq(16));
    sg_assert(rv_f32.debug_eq(32) && f32.debug_eq(16));
    sg_assert(rv_f64.debug_eq(32) && f64.debug_eq(16));

    rv_pi32 = (pi32 /= 2); rv_pi64 = (pi64 /= 2);
    rv_ps = (ps /= 2.0f); rv_pd = (pd /= 2.0);
    rv_s32 = (s32 /= 2); rv_s64 = (s64 /= 2);
    rv_f32 = (f32 /= 2); rv_f64 = (f64 /= 2);
    sg_assert(rv_pi32.debug_eq(8) && pi32.debug_eq(8));
    sg_assert(rv_pi64.debug_eq(8) && pi64.debug_eq(8));
    sg_assert(rv_ps.debug_eq(8.0f) && ps.debug_eq(8.0f));
    sg_assert(rv_pd.debug_eq(8.0) && pd.debug_eq(8.0));
    sg_assert(rv_s32.debug_eq(8) && s32.debug_eq(8));
    sg_assert(rv_s64.debug_eq(8) && s64.debug_eq(8));
    sg_assert(rv_f32.debug_eq(8) && f32.debug_eq(8));
    sg_assert(rv_f64.debug_eq(8) && f64.debug_eq(8));

    rv_pi32 = pi32 / 2; rv_pi64 = pi64 / 2;
    rv_ps = ps / 2.0f; rv_pd = pd / 2.0;
    rv_s32 = s32 / 2; rv_s64 = s64 / 2;
    rv_f32 = f32 / 2; rv_f64 = f64/ 2;
    sg_assert(rv_pi32.debug_eq(4) && pi32.debug_eq(8));
    sg_assert(rv_pi64.debug_eq(4) && pi64.debug_eq(8));
    sg_assert(rv_ps.debug_eq(4.0f) && ps.debug_eq(8.0f));
    sg_assert(rv_pd.debug_eq(4.0) && pd.debug_eq(8.0));
    sg_assert(rv_s32.debug_eq(4) && s32.debug_eq(8));
    sg_assert(rv_s64.debug_eq(4) && s64.debug_eq(8));
    sg_assert(rv_f32.debug_eq(4) && f32.debug_eq(8));
    sg_assert(rv_f64.debug_eq(4) && f64.debug_eq(8));

    // Tes mul add
    sg_assert((Vec_ps{1.0f, 2.0f, 3.0f, 4.0f}.mul_add(
        Vec_ps{5.0f, 6.0f, 7.0f, 8.0f}, Vec_ps{9.0f, 10.0f, 11.0f, 12.0f})
        .debug_eq(14.0f, 22.0f, 32.0f, 44.0f)));
    sg_assert((Vec_pd{1.0, 2.0}.mul_add(Vec_pd{5.0, 6.0}, Vec_pd{9.0, 10.0})
        .debug_eq(14.0, 22.0)));
    sg_assert((Vec_f32x1{4}.mul_add(8, 12).debug_eq(44)));
    sg_assert((Vec_f64x1{4}.mul_add(8, 12).debug_eq(44)));

    // Bitwise logic
    for (int32_t i1 = 0; i1 < 2; ++i1) {
    for (int32_t i2 = 0; i2 < 2; ++i2) {
        const float f1 = static_cast<float>(i1),
            f2 = static_cast<float>(i2);
        const double d1 = static_cast<double>(i1),
            d2 = static_cast<double>(i2);

        // and
        float andf = static_cast<bool>(i1 & i2) ? 1.0f : 0.0f;
        double andd = static_cast<bool>(i1 & i2) ? 1.0 : 0.0;
        sg_assert((Vec_pi32{i1} & Vec_pi32{i2}).debug_eq(i1 & i2));
        sg_assert((Vec_pi64{i1} & Vec_pi64{i2}).debug_eq(i1 & i2));
        sg_assert((Vec_ps{f1} & Vec_ps{f2}).debug_eq(andf));
        sg_assert((Vec_pd{d1} & Vec_pd{d2}).debug_eq(andd));
        sg_assert((Vec_s32x1{i1} & Vec_s32x1{i2}).debug_eq(i1 & i2));
        sg_assert((Vec_s64x1{i1} & Vec_s64x1{i2}).debug_eq(i1 & i2));
        sg_assert((Vec_f32x1{f1} & Vec_f32x1{f2}).debug_eq(andf));
        sg_assert((Vec_f64x1{f1} & Vec_f64x1{f2}).debug_eq(andd));

        pi32 = i1; pi64 = i1; ps = f1; pd = d1;
        s32 = i1; s64 = i1; f32 = f1; f64 = d1;
        rv_pi32 = pi32 &= i2; rv_pi64 = pi64 &= i2;
        rv_ps = ps &= f2; rv_pd = pd &= d2;
        rv_s32 = s32 &= i2; rv_s64 = s64 &= i2;
        rv_f32 = f32 &= f2; rv_f64 = f64 &= d2;
        sg_assert(pi32.debug_eq(i1 & i2) && rv_pi32.debug_eq(i1 & i2));
        sg_assert(pi64.debug_eq(i1 & i2) && rv_pi64.debug_eq(i1 & i2));
        sg_assert(ps.debug_eq(andf) && rv_ps.debug_eq(andf));
        sg_assert(pd.debug_eq(andd) && rv_pd.debug_eq(andd));
        sg_assert(s32.debug_eq(i1 & i2) && rv_s32.debug_eq(i1 & i2));
        sg_assert(s64.debug_eq(i1 & i2) && rv_s64.debug_eq(i1 & i2));
        sg_assert(f32.debug_eq(andf) && rv_f32.debug_eq(andf));
        sg_assert(f64.debug_eq(andd) && rv_f64.debug_eq(andd));

        // or
        float orf = static_cast<bool>(i1 | i2) ? 1.0f : 0.0f;
        double ord = static_cast<bool>(i1 | i2) ? 1.0 : 0.0;
        sg_assert((Vec_pi32{i1} | Vec_pi32{i2}).debug_eq(i1 | i2));
        sg_assert((Vec_pi64{i1} | Vec_pi64{i2}).debug_eq(i1 | i2));
        sg_assert((Vec_ps{f1} | Vec_ps{f2}).debug_eq(orf));
        sg_assert((Vec_pd{d1} | Vec_pd{d2}).debug_eq(ord));
        sg_assert((Vec_s32x1{i1} | Vec_s32x1{i2}).debug_eq(i1 | i2));
        sg_assert((Vec_s64x1{i1} | Vec_s64x1{i2}).debug_eq(i1 | i2));
        sg_assert((Vec_f32x1{f1} | Vec_f32x1{f2}).debug_eq(orf));
        sg_assert((Vec_f64x1{f1} | Vec_f64x1{f2}).debug_eq(ord));

        pi32 = i1; pi64 = i1; ps = f1; pd = d1;
        s32 = i1; s64 = i1; f32 = f1; f64 = d1;
        rv_pi32 = pi32 |= i2; rv_pi64 = pi64 |= i2;
        rv_ps = ps |= f2; rv_pd = pd |= d2;
        rv_s32 = s32 |= i2; rv_s64 = s64 |= i2;
        rv_f32 = f32 |= f2; rv_f64 = f64 |= d2;
        sg_assert(pi32.debug_eq(i1 | i2) && rv_pi32.debug_eq(i1 | i2));
        sg_assert(pi64.debug_eq(i1 | i2) && rv_pi64.debug_eq(i1 | i2));
        sg_assert(ps.debug_eq(orf) && rv_ps.debug_eq(orf));
        sg_assert(pd.debug_eq(ord) && rv_pd.debug_eq(ord));
        sg_assert(s32.debug_eq(i1 | i2) && rv_s32.debug_eq(i1 | i2));
        sg_assert(s64.debug_eq(i1 | i2) && rv_s64.debug_eq(i1 | i2));
        sg_assert(f32.debug_eq(orf) && rv_f32.debug_eq(orf));
        sg_assert(f64.debug_eq(ord) && rv_f64.debug_eq(ord));

        // xor
        float xorf = static_cast<bool>(i1 ^ i2) ? 1.0f : 0.0f;
        double xord = static_cast<bool>(i1 ^ i2) ? 1.0 : 0.0;
        sg_assert((Vec_pi32{i1} ^ Vec_pi32{i2}).debug_eq(i1 ^ i2));
        sg_assert((Vec_pi64{i1} ^ Vec_pi64{i2}).debug_eq(i1 ^ i2));
        sg_assert((Vec_ps{f1} ^ Vec_ps{f2}).debug_eq(xorf));
        sg_assert((Vec_pd{d1} ^ Vec_pd{d2}).debug_eq(xord));
        sg_assert((Vec_s32x1{i1} ^ Vec_s32x1{i2}).debug_eq(i1 ^ i2));
        sg_assert((Vec_s64x1{i1} ^ Vec_s64x1{i2}).debug_eq(i1 ^ i2));
        sg_assert((Vec_f32x1{f1} ^ Vec_f32x1{f2}).debug_eq(xorf));
        sg_assert((Vec_f64x1{f1} ^ Vec_f64x1{f2}).debug_eq(xord));

        pi32 = i1; pi64 = i1; ps = f1; pd = d1;
        s32 = i1; s64 = i1; f32 = f1; f64 = d1;
        rv_pi32 = pi32 ^= i2; rv_pi64 = pi64 ^= i2;
        rv_ps = ps ^= f2; rv_pd = pd ^= d2;
        rv_s32 = s32 ^= i2; rv_s64 = s64 ^= i2;
        rv_f32 = f32 ^= f2; rv_f64 = f64 ^= d2;
        sg_assert(pi32.debug_eq(i1 ^ i2) && rv_pi32.debug_eq(i1 ^ i2));
        sg_assert(pi64.debug_eq(i1 ^ i2) && rv_pi64.debug_eq(i1 ^ i2));
        sg_assert(ps.debug_eq(xorf) && rv_ps.debug_eq(xorf));
        sg_assert(pd.debug_eq(xord) && rv_pd.debug_eq(xord));
        sg_assert(s32.debug_eq(i1 ^ i2) && rv_s32.debug_eq(i1 ^ i2));
        sg_assert(s64.debug_eq(i1 ^ i2) && rv_s64.debug_eq(i1 ^ i2));
        sg_assert(f32.debug_eq(xorf) && rv_f32.debug_eq(xorf));
        sg_assert(f64.debug_eq(xord) && rv_f64.debug_eq(xord));

        // not
        sg_assert((~Vec_pi32{i1}).debug_eq(~i1));
        sg_assert((~Vec_pi64{i1}).debug_eq(~(int64_t) i1));
        sg_assert((~Vec_ps{f1}).debug_eq(
            sg_bitcast_u32x1_f32x1(~sg_bitcast_f32x1_u32x1(f1))));
        sg_assert((~Vec_pd{d1}).debug_eq(
            sg_bitcast_u64x1_f64x1(~sg_bitcast_f64x1_u64x1(d1))));
        sg_assert((~Vec_s32x1{i1}).debug_eq(~i1));
        sg_assert((~Vec_s64x1{i1}).debug_eq(~(int64_t) i1));
        sg_assert((~Vec_f32x1{f1}).debug_eq(
            sg_bitcast_u32x1_f32x1(~sg_bitcast_f32x1_u32x1(f1))));
        sg_assert((~Vec_f64x1{d1}).debug_eq(
            sg_bitcast_u64x1_f64x1(~sg_bitcast_f64x1_u64x1(d1))));
    } }

    // Comparison
    sg_assert((Vec_pi32{1} < 2).debug_valid_eq(true));
    sg_assert((Vec_pi32{2} < 2).debug_valid_eq(false));
    sg_assert((Vec_pi32{3} < 2).debug_valid_eq(false));
    sg_assert((Vec_pi64{1} < 2).debug_valid_eq(true));
    sg_assert((Vec_pi64{2} < 2).debug_valid_eq(false));
    sg_assert((Vec_pi64{3} < 2).debug_valid_eq(false));
    sg_assert((Vec_ps{1.0f} < 2.0f).debug_valid_eq(true));
    sg_assert((Vec_ps{2.0f} < 2.0f).debug_valid_eq(false));
    sg_assert((Vec_ps{3.0f} < 2.0f).debug_valid_eq(false));
    sg_assert((Vec_pd{1.0} < 2.0).debug_valid_eq(true));
    sg_assert((Vec_pd{2.0} < 2.0).debug_valid_eq(false));
    sg_assert((Vec_pd{3.0} < 2.0).debug_valid_eq(false));
    sg_assert((Vec_s32x1{1} < 2).debug_valid_eq(true));
    sg_assert((Vec_s32x1{2} < 2).debug_valid_eq(false));
    sg_assert((Vec_s32x1{3} < 2).debug_valid_eq(false));
    sg_assert((Vec_s64x1{1} < 2).debug_valid_eq(true));
    sg_assert((Vec_s64x1{2} < 2).debug_valid_eq(false));
    sg_assert((Vec_s64x1{3} < 2).debug_valid_eq(false));
    sg_assert((Vec_f32x1{1} < 2).debug_valid_eq(true));
    sg_assert((Vec_f32x1{2} < 2).debug_valid_eq(false));
    sg_assert((Vec_f32x1{3} < 2).debug_valid_eq(false));
    sg_assert((Vec_f64x1{1} < 2).debug_valid_eq(true));
    sg_assert((Vec_f64x1{2} < 2).debug_valid_eq(false));
    sg_assert((Vec_f64x1{3} < 2).debug_valid_eq(false));

    sg_assert((Vec_pi32{1} <= 2).debug_valid_eq(true));
    sg_assert((Vec_pi32{2} <= 2).debug_valid_eq(true));
    sg_assert((Vec_pi32{3} <= 2).debug_valid_eq(false));
    sg_assert((Vec_pi64{1} <= 2).debug_valid_eq(true));
    sg_assert((Vec_pi64{2} <= 2).debug_valid_eq(true));
    sg_assert((Vec_pi64{3} <= 2).debug_valid_eq(false));
    sg_assert((Vec_ps{1.0f} <= 2.0f).debug_valid_eq(true));
    sg_assert((Vec_ps{2.0f} <= 2.0f).debug_valid_eq(true));
    sg_assert((Vec_ps{3.0f} <= 2.0f).debug_valid_eq(false));
    sg_assert((Vec_pd{1.0} <= 2.0).debug_valid_eq(true));
    sg_assert((Vec_pd{2.0} <= 2.0).debug_valid_eq(true));
    sg_assert((Vec_pd{3.0} <= 2.0).debug_valid_eq(false));
    sg_assert((Vec_s32x1{1} <= 2).debug_valid_eq(true));
    sg_assert((Vec_s32x1{2} <= 2).debug_valid_eq(true));
    sg_assert((Vec_s32x1{3} <= 2).debug_valid_eq(false));
    sg_assert((Vec_s64x1{1} <= 2).debug_valid_eq(true));
    sg_assert((Vec_s64x1{2} <= 2).debug_valid_eq(true));
    sg_assert((Vec_s64x1{3} <= 2).debug_valid_eq(false));
    sg_assert((Vec_f32x1{1} <= 2).debug_valid_eq(true));
    sg_assert((Vec_f32x1{2} <= 2).debug_valid_eq(true));
    sg_assert((Vec_f32x1{3} <= 2).debug_valid_eq(false));
    sg_assert((Vec_f64x1{1} <= 2).debug_valid_eq(true));
    sg_assert((Vec_f64x1{2} <= 2).debug_valid_eq(true));
    sg_assert((Vec_f64x1{3} <= 2).debug_valid_eq(false));

    sg_assert((Vec_pi32{1} == 2).debug_valid_eq(false));
    sg_assert((Vec_pi32{2} == 2).debug_valid_eq(true));
    sg_assert((Vec_pi32{3} == 2).debug_valid_eq(false));
    sg_assert((Vec_pi64{1} == 2).debug_valid_eq(false));
    sg_assert((Vec_pi64{2} == 2).debug_valid_eq(true));
    sg_assert((Vec_pi64{3} == 2).debug_valid_eq(false));
    sg_assert((Vec_ps{1.0f} == 2.0f).debug_valid_eq(false));
    sg_assert((Vec_ps{2.0f} == 2.0f).debug_valid_eq(true));
    sg_assert((Vec_ps{3.0f} == 2.0f).debug_valid_eq(false));
    sg_assert((Vec_pd{1.0} == 2.0).debug_valid_eq(false));
    sg_assert((Vec_pd{2.0} == 2.0).debug_valid_eq(true));
    sg_assert((Vec_pd{3.0} == 2.0).debug_valid_eq(false));
    sg_assert((Vec_s32x1{1} == 2).debug_valid_eq(false));
    sg_assert((Vec_s32x1{2} == 2).debug_valid_eq(true));
    sg_assert((Vec_s32x1{3} == 2).debug_valid_eq(false));
    sg_assert((Vec_s64x1{1} == 2).debug_valid_eq(false));
    sg_assert((Vec_s64x1{2} == 2).debug_valid_eq(true));
    sg_assert((Vec_s64x1{3} == 2).debug_valid_eq(false));
    sg_assert((Vec_f32x1{1} == 2).debug_valid_eq(false));
    sg_assert((Vec_f32x1{2} == 2).debug_valid_eq(true));
    sg_assert((Vec_f32x1{3} == 2).debug_valid_eq(false));
    sg_assert((Vec_f64x1{1} == 2).debug_valid_eq(false));
    sg_assert((Vec_f64x1{2} == 2).debug_valid_eq(true));
    sg_assert((Vec_f64x1{3} == 2).debug_valid_eq(false));

    sg_assert((Vec_pi32{1} >= 2).debug_valid_eq(false));
    sg_assert((Vec_pi32{2} >= 2).debug_valid_eq(true));
    sg_assert((Vec_pi32{3} >= 2).debug_valid_eq(true));
    sg_assert((Vec_pi64{1} >= 2).debug_valid_eq(false));
    sg_assert((Vec_pi64{2} >= 2).debug_valid_eq(true));
    sg_assert((Vec_pi64{3} >= 2).debug_valid_eq(true));
    sg_assert((Vec_ps{1.0f} >= 2.0f).debug_valid_eq(false));
    sg_assert((Vec_ps{2.0f} >= 2.0f).debug_valid_eq(true));
    sg_assert((Vec_ps{3.0f} >= 2.0f).debug_valid_eq(true));
    sg_assert((Vec_pd{1.0} >= 2.0).debug_valid_eq(false));
    sg_assert((Vec_pd{2.0} >= 2.0).debug_valid_eq(true));
    sg_assert((Vec_pd{3.0} >= 2.0).debug_valid_eq(true));
    sg_assert((Vec_s32x1{1} >= 2).debug_valid_eq(false));
    sg_assert((Vec_s32x1{2} >= 2).debug_valid_eq(true));
    sg_assert((Vec_s32x1{3} >= 2).debug_valid_eq(true));
    sg_assert((Vec_s64x1{1} >= 2).debug_valid_eq(false));
    sg_assert((Vec_s64x1{2} >= 2).debug_valid_eq(true));
    sg_assert((Vec_s64x1{3} >= 2).debug_valid_eq(true));
    sg_assert((Vec_f32x1{1} >= 2).debug_valid_eq(false));
    sg_assert((Vec_f32x1{2} >= 2).debug_valid_eq(true));
    sg_assert((Vec_f32x1{3} >= 2).debug_valid_eq(true));
    sg_assert((Vec_f64x1{1} >= 2).debug_valid_eq(false));
    sg_assert((Vec_f64x1{2} >= 2).debug_valid_eq(true));
    sg_assert((Vec_f64x1{3} >= 2).debug_valid_eq(true));

    sg_assert((Vec_pi32{1} > 2).debug_valid_eq(false));
    sg_assert((Vec_pi32{2} > 2).debug_valid_eq(false));
    sg_assert((Vec_pi32{3} > 2).debug_valid_eq(true));
    sg_assert((Vec_pi64{1} > 2).debug_valid_eq(false));
    sg_assert((Vec_pi64{2} > 2).debug_valid_eq(false));
    sg_assert((Vec_pi64{3} > 2).debug_valid_eq(true));
    sg_assert((Vec_ps{1.0f} > 2.0f).debug_valid_eq(false));
    sg_assert((Vec_ps{2.0f} > 2.0f).debug_valid_eq(false));
    sg_assert((Vec_ps{3.0f} > 2.0f).debug_valid_eq(true));
    sg_assert((Vec_pd{1.0} > 2.0).debug_valid_eq(false));
    sg_assert((Vec_pd{2.0} > 2.0).debug_valid_eq(false));
    sg_assert((Vec_pd{3.0} > 2.0).debug_valid_eq(true));
    sg_assert((Vec_s32x1{1} > 2).debug_valid_eq(false));
    sg_assert((Vec_s32x1{2} > 2).debug_valid_eq(false));
    sg_assert((Vec_s32x1{3} > 2).debug_valid_eq(true));
    sg_assert((Vec_s64x1{1} > 2).debug_valid_eq(false));
    sg_assert((Vec_s64x1{2} > 2).debug_valid_eq(false));
    sg_assert((Vec_s64x1{3} > 2).debug_valid_eq(true));
    sg_assert((Vec_f32x1{1} > 2).debug_valid_eq(false));
    sg_assert((Vec_f32x1{2} > 2).debug_valid_eq(false));
    sg_assert((Vec_f32x1{3} > 2).debug_valid_eq(true));
    sg_assert((Vec_f64x1{1} > 2).debug_valid_eq(false));
    sg_assert((Vec_f64x1{2} > 2).debug_valid_eq(false));
    sg_assert((Vec_f64x1{3} > 2).debug_valid_eq(true));

    // Shift
    sg_assert(Vec_pi32{1}.shift_l_imm<1>().debug_eq(2));
    sg_assert(Vec_pi32{1}.shift_l(1).debug_eq(2));
    sg_assert(Vec_pi32{2}.shift_rl_imm<1>().debug_eq(1));
    sg_assert(Vec_pi32{2}.shift_rl(1).debug_eq(1));
    sg_assert(Vec_pi32{-2}.shift_rl_imm<1>().debug_eq(2147483647));
    sg_assert(Vec_pi32{-2}.shift_rl(1).debug_eq(2147483647));
    sg_assert(Vec_pi32{-2}.shift_ra_imm<1>().debug_eq(-1));
    sg_assert(Vec_pi32{-2}.shift_ra(1).debug_eq(-1));
    sg_assert(Vec_s32x1{1}.shift_l_imm<1>().debug_eq(2));
    sg_assert(Vec_s32x1{1}.shift_l(1).debug_eq(2));
    sg_assert(Vec_s32x1{2}.shift_rl_imm<1>().debug_eq(1));
    sg_assert(Vec_s32x1{2}.shift_rl(1).debug_eq(1));
    sg_assert(Vec_s32x1{-2}.shift_rl_imm<1>().debug_eq(2147483647));
    sg_assert(Vec_s32x1{-2}.shift_rl(1).debug_eq(2147483647));
    sg_assert(Vec_s32x1{-2}.shift_ra_imm<1>().debug_eq(-1));
    sg_assert(Vec_s32x1{-2}.shift_ra(1).debug_eq(-1));

    sg_assert(Vec_pi64{1}.shift_l_imm<1>().debug_eq(2));
    sg_assert(Vec_pi64{1}.shift_l(1).debug_eq(2));
    sg_assert(Vec_pi64{2}.shift_rl_imm<1>().debug_eq(1));
    sg_assert(Vec_pi64{2}.shift_rl(1).debug_eq(1));
    sg_assert(Vec_pi64{-2}.shift_rl_imm<1>().debug_eq(9223372036854775807));
    sg_assert(Vec_pi64{-2}.shift_rl(1).debug_eq(9223372036854775807));
    sg_assert(Vec_pi64{-2}.shift_ra_imm<1>().debug_eq(-1));
    sg_assert(Vec_pi64{-2}.shift_ra(1).debug_eq(-1));
    sg_assert(Vec_s64x1{1}.shift_l_imm<1>().debug_eq(2));
    sg_assert(Vec_s64x1{1}.shift_l(1).debug_eq(2));
    sg_assert(Vec_s64x1{2}.shift_rl_imm<1>().debug_eq(1));
    sg_assert(Vec_s64x1{2}.shift_rl(1).debug_eq(1));
    sg_assert(Vec_s64x1{-2}.shift_rl_imm<1>().debug_eq(9223372036854775807));
    sg_assert(Vec_s64x1{-2}.shift_rl(1).debug_eq(9223372036854775807));
    sg_assert(Vec_s64x1{-2}.shift_ra_imm<1>().debug_eq(-1));
    sg_assert(Vec_s64x1{-2}.shift_ra(1).debug_eq(-1));

    // Shuffle
    sg_assert((Vec_pi32{3, 2, 1, 0}.shuffle<0, 1, 2, 3>()
        .debug_eq(0, 1, 2, 3)));
    sg_assert((Vec_pi64{1, 0}.shuffle<0, 1>().debug_eq(0, 1)));
    sg_assert((Vec_ps{3.0f, 2.0f, 1.0f, 0.0f}.shuffle<0, 1, 2, 3>()
        .debug_eq(0.0f, 1.0f, 2.0f, 3.0f)));
    sg_assert((Vec_pd{1.0, 0.0}.shuffle<0, 1>().debug_eq(0.0, 1.0)));

    // Safe div
    sg_assert((Vec_pi32{8}.safe_divide_by(2).debug_eq(4)));
    sg_assert((Vec_pi32{8}.safe_divide_by(0).debug_eq(8)));
    sg_assert((Vec_pi64{8}.safe_divide_by(2).debug_eq(4)));
    sg_assert((Vec_pi64{8}.safe_divide_by(0).debug_eq(8)));
    sg_assert((Vec_ps{8.0f}.safe_divide_by(2.0f).debug_eq(4.0f)));
    sg_assert((Vec_ps{8.0f}.safe_divide_by(0.0f).debug_eq(8.0f)));
    sg_assert((Vec_ps{8.0f}.safe_divide_by(-0.0f).debug_eq(8.0f)));
    sg_assert((Vec_pd{8.0}.safe_divide_by(2.0).debug_eq(4.0)));
    sg_assert((Vec_pd{8.0}.safe_divide_by(0.0).debug_eq(8.0)));
    sg_assert((Vec_pd{8.0}.safe_divide_by(-0.0).debug_eq(8.0)));
    sg_assert((Vec_s32x1{8}.safe_divide_by(2).debug_eq(4)));
    sg_assert((Vec_s32x1{8}.safe_divide_by(0).debug_eq(8)));
    sg_assert((Vec_s64x1{8}.safe_divide_by(2).debug_eq(4)));
    sg_assert((Vec_s64x1{8}.safe_divide_by(0).debug_eq(8)));
    sg_assert((Vec_f32x1{8}.safe_divide_by(2).debug_eq(4)));
    sg_assert((Vec_f32x1{8}.safe_divide_by(0).debug_eq(8)));
    sg_assert((Vec_f64x1{8}.safe_divide_by(2).debug_eq(4)));
    sg_assert((Vec_f64x1{8}.safe_divide_by(0).debug_eq(8)));

    // Abs
    sg_assert((Vec_pi32{1}.abs().debug_eq(1)));
    sg_assert((Vec_pi32{-1}.abs().debug_eq(1)));
    sg_assert((Vec_pi64{1}.abs().debug_eq(1)));
    sg_assert((Vec_pi64{-1}.abs().debug_eq(1)));
    sg_assert((Vec_ps{1.0f}.abs().debug_eq(1.0f)));
    sg_assert((Vec_ps{-1.0f}.abs().debug_eq(1.0f)));
    sg_assert((Vec_ps{-0.0f}.abs().debug_eq(0.0f)));
    sg_assert((Vec_pd{1.0}.abs().debug_eq(1.0)));
    sg_assert((Vec_pd{-1.0}.abs().debug_eq(1.0)));
    sg_assert((Vec_pd{-0.0}.abs().debug_eq(0.0)));
    sg_assert((Vec_s32x1{1}.abs().debug_eq(1)));
    sg_assert((Vec_s32x1{-1}.abs().debug_eq(1)));
    sg_assert((Vec_s64x1{1}.abs().debug_eq(1)));
    sg_assert((Vec_s64x1{-1}.abs().debug_eq(1)));
    sg_assert((Vec_f32x1{1.0f}.abs().debug_eq(1.0f)));
    sg_assert((Vec_f32x1{-1.0f}.abs().debug_eq(1.0f)));
    sg_assert((Vec_f32x1{-0.0f}.abs().debug_eq(0.0f)));
    sg_assert((Vec_f64x1{1.0}.abs().debug_eq(1.0)));
    sg_assert((Vec_f64x1{-1.0}.abs().debug_eq(1.0)));
    sg_assert((Vec_f64x1{-0.0}.abs().debug_eq(0.0)));

    // Remove signed zero
    sg_assert(Vec_ps{-0.0f}.remove_signed_zero().debug_eq(0.0f));
    sg_assert(!(Vec_ps{-0.0f}.debug_eq(0.0f)));
    sg_assert(Vec_pd{0.0}.remove_signed_zero().debug_eq(0.0));
    sg_assert(!(Vec_pd{-0.0}.debug_eq(0.0)));

    sg_assert(Vec_f32x1{-0.0}.remove_signed_zero().debug_eq(0.0));
    sg_assert(Vec_f32x1{0.0}.remove_signed_zero().debug_eq(0.0));
    sg_assert(Vec_f32x1{-3.0}.remove_signed_zero().debug_eq(-3.0));
    sg_assert(!(Vec_f32x1{-0.0}.debug_eq(0.0)));
    sg_assert(Vec_f64x1{-0.0}.remove_signed_zero().debug_eq(0.0));
    sg_assert(Vec_f64x1{0.0}.remove_signed_zero().debug_eq(0.0));
    sg_assert(Vec_f64x1{-3.0}.remove_signed_zero().debug_eq(-3.0));
    sg_assert(!(Vec_f64x1{-0.0}.debug_eq(0.0)));

    // Constrain
    sg_assert((Vec_pi32{-3}.constrain(-2, 2).debug_eq(-2)));
    sg_assert((Vec_pi64{-3}.constrain(-2, 2).debug_eq(-2)));
    sg_assert((Vec_ps{-3.0f}.constrain(-2.0f, 2.0f).debug_eq(-2.0f)));
    sg_assert((Vec_pd{-3.0}.constrain(-2.0, 2.0).debug_eq(-2.0)));
    sg_assert((Vec_s32x1{-3}.constrain(-2, 2).debug_eq(-2)));
    sg_assert((Vec_s64x1{-3}.constrain(-2, 2).debug_eq(-2)));
    sg_assert((Vec_f32x1{-3}.constrain(-2, 2).debug_eq(-2)));
    sg_assert((Vec_f64x1{-3}.constrain(-2, 2).debug_eq(-2)));

    // Min and max
    sg_assert(Vec_pi32::min(Vec_pi32{1}, Vec_pi32{2}).debug_eq(1));
    sg_assert(Vec_pi32::max(Vec_pi32{1}, Vec_pi32{2}).debug_eq(2));
    sg_assert(Vec_pi64::min(Vec_pi64{1}, Vec_pi64{2}).debug_eq(1));
    sg_assert(Vec_pi64::max(Vec_pi64{1}, Vec_pi64{2}).debug_eq(2));
    sg_assert(Vec_ps::min(Vec_ps{1.0f}, Vec_ps{2.0f}).debug_eq(1.0f));
    sg_assert(Vec_ps::max(Vec_ps{1.0f}, Vec_ps{2.0f}).debug_eq(2.0f));
    sg_assert(Vec_pd::min(Vec_pd{1.0}, Vec_pd{2.0}).debug_eq(1.0));
    sg_assert(Vec_pd::max(Vec_pd{1.0}, Vec_pd{2.0}).debug_eq(2.0));
    sg_assert(Vec_s32x1::min(Vec_s32x1{1}, Vec_s32x1{2}).debug_eq(1));
    sg_assert(Vec_s32x1::max(Vec_s32x1{1}, Vec_s32x1{2}).debug_eq(2));
    sg_assert(Vec_s64x1::min(Vec_s64x1{1}, Vec_s64x1{2}).debug_eq(1));
    sg_assert(Vec_s64x1::max(Vec_s64x1{1}, Vec_s64x1{2}).debug_eq(2));
    sg_assert(Vec_f32x1::min(Vec_f32x1{1}, Vec_f32x1{2}).debug_eq(1));
    sg_assert(Vec_f32x1::max(Vec_f32x1{1}, Vec_f32x1{2}).debug_eq(2));
    sg_assert(Vec_f64x1::min(Vec_f64x1{1}, Vec_f64x1{2}).debug_eq(1));
    sg_assert(Vec_f64x1::max(Vec_f64x1{1}, Vec_f64x1{2}).debug_eq(2));

    // Bitcast
    sg_assert(Vec_pi32{1}.bitcast<Vec_pi64>().bitcast<Vec_pi32>().debug_eq(1));
    sg_assert(Vec_pi32{1}.bitcast<Vec_ps>().bitcast<Vec_pi32>().debug_eq(1));
    sg_assert(Vec_pi32{1}.bitcast<Vec_pd>().bitcast<Vec_pi32>().debug_eq(1));
    //sg_assert(Vec_s32x1{1}.bitcast<Vec_s64x1>().bitcast<Vec_s32x1>()
        //.debug_eq(1));
    sg_assert(Vec_s32x1{1}.bitcast<Vec_f32x1>().bitcast<Vec_s32x1>()
        .debug_eq(1));
    //sg_assert(Vec_s32x1{1}.bitcast<Vec_f64x1>().bitcast<Vec_s32x1>()
        //.debug_eq(1));

    sg_assert(Vec_pi64{1}.bitcast<Vec_ps>().bitcast<Vec_pi64>().debug_eq(1));
    sg_assert(Vec_pi64{1}.bitcast<Vec_pd>().bitcast<Vec_pi64>().debug_eq(1));
    //sg_assert(Vec_s64x1{1}.bitcast<Vec_f32x1>().bitcast<Vec_s64x1>()
        //.debug_eq(1));
    sg_assert(Vec_s64x1{1}.bitcast<Vec_f64x1>().bitcast<Vec_s64x1>()
        .debug_eq(1));

    sg_assert(Vec_ps{1.0f}.bitcast<Vec_pd>().bitcast<Vec_ps>().debug_eq(1));
    //sg_assert(Vec_f32x1{1}.bitcast<Vec_f64x1>().bitcast<Vec_f32x1>()
        //.debug_eq(1));

    // Convert
    sg_assert(Vec_pi32{1}.to<Vec_pi64>().debug_eq(1));
    sg_assert(Vec_pi32{1}.to<Vec_ps>().debug_eq(1.0f));
    sg_assert(Vec_pi32{1}.to<Vec_pd>().debug_eq(1.0));
    sg_assert(Vec_s32x1{1}.to<Vec_s64x1>().debug_eq(1));
    sg_assert(Vec_s32x1{1}.to<Vec_f32x1>().debug_eq(1.0f));
    sg_assert(Vec_s32x1{1}.to<Vec_f64x1>().debug_eq(1.0));
    sg_assert(Vec_pi64{1}.to<Vec_pi32>().debug_eq(0, 0, 1, 1));
    sg_assert(Vec_pi64{1}.to<Vec_ps>().debug_eq(0.0f, 0.0f, 1.0f, 1.0f));
    sg_assert(Vec_pi64{1}.to<Vec_pd>().debug_eq(1.0));
    sg_assert(Vec_s64x1{1}.to<Vec_s32x1>().debug_eq(1));
    sg_assert(Vec_s64x1{1}.to<Vec_f32x1>().debug_eq(1.0f));
    sg_assert(Vec_s64x1{1}.to<Vec_s64x1>().debug_eq(1.0));

    sg_assert(Vec_ps{1.7f}.nearest<Vec_pi32>().debug_eq(2));
    sg_assert(Vec_ps{1.7f}.truncate<Vec_pi32>().debug_eq(1));
    sg_assert(Vec_ps{1.7f}.floor<Vec_pi32>().debug_eq(1));
    sg_assert(Vec_ps{-1.7f}.floor<Vec_pi32>().debug_eq(-2));
    sg_assert(Vec_ps{1.7f}.nearest<Vec_pi64>().debug_eq(2));
    sg_assert(Vec_ps{1.7f}.truncate<Vec_pi64>().debug_eq(1));
    sg_assert(Vec_ps{1.7f}.floor<Vec_pi64>().debug_eq(1));
    sg_assert(Vec_ps{-1.7f}.floor<Vec_pi64>().debug_eq(-2));
    sg_assert(Vec_ps{1.0f}.to<Vec_pd>().debug_eq(1.0));

    sg_assert(Vec_f32x1{1.7f}.nearest<Vec_s32x1>().debug_eq(2));
    sg_assert(Vec_f32x1{1.7f}.truncate<Vec_s32x1>().debug_eq(1));
    sg_assert(Vec_f32x1{1.7f}.floor<Vec_s32x1>().debug_eq(1));
    sg_assert(Vec_f32x1{-1.7f}.floor<Vec_s32x1>().debug_eq(-2));
    sg_assert(Vec_f32x1{1.7f}.nearest<Vec_s64x1>().debug_eq(2));
    sg_assert(Vec_f32x1{1.7f}.truncate<Vec_s64x1>().debug_eq(1));
    sg_assert(Vec_f32x1{1.7f}.floor<Vec_s64x1>().debug_eq(1));
    sg_assert(Vec_f32x1{-1.7f}.floor<Vec_s64x1>().debug_eq(-2));
    sg_assert(Vec_f32x1{1.0f}.to<Vec_f64x1>().debug_eq(1.0));

    sg_assert(Vec_pd{1.7}.nearest<Vec_pi32>().debug_eq(0, 0, 2, 2));
    sg_assert(Vec_pd{1.7}.truncate<Vec_pi32>().debug_eq(0, 0, 1, 1));
    sg_assert(Vec_pd{1.7}.floor<Vec_pi32>().debug_eq(0, 0, 1, 1));
    sg_assert(Vec_pd{-1.7}.floor<Vec_pi32>().debug_eq(0, 0, -2, -2));
    sg_assert(Vec_pd{1.7}.nearest<Vec_pi64>().debug_eq(2));
    sg_assert(Vec_pd{1.7}.truncate<Vec_pi64>().debug_eq(1));
    sg_assert(Vec_pd{1.7}.floor<Vec_pi64>().debug_eq(1));
    sg_assert(Vec_pd{-1.7}.floor<Vec_pi64>().debug_eq(-2));
    sg_assert(Vec_pd{1.0}.to<Vec_ps>().debug_eq(0.0f, 0.0f, 1.0f, 1.0f));

    sg_assert(Vec_f64x1{1.7}.nearest<Vec_s32x1>().debug_eq(2));
    sg_assert(Vec_f64x1{1.7}.truncate<Vec_s32x1>().debug_eq(1));
    sg_assert(Vec_f64x1{1.7}.floor<Vec_s32x1>().debug_eq(1));
    sg_assert(Vec_f64x1{-1.7}.floor<Vec_s32x1>().debug_eq(-2));
    sg_assert(Vec_f64x1{1.7}.nearest<Vec_s64x1>().debug_eq(2));
    sg_assert(Vec_f64x1{1.7}.truncate<Vec_s64x1>().debug_eq(1));
    sg_assert(Vec_f64x1{1.7}.floor<Vec_s64x1>().debug_eq(1));
    sg_assert(Vec_f64x1{-1.7}.floor<Vec_s64x1>().debug_eq(-2));
    sg_assert(Vec_f64x1{1.0}.to<Vec_f32x1>().debug_eq(1.0f));

    //printf("Vector operator overloading test succeeded\n");
}

static void test_opover_cmp() {
    sg_assert(Compare_pi32{}.debug_valid_eq(false));
    sg_assert(Compare_pi64{}.debug_valid_eq(false));
    sg_assert(Compare_ps{}.debug_valid_eq(false));
    sg_assert(Compare_pd{}.debug_valid_eq(false));
    sg_assert(Compare_s32x1{}.debug_valid_eq(false));
    sg_assert(Compare_s64x1{}.debug_valid_eq(false));
    sg_assert(Compare_f32x1{}.debug_valid_eq(false));
    sg_assert(Compare_f64x1{}.debug_valid_eq(false));

    sg_assert(Compare_pi32{true}.debug_valid_eq(true));
    sg_assert(Compare_pi64{true}.debug_valid_eq(true));
    sg_assert(Compare_ps{true}.debug_valid_eq(true));
    sg_assert(Compare_pd{true}.debug_valid_eq(true));
    sg_assert(Compare_s32x1{true}.debug_valid_eq(true));
    sg_assert(Compare_s64x1{true}.debug_valid_eq(true));
    sg_assert(Compare_f32x1{true}.debug_valid_eq(true));
    sg_assert(Compare_f64x1{true}.debug_valid_eq(true));

    sg_assert((Compare_pi32{true, false, false, true}
        .debug_valid_eq(true, false, false, true)));
    sg_assert((Compare_pi32{false, true, true, false}
        .debug_valid_eq(false, true, true, false)));
    sg_assert((Compare_pi64{true, false}.debug_valid_eq(true, false)));
    sg_assert((Compare_pi64{false, true}.debug_valid_eq(false, true)));
    sg_assert((Compare_ps{true, false, false, true}
        .debug_valid_eq(true, false, false, true)));
    sg_assert((Compare_ps{false, true, true, false}
        .debug_valid_eq(false, true, true, false)));
    sg_assert((Compare_pd{true, false}.debug_valid_eq(true, false)));
    sg_assert((Compare_pd{false, true}.debug_valid_eq(false, true)));

    sg_assert((Compare_pi32{true, true, false, true}
        .to<Compare_pi64>().debug_valid_eq(false, true)));
    sg_assert((Compare_pi32{true, true, false, true}
        .to<Compare_ps>().debug_valid_eq(true, true, false, true)));
    sg_assert((Compare_pi32{true, true, false, true}
        .to<Compare_pd>().debug_valid_eq(false, true)));

    sg_assert((Compare_pi64{false, true}
        .to<Compare_pi32>().debug_valid_eq(false, false, false, true)));
    sg_assert((Compare_pi64{false, true}
        .to<Compare_ps>().debug_valid_eq(false, false, false, true)));
    sg_assert((Compare_pi64{false, true}
        .to<Compare_pd>().debug_valid_eq(false, true)));

    sg_assert((Compare_ps{true, true, false, true}
        .to<Compare_pi32>().debug_valid_eq(true, true, false, true)));
    sg_assert((Compare_ps{true, true, false, true}
        .to<Compare_pi64>().debug_valid_eq(false, true)));
    sg_assert((Compare_ps{true, true, false, true}
        .to<Compare_pd>().debug_valid_eq(false, true)));

    sg_assert((Compare_pd{false, true}
        .to<Compare_pi32>().debug_valid_eq(false, false, false, true)));
    sg_assert((Compare_pd{false, true}
        .to<Compare_pi64>().debug_valid_eq(false, true)));
    sg_assert((Compare_pd{false, true}
        .to<Compare_ps>().debug_valid_eq(false, false, false, true)));

    for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
        const bool b1 = static_cast<bool>(i), b2 = static_cast<bool>(j);

        // AND
        sg_assert((Compare_pi32{b1} && Compare_pi32{b2})
            .debug_valid_eq(b1 && b2));

        sg_assert((Compare_pi64{b1} && Compare_pi64{b2})
            .debug_valid_eq(b1 && b2));

        sg_assert((Compare_ps{b1} && Compare_ps{b2})
            .debug_valid_eq(b1 && b2));

        sg_assert((Compare_pd{b1} && Compare_pd{b2})
            .debug_valid_eq(b1 && b2));

        sg_assert((Compare_s32x1{b1} && Compare_s32x1{b2})
            .debug_valid_eq(b1 && b2));
        sg_assert((Compare_s64x1{b1} && Compare_s64x1{b2})
            .debug_valid_eq(b1 && b2));
        sg_assert((Compare_f32x1{b1} && Compare_f32x1{b2})
            .debug_valid_eq(b1 && b2));
        sg_assert((Compare_f64x1{b1} && Compare_f64x1{b2})
            .debug_valid_eq(b1 && b2));

        // OR
        sg_assert((Compare_pi32{b1} || Compare_pi32{b2})
            .debug_valid_eq(b1 || b2));

        sg_assert((Compare_pi64{b1} || Compare_pi64{b2})
            .debug_valid_eq(b1 || b2));

        sg_assert((Compare_ps{b1} || Compare_ps{b2})
            .debug_valid_eq(b1 || b2));

        sg_assert((Compare_pd{b1} || Compare_pd{b2})
            .debug_valid_eq(b1 || b2));

        sg_assert((Compare_s32x1{b1} || Compare_s32x1{b2})
            .debug_valid_eq(b1 || b2));
        sg_assert((Compare_s64x1{b1} || Compare_s64x1{b2})
            .debug_valid_eq(b1 || b2));
        sg_assert((Compare_f32x1{b1} || Compare_f32x1{b2})
            .debug_valid_eq(b1 || b2));
        sg_assert((Compare_f64x1{b1} || Compare_f64x1{b2})
            .debug_valid_eq(b1 || b2));

        // Equal
        sg_assert((Compare_pi32{b1} == Compare_pi32{b2})
            .debug_valid_eq(b1 == b2));
        sg_assert((Compare_pi64{b1} == Compare_pi64{b2})
            .debug_valid_eq(b1 == b2));
        sg_assert((Compare_ps{b1} == Compare_ps{b2})
            .debug_valid_eq(b1 == b2));
        sg_assert((Compare_pd{b1} == Compare_pd{b2})
            .debug_valid_eq(b1 == b2));

        sg_assert((Compare_s32x1{b1} == (Compare_s32x1{b2}))
            .debug_valid_eq(b1 == b2));
        sg_assert((Compare_s64x1{b1} == (Compare_s64x1{b2}))
            .debug_valid_eq(b1 == b2));
        sg_assert((Compare_f32x1{b1} == (Compare_f32x1{b2}))
            .debug_valid_eq(b1 == b2));
        sg_assert((Compare_f64x1{b1} == (Compare_f64x1{b2}))
            .debug_valid_eq(b1 == b2));

        // Not equal (XOR)
        sg_assert((Compare_pi32{b1} != (Compare_pi32{b2}))
            .debug_valid_eq(b1 != b2));
        sg_assert((Compare_pi64{b1} != (Compare_pi64{b2}))
            .debug_valid_eq(b1 != b2));
        sg_assert((Compare_ps{b1} != (Compare_ps{b2}))
            .debug_valid_eq(b1 != b2));
        sg_assert((Compare_pd{b1} != (Compare_pd{b2}))
            .debug_valid_eq(b1 != b2));

        sg_assert((Compare_s32x1{b1} != (Compare_s32x1{b2}))
            .debug_valid_eq(b1 != b2));
        sg_assert((Compare_s64x1{b1} != (Compare_s64x1{b2}))
            .debug_valid_eq(b1 != b2));
        sg_assert((Compare_f32x1{b1} != (Compare_f32x1{b2}))
            .debug_valid_eq(b1 != b2));
        sg_assert((Compare_f64x1{b1} != (Compare_f64x1{b2}))
            .debug_valid_eq(b1 != b2));

        // NOT
        sg_assert((!Compare_pi32{b1}).debug_valid_eq(!b1));
        sg_assert((!Compare_pi64{b1}).debug_valid_eq(!b1));
        sg_assert((!Compare_ps{b1}).debug_valid_eq(!b1));
        sg_assert((!Compare_pd{b1}).debug_valid_eq(!b1));

        sg_assert((!Compare_s32x1{b1}).debug_valid_eq(!b1));
        sg_assert((!Compare_s64x1{b1}).debug_valid_eq(!b1));
        sg_assert((!Compare_f32x1{b1}).debug_valid_eq(!b1));
        sg_assert((!Compare_f64x1{b1}).debug_valid_eq(!b1));

    } }

    // Choose or zero
    sg_assert((Compare_pi32{false}.choose_else_zero(2).debug_eq(0)));
    sg_assert((Compare_pi32{true}.choose_else_zero(2).debug_eq(2)));
    sg_assert((Compare_pi64{false}.choose_else_zero(2).debug_eq(0)));
    sg_assert((Compare_pi64{true}.choose_else_zero(2).debug_eq(2)));
    sg_assert((Compare_ps{false}.choose_else_zero(2.0f).debug_eq(0.0f)));
    sg_assert((Compare_ps{true}.choose_else_zero(2.0f).debug_eq(2.0f)));
    sg_assert((Compare_pd{false}.choose_else_zero(2.0).debug_eq(0.0)));
    sg_assert((Compare_pd{true}.choose_else_zero(2.0).debug_eq(2.0)));

    sg_assert((Compare_s32x1{false}.choose_else_zero(2).debug_eq(0)));
    sg_assert((Compare_s32x1{true}.choose_else_zero(2).debug_eq(2)));
    sg_assert((Compare_s64x1{false}.choose_else_zero(2).debug_eq(0)));
    sg_assert((Compare_s64x1{true}.choose_else_zero(2).debug_eq(2)));
    sg_assert((Compare_f32x1{false}.choose_else_zero(2.0f).debug_eq(0.0f)));
    sg_assert((Compare_f32x1{true}.choose_else_zero(2.0f).debug_eq(2.0f)));
    sg_assert((Compare_f64x1{false}.choose_else_zero(2.0).debug_eq(0.0)));
    sg_assert((Compare_f64x1{true}.choose_else_zero(2.0).debug_eq(2.0)));

    // Choose
    sg_assert(Compare_pi32{false}.choose(2, 3).debug_eq(3));
    sg_assert(Compare_pi32{true}.choose(2, 3).debug_eq(2));
    sg_assert(Compare_pi64{false}.choose(2, 3).debug_eq(3));
    sg_assert(Compare_pi64{true}.choose(2, 3).debug_eq(2));
    sg_assert(Compare_ps{false}.choose(2.0f, 3.0f).debug_eq(3.0f));
    sg_assert(Compare_ps{true}.choose(2.0f, 3.0f).debug_eq(2.0f));
    sg_assert(Compare_pd{false}.choose(2.0, 3.0).debug_eq(3.0));
    sg_assert(Compare_pd{true}.choose(2.0, 3.0).debug_eq(2.0));

    sg_assert(Compare_s32x1{false}.choose(2, 3).debug_eq(3));
    sg_assert(Compare_s32x1{true}.choose(2, 3).debug_eq(2));
    sg_assert(Compare_s64x1{false}.choose(2, 3).debug_eq(3));
    sg_assert(Compare_s64x1{true}.choose(2, 3).debug_eq(2));
    sg_assert(Compare_f32x1{false}.choose(2.0f, 3.0f).debug_eq(3.0f));
    sg_assert(Compare_f32x1{true}.choose(2.0f, 3.0f).debug_eq(2.0f));
    sg_assert(Compare_f64x1{false}.choose(2.0, 3.0).debug_eq(3.0));
    sg_assert(Compare_f64x1{true}.choose(2.0, 3.0).debug_eq(2.0));

    //printf("Comparison operator overloading test succeeded\n");
}
#endif
