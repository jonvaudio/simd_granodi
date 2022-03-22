/*

Copyright (c) 2021-2022 Jon Ville

Very thin conversion layer for compiling old SSE2 code on NEON, used by
author

- Does NOT include every intrinsic.
- Some intrinsics are NOT implemented efficiently.
- __m128i is represented as int32x4_t, so intrinsics that use int64x2_t just
  reinterpret everywhere.

*/

#include "simd_granodi.h"

#ifdef SIMD_GRANODI_NEON

typedef int32x4_t __m128i; // The most common, then reinterpret for other ints
typedef float32x4_t __m128;
typedef float64x2_t __m128d;
#define _mm_castsi128_ps sg_cast_pi32_ps
#define _mm_castps_si128 sg_cast_ps_pi32
#define _mm_castsi128_pd sg_cast_pi32_pd
#define _mm_castpd_si128 sg_cast_pd_pi32
#define _mm_castps_pd sg_cast_ps_pd
#define _mm_castpd_ps sg_cast_pd_ps
#define _mm_shuffle_epi32 sg_shuffle_pi32_switch_
static inline float32x4_t _mm_shuffle_ps(const float32x4_t a,
    const float32x4_t b, const int imm8_compile_time_constant) {
    return vcombine_f32(
        vget_low_f32(sg_shuffle_ps_switch_(a, imm8_compile_time_constant)),
        vget_high_f32(sg_shuffle_ps_switch_(b, imm8_compile_time_constant))); }
static inline float64x2_t _mm_shuffle_pd(const float64x2_t a,
    const float64x2_t b, const int imm8_compile_time_constant) {
    return vcombine_f64(
        vget_low_f64(sg_shuffle_pd_switch_(a, imm8_compile_time_constant)),
        vget_high_f64(sg_shuffle_pd_switch_(b, imm8_compile_time_constant))); }
#define _mm_setzero_si128 sg_setzero_pi32
#define _mm_setzero_ps sg_setzero_ps
#define _mm_setzero_pd sg_setzero_pd
#define _mm_set_ss(a) vsetq_lane_f32(a, sg_setzero_ps(), 0)
#define _mm_set_sd(a) vsetq_lane_f64(a, sg_setzero_pd(), 0)
#define _mm_set1_epi32 sg_set1_pi32
#define _mm_set1_epi64x(a) sg_cast_pi64_pi32(sg_set1_pi64(a))
#define _mm_set1_ps sg_set1_ps
#define _mm_set_ps1 sg_set1_ps
#define _mm_set1_pd sg_set1_pd
#define _mm_set_pd1 sg_set1_pd
#define _mm_set_epi32 sg_set_pi32
#define _mm_set_epi64x(si1, si0) sg_cast_pi64_pi32(sg_set_pi64(si1, si0))
#define _mm_set_ps sg_set_ps
#define _mm_set_pd sg_set_pd
#define _mm_cvtsi128_si32 sg_get0_pi32
#define _mm_cvtsi128_si64(a) sg_get0_pi64(sg_cast_pi32_pi64(a))
#define _mm_cvtss_f32 sg_get0_ps
#define _mm_cvtsd_f64 sg_get0_pd
#define _mm_cvtsi64_ss(a, b) vsetq_lane_f32((float) (b), a, 0)
#define _mm_cvtsi64_sd(a, b) vsetq_lane_f64((double) (b), a, 0)
#define _mm_cvtsd_si64(a) sg_get0_pi64(sg_cvt_pd_pi64(a))
#define _mm_cvtss_si32(a) sg_get0_pi32(sg_cvt_ps_pi32(a))
#define _mm_cvtepi32_ps sg_cvt_pi32_ps
#define _mm_cvtepi32_pd sg_cvt_pi32_pd
#define _mm_cvtps_epi32 sg_cvt_ps_pi32
#define _mm_cvttps_epi32 sg_cvtt_ps_pi32
#define _mm_cvtpd_epi32 sg_cvt_pd_pi32
#define _mm_cvttpd_epi32 sg_cvtt_pd_pi32
#define _mm_cvtpd_ps sg_cvt_pd_ps
#define _mm_cvtps_pd sg_cvt_ps_pd
#define _mm_unpackhi_pd(a, b) vcombine_f64(vget_high_f64(a), vget_high_f64(b))
#define _mm_unpackhi_epi64(a, b) sg_cast_pi64_pi32(vcombine_s64( \
    vget_high_s64(sg_cast_pi32_pi64(a)), \
    vget_high_s64(sg_cast_pi32_pi64(b))))
#define _mm_add_epi32 sg_add_pi32
#define _mm_add_epi64(a, b) sg_cast_pi64_pi32( \
    sg_add_pi64(sg_cast_pi32_pi64(a), sg_cast_pi32_pi64(b)))
#define _mm_add_ps sg_add_ps
#define _mm_add_pd sg_add_pd
#define _mm_sub_epi32 sg_sub_pi32
#define _mm_sub_epi64(a, b) sg_cast_pi64_pi32( \
    sg_sub_pi64(sg_cast_pi32_pi64(a), sg_cast_pi32_pi64(b)))
#define _mm_sub_ps sg_sub_ps
#define _mm_sub_pd sg_sub_pd
#define _mm_mul_ps sg_mul_ps
#define _mm_mul_pd sg_mul_pd
#define _mm_div_ps sg_div_ps
#define _mm_div_pd sg_div_pd
#define _mm_andnot_si128 sg_andnot_pi32 // Yes, arg swap is correct!
#define _mm_andnot_ps sg_andnot_ps
#define _mm_andnot_pd sg_andnot_pd
#define _mm_and_si128 sg_and_pi32
#define _mm_and_ps sg_and_ps
#define _mm_and_pd sg_and_pd
#define _mm_or_si128 sg_or_pi32
#define _mm_or_ps sg_or_ps
#define _mm_or_pd sg_or_pd
#define _mm_xor_si128 sg_xor_pi32
#define _mm_xor_ps sg_xor_ps
#define _mm_xor_pd sg_xor_pd
#define _mm_slli_epi32 sg_sl_imm_pi32
#define _mm_slli_epi64(a, imm) sg_cast_pi64_pi32( \
    sg_sl_imm_pi64(sg_cast_pi32_pi64(a), imm))
#define _mm_srli_epi32 sg_srl_imm_pi32
#define _mm_srli_epi64(a, imm) sg_cast_pi64_pi32(sg_srl_imm_pi64( \
    sg_cast_pi32_pi64(a), imm))
#define _mm_srai_epi32 sg_sra_imm_pi32
#define _mm_cmplt_epi32 sg_cmplt_pi32
#define _mm_cmplt_ps sg_cmplt_ps
#define _mm_cmplt_pd sg_cmplt_pd
#define _mm_cmple_ps sg_cmplte_ps
#define _mm_cmple_pd sg_cmplte_pd
#define _mm_cmpeq_epi32 sg_cmpeq_pi32
#define _mm_cmpeq_ps sg_cmpeq_ps
#define _mm_cmpeq_pd sg_cmpeq_pd
#define _mm_cmpneq_ps sg_cmpneq_ps
#define _mm_cmpneq_pd sg_cmpneq_pd
#define _mm_cmpge_ps sg_cmpgte_ps
#define _mm_cmpge_pd sg_cmpgte_pd
#define _mm_cmpgt_epi32 sg_cmpgt_pi32
#define _mm_cmpgt_ps sg_cmpgt_ps
#define _mm_cmpgt_pd sg_cmpgt_pd
// Note: these handle signed zero differently than on intel, but shouldn't
// matter!
#define _mm_min_ps sg_min_fast_ps
#define _mm_min_pd sg_min_fast_pd
#define _mm_max_ps sg_max_fast_ps
#define _mm_max_pd sg_max_fast_pd

#endif
