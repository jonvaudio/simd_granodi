/*

Copyright (c) 2021-2022 Jon Ville

Very thin conversion layer for compiling old SSE2 code on NEON, used by
author (in future, can be spun into different header).

- Does NOT include every intrinsic.
- Some intrinsics are NOT implemented efficiently.
__m128i is represented as int32x4_t, so intrinsics that use int64x2_t just
reinterpret everywhere.

*/

#include "simd_granodi.h"

#ifdef SIMD_GRANODI_NEON

#ifdef __cplusplus
namespace simd_granodi {
#endif

typedef int32x4_t __m128i; // The most common, then reinterpret for other ints
typedef float32x4_t __m128;
typedef float64x2_t __m128d;
#define _mm_castsi128_ps vreinterpretq_f32_s32
#define _mm_castps_si128 vreinterpretq_s32_f32
#define _mm_castsi128_pd vreinterpretq_f64_s32
#define _mm_castpd_si128 vreinterpretq_s32_f64
#define _mm_castps_pd vreinterpretq_f64_f32
#define _mm_castpd_ps vreinterpretq_f32_f64
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
#define _mm_cvtsi128_si64(a) vgetq_lane_s64(vreinterpretq_s64_s32(a), 0)
#define _mm_cvtss_f32(a) vgetq_lane_f32(a, 0)
#define _mm_cvtsd_f64(a) vgetq_lane_f64(a, 0)
#define _mm_cvtsi64_ss(a, b) vsetq_lane_f32((float) (b), a, 0)
#define _mm_cvtsi64_sd(a, b) vsetq_lane_f64((double) (b), a, 0)
#define _mm_cvtsd_si64(a) ((int64_t) vgetq_lane_f64(a, 0))
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
#define _mm_andnot_ps sg_andnot_ps
#define _mm_andnot_pd sg_andnot_pd
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
#define _mm_srai_epi32 vshrq_n_s32
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

#ifdef __cplusplus
}
#endif

#endif
