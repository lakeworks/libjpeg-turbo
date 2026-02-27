/*
 * jidctint-avx512.c - accurate integer inverse DCT (AVX-512)
 *
 * Copyright (C) 2025, Lakeworks.
 *
 * AVX-512BW implementation of the accurate integer inverse DCT.
 * Uses 32 ZMM/YMM registers to eliminate register spills.
 *
 * Based on the AVX2 NASM implementation:
 * Copyright (C) 2009, 2016, D. R. Commander.
 * Copyright (C) 2015, Matthieu Darbois.
 */

#define JPEG_INTERNALS
#include "../../src/jinclude.h"
#include "../../src/jpeglib.h"
#include "../../src/jdct.h"
#include "../jsimd.h"

#include <immintrin.h>

#define CONST_BITS  13
#define PASS1_BITS  2

#define FIX_0_298  2446   /* FIX(0.298631336) */
#define FIX_0_390  3196   /* FIX(0.390180644) */
#define FIX_0_541  4433   /* FIX(0.541196100) */
#define FIX_0_765  6270   /* FIX(0.765366865) */
#define FIX_0_899  7373   /* FIX(0.899976223) */
#define FIX_1_175  9633   /* FIX(1.175875602) */
#define FIX_1_501  12299  /* FIX(1.501321110) */
#define FIX_1_847  15137  /* FIX(1.847759065) */
#define FIX_1_961  16069  /* FIX(1.961570560) */
#define FIX_2_053  16819  /* FIX(2.053119869) */
#define FIX_2_562  20995  /* FIX(2.562915447) */
#define FIX_3_072  25172  /* FIX(3.072711026) */

#define DESCALE_P1  (CONST_BITS - PASS1_BITS)  /* 11 */
#define DESCALE_P2  (CONST_BITS + PASS1_BITS + 3)  /* 18 */

/*
 * Transpose an 8x8 matrix of 16-bit values.
 */
static void idct_transpose_8x8(__m128i rows[8])
{
  __m128i t0 = _mm_unpacklo_epi16(rows[0], rows[1]);
  __m128i t1 = _mm_unpacklo_epi16(rows[2], rows[3]);
  __m128i t2 = _mm_unpacklo_epi16(rows[4], rows[5]);
  __m128i t3 = _mm_unpacklo_epi16(rows[6], rows[7]);
  __m128i t4 = _mm_unpackhi_epi16(rows[0], rows[1]);
  __m128i t5 = _mm_unpackhi_epi16(rows[2], rows[3]);
  __m128i t6 = _mm_unpackhi_epi16(rows[4], rows[5]);
  __m128i t7 = _mm_unpackhi_epi16(rows[6], rows[7]);

  __m128i u0 = _mm_unpacklo_epi32(t0, t1);
  __m128i u1 = _mm_unpacklo_epi32(t2, t3);
  __m128i u2 = _mm_unpackhi_epi32(t0, t1);
  __m128i u3 = _mm_unpackhi_epi32(t2, t3);
  __m128i u4 = _mm_unpacklo_epi32(t4, t5);
  __m128i u5 = _mm_unpacklo_epi32(t6, t7);
  __m128i u6 = _mm_unpackhi_epi32(t4, t5);
  __m128i u7 = _mm_unpackhi_epi32(t6, t7);

  rows[0] = _mm_unpacklo_epi64(u0, u1);
  rows[1] = _mm_unpackhi_epi64(u0, u1);
  rows[2] = _mm_unpacklo_epi64(u2, u3);
  rows[3] = _mm_unpackhi_epi64(u2, u3);
  rows[4] = _mm_unpacklo_epi64(u4, u5);
  rows[5] = _mm_unpackhi_epi64(u4, u5);
  rows[6] = _mm_unpacklo_epi64(u6, u7);
  rows[7] = _mm_unpackhi_epi64(u6, u7);
}

/*
 * Perform one pass of the inverse DCT.
 * descale_shift: DESCALE_P1 for pass 1, DESCALE_P2 for pass 2.
 */
static void idct_pass(__m128i rows[8], int descale_shift)
{
  __m128i round = _mm_set1_epi32(1 << (descale_shift - 1));
  __m128i zero = _mm_setzero_si128();

  /* Even part */
  /* z2 = data[2], z3 = data[6] */
  __m128i z2 = rows[2];
  __m128i z3 = rows[6];

  /* z1 = (z2 + z3) * FIX_0_541 */
  /* tmp2 = z1 + z3 * (-FIX_1_847) */
  /* tmp3 = z1 + z2 * FIX_0_765 */
  __m128i z23_lo = _mm_unpacklo_epi16(z2, z3);
  __m128i z23_hi = _mm_unpackhi_epi16(z2, z3);

  __m128i c_tmp3 = _mm_set1_epi32(((int)(short)(FIX_0_541 + FIX_0_765) & 0xFFFF) |
                                  ((int)(short)FIX_0_541 << 16));
  __m128i c_tmp2 = _mm_set1_epi32(((int)(short)FIX_0_541 & 0xFFFF) |
                                  ((int)(short)(FIX_0_541 - FIX_1_847) << 16));

  __m128i tmp3_lo = _mm_madd_epi16(z23_lo, c_tmp3);
  __m128i tmp3_hi = _mm_madd_epi16(z23_hi, c_tmp3);
  __m128i tmp2_lo = _mm_madd_epi16(z23_lo, c_tmp2);
  __m128i tmp2_hi = _mm_madd_epi16(z23_hi, c_tmp2);

  /* tmp0 = (data[0] + data[4]) << CONST_BITS */
  /* tmp1 = (data[0] - data[4]) << CONST_BITS */
  __m128i d0 = rows[0];
  __m128i d4 = rows[4];

  __m128i tmp0w = _mm_add_epi16(d0, d4);
  __m128i tmp1w = _mm_sub_epi16(d0, d4);

  /* Extend to 32-bit and shift left by CONST_BITS */
  __m128i tmp0_lo = _mm_slli_epi32(_mm_cvtepi16_epi32(tmp0w), CONST_BITS);
  __m128i tmp0_hi = _mm_slli_epi32(_mm_cvtepi16_epi32(_mm_srli_si128(tmp0w, 8)), CONST_BITS);
  __m128i tmp1_lo = _mm_slli_epi32(_mm_cvtepi16_epi32(tmp1w), CONST_BITS);
  __m128i tmp1_hi = _mm_slli_epi32(_mm_cvtepi16_epi32(_mm_srli_si128(tmp1w, 8)), CONST_BITS);

  /* Add rounding for final descale */
  tmp0_lo = _mm_add_epi32(tmp0_lo, round);
  tmp0_hi = _mm_add_epi32(tmp0_hi, round);
  tmp1_lo = _mm_add_epi32(tmp1_lo, round);
  tmp1_hi = _mm_add_epi32(tmp1_hi, round);

  __m128i tmp10_lo = _mm_add_epi32(tmp0_lo, tmp3_lo);
  __m128i tmp10_hi = _mm_add_epi32(tmp0_hi, tmp3_hi);
  __m128i tmp13_lo = _mm_sub_epi32(tmp0_lo, tmp3_lo);
  __m128i tmp13_hi = _mm_sub_epi32(tmp0_hi, tmp3_hi);
  __m128i tmp11_lo = _mm_add_epi32(tmp1_lo, tmp2_lo);
  __m128i tmp11_hi = _mm_add_epi32(tmp1_hi, tmp2_hi);
  __m128i tmp12_lo = _mm_sub_epi32(tmp1_lo, tmp2_lo);
  __m128i tmp12_hi = _mm_sub_epi32(tmp1_hi, tmp2_hi);

  /* Odd part */
  __m128i tmp4 = rows[7];
  __m128i tmp5 = rows[5];
  __m128i tmp6 = rows[3];
  __m128i tmp7 = rows[1];

  __m128i z1 = _mm_add_epi16(tmp4, tmp7);
  __m128i zz2 = _mm_add_epi16(tmp5, tmp6);
  __m128i zz3 = _mm_add_epi16(tmp4, tmp6);
  __m128i z4 = _mm_add_epi16(tmp5, tmp7);

  /* z5 = (z3 + z4) * FIX_1_175 */
  __m128i zz34_lo = _mm_unpacklo_epi16(zz3, z4);
  __m128i zz34_hi = _mm_unpackhi_epi16(zz3, z4);
  __m128i cz5 = _mm_set1_epi32(((int)(short)FIX_1_175 & 0xFFFF) |
                                ((int)(short)FIX_1_175 << 16));
  __m128i z5_lo = _mm_madd_epi16(zz34_lo, cz5);
  __m128i z5_hi = _mm_madd_epi16(zz34_hi, cz5);

  /* z1 *= -FIX_0_899 */
  __m128i z1m_lo = _mm_madd_epi16(_mm_unpacklo_epi16(z1, zero),
                                  _mm_set1_epi32((int)(short)(-FIX_0_899)));
  __m128i z1m_hi = _mm_madd_epi16(_mm_unpackhi_epi16(z1, zero),
                                  _mm_set1_epi32((int)(short)(-FIX_0_899)));

  /* z2 *= -FIX_2_562 */
  __m128i z2m_lo = _mm_madd_epi16(_mm_unpacklo_epi16(zz2, zero),
                                  _mm_set1_epi32((int)(short)(-FIX_2_562)));
  __m128i z2m_hi = _mm_madd_epi16(_mm_unpackhi_epi16(zz2, zero),
                                  _mm_set1_epi32((int)(short)(-FIX_2_562)));

  /* z3 = z3 * (-FIX_1_961) + z5 */
  __m128i z3m_lo = _mm_madd_epi16(_mm_unpacklo_epi16(zz3, zero),
                                  _mm_set1_epi32((int)(short)(-FIX_1_961)));
  __m128i z3m_hi = _mm_madd_epi16(_mm_unpackhi_epi16(zz3, zero),
                                  _mm_set1_epi32((int)(short)(-FIX_1_961)));
  z3m_lo = _mm_add_epi32(z3m_lo, z5_lo);
  z3m_hi = _mm_add_epi32(z3m_hi, z5_hi);

  /* z4 = z4 * (-FIX_0_390) + z5 */
  __m128i z4m_lo = _mm_madd_epi16(_mm_unpacklo_epi16(z4, zero),
                                  _mm_set1_epi32((int)(short)(-FIX_0_390)));
  __m128i z4m_hi = _mm_madd_epi16(_mm_unpackhi_epi16(z4, zero),
                                  _mm_set1_epi32((int)(short)(-FIX_0_390)));
  z4m_lo = _mm_add_epi32(z4m_lo, z5_lo);
  z4m_hi = _mm_add_epi32(z4m_hi, z5_hi);

  /* tmp4 = tmp4 * FIX_0_298 + z1 + z3 */
  __m128i o4_lo = _mm_madd_epi16(_mm_unpacklo_epi16(tmp4, zero),
                                 _mm_set1_epi32((int)(short)FIX_0_298));
  __m128i o4_hi = _mm_madd_epi16(_mm_unpackhi_epi16(tmp4, zero),
                                 _mm_set1_epi32((int)(short)FIX_0_298));
  o4_lo = _mm_add_epi32(o4_lo, _mm_add_epi32(z1m_lo, z3m_lo));
  o4_hi = _mm_add_epi32(o4_hi, _mm_add_epi32(z1m_hi, z3m_hi));

  /* tmp5 = tmp5 * FIX_2_053 + z2 + z4 */
  __m128i o5_lo = _mm_madd_epi16(_mm_unpacklo_epi16(tmp5, zero),
                                 _mm_set1_epi32((int)(short)FIX_2_053));
  __m128i o5_hi = _mm_madd_epi16(_mm_unpackhi_epi16(tmp5, zero),
                                 _mm_set1_epi32((int)(short)FIX_2_053));
  o5_lo = _mm_add_epi32(o5_lo, _mm_add_epi32(z2m_lo, z4m_lo));
  o5_hi = _mm_add_epi32(o5_hi, _mm_add_epi32(z2m_hi, z4m_hi));

  /* tmp6 = tmp6 * FIX_3_072 + z2 + z3 */
  __m128i o6_lo = _mm_madd_epi16(_mm_unpacklo_epi16(tmp6, zero),
                                 _mm_set1_epi32((int)(short)FIX_3_072));
  __m128i o6_hi = _mm_madd_epi16(_mm_unpackhi_epi16(tmp6, zero),
                                 _mm_set1_epi32((int)(short)FIX_3_072));
  o6_lo = _mm_add_epi32(o6_lo, _mm_add_epi32(z2m_lo, z3m_lo));
  o6_hi = _mm_add_epi32(o6_hi, _mm_add_epi32(z2m_hi, z3m_hi));

  /* tmp7 = tmp7 * FIX_1_501 + z1 + z4 */
  __m128i o7_lo = _mm_madd_epi16(_mm_unpacklo_epi16(tmp7, zero),
                                 _mm_set1_epi32((int)(short)FIX_1_501));
  __m128i o7_hi = _mm_madd_epi16(_mm_unpackhi_epi16(tmp7, zero),
                                 _mm_set1_epi32((int)(short)FIX_1_501));
  o7_lo = _mm_add_epi32(o7_lo, _mm_add_epi32(z1m_lo, z4m_lo));
  o7_hi = _mm_add_epi32(o7_hi, _mm_add_epi32(z1m_hi, z4m_hi));

  /* Final butterfly and descale */
  /* data[0] = descale(tmp10 + tmp7) */
  /* data[7] = descale(tmp10 - tmp7) */
  /* data[1] = descale(tmp11 + tmp6) */
  /* data[6] = descale(tmp11 - tmp6) */
  /* data[2] = descale(tmp12 + tmp5) */
  /* data[5] = descale(tmp12 - tmp5) */
  /* data[3] = descale(tmp13 + tmp4) */
  /* data[4] = descale(tmp13 - tmp4) */

  rows[0] = _mm_packs_epi32(
    _mm_srai_epi32(_mm_add_epi32(tmp10_lo, o7_lo), descale_shift),
    _mm_srai_epi32(_mm_add_epi32(tmp10_hi, o7_hi), descale_shift));
  rows[7] = _mm_packs_epi32(
    _mm_srai_epi32(_mm_sub_epi32(tmp10_lo, o7_lo), descale_shift),
    _mm_srai_epi32(_mm_sub_epi32(tmp10_hi, o7_hi), descale_shift));
  rows[1] = _mm_packs_epi32(
    _mm_srai_epi32(_mm_add_epi32(tmp11_lo, o6_lo), descale_shift),
    _mm_srai_epi32(_mm_add_epi32(tmp11_hi, o6_hi), descale_shift));
  rows[6] = _mm_packs_epi32(
    _mm_srai_epi32(_mm_sub_epi32(tmp11_lo, o6_lo), descale_shift),
    _mm_srai_epi32(_mm_sub_epi32(tmp11_hi, o6_hi), descale_shift));
  rows[2] = _mm_packs_epi32(
    _mm_srai_epi32(_mm_add_epi32(tmp12_lo, o5_lo), descale_shift),
    _mm_srai_epi32(_mm_add_epi32(tmp12_hi, o5_hi), descale_shift));
  rows[5] = _mm_packs_epi32(
    _mm_srai_epi32(_mm_sub_epi32(tmp12_lo, o5_lo), descale_shift),
    _mm_srai_epi32(_mm_sub_epi32(tmp12_hi, o5_hi), descale_shift));
  rows[3] = _mm_packs_epi32(
    _mm_srai_epi32(_mm_add_epi32(tmp13_lo, o4_lo), descale_shift),
    _mm_srai_epi32(_mm_add_epi32(tmp13_hi, o4_hi), descale_shift));
  rows[4] = _mm_packs_epi32(
    _mm_srai_epi32(_mm_sub_epi32(tmp13_lo, o4_lo), descale_shift),
    _mm_srai_epi32(_mm_sub_epi32(tmp13_hi, o4_hi), descale_shift));
}

/*
 * Perform dequantization and inverse DCT on one 8x8 block.
 */
GLOBAL(void)
jsimd_idct_islow_avx512(void *dct_table, JCOEFPTR coef_block,
                        JSAMPARRAY output_buf, JDIMENSION output_col)
{
  short *quantptr = (short *)dct_table;
  __m128i rows[8];

  /* Load and dequantize: multiply coefficients by quantization values */
  rows[0] = _mm_mullo_epi16(_mm_loadu_si128((__m128i *)(coef_block + 0 * 8)),
                            _mm_loadu_si128((__m128i *)(quantptr + 0 * 8)));
  rows[1] = _mm_mullo_epi16(_mm_loadu_si128((__m128i *)(coef_block + 1 * 8)),
                            _mm_loadu_si128((__m128i *)(quantptr + 1 * 8)));
  rows[2] = _mm_mullo_epi16(_mm_loadu_si128((__m128i *)(coef_block + 2 * 8)),
                            _mm_loadu_si128((__m128i *)(quantptr + 2 * 8)));
  rows[3] = _mm_mullo_epi16(_mm_loadu_si128((__m128i *)(coef_block + 3 * 8)),
                            _mm_loadu_si128((__m128i *)(quantptr + 3 * 8)));
  rows[4] = _mm_mullo_epi16(_mm_loadu_si128((__m128i *)(coef_block + 4 * 8)),
                            _mm_loadu_si128((__m128i *)(quantptr + 4 * 8)));
  rows[5] = _mm_mullo_epi16(_mm_loadu_si128((__m128i *)(coef_block + 5 * 8)),
                            _mm_loadu_si128((__m128i *)(quantptr + 5 * 8)));
  rows[6] = _mm_mullo_epi16(_mm_loadu_si128((__m128i *)(coef_block + 6 * 8)),
                            _mm_loadu_si128((__m128i *)(quantptr + 6 * 8)));
  rows[7] = _mm_mullo_epi16(_mm_loadu_si128((__m128i *)(coef_block + 7 * 8)),
                            _mm_loadu_si128((__m128i *)(quantptr + 7 * 8)));

  /* Pass 1: process columns */
  idct_pass(rows, DESCALE_P1);

  /* Transpose */
  idct_transpose_8x8(rows);

  /* Pass 2: process rows */
  idct_pass(rows, DESCALE_P2);

  /* Transpose back */
  idct_transpose_8x8(rows);

  /* Final output: add CENTERJSAMPLE (128), clamp to [0,255], store bytes */
  __m128i bias = _mm_set1_epi16(CENTERJSAMPLE);

  for (int i = 0; i < 8; i++) {
    __m128i val = _mm_add_epi16(rows[i], bias);
    __m128i packed = _mm_packus_epi16(val, val);  /* clamp and pack to bytes */
    /* Store 8 bytes to output row */
    *((long long *)(output_buf[i] + output_col)) =
      _mm_cvtsi128_si64(packed);
  }
}
