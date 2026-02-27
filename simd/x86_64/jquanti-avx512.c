/*
 * jquanti-avx512.c - sample data conversion and quantization (AVX-512)
 *
 * Copyright (C) 2025, Lakeworks.
 *
 * AVX-512BW implementation of sample conversion and quantization.
 * Processes the entire 64-coefficient DCT block in 2 ZMM loads instead of 4
 * YMM loads.
 *
 * Based on the AVX2 NASM implementation:
 * Copyright (C) 2016, D. R. Commander.
 * Copyright (C) 2016, Matthieu Darbois.
 */

#define JPEG_INTERNALS
#include "../../src/jinclude.h"
#include "../../src/jpeglib.h"
#include "../../src/jdct.h"
#include "../jsimd.h"

#include <immintrin.h>

/*
 * Convert 8x8 sample block to 16-bit workspace, subtracting CENTERJSAMPLE.
 * Loads 8 rows of 8 bytes each, zero-extends to 16-bit, subtracts 128.
 */
GLOBAL(void)
jsimd_convsamp_avx512(JSAMPARRAY sample_data, JDIMENSION start_col,
                      DCTELEM *workspace)
{
  const __m512i bias = _mm512_set1_epi16(-CENTERJSAMPLE);  /* -128 = 0xFF80 */

  /* Load 8 rows of 8 bytes each into two ZMM registers (4 rows per ZMM).
   * Each row produces 8 words = 16 bytes = 128 bits.
   * 4 rows = 512 bits = 1 ZMM register.
   */

  /* Rows 0-3 */
  __m128i r0 = _mm_loadl_epi64((__m128i *)(sample_data[0] + start_col));
  __m128i r1 = _mm_loadl_epi64((__m128i *)(sample_data[1] + start_col));
  __m128i r2 = _mm_loadl_epi64((__m128i *)(sample_data[2] + start_col));
  __m128i r3 = _mm_loadl_epi64((__m128i *)(sample_data[3] + start_col));

  /* Combine row pairs into 128-bit chunks, then into 256-bit, then 512-bit */
  __m256i lo01 = _mm256_cvtepu8_epi16(r0);
  __m256i lo23 = _mm256_cvtepu8_epi16(r2);
  __m256i hi01 = _mm256_cvtepu8_epi16(r1);
  __m256i hi23 = _mm256_cvtepu8_epi16(r3);

  /* Actually, simpler approach: use individual cvt then combine */
  __m128i w0 = _mm_cvtepu8_epi16(r0);
  __m128i w1 = _mm_cvtepu8_epi16(r1);
  __m128i w2 = _mm_cvtepu8_epi16(r2);
  __m128i w3 = _mm_cvtepu8_epi16(r3);

  __m256i v01 = _mm256_setr_m128i(w0, w1);
  __m256i v23 = _mm256_setr_m128i(w2, w3);
  __m512i rows03 = _mm512_inserti64x4(_mm512_castsi256_si512(v01), v23, 1);

  /* Rows 4-7 */
  __m128i r4 = _mm_loadl_epi64((__m128i *)(sample_data[4] + start_col));
  __m128i r5 = _mm_loadl_epi64((__m128i *)(sample_data[5] + start_col));
  __m128i r6 = _mm_loadl_epi64((__m128i *)(sample_data[6] + start_col));
  __m128i r7 = _mm_loadl_epi64((__m128i *)(sample_data[7] + start_col));

  __m128i w4 = _mm_cvtepu8_epi16(r4);
  __m128i w5 = _mm_cvtepu8_epi16(r5);
  __m128i w6 = _mm_cvtepu8_epi16(r6);
  __m128i w7 = _mm_cvtepu8_epi16(r7);

  __m256i v45 = _mm256_setr_m128i(w4, w5);
  __m256i v67 = _mm256_setr_m128i(w6, w7);
  __m512i rows47 = _mm512_inserti64x4(_mm512_castsi256_si512(v45), v67, 1);

  /* Subtract CENTERJSAMPLE (128) and store */
  rows03 = _mm512_add_epi16(rows03, bias);
  rows47 = _mm512_add_epi16(rows47, bias);

  _mm512_storeu_si512((__m512i *)(workspace + 0), rows03);
  _mm512_storeu_si512((__m512i *)(workspace + 32), rows47);
}

/*
 * Quantize/descale the coefficients, and store into coef_block.
 *
 * The divisors table contains:
 *   [0..63]   = reciprocal (DCTELEM)
 *   [64..127] = correction (DCTELEM)
 *   [128..191] = scale (DCTELEM)
 *
 * Algorithm (Agner Fog method, same as AVX2):
 *   1. Take absolute value
 *   2. Add correction
 *   3. Multiply by reciprocal (unsigned high multiply)
 *   4. Multiply by scale (unsigned high multiply)
 *   5. Restore original sign
 */
GLOBAL(void)
jsimd_quantize_avx512(JCOEFPTR coef_block, DCTELEM *divisors,
                      DCTELEM *workspace)
{
  DCTELEM *recip = divisors + 0 * DCTSIZE2;
  DCTELEM *corr  = divisors + 1 * DCTSIZE2;
  DCTELEM *scale = divisors + 2 * DCTSIZE2;

  /* Process all 64 coefficients in 2 ZMM iterations (32 per ZMM) */

  /* First 32 coefficients (rows 0-3) */
  __m512i val0 = _mm512_loadu_si512((__m512i *)(workspace + 0));
  __m512i rec0 = _mm512_loadu_si512((__m512i *)(recip + 0));
  __m512i cor0 = _mm512_loadu_si512((__m512i *)(corr + 0));
  __m512i sca0 = _mm512_loadu_si512((__m512i *)(scale + 0));

  __m512i abs0 = _mm512_abs_epi16(val0);
  abs0 = _mm512_add_epi16(abs0, cor0);
  abs0 = _mm512_mulhi_epu16(abs0, rec0);
  abs0 = _mm512_mulhi_epu16(abs0, sca0);
  /* Restore sign: negate where original was negative, zero where zero */
  __mmask32 neg0 = _mm512_cmpgt_epi16_mask(_mm512_setzero_si512(), val0);
  __m512i res0 = _mm512_mask_sub_epi16(abs0, neg0,
                                       _mm512_setzero_si512(), abs0);

  _mm512_storeu_si512((__m512i *)(coef_block + 0), res0);

  /* Second 32 coefficients (rows 4-7) */
  __m512i val1 = _mm512_loadu_si512((__m512i *)(workspace + 32));
  __m512i rec1 = _mm512_loadu_si512((__m512i *)(recip + 32));
  __m512i cor1 = _mm512_loadu_si512((__m512i *)(corr + 32));
  __m512i sca1 = _mm512_loadu_si512((__m512i *)(scale + 32));

  __m512i abs1 = _mm512_abs_epi16(val1);
  abs1 = _mm512_add_epi16(abs1, cor1);
  abs1 = _mm512_mulhi_epu16(abs1, rec1);
  abs1 = _mm512_mulhi_epu16(abs1, sca1);
  __mmask32 neg1 = _mm512_cmpgt_epi16_mask(_mm512_setzero_si512(), val1);
  __m512i res1 = _mm512_mask_sub_epi16(abs1, neg1,
                                       _mm512_setzero_si512(), abs1);

  _mm512_storeu_si512((__m512i *)(coef_block + 32), res1);
}
