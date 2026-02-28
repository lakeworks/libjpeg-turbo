/*
 * jquanti-avx512.c - quantization (AVX-512)
 *
 * Copyright (C) 2025, Lakeworks.
 *
 * AVX-512BW implementation of quantization.
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

  const __m512i zero = _mm512_setzero_si512();

  /* Load all data upfront — gives OOO engine maximum scheduling freedom */
  __m512i val0 = _mm512_loadu_si512((__m512i *)(workspace + 0));
  __m512i val1 = _mm512_loadu_si512((__m512i *)(workspace + 32));
  __m512i rec0 = _mm512_loadu_si512((__m512i *)(recip + 0));
  __m512i rec1 = _mm512_loadu_si512((__m512i *)(recip + 32));
  __m512i cor0 = _mm512_loadu_si512((__m512i *)(corr + 0));
  __m512i cor1 = _mm512_loadu_si512((__m512i *)(corr + 32));
  __m512i sca0 = _mm512_loadu_si512((__m512i *)(scale + 0));
  __m512i sca1 = _mm512_loadu_si512((__m512i *)(scale + 32));

  /* Interleave half0/half1 computation to hide mulhi_epu16 5-cycle latency.
   * While half0's multiply retires, half1's instructions fill the pipeline. */
  __m512i abs0 = _mm512_abs_epi16(val0);
  __m512i abs1 = _mm512_abs_epi16(val1);

  abs0 = _mm512_add_epi16(abs0, cor0);
  abs1 = _mm512_add_epi16(abs1, cor1);

  abs0 = _mm512_mulhi_epu16(abs0, rec0);   /* 5c latency — half1 fills gap */
  abs1 = _mm512_mulhi_epu16(abs1, rec1);

  abs0 = _mm512_mulhi_epu16(abs0, sca0);   /* 5c latency */
  abs1 = _mm512_mulhi_epu16(abs1, sca1);

  /* Restore sign: negate where original was negative */
  __mmask32 neg0 = _mm512_cmpgt_epi16_mask(zero, val0);
  __mmask32 neg1 = _mm512_cmpgt_epi16_mask(zero, val1);

  __m512i res0 = _mm512_mask_sub_epi16(abs0, neg0, zero, abs0);
  __m512i res1 = _mm512_mask_sub_epi16(abs1, neg1, zero, abs1);

  _mm512_storeu_si512((__m512i *)(coef_block + 0), res0);
  _mm512_storeu_si512((__m512i *)(coef_block + 32), res1);
}
