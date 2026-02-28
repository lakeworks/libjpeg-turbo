/*
 * jdsample-avx512.c - chroma upsampling (AVX-512 VBMI)
 *
 * Copyright (C) 2025, Lakeworks.
 *
 * AVX-512BW+VBMI implementation of chroma upsampling (simple box filter).
 * Uses vpermb (VBMI) for single-instruction byte duplication:
 *   64 input bytes → 128 output bytes per iteration (2 vpermb).
 *
 * Based on the AVX2 NASM implementation:
 * Copyright (C) 2009, 2014-2015, D. R. Commander.
 * Copyright (C) 2015, Matthieu Darbois.
 */

#define JPEG_INTERNALS
#include "../../src/jinclude.h"
#include "../../src/jpeglib.h"
#include "../../src/jdct.h"
#include "../jsimd.h"

#include <immintrin.h>

/* Build marker — detectable via: strings php8ts.dll | findstr LAKEWORKS
 * or from PHP: php -r "echo bin2hex(file_get_contents('php8ts.dll'));" | findstr pattern
 */
const char jsimd_avx512_build_id[] =
  "LAKEWORKS_LIBJPEG_TURBO_AVX512_ZEN5_V1";

/*
 * Upsample (simple box filter) for 2:1 horizontal, 1:1 vertical.
 * Each input pixel is duplicated to produce two output pixels.
 *
 * Uses vpermb (VBMI) for byte-level permutation: 2 ops instead of
 * 4 (unpacklo+unpackhi+permutex2var x2).  Zen 5: 1c latency, port 2/3.
 */
GLOBAL(void)
jsimd_h2v1_upsample_avx512(int max_v_samp_factor, JDIMENSION output_width,
                            JSAMPARRAY input_data,
                            JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;

  /* Byte duplication indices (hoisted): byte i → positions 2i, 2i+1.
   * _mm512_set_epi8 is HIGH-to-LOW: first arg = byte 63, last = byte 0. */
  const __m512i idx_lo = _mm512_set_epi8(
    31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24,
    23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16,
    15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10,  9,  9,  8,  8,
     7,  7,  6,  6,  5,  5,  4,  4,  3,  3,  2,  2,  1,  1,  0,  0);
  const __m512i idx_hi = _mm512_set_epi8(
    63, 63, 62, 62, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 56, 56,
    55, 55, 54, 54, 53, 53, 52, 52, 51, 51, 50, 50, 49, 49, 48, 48,
    47, 47, 46, 46, 45, 45, 44, 44, 43, 43, 42, 42, 41, 41, 40, 40,
    39, 39, 38, 38, 37, 37, 36, 36, 35, 35, 34, 34, 33, 33, 32, 32);

  for (int inrow = 0; inrow < max_v_samp_factor; inrow++) {
    JSAMPROW inptr = input_data[inrow];
    JSAMPROW outptr = output_data[inrow];
    JDIMENSION colctr;

    /* Process 64 input bytes → 128 output bytes per iteration */
    for (colctr = 0; colctr + 128 <= output_width; colctr += 128) {
      __m512i in = _mm512_loadu_si512((__m512i *)(inptr + colctr / 2));

      /* vpermb: each result byte = in[idx[i] & 63] */
      __m512i out_lo = _mm512_permutexvar_epi8(idx_lo, in);  /* bytes 0-31 → 0-63 */
      __m512i out_hi = _mm512_permutexvar_epi8(idx_hi, in);  /* bytes 32-63 → 64-127 */

      _mm512_storeu_si512((__m512i *)(outptr + colctr), out_lo);
      _mm512_storeu_si512((__m512i *)(outptr + colctr + 64), out_hi);
    }

    /* Handle remaining with scalar */
    for (; colctr + 2 <= output_width; colctr += 2) {
      JSAMPLE val = inptr[colctr / 2];
      outptr[colctr] = val;
      outptr[colctr + 1] = val;
    }
  }
}

/*
 * Upsample (simple box filter) for 2:1 horizontal, 2:1 vertical.
 * Each input pixel is duplicated to a 2x2 block.
 */
GLOBAL(void)
jsimd_h2v2_upsample_avx512(int max_v_samp_factor, JDIMENSION output_width,
                            JSAMPARRAY input_data,
                            JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  int inrow = 0, outrow = 0;

  /* Same duplication indices as h2v1 */
  const __m512i idx_lo = _mm512_set_epi8(
    31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24,
    23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16,
    15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10,  9,  9,  8,  8,
     7,  7,  6,  6,  5,  5,  4,  4,  3,  3,  2,  2,  1,  1,  0,  0);
  const __m512i idx_hi = _mm512_set_epi8(
    63, 63, 62, 62, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 56, 56,
    55, 55, 54, 54, 53, 53, 52, 52, 51, 51, 50, 50, 49, 49, 48, 48,
    47, 47, 46, 46, 45, 45, 44, 44, 43, 43, 42, 42, 41, 41, 40, 40,
    39, 39, 38, 38, 37, 37, 36, 36, 35, 35, 34, 34, 33, 33, 32, 32);

  while (outrow < max_v_samp_factor) {
    JSAMPROW inptr = input_data[inrow];
    JSAMPROW outptr0 = output_data[outrow];
    JSAMPROW outptr1 = output_data[outrow + 1];
    JDIMENSION colctr;

    /* Process 64 input bytes → 128 output bytes, write to 2 rows */
    for (colctr = 0; colctr + 128 <= output_width; colctr += 128) {
      __m512i in = _mm512_loadu_si512((__m512i *)(inptr + colctr / 2));

      __m512i out_lo = _mm512_permutexvar_epi8(idx_lo, in);
      __m512i out_hi = _mm512_permutexvar_epi8(idx_hi, in);

      /* Write same data to both output rows */
      _mm512_storeu_si512((__m512i *)(outptr0 + colctr), out_lo);
      _mm512_storeu_si512((__m512i *)(outptr0 + colctr + 64), out_hi);
      _mm512_storeu_si512((__m512i *)(outptr1 + colctr), out_lo);
      _mm512_storeu_si512((__m512i *)(outptr1 + colctr + 64), out_hi);
    }

    /* Handle remaining */
    for (; colctr + 2 <= output_width; colctr += 2) {
      JSAMPLE val = inptr[colctr / 2];
      outptr0[colctr] = val;
      outptr0[colctr + 1] = val;
      outptr1[colctr] = val;
      outptr1[colctr + 1] = val;
    }

    inrow++;
    outrow += 2;
  }
}
