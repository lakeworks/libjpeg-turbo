/*
 * jdsample-avx512.c - chroma upsampling (AVX-512)
 *
 * Copyright (C) 2025, Lakeworks.
 *
 * AVX-512BW implementation of chroma upsampling (simple box filter).
 * Processes 64 input bytes → 128 output bytes per iteration.
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
 */
GLOBAL(void)
jsimd_h2v1_upsample_avx512(int max_v_samp_factor, JDIMENSION output_width,
                            JSAMPARRAY input_data,
                            JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;

  for (int inrow = 0; inrow < max_v_samp_factor; inrow++) {
    JSAMPROW inptr = input_data[inrow];
    JSAMPROW outptr = output_data[inrow];
    JDIMENSION colctr;

    /* Process 64 input bytes → 128 output bytes per iteration */
    for (colctr = 0; colctr + 128 <= output_width; colctr += 128) {
      __m512i in = _mm512_loadu_si512((__m512i *)(inptr + colctr / 2));

      /* Duplicate each byte: AABB CCDD ... using unpack with self */
      __m512i lo = _mm512_unpacklo_epi8(in, in);  /* duplicate bytes in pairs */
      __m512i hi = _mm512_unpackhi_epi8(in, in);

      /* Fix lane ordering: unpack interleaves within 128-bit lanes */
      /* We need: bytes 0-15 duplicated, then bytes 16-31, 32-47, 48-63 */
      /* unpacklo gives us: lane0_lo_dup, lane1_lo_dup, lane2_lo_dup, lane3_lo_dup */
      /* unpackhi gives us: lane0_hi_dup, lane1_hi_dup, lane2_hi_dup, lane3_hi_dup */
      /* We need to interleave the lo and hi halves of each lane */
      __m512i perm_lo = _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11), hi);
      __m512i perm_hi = _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15), hi);

      _mm512_storeu_si512((__m512i *)(outptr + colctr), perm_lo);
      _mm512_storeu_si512((__m512i *)(outptr + colctr + 64), perm_hi);
    }

    /* Handle remaining with YMM or scalar */
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

  while (outrow < max_v_samp_factor) {
    JSAMPROW inptr = input_data[inrow];
    JSAMPROW outptr0 = output_data[outrow];
    JSAMPROW outptr1 = output_data[outrow + 1];
    JDIMENSION colctr;

    /* Process 64 input bytes → 128 output bytes, write to 2 rows */
    for (colctr = 0; colctr + 128 <= output_width; colctr += 128) {
      __m512i in = _mm512_loadu_si512((__m512i *)(inptr + colctr / 2));

      __m512i lo = _mm512_unpacklo_epi8(in, in);
      __m512i hi = _mm512_unpackhi_epi8(in, in);

      __m512i perm_lo = _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11), hi);
      __m512i perm_hi = _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15), hi);

      /* Write same data to both output rows */
      _mm512_storeu_si512((__m512i *)(outptr0 + colctr), perm_lo);
      _mm512_storeu_si512((__m512i *)(outptr0 + colctr + 64), perm_hi);
      _mm512_storeu_si512((__m512i *)(outptr1 + colctr), perm_lo);
      _mm512_storeu_si512((__m512i *)(outptr1 + colctr + 64), perm_hi);
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
