/*
 * jcsample-avx512.c - chroma downsampling (AVX-512)
 *
 * Copyright (C) 2025, Lakeworks.
 *
 * AVX-512BW implementation of chroma downsampling.
 * Processes 128 input pixels → 64 output pixels per iteration.
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

/*
 * Downsample pixel values of a single component.
 * This version handles the common case of 2:1 horizontal and 1:1 vertical,
 * without smoothing.
 *
 * A://input row is padded to a multiple of the DCT block size (8).
 */
GLOBAL(void)
jsimd_h2v1_downsample_avx512(JDIMENSION image_width, int max_v_samp_factor,
                              JDIMENSION v_samp_factor,
                              JDIMENSION width_in_blocks,
                              JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  JDIMENSION output_cols = width_in_blocks * DCTSIZE;  /* 8 * blocks */
  const __m512i mask_even = _mm512_set1_epi16(0x00FF);
  const __m512i bias = _mm512_set1_epi16(1);  /* rounding bias */
  /* packus interleaves 128-bit lanes — fix with lane permute (hoisted) */
  const __m512i pack_perm = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

  for (int outrow = 0; outrow < (int)v_samp_factor; outrow++) {
    JSAMPROW inptr = input_data[outrow];
    JSAMPROW outptr = output_data[outrow];
    JDIMENSION outcol;

    /* Process 128 input bytes → 64 output bytes per iteration */
    for (outcol = 0; outcol + 64 <= output_cols; outcol += 64) {
      /* Load 128 input bytes (64 pixel pairs) */
      __m512i in0 = _mm512_loadu_si512((__m512i *)(inptr + outcol * 2));
      __m512i in1 = _mm512_loadu_si512((__m512i *)(inptr + outcol * 2 + 64));

      /* Extract even bytes (pixel[2n]) and odd bytes (pixel[2n+1]) */
      __m512i even0 = _mm512_and_si512(in0, mask_even);
      __m512i odd0 = _mm512_srli_epi16(in0, 8);
      __m512i even1 = _mm512_and_si512(in1, mask_even);
      __m512i odd1 = _mm512_srli_epi16(in1, 8);

      /* Average: (even + odd + 1) >> 1 */
      __m512i sum0 = _mm512_add_epi16(even0, odd0);
      sum0 = _mm512_add_epi16(sum0, bias);
      sum0 = _mm512_srli_epi16(sum0, 1);

      __m512i sum1 = _mm512_add_epi16(even1, odd1);
      sum1 = _mm512_add_epi16(sum1, bias);
      sum1 = _mm512_srli_epi16(sum1, 1);

      /* Pack 16-bit results back to 8-bit */
      __m512i result = _mm512_packus_epi16(sum0, sum1);
      result = _mm512_permutexvar_epi64(pack_perm, result);

      _mm512_storeu_si512((__m512i *)(outptr + outcol), result);
    }

    /* Handle remaining columns with scalar */
    for (; outcol < output_cols; outcol++) {
      int in_off = outcol * 2;
      int val = (int)inptr[in_off] + (int)inptr[in_off + 1] + 1;
      outptr[outcol] = (JSAMPLE)(val >> 1);
    }
  }
}

/*
 * Downsample pixel values of a single component.
 * This version handles the common case of 2:1 horizontal and 2:1 vertical,
 * without smoothing.
 */
GLOBAL(void)
jsimd_h2v2_downsample_avx512(JDIMENSION image_width, int max_v_samp_factor,
                              JDIMENSION v_samp_factor,
                              JDIMENSION width_in_blocks,
                              JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  JDIMENSION output_cols = width_in_blocks * DCTSIZE;
  const __m512i mask_even = _mm512_set1_epi16(0x00FF);
  const __m512i bias = _mm512_set1_epi16(2);  /* rounding bias for /4 */
  const __m512i pack_perm = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

  for (int outrow = 0; outrow < (int)v_samp_factor; outrow++) {
    JSAMPROW inptr0 = input_data[outrow * 2];
    JSAMPROW inptr1 = input_data[outrow * 2 + 1];
    JSAMPROW outptr = output_data[outrow];
    JDIMENSION outcol;

    /* Process 128 input bytes → 64 output bytes per iteration */
    for (outcol = 0; outcol + 64 <= output_cols; outcol += 64) {
      /* Load 128 input bytes from each of 2 rows */
      __m512i r0_0 = _mm512_loadu_si512((__m512i *)(inptr0 + outcol * 2));
      __m512i r0_1 = _mm512_loadu_si512((__m512i *)(inptr0 + outcol * 2 + 64));
      __m512i r1_0 = _mm512_loadu_si512((__m512i *)(inptr1 + outcol * 2));
      __m512i r1_1 = _mm512_loadu_si512((__m512i *)(inptr1 + outcol * 2 + 64));

      /* Row 0: extract even+odd */
      __m512i e0_0 = _mm512_and_si512(r0_0, mask_even);
      __m512i o0_0 = _mm512_srli_epi16(r0_0, 8);
      __m512i e0_1 = _mm512_and_si512(r0_1, mask_even);
      __m512i o0_1 = _mm512_srli_epi16(r0_1, 8);

      /* Row 1: extract even+odd */
      __m512i e1_0 = _mm512_and_si512(r1_0, mask_even);
      __m512i o1_0 = _mm512_srli_epi16(r1_0, 8);
      __m512i e1_1 = _mm512_and_si512(r1_1, mask_even);
      __m512i o1_1 = _mm512_srli_epi16(r1_1, 8);

      /* Sum all 4 pixels in the 2x2 block: (e0+o0+e1+o1+2) >> 2 */
      __m512i sum0 = _mm512_add_epi16(e0_0, o0_0);
      sum0 = _mm512_add_epi16(sum0, e1_0);
      sum0 = _mm512_add_epi16(sum0, o1_0);
      sum0 = _mm512_add_epi16(sum0, bias);
      sum0 = _mm512_srli_epi16(sum0, 2);

      __m512i sum1 = _mm512_add_epi16(e0_1, o0_1);
      sum1 = _mm512_add_epi16(sum1, e1_1);
      sum1 = _mm512_add_epi16(sum1, o1_1);
      sum1 = _mm512_add_epi16(sum1, bias);
      sum1 = _mm512_srli_epi16(sum1, 2);

      /* Pack back to 8-bit */
      __m512i result = _mm512_packus_epi16(sum0, sum1);
      result = _mm512_permutexvar_epi64(pack_perm, result);

      _mm512_storeu_si512((__m512i *)(outptr + outcol), result);
    }

    /* Handle remaining columns */
    for (; outcol < output_cols; outcol++) {
      int in_off = outcol * 2;
      int val = (int)inptr0[in_off] + (int)inptr0[in_off + 1] +
                (int)inptr1[in_off] + (int)inptr1[in_off + 1] + 2;
      outptr[outcol] = (JSAMPLE)(val >> 2);
    }
  }
}
