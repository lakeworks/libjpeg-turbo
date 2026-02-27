/*
 * Merged upsampling/color conversion (64-bit AVX-512BW/VBMI) -- template
 *
 * Copyright (C) 2025, Lakeworks.
 *
 * This file is included by jdmerge-avx512.c multiple times with different
 * RGB_RED, RGB_GREEN, RGB_BLUE, RGB_PIXELSIZE, and RGB_ALPHA definitions to
 * generate all 7 pixel format variants.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

/* This file is included by jdmerge-avx512.c. */


/* These routines combine simple (non-fancy, i.e. non-smooth) h2v1 or h2v2
 * chroma upsampling and YCbCr -> RGB color conversion into a single function.
 *
 * As with the standalone functions, YCbCr -> RGB conversion is defined by the
 * following equations:
 *    R = Y                        + 1.40200 * (Cr - 128)
 *    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
 *    B = Y + 1.77200 * (Cb - 128)
 *
 * Implemented using the same alternate fixed-point form as the AVX2 NASM code:
 *    R = Y + 0.40200 * Cr' + Cr'
 *    G = Y + madd(-0.34414 * Cb', 0.28586 * Cr') - Cr'
 *    B = Y - 0.22800 * Cb' + 2 * Cb'
 *
 * where Cb' = Cb - 128, Cr' = Cr - 128.
 *
 * Constants are defined in jdmerge-avx512.c.
 */

/* Notes on safe memory access for merged upsampling/YCbCr -> RGB conversion
 * routines:
 *
 * Input memory buffers can be safely overread up to the next multiple of
 * ALIGN_SIZE bytes, since they are always allocated by alloc_sarray() in
 * jmemmgr.c.
 *
 * The output buffer cannot safely be written beyond output_width, since
 * output_buf points to a possibly unpadded row in the decompressed image
 * buffer allocated by the calling program.
 */


/*
 * Process one row: compute color diffs from Cb/Cr, apply to Y, interleave
 * into pixel format, and store to output buffer.
 *
 * Processes up to 32 output pixels (16 chroma samples).
 */
static INLINE void
jsimd_h2v1_merged_upsample_avx512_row(JSAMPROW inptr0, JSAMPROW inptr1,
                                       JSAMPROW inptr2, JSAMPROW outptr,
                                       int num_cols)
{
  /* Load 16 Cb and 16 Cr bytes, zero-extend to 16-bit words */
  __m128i cb_raw = _mm_loadu_si128((__m128i *)inptr1);
  __m128i cr_raw = _mm_loadu_si128((__m128i *)inptr2);

  __m256i cb_16 = _mm256_cvtepu8_epi16(cb_raw);
  __m256i cr_16 = _mm256_cvtepu8_epi16(cr_raw);

  /* Subtract 128: Cb' = Cb - 128, Cr' = Cr - 128 (signed) */
  __m256i bias = _mm256_set1_epi16((short)0xFF80);
  __m256i cb = _mm256_add_epi16(cb_16, bias);
  __m256i cr = _mm256_add_epi16(cr_16, bias);

  /* === Compute (B - Y) = Cb' * FIX(1.772) === */
  /* mulhi(2*Cb', -FIX(0.228)) gives (2*Cb' * -14942) >> 16.
   * Round by adding 1 and shifting right 1.
   * Then add 2*Cb' to get Cb' * (2 - 0.228*2) ... no, let's trace:
   *   mulhi(2*Cb', -F_0_228) = approximately 2*Cb' * (-0.228) = -0.456*Cb'
   *   After (x + 1) >> 1:  approximately -0.228*Cb'
   *   Then + Cb' + Cb' = -0.228*Cb' + 2*Cb' = 1.772*Cb'   CORRECT
   */
  __m256i cb2 = _mm256_add_epi16(cb, cb);
  __m256i pw_mf0228 = _mm256_set1_epi16(-F_0_228);
  __m256i b_tmp = _mm256_mulhi_epi16(cb2, pw_mf0228);
  __m256i pw_one = _mm256_set1_epi16(1);
  b_tmp = _mm256_add_epi16(b_tmp, pw_one);
  b_tmp = _mm256_srai_epi16(b_tmp, 1);
  b_tmp = _mm256_add_epi16(b_tmp, cb);
  __m256i b_diff = _mm256_add_epi16(b_tmp, cb);

  /* === Compute (R - Y) = Cr' * FIX(1.402) === */
  /* mulhi(2*Cr', FIX(0.402)) gives approximately 2*Cr' * 0.402 = 0.804*Cr'
   * After (x + 1) >> 1:  approximately 0.402*Cr'
   * Then + Cr' = 0.402*Cr' + Cr' = 1.402*Cr'   CORRECT
   */
  __m256i cr2 = _mm256_add_epi16(cr, cr);
  __m256i pw_f0402 = _mm256_set1_epi16(F_0_402);
  __m256i r_tmp = _mm256_mulhi_epi16(cr2, pw_f0402);
  r_tmp = _mm256_add_epi16(r_tmp, pw_one);
  r_tmp = _mm256_srai_epi16(r_tmp, 1);
  __m256i r_diff = _mm256_add_epi16(r_tmp, cr);

  /* === Compute (G - Y) = -0.34414 * Cb' - 0.71414 * Cr' === */
  /* Interleave Cb' and Cr' into word pairs for vpmaddwd:
   *   {Cb'[0], Cr'[0], Cb'[1], Cr'[1], ...}
   * Then madd with {-F_0_344, F_0_285, -F_0_344, F_0_285, ...}
   *   = -0.344*Cb'[i] + 0.286*Cr'[i]  (32-bit result)
   * Then >> SCALEBITS, pack to 16-bit, subtract Cr' to complete:
   *   -0.344*Cb' + 0.286*Cr' - Cr' = -0.344*Cb' - 0.714*Cr'   CORRECT
   */
  __m256i cb_cr_lo = _mm256_unpacklo_epi16(cb, cr);
  __m256i cb_cr_hi = _mm256_unpackhi_epi16(cb, cr);

  /* Pack constant: each dword has low word = -F_0_344, high word = F_0_285 */
  __m256i pw_mf0344_f0285 = _mm256_set1_epi32(
    (int)(((unsigned int)(unsigned short)F_0_285 << 16) |
          (unsigned short)(-F_0_344)));

  __m256i g_lo_32 = _mm256_madd_epi16(cb_cr_lo, pw_mf0344_f0285);
  __m256i g_hi_32 = _mm256_madd_epi16(cb_cr_hi, pw_mf0344_f0285);

  __m256i pd_onehalf = _mm256_set1_epi32((int)ONE_HALF);
  g_lo_32 = _mm256_add_epi32(g_lo_32, pd_onehalf);
  g_hi_32 = _mm256_add_epi32(g_hi_32, pd_onehalf);
  g_lo_32 = _mm256_srai_epi32(g_lo_32, SCALEBITS);
  g_hi_32 = _mm256_srai_epi32(g_hi_32, SCALEBITS);

  /* Pack 32-bit -> 16-bit with signed saturation.
   * _mm256_packs_epi32 operates within 128-bit lanes, producing:
   *   lane0: lo_lane0[0..3], hi_lane0[0..3]
   *   lane1: lo_lane1[4..7], hi_lane1[4..7]
   * This matches the lane-interleaved order from unpacklo/unpackhi.
   */
  __m256i g_diff = _mm256_packs_epi32(g_lo_32, g_hi_32);
  g_diff = _mm256_sub_epi16(g_diff, cr);

  /* === Duplicate each 16-word diff to 32 words === */
  /* Each color diff has 16 values (one per chroma sample).  For h2v1, each
   * chroma sample covers 2 adjacent Y pixels, so we duplicate each value.
   * Use _mm512_permutexvar_epi16 with index {0,0,1,1,2,2,...,15,15}.
   */
  static const short dup_idx[32] = {
     0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,
     8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15
  };
  __m512i vidx_dup = _mm512_loadu_si512((const __m512i *)dup_idx);

  __m512i r_sub_y = _mm512_permutexvar_epi16(vidx_dup,
                      _mm512_castsi256_si512(r_diff));
  __m512i g_sub_y = _mm512_permutexvar_epi16(vidx_dup,
                      _mm512_castsi256_si512(g_diff));
  __m512i b_sub_y = _mm512_permutexvar_epi16(vidx_dup,
                      _mm512_castsi256_si512(b_diff));

  /* === Load 32 Y bytes and add color differences === */
  __m256i y_raw = _mm256_loadu_si256((__m256i *)inptr0);
  __m512i y_16 = _mm512_cvtepu8_epi16(y_raw);

  __m512i r_16 = _mm512_add_epi16(y_16, r_sub_y);
  __m512i g_16 = _mm512_add_epi16(y_16, g_sub_y);
  __m512i b_16 = _mm512_add_epi16(y_16, b_sub_y);

  /* Clamp to [0, 255] */
  __m512i zero = _mm512_setzero_si512();
  __m512i maxval = _mm512_set1_epi16(255);

  r_16 = _mm512_max_epi16(r_16, zero);
  r_16 = _mm512_min_epi16(r_16, maxval);
  g_16 = _mm512_max_epi16(g_16, zero);
  g_16 = _mm512_min_epi16(g_16, maxval);
  b_16 = _mm512_max_epi16(b_16, zero);
  b_16 = _mm512_min_epi16(b_16, maxval);

  /* Narrow from 16-bit to 8-bit using vpmovwb (_mm512_cvtepi16_epi8).
   * Values are already in [0,255] so this is a simple truncation.
   */
  __m256i r_8 = _mm512_cvtepi16_epi8(r_16);
  __m256i g_8 = _mm512_cvtepi16_epi8(g_16);
  __m256i b_8 = _mm512_cvtepi16_epi8(b_16);

  /* === Interleave R, G, B (and possibly X/alpha) into output pixels === */
  /* Use _mm512_permutex2var_epi8 (AVX-512 VBMI) to perform a 128-byte
   * two-source byte permute.  This selects arbitrary bytes from two 64-byte
   * source registers using a 7-bit index per output byte (bit 6 selects
   * source a vs b, bits 5:0 select byte within that source).
   *
   * Source layout:
   *   src_a: bytes [0..31] = R[0..31], bytes [32..63] = G[0..31]
   *   src_b: bytes [0..31] = B[0..31], bytes [32..63] = X[0..31] (or unused)
   *
   * Index encoding:
   *   R[j] -> src_a, byte j        -> index = j
   *   G[j] -> src_a, byte 32+j     -> index = 32 + j
   *   B[j] -> src_b, byte j        -> index = 64 + j  (bit 6 set)
   *   X[j] -> src_b, byte 32+j     -> index = 96 + j  (bit 6 set)
   */

#if RGB_PIXELSIZE == 3

  /* 3 bytes per pixel: 32 pixels * 3 = 96 output bytes = 1.5 ZMM registers.
   *
   * For output byte position p:  pixel = p / 3, component = p % 3
   * Component at offset k within pixel maps to color channel:
   *   k == RGB_RED   -> R[pixel]  -> index = pixel
   *   k == RGB_GREEN -> G[pixel]  -> index = 32 + pixel
   *   k == RGB_BLUE  -> B[pixel]  -> index = 64 + pixel
   */
  {
    __m512i src_a = _mm512_inserti64x4(_mm512_castsi256_si512(r_8), g_8, 1);
    __m512i src_b = _mm512_castsi256_si512(b_8);

#define IDX3(p) \
    ((char)(((p) % 3 == RGB_RED)   ? ((p) / 3) : \
            ((p) % 3 == RGB_GREEN) ? ((p) / 3 + 32) : \
                                     ((p) / 3 + 64)))

    /* First 64 output bytes (pixels 0-20 complete + pixel 21 byte 0) */
    static const char ALIGN(64) idx0[64] = {
      IDX3(0),  IDX3(1),  IDX3(2),  IDX3(3),  IDX3(4),  IDX3(5),
      IDX3(6),  IDX3(7),  IDX3(8),  IDX3(9),  IDX3(10), IDX3(11),
      IDX3(12), IDX3(13), IDX3(14), IDX3(15), IDX3(16), IDX3(17),
      IDX3(18), IDX3(19), IDX3(20), IDX3(21), IDX3(22), IDX3(23),
      IDX3(24), IDX3(25), IDX3(26), IDX3(27), IDX3(28), IDX3(29),
      IDX3(30), IDX3(31), IDX3(32), IDX3(33), IDX3(34), IDX3(35),
      IDX3(36), IDX3(37), IDX3(38), IDX3(39), IDX3(40), IDX3(41),
      IDX3(42), IDX3(43), IDX3(44), IDX3(45), IDX3(46), IDX3(47),
      IDX3(48), IDX3(49), IDX3(50), IDX3(51), IDX3(52), IDX3(53),
      IDX3(54), IDX3(55), IDX3(56), IDX3(57), IDX3(58), IDX3(59),
      IDX3(60), IDX3(61), IDX3(62), IDX3(63)
    };
    /* Second 32 output bytes (pixels 21 cont'd through 31) */
    static const char ALIGN(64) idx1[64] = {
      IDX3(64), IDX3(65), IDX3(66), IDX3(67), IDX3(68), IDX3(69),
      IDX3(70), IDX3(71), IDX3(72), IDX3(73), IDX3(74), IDX3(75),
      IDX3(76), IDX3(77), IDX3(78), IDX3(79), IDX3(80), IDX3(81),
      IDX3(82), IDX3(83), IDX3(84), IDX3(85), IDX3(86), IDX3(87),
      IDX3(88), IDX3(89), IDX3(90), IDX3(91), IDX3(92), IDX3(93),
      IDX3(94), IDX3(95),
      /* Padding (not stored) */
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
#undef IDX3

    __m512i vidx0 = _mm512_load_si512((const __m512i *)idx0);
    __m512i vidx1 = _mm512_load_si512((const __m512i *)idx1);

    __m512i out0 = _mm512_permutex2var_epi8(src_a, vidx0, src_b);
    __m512i out1 = _mm512_permutex2var_epi8(src_a, vidx1, src_b);

    int out_bytes = num_cols * 3;

    if (out_bytes >= 64) {
      _mm512_storeu_si512((__m512i *)outptr, out0);
      int remaining = out_bytes - 64;
      if (remaining > 0) {
        __mmask64 mask = (remaining >= 64) ? (__mmask64)-1 :
                         ((__mmask64)1 << remaining) - 1;
        _mm512_mask_storeu_epi8(outptr + 64, mask, out1);
      }
    } else {
      __mmask64 mask = ((__mmask64)1 << out_bytes) - 1;
      _mm512_mask_storeu_epi8(outptr, mask, out0);
    }
  }

#else  /* RGB_PIXELSIZE == 4 */

  /* 4 bytes per pixel: 32 pixels * 4 = 128 output bytes = 2 ZMM registers.
   *
   * For output byte position p:  pixel = p / 4, component = p % 4
   * Component at offset k within pixel maps to color channel:
   *   k == RGB_RED   -> R[pixel]  -> index = pixel
   *   k == RGB_GREEN -> G[pixel]  -> index = 32 + pixel
   *   k == RGB_BLUE  -> B[pixel]  -> index = 64 + pixel
   *   k == RGB_ALPHA -> X[pixel]  -> index = 96 + pixel
   */
  {
    __m256i x_8;
#ifdef RGBX_FILLER_0XFF
    x_8 = _mm256_set1_epi8((char)0xFF);
#else
    x_8 = _mm256_setzero_si256();
#endif

    __m512i src_a = _mm512_inserti64x4(_mm512_castsi256_si512(r_8), g_8, 1);
    __m512i src_b = _mm512_inserti64x4(_mm512_castsi256_si512(b_8), x_8, 1);

    /* Determine the alpha offset.  For RGBX and BGRX, alpha is at offset 3.
     * For XRGB and XBGR, alpha is at offset 0.
     */
#ifdef RGB_ALPHA
#define ALPHA_OFFSET  RGB_ALPHA
#else
#define ALPHA_OFFSET  3
#endif

#define IDX4(p) \
    ((char)(((p) % 4 == RGB_RED)      ? ((p) / 4) : \
            ((p) % 4 == RGB_GREEN)    ? ((p) / 4 + 32) : \
            ((p) % 4 == RGB_BLUE)     ? ((p) / 4 + 64) : \
                                        ((p) / 4 + 96)))

    /* First 64 output bytes = pixels 0-15 */
    static const char ALIGN(64) idx0[64] = {
      IDX4(0),  IDX4(1),  IDX4(2),  IDX4(3),  IDX4(4),  IDX4(5),
      IDX4(6),  IDX4(7),  IDX4(8),  IDX4(9),  IDX4(10), IDX4(11),
      IDX4(12), IDX4(13), IDX4(14), IDX4(15), IDX4(16), IDX4(17),
      IDX4(18), IDX4(19), IDX4(20), IDX4(21), IDX4(22), IDX4(23),
      IDX4(24), IDX4(25), IDX4(26), IDX4(27), IDX4(28), IDX4(29),
      IDX4(30), IDX4(31), IDX4(32), IDX4(33), IDX4(34), IDX4(35),
      IDX4(36), IDX4(37), IDX4(38), IDX4(39), IDX4(40), IDX4(41),
      IDX4(42), IDX4(43), IDX4(44), IDX4(45), IDX4(46), IDX4(47),
      IDX4(48), IDX4(49), IDX4(50), IDX4(51), IDX4(52), IDX4(53),
      IDX4(54), IDX4(55), IDX4(56), IDX4(57), IDX4(58), IDX4(59),
      IDX4(60), IDX4(61), IDX4(62), IDX4(63)
    };
    /* Second 64 output bytes = pixels 16-31 */
    static const char ALIGN(64) idx1[64] = {
      IDX4(64),  IDX4(65),  IDX4(66),  IDX4(67),  IDX4(68),  IDX4(69),
      IDX4(70),  IDX4(71),  IDX4(72),  IDX4(73),  IDX4(74),  IDX4(75),
      IDX4(76),  IDX4(77),  IDX4(78),  IDX4(79),  IDX4(80),  IDX4(81),
      IDX4(82),  IDX4(83),  IDX4(84),  IDX4(85),  IDX4(86),  IDX4(87),
      IDX4(88),  IDX4(89),  IDX4(90),  IDX4(91),  IDX4(92),  IDX4(93),
      IDX4(94),  IDX4(95),  IDX4(96),  IDX4(97),  IDX4(98),  IDX4(99),
      IDX4(100), IDX4(101), IDX4(102), IDX4(103), IDX4(104), IDX4(105),
      IDX4(106), IDX4(107), IDX4(108), IDX4(109), IDX4(110), IDX4(111),
      IDX4(112), IDX4(113), IDX4(114), IDX4(115), IDX4(116), IDX4(117),
      IDX4(118), IDX4(119), IDX4(120), IDX4(121), IDX4(122), IDX4(123),
      IDX4(124), IDX4(125), IDX4(126), IDX4(127)
    };
#undef IDX4
#undef ALPHA_OFFSET

    __m512i vidx0 = _mm512_load_si512((const __m512i *)idx0);
    __m512i vidx1 = _mm512_load_si512((const __m512i *)idx1);

    __m512i out0 = _mm512_permutex2var_epi8(src_a, vidx0, src_b);
    __m512i out1 = _mm512_permutex2var_epi8(src_a, vidx1, src_b);

    int out_bytes = num_cols * 4;

    if (out_bytes >= 128) {
      _mm512_storeu_si512((__m512i *)outptr, out0);
      _mm512_storeu_si512((__m512i *)(outptr + 64), out1);
    } else if (out_bytes >= 64) {
      _mm512_storeu_si512((__m512i *)outptr, out0);
      int remaining = out_bytes - 64;
      if (remaining > 0) {
        __mmask64 mask = ((__mmask64)1 << remaining) - 1;
        _mm512_mask_storeu_epi8(outptr + 64, mask, out1);
      }
    } else {
      __mmask64 mask = ((__mmask64)1 << out_bytes) - 1;
      _mm512_mask_storeu_epi8(outptr, mask, out0);
    }
  }

#endif  /* RGB_PIXELSIZE */
}


/* Upsample and color convert for the case of 2:1 horizontal and 1:1 vertical.
 */

GLOBAL(void)
jsimd_h2v1_merged_upsample_avx512(JDIMENSION output_width,
                                   JSAMPIMAGE input_buf,
                                   JDIMENSION in_row_group_ctr,
                                   JSAMPARRAY output_buf)
{
  JSAMPROW outptr;
  JSAMPROW inptr0, inptr1, inptr2;

  inptr0 = input_buf[0][in_row_group_ctr];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr = output_buf[0];

  int cols_remaining = (int)output_width;

  /* Main loop: process 32 output pixels (16 chroma samples) per iteration */
  for (; cols_remaining >= 32; cols_remaining -= 32) {
    jsimd_h2v1_merged_upsample_avx512_row(inptr0, inptr1, inptr2,
                                           outptr, 32);
    inptr0 += 32;
    inptr1 += 16;
    inptr2 += 16;
    outptr += 32 * RGB_PIXELSIZE;
  }

  /* Handle remaining columns (< 32 output pixels).
   * Input buffers can be safely overread (padded by alloc_sarray).
   * Output is written with masked stores to avoid buffer overrun.
   */
  if (cols_remaining > 0) {
    jsimd_h2v1_merged_upsample_avx512_row(inptr0, inptr1, inptr2,
                                           outptr, cols_remaining);
  }
}


/* Upsample and color convert for the case of 2:1 horizontal and 2:1 vertical.
 *
 * Processes two Y rows that share the same Cb/Cr row.
 */

GLOBAL(void)
jsimd_h2v2_merged_upsample_avx512(JDIMENSION output_width,
                                   JSAMPIMAGE input_buf,
                                   JDIMENSION in_row_group_ctr,
                                   JSAMPARRAY output_buf)
{
  JSAMPROW outptr0, outptr1;
  JSAMPROW inptr0_0, inptr0_1, inptr1, inptr2;

  inptr0_0 = input_buf[0][in_row_group_ctr * 2];
  inptr0_1 = input_buf[0][in_row_group_ctr * 2 + 1];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr0 = output_buf[0];
  outptr1 = output_buf[1];

  int cols_remaining = (int)output_width;

  /* Main loop: process 32 output pixels (16 chroma samples) per iteration.
   * Both Y rows are processed with the same chroma data.
   */
  for (; cols_remaining >= 32; cols_remaining -= 32) {
    /* Row 0 */
    jsimd_h2v1_merged_upsample_avx512_row(inptr0_0, inptr1, inptr2,
                                           outptr0, 32);
    /* Row 1 (same Cb/Cr) */
    jsimd_h2v1_merged_upsample_avx512_row(inptr0_1, inptr1, inptr2,
                                           outptr1, 32);

    inptr0_0 += 32;
    inptr0_1 += 32;
    inptr1 += 16;
    inptr2 += 16;
    outptr0 += 32 * RGB_PIXELSIZE;
    outptr1 += 32 * RGB_PIXELSIZE;
  }

  /* Handle remaining columns */
  if (cols_remaining > 0) {
    jsimd_h2v1_merged_upsample_avx512_row(inptr0_0, inptr1, inptr2,
                                           outptr0, cols_remaining);
    jsimd_h2v1_merged_upsample_avx512_row(inptr0_1, inptr1, inptr2,
                                           outptr1, cols_remaining);
  }
}
