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
 *    R = Y + mulhi(2*Cr', F_0_402) + Cr'
 *    G = Y + (madd(Cb', -F_0_344) + madd(Cr', F_0_285)) >> 16 - Cr'
 *    B = Y + mulhi(2*Cb', -F_0_228) + 2*Cb'
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

  /* Duplication index: expand 16 chroma diffs to 32 words
   * {0,0,1,1,2,2,...,15,15}
   */
  static const short ALIGN(64) dup_idx[32] = {
     0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,
     8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15
  };

  /* Pixel interleave indices for _mm512_permutex2var_epi8.
   *
   * Source layout:
   *   src_a: bytes [0..31] = R[0..31], bytes [32..63] = G[0..31]
   *   src_b: bytes [0..31] = B[0..31], bytes [32..63] = X[0..31]
   *
   * Index encoding: bit 6 selects source (0=a, 1=b), bits 5:0 select byte.
   *   R[pixel] -> pixel        (src_a)
   *   G[pixel] -> 32 + pixel   (src_a)
   *   B[pixel] -> 64 + pixel   (src_b, bit6 set, offset=pixel)
   *   X[pixel] -> 96 + pixel   (src_b, bit6 set, offset=32+pixel)
   */
#if RGB_PIXELSIZE == 3
#define IDXBYTE(p) \
  ((char)(((p) % 3 == RGB_RED)   ? ((p) / 3) : \
          ((p) % 3 == RGB_GREEN) ? ((p) / 3 + 32) : \
                                   ((p) / 3 + 64)))
  /* 32 pixels * 3 bytes = 96 bytes = first ZMM (64) + second ZMM (32) */
  static const char ALIGN(64) perm_idx0[64] = {
    IDXBYTE(0),  IDXBYTE(1),  IDXBYTE(2),  IDXBYTE(3),
    IDXBYTE(4),  IDXBYTE(5),  IDXBYTE(6),  IDXBYTE(7),
    IDXBYTE(8),  IDXBYTE(9),  IDXBYTE(10), IDXBYTE(11),
    IDXBYTE(12), IDXBYTE(13), IDXBYTE(14), IDXBYTE(15),
    IDXBYTE(16), IDXBYTE(17), IDXBYTE(18), IDXBYTE(19),
    IDXBYTE(20), IDXBYTE(21), IDXBYTE(22), IDXBYTE(23),
    IDXBYTE(24), IDXBYTE(25), IDXBYTE(26), IDXBYTE(27),
    IDXBYTE(28), IDXBYTE(29), IDXBYTE(30), IDXBYTE(31),
    IDXBYTE(32), IDXBYTE(33), IDXBYTE(34), IDXBYTE(35),
    IDXBYTE(36), IDXBYTE(37), IDXBYTE(38), IDXBYTE(39),
    IDXBYTE(40), IDXBYTE(41), IDXBYTE(42), IDXBYTE(43),
    IDXBYTE(44), IDXBYTE(45), IDXBYTE(46), IDXBYTE(47),
    IDXBYTE(48), IDXBYTE(49), IDXBYTE(50), IDXBYTE(51),
    IDXBYTE(52), IDXBYTE(53), IDXBYTE(54), IDXBYTE(55),
    IDXBYTE(56), IDXBYTE(57), IDXBYTE(58), IDXBYTE(59),
    IDXBYTE(60), IDXBYTE(61), IDXBYTE(62), IDXBYTE(63)
  };
  static const char ALIGN(64) perm_idx1[64] = {
    IDXBYTE(64), IDXBYTE(65), IDXBYTE(66), IDXBYTE(67),
    IDXBYTE(68), IDXBYTE(69), IDXBYTE(70), IDXBYTE(71),
    IDXBYTE(72), IDXBYTE(73), IDXBYTE(74), IDXBYTE(75),
    IDXBYTE(76), IDXBYTE(77), IDXBYTE(78), IDXBYTE(79),
    IDXBYTE(80), IDXBYTE(81), IDXBYTE(82), IDXBYTE(83),
    IDXBYTE(84), IDXBYTE(85), IDXBYTE(86), IDXBYTE(87),
    IDXBYTE(88), IDXBYTE(89), IDXBYTE(90), IDXBYTE(91),
    IDXBYTE(92), IDXBYTE(93), IDXBYTE(94), IDXBYTE(95),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };
#undef IDXBYTE
#else  /* RGB_PIXELSIZE == 4 */
#define IDXBYTE(p) \
  ((char)(((p) % 4 == RGB_RED)   ? ((p) / 4) : \
          ((p) % 4 == RGB_GREEN) ? ((p) / 4 + 32) : \
          ((p) % 4 == RGB_BLUE)  ? ((p) / 4 + 64) : \
                                   ((p) / 4 + 96)))
  /* 32 pixels * 4 bytes = 128 bytes = 2 ZMM registers */
  static const char ALIGN(64) perm_idx0[64] = {
    IDXBYTE(0),  IDXBYTE(1),  IDXBYTE(2),  IDXBYTE(3),
    IDXBYTE(4),  IDXBYTE(5),  IDXBYTE(6),  IDXBYTE(7),
    IDXBYTE(8),  IDXBYTE(9),  IDXBYTE(10), IDXBYTE(11),
    IDXBYTE(12), IDXBYTE(13), IDXBYTE(14), IDXBYTE(15),
    IDXBYTE(16), IDXBYTE(17), IDXBYTE(18), IDXBYTE(19),
    IDXBYTE(20), IDXBYTE(21), IDXBYTE(22), IDXBYTE(23),
    IDXBYTE(24), IDXBYTE(25), IDXBYTE(26), IDXBYTE(27),
    IDXBYTE(28), IDXBYTE(29), IDXBYTE(30), IDXBYTE(31),
    IDXBYTE(32), IDXBYTE(33), IDXBYTE(34), IDXBYTE(35),
    IDXBYTE(36), IDXBYTE(37), IDXBYTE(38), IDXBYTE(39),
    IDXBYTE(40), IDXBYTE(41), IDXBYTE(42), IDXBYTE(43),
    IDXBYTE(44), IDXBYTE(45), IDXBYTE(46), IDXBYTE(47),
    IDXBYTE(48), IDXBYTE(49), IDXBYTE(50), IDXBYTE(51),
    IDXBYTE(52), IDXBYTE(53), IDXBYTE(54), IDXBYTE(55),
    IDXBYTE(56), IDXBYTE(57), IDXBYTE(58), IDXBYTE(59),
    IDXBYTE(60), IDXBYTE(61), IDXBYTE(62), IDXBYTE(63)
  };
  static const char ALIGN(64) perm_idx1[64] = {
    IDXBYTE(64),  IDXBYTE(65),  IDXBYTE(66),  IDXBYTE(67),
    IDXBYTE(68),  IDXBYTE(69),  IDXBYTE(70),  IDXBYTE(71),
    IDXBYTE(72),  IDXBYTE(73),  IDXBYTE(74),  IDXBYTE(75),
    IDXBYTE(76),  IDXBYTE(77),  IDXBYTE(78),  IDXBYTE(79),
    IDXBYTE(80),  IDXBYTE(81),  IDXBYTE(82),  IDXBYTE(83),
    IDXBYTE(84),  IDXBYTE(85),  IDXBYTE(86),  IDXBYTE(87),
    IDXBYTE(88),  IDXBYTE(89),  IDXBYTE(90),  IDXBYTE(91),
    IDXBYTE(92),  IDXBYTE(93),  IDXBYTE(94),  IDXBYTE(95),
    IDXBYTE(96),  IDXBYTE(97),  IDXBYTE(98),  IDXBYTE(99),
    IDXBYTE(100), IDXBYTE(101), IDXBYTE(102), IDXBYTE(103),
    IDXBYTE(104), IDXBYTE(105), IDXBYTE(106), IDXBYTE(107),
    IDXBYTE(108), IDXBYTE(109), IDXBYTE(110), IDXBYTE(111),
    IDXBYTE(112), IDXBYTE(113), IDXBYTE(114), IDXBYTE(115),
    IDXBYTE(116), IDXBYTE(117), IDXBYTE(118), IDXBYTE(119),
    IDXBYTE(120), IDXBYTE(121), IDXBYTE(122), IDXBYTE(123),
    IDXBYTE(124), IDXBYTE(125), IDXBYTE(126), IDXBYTE(127)
  };
#undef IDXBYTE
#endif  /* RGB_PIXELSIZE */

  __m512i vidx_dup = _mm512_load_si512((const __m512i *)dup_idx);
  __m512i vidx0 = _mm512_load_si512((const __m512i *)perm_idx0);
  __m512i vidx1 = _mm512_load_si512((const __m512i *)perm_idx1);

  int cols_remaining = (int)output_width;

  for (; cols_remaining > 0; cols_remaining -= 32) {
    int num_cols = (cols_remaining >= 32) ? 32 : cols_remaining;

    /* === Load 16 Cb/Cr bytes, zero-extend to 16-bit === */
    __m128i cb_raw = _mm_loadu_si128((__m128i *)inptr1);
    __m128i cr_raw = _mm_loadu_si128((__m128i *)inptr2);

    __m256i cb_16 = _mm256_cvtepu8_epi16(cb_raw);
    __m256i cr_16 = _mm256_cvtepu8_epi16(cr_raw);

    /* Subtract 128: Cb' = Cb - 128, Cr' = Cr - 128 */
    __m256i bias = _mm256_set1_epi16((short)0xFF80);
    __m256i cb = _mm256_add_epi16(cb_16, bias);
    __m256i cr = _mm256_add_epi16(cr_16, bias);

    /* === (B - Y) = Cb' * 1.772 === */
    __m256i cb2 = _mm256_add_epi16(cb, cb);
    __m256i pw_one = _mm256_set1_epi16(1);
    __m256i b_tmp = _mm256_mulhi_epi16(cb2, _mm256_set1_epi16(-F_0_228));
    b_tmp = _mm256_add_epi16(b_tmp, pw_one);
    b_tmp = _mm256_srai_epi16(b_tmp, 1);
    b_tmp = _mm256_add_epi16(b_tmp, cb);
    __m256i b_diff = _mm256_add_epi16(b_tmp, cb);

    /* === (R - Y) = Cr' * 1.402 === */
    __m256i cr2 = _mm256_add_epi16(cr, cr);
    __m256i r_tmp = _mm256_mulhi_epi16(cr2, _mm256_set1_epi16(F_0_402));
    r_tmp = _mm256_add_epi16(r_tmp, pw_one);
    r_tmp = _mm256_srai_epi16(r_tmp, 1);
    __m256i r_diff = _mm256_add_epi16(r_tmp, cr);

    /* === (G - Y) = -0.34414 * Cb' - 0.71414 * Cr' === */
    __m256i cb_cr_lo = _mm256_unpacklo_epi16(cb, cr);
    __m256i cb_cr_hi = _mm256_unpackhi_epi16(cb, cr);
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
    __m256i g_diff = _mm256_packs_epi32(g_lo_32, g_hi_32);
    g_diff = _mm256_sub_epi16(g_diff, cr);

    /* === Duplicate 16 diffs -> 32 words === */
    __m512i r_sub_y = _mm512_permutexvar_epi16(vidx_dup,
                        _mm512_castsi256_si512(r_diff));
    __m512i g_sub_y = _mm512_permutexvar_epi16(vidx_dup,
                        _mm512_castsi256_si512(g_diff));
    __m512i b_sub_y = _mm512_permutexvar_epi16(vidx_dup,
                        _mm512_castsi256_si512(b_diff));

    /* === Load 32 Y, add diffs, clamp [0,255] === */
    __m256i y_raw = _mm256_loadu_si256((__m256i *)inptr0);
    __m512i y_16 = _mm512_cvtepu8_epi16(y_raw);

    __m512i zero = _mm512_setzero_si512();
    __m512i maxval = _mm512_set1_epi16(255);

    __m512i r_16 = _mm512_max_epi16(_mm512_min_epi16(
                     _mm512_add_epi16(y_16, r_sub_y), maxval), zero);
    __m512i g_16 = _mm512_max_epi16(_mm512_min_epi16(
                     _mm512_add_epi16(y_16, g_sub_y), maxval), zero);
    __m512i b_16 = _mm512_max_epi16(_mm512_min_epi16(
                     _mm512_add_epi16(y_16, b_sub_y), maxval), zero);

    /* Narrow 16-bit -> 8-bit (vpmovwb) */
    __m256i r_8 = _mm512_cvtepi16_epi8(r_16);
    __m256i g_8 = _mm512_cvtepi16_epi8(g_16);
    __m256i b_8 = _mm512_cvtepi16_epi8(b_16);

    /* === Interleave and store === */
    __m512i src_a = _mm512_inserti64x4(_mm512_castsi256_si512(r_8), g_8, 1);
#if RGB_PIXELSIZE == 3
    __m512i src_b = _mm512_castsi256_si512(b_8);
#else
    __m256i x_8;
#ifdef RGBX_FILLER_0XFF
    x_8 = _mm256_set1_epi8((char)0xFF);
#else
    x_8 = _mm256_setzero_si256();
#endif
    __m512i src_b = _mm512_inserti64x4(_mm512_castsi256_si512(b_8), x_8, 1);
#endif

    __m512i out0 = _mm512_permutex2var_epi8(src_a, vidx0, src_b);
    __m512i out1 = _mm512_permutex2var_epi8(src_a, vidx1, src_b);

    int out_bytes = num_cols * RGB_PIXELSIZE;

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

    inptr0 += 32;
    inptr1 += 16;
    inptr2 += 16;
    outptr += out_bytes;
  }
}


/* Upsample and color convert for the case of 2:1 horizontal and 2:1 vertical.
 *
 * Processes two Y rows that share the same Cb/Cr row.  The chroma color
 * differences are computed once and applied to both Y rows.
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

  static const short ALIGN(64) dup_idx[32] = {
     0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,
     8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15
  };

#if RGB_PIXELSIZE == 3
#define IDXBYTE(p) \
  ((char)(((p) % 3 == RGB_RED)   ? ((p) / 3) : \
          ((p) % 3 == RGB_GREEN) ? ((p) / 3 + 32) : \
                                   ((p) / 3 + 64)))
  static const char ALIGN(64) perm_idx0[64] = {
    IDXBYTE(0),  IDXBYTE(1),  IDXBYTE(2),  IDXBYTE(3),
    IDXBYTE(4),  IDXBYTE(5),  IDXBYTE(6),  IDXBYTE(7),
    IDXBYTE(8),  IDXBYTE(9),  IDXBYTE(10), IDXBYTE(11),
    IDXBYTE(12), IDXBYTE(13), IDXBYTE(14), IDXBYTE(15),
    IDXBYTE(16), IDXBYTE(17), IDXBYTE(18), IDXBYTE(19),
    IDXBYTE(20), IDXBYTE(21), IDXBYTE(22), IDXBYTE(23),
    IDXBYTE(24), IDXBYTE(25), IDXBYTE(26), IDXBYTE(27),
    IDXBYTE(28), IDXBYTE(29), IDXBYTE(30), IDXBYTE(31),
    IDXBYTE(32), IDXBYTE(33), IDXBYTE(34), IDXBYTE(35),
    IDXBYTE(36), IDXBYTE(37), IDXBYTE(38), IDXBYTE(39),
    IDXBYTE(40), IDXBYTE(41), IDXBYTE(42), IDXBYTE(43),
    IDXBYTE(44), IDXBYTE(45), IDXBYTE(46), IDXBYTE(47),
    IDXBYTE(48), IDXBYTE(49), IDXBYTE(50), IDXBYTE(51),
    IDXBYTE(52), IDXBYTE(53), IDXBYTE(54), IDXBYTE(55),
    IDXBYTE(56), IDXBYTE(57), IDXBYTE(58), IDXBYTE(59),
    IDXBYTE(60), IDXBYTE(61), IDXBYTE(62), IDXBYTE(63)
  };
  static const char ALIGN(64) perm_idx1[64] = {
    IDXBYTE(64), IDXBYTE(65), IDXBYTE(66), IDXBYTE(67),
    IDXBYTE(68), IDXBYTE(69), IDXBYTE(70), IDXBYTE(71),
    IDXBYTE(72), IDXBYTE(73), IDXBYTE(74), IDXBYTE(75),
    IDXBYTE(76), IDXBYTE(77), IDXBYTE(78), IDXBYTE(79),
    IDXBYTE(80), IDXBYTE(81), IDXBYTE(82), IDXBYTE(83),
    IDXBYTE(84), IDXBYTE(85), IDXBYTE(86), IDXBYTE(87),
    IDXBYTE(88), IDXBYTE(89), IDXBYTE(90), IDXBYTE(91),
    IDXBYTE(92), IDXBYTE(93), IDXBYTE(94), IDXBYTE(95),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };
#undef IDXBYTE
#else  /* RGB_PIXELSIZE == 4 */
#define IDXBYTE(p) \
  ((char)(((p) % 4 == RGB_RED)   ? ((p) / 4) : \
          ((p) % 4 == RGB_GREEN) ? ((p) / 4 + 32) : \
          ((p) % 4 == RGB_BLUE)  ? ((p) / 4 + 64) : \
                                   ((p) / 4 + 96)))
  static const char ALIGN(64) perm_idx0[64] = {
    IDXBYTE(0),  IDXBYTE(1),  IDXBYTE(2),  IDXBYTE(3),
    IDXBYTE(4),  IDXBYTE(5),  IDXBYTE(6),  IDXBYTE(7),
    IDXBYTE(8),  IDXBYTE(9),  IDXBYTE(10), IDXBYTE(11),
    IDXBYTE(12), IDXBYTE(13), IDXBYTE(14), IDXBYTE(15),
    IDXBYTE(16), IDXBYTE(17), IDXBYTE(18), IDXBYTE(19),
    IDXBYTE(20), IDXBYTE(21), IDXBYTE(22), IDXBYTE(23),
    IDXBYTE(24), IDXBYTE(25), IDXBYTE(26), IDXBYTE(27),
    IDXBYTE(28), IDXBYTE(29), IDXBYTE(30), IDXBYTE(31),
    IDXBYTE(32), IDXBYTE(33), IDXBYTE(34), IDXBYTE(35),
    IDXBYTE(36), IDXBYTE(37), IDXBYTE(38), IDXBYTE(39),
    IDXBYTE(40), IDXBYTE(41), IDXBYTE(42), IDXBYTE(43),
    IDXBYTE(44), IDXBYTE(45), IDXBYTE(46), IDXBYTE(47),
    IDXBYTE(48), IDXBYTE(49), IDXBYTE(50), IDXBYTE(51),
    IDXBYTE(52), IDXBYTE(53), IDXBYTE(54), IDXBYTE(55),
    IDXBYTE(56), IDXBYTE(57), IDXBYTE(58), IDXBYTE(59),
    IDXBYTE(60), IDXBYTE(61), IDXBYTE(62), IDXBYTE(63)
  };
  static const char ALIGN(64) perm_idx1[64] = {
    IDXBYTE(64),  IDXBYTE(65),  IDXBYTE(66),  IDXBYTE(67),
    IDXBYTE(68),  IDXBYTE(69),  IDXBYTE(70),  IDXBYTE(71),
    IDXBYTE(72),  IDXBYTE(73),  IDXBYTE(74),  IDXBYTE(75),
    IDXBYTE(76),  IDXBYTE(77),  IDXBYTE(78),  IDXBYTE(79),
    IDXBYTE(80),  IDXBYTE(81),  IDXBYTE(82),  IDXBYTE(83),
    IDXBYTE(84),  IDXBYTE(85),  IDXBYTE(86),  IDXBYTE(87),
    IDXBYTE(88),  IDXBYTE(89),  IDXBYTE(90),  IDXBYTE(91),
    IDXBYTE(92),  IDXBYTE(93),  IDXBYTE(94),  IDXBYTE(95),
    IDXBYTE(96),  IDXBYTE(97),  IDXBYTE(98),  IDXBYTE(99),
    IDXBYTE(100), IDXBYTE(101), IDXBYTE(102), IDXBYTE(103),
    IDXBYTE(104), IDXBYTE(105), IDXBYTE(106), IDXBYTE(107),
    IDXBYTE(108), IDXBYTE(109), IDXBYTE(110), IDXBYTE(111),
    IDXBYTE(112), IDXBYTE(113), IDXBYTE(114), IDXBYTE(115),
    IDXBYTE(116), IDXBYTE(117), IDXBYTE(118), IDXBYTE(119),
    IDXBYTE(120), IDXBYTE(121), IDXBYTE(122), IDXBYTE(123),
    IDXBYTE(124), IDXBYTE(125), IDXBYTE(126), IDXBYTE(127)
  };
#undef IDXBYTE
#endif

  __m512i vidx_dup = _mm512_load_si512((const __m512i *)dup_idx);
  __m512i vidx0 = _mm512_load_si512((const __m512i *)perm_idx0);
  __m512i vidx1 = _mm512_load_si512((const __m512i *)perm_idx1);

  int cols_remaining = (int)output_width;

  for (; cols_remaining > 0; cols_remaining -= 32) {
    int num_cols = (cols_remaining >= 32) ? 32 : cols_remaining;

    /* === Chroma computation (shared by both rows) === */
    __m128i cb_raw = _mm_loadu_si128((__m128i *)inptr1);
    __m128i cr_raw = _mm_loadu_si128((__m128i *)inptr2);

    __m256i cb_16 = _mm256_cvtepu8_epi16(cb_raw);
    __m256i cr_16 = _mm256_cvtepu8_epi16(cr_raw);

    __m256i bias = _mm256_set1_epi16((short)0xFF80);
    __m256i cb = _mm256_add_epi16(cb_16, bias);
    __m256i cr = _mm256_add_epi16(cr_16, bias);

    __m256i cb2 = _mm256_add_epi16(cb, cb);
    __m256i pw_one = _mm256_set1_epi16(1);
    __m256i b_tmp = _mm256_mulhi_epi16(cb2, _mm256_set1_epi16(-F_0_228));
    b_tmp = _mm256_add_epi16(b_tmp, pw_one);
    b_tmp = _mm256_srai_epi16(b_tmp, 1);
    b_tmp = _mm256_add_epi16(b_tmp, cb);
    __m256i b_diff = _mm256_add_epi16(b_tmp, cb);

    __m256i cr2 = _mm256_add_epi16(cr, cr);
    __m256i r_tmp = _mm256_mulhi_epi16(cr2, _mm256_set1_epi16(F_0_402));
    r_tmp = _mm256_add_epi16(r_tmp, pw_one);
    r_tmp = _mm256_srai_epi16(r_tmp, 1);
    __m256i r_diff = _mm256_add_epi16(r_tmp, cr);

    __m256i cb_cr_lo = _mm256_unpacklo_epi16(cb, cr);
    __m256i cb_cr_hi = _mm256_unpackhi_epi16(cb, cr);
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
    __m256i g_diff = _mm256_packs_epi32(g_lo_32, g_hi_32);
    g_diff = _mm256_sub_epi16(g_diff, cr);

    __m512i r_sub_y = _mm512_permutexvar_epi16(vidx_dup,
                        _mm512_castsi256_si512(r_diff));
    __m512i g_sub_y = _mm512_permutexvar_epi16(vidx_dup,
                        _mm512_castsi256_si512(g_diff));
    __m512i b_sub_y = _mm512_permutexvar_epi16(vidx_dup,
                        _mm512_castsi256_si512(b_diff));

    __m512i zero = _mm512_setzero_si512();
    __m512i maxval = _mm512_set1_epi16(255);
    int out_bytes = num_cols * RGB_PIXELSIZE;

    /* === Row 0 === */
    {
      __m256i y_raw = _mm256_loadu_si256((__m256i *)inptr0_0);
      __m512i y_16 = _mm512_cvtepu8_epi16(y_raw);

      __m512i r_16 = _mm512_max_epi16(_mm512_min_epi16(
                       _mm512_add_epi16(y_16, r_sub_y), maxval), zero);
      __m512i g_16 = _mm512_max_epi16(_mm512_min_epi16(
                       _mm512_add_epi16(y_16, g_sub_y), maxval), zero);
      __m512i b_16 = _mm512_max_epi16(_mm512_min_epi16(
                       _mm512_add_epi16(y_16, b_sub_y), maxval), zero);

      __m256i r_8 = _mm512_cvtepi16_epi8(r_16);
      __m256i g_8 = _mm512_cvtepi16_epi8(g_16);
      __m256i b_8 = _mm512_cvtepi16_epi8(b_16);

      __m512i src_a = _mm512_inserti64x4(_mm512_castsi256_si512(r_8), g_8, 1);
#if RGB_PIXELSIZE == 3
      __m512i src_b = _mm512_castsi256_si512(b_8);
#else
      __m256i x_8;
#ifdef RGBX_FILLER_0XFF
      x_8 = _mm256_set1_epi8((char)0xFF);
#else
      x_8 = _mm256_setzero_si256();
#endif
      __m512i src_b = _mm512_inserti64x4(_mm512_castsi256_si512(b_8), x_8, 1);
#endif

      __m512i out0 = _mm512_permutex2var_epi8(src_a, vidx0, src_b);
      __m512i out1 = _mm512_permutex2var_epi8(src_a, vidx1, src_b);

      if (out_bytes >= 64) {
        _mm512_storeu_si512((__m512i *)outptr0, out0);
        int remaining = out_bytes - 64;
        if (remaining > 0) {
          __mmask64 mask = (remaining >= 64) ? (__mmask64)-1 :
                           ((__mmask64)1 << remaining) - 1;
          _mm512_mask_storeu_epi8(outptr0 + 64, mask, out1);
        }
      } else {
        __mmask64 mask = ((__mmask64)1 << out_bytes) - 1;
        _mm512_mask_storeu_epi8(outptr0, mask, out0);
      }
    }

    /* === Row 1 (same color diffs) === */
    {
      __m256i y_raw = _mm256_loadu_si256((__m256i *)inptr0_1);
      __m512i y_16 = _mm512_cvtepu8_epi16(y_raw);

      __m512i r_16 = _mm512_max_epi16(_mm512_min_epi16(
                       _mm512_add_epi16(y_16, r_sub_y), maxval), zero);
      __m512i g_16 = _mm512_max_epi16(_mm512_min_epi16(
                       _mm512_add_epi16(y_16, g_sub_y), maxval), zero);
      __m512i b_16 = _mm512_max_epi16(_mm512_min_epi16(
                       _mm512_add_epi16(y_16, b_sub_y), maxval), zero);

      __m256i r_8 = _mm512_cvtepi16_epi8(r_16);
      __m256i g_8 = _mm512_cvtepi16_epi8(g_16);
      __m256i b_8 = _mm512_cvtepi16_epi8(b_16);

      __m512i src_a = _mm512_inserti64x4(_mm512_castsi256_si512(r_8), g_8, 1);
#if RGB_PIXELSIZE == 3
      __m512i src_b = _mm512_castsi256_si512(b_8);
#else
      __m256i x_8;
#ifdef RGBX_FILLER_0XFF
      x_8 = _mm256_set1_epi8((char)0xFF);
#else
      x_8 = _mm256_setzero_si256();
#endif
      __m512i src_b = _mm512_inserti64x4(_mm512_castsi256_si512(b_8), x_8, 1);
#endif

      __m512i out0 = _mm512_permutex2var_epi8(src_a, vidx0, src_b);
      __m512i out1 = _mm512_permutex2var_epi8(src_a, vidx1, src_b);

      if (out_bytes >= 64) {
        _mm512_storeu_si512((__m512i *)outptr1, out0);
        int remaining = out_bytes - 64;
        if (remaining > 0) {
          __mmask64 mask = (remaining >= 64) ? (__mmask64)-1 :
                           ((__mmask64)1 << remaining) - 1;
          _mm512_mask_storeu_epi8(outptr1 + 64, mask, out1);
        }
      } else {
        __mmask64 mask = ((__mmask64)1 << out_bytes) - 1;
        _mm512_mask_storeu_epi8(outptr1, mask, out0);
      }
    }

    inptr0_0 += 32;
    inptr0_1 += 32;
    inptr1 += 16;
    inptr2 += 16;
    outptr0 += out_bytes;
    outptr1 += out_bytes;
  }
}
