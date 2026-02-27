/*
 * jdcolor-avx512.c - YCbCr -> RGB color conversion (AVX-512BW)
 *
 * Copyright (C) 2025, Lakeworks.
 *
 * AVX-512BW implementation of YCbCr -> RGB color conversion for
 * decompression.  Processes 32 pixels per iteration using 512-bit registers.
 *
 * Based on the AVX2 NASM implementation (jdcolext-avx2.asm):
 * Copyright (C) 2009, 2012, 2016, 2024-2025, D. R. Commander.
 * Copyright (C) 2015, Intel Corporation.
 * Copyright (C) 2018, Matthias Raencker.
 * Copyright (C) 2023, Aliaksiej Kandracienka.
 *
 * Algorithm:
 *   R = Y                + 1.40200 * Cr
 *   G = Y - 0.34414 * Cb - 0.71414 * Cr
 *   B = Y + 1.77200 * Cb
 *
 * Rewritten for integer arithmetic (same decomposition as AVX2 NASM):
 *   R = Y + 0.40200 * Cr + Cr              = Y + Cr + round(Cr * 26345 >> 16)
 *   G = Y + madd(Cb,Cr, {-22554, 18734}) - Cr  (madd at SCALEBITS=16)
 *   B = Y - 0.22800 * Cb + Cb + Cb         = Y + 2*Cb + round(Cb * -14942 >> 16)
 */

#include "../../src/jinclude.h"
#include "../../src/jpeglib.h"
#include "../../src/jdct.h"
#include "../jsimd.h"

#include <immintrin.h>

/* Portable alignment macro (matches simd/arm/align.h) */
#ifndef ALIGN
#if defined(_MSC_VER)
#define ALIGN(n)  __declspec(align(n))
#elif defined(__clang__) || defined(__GNUC__)
#define ALIGN(n)  __attribute__((aligned(n)))
#else
#error "Unknown compiler"
#endif
#endif

/*
 * Fixed-point constants (SCALEBITS = 16):
 *
 * FIX(1.40200) = 91881
 * FIX(0.34414) = 22554
 * FIX(0.71414) = 46802
 * FIX(1.77200) = 116130
 *
 * Derived (to keep multipliers within signed 16-bit range):
 * FIX(0.40200) = FIX(1.40200) - 65536 = 26345
 * FIX(0.28586) = 65536 - FIX(0.71414) = 18734
 * FIX(0.22800) = 2*65536 - FIX(1.77200) = 14942
 */
#define SCALEBITS       16
#define ONE_HALF        ((int)1 << (SCALEBITS - 1))

#define PW_F0402        26345   /* FIX(0.40200) */
#define PW_MF0228       (-14942) /* -FIX(0.22800) */
#define PW_MF0344       (-22554) /* -FIX(0.34414) */
#define PW_F0285        18734   /* FIX(0.28586) */
#define PW_ONE          1
#define PD_ONEHALF      ONE_HALF


/*
 * Compute the color-difference values R-Y, G-Y, B-Y for 32 pixels.
 *
 * Inputs:  cb, cr  -- 32 x int16 (zero-extended from uint8, then -128)
 * Outputs: r_y, g_y, b_y -- 32 x int16 color differences
 */
static INLINE void
ycc_rgb_convert_avx512_compute(const __m512i cb, const __m512i cr,
                               __m512i *r_y, __m512i *g_y, __m512i *b_y)
{
  const __m512i pw_f0402 = _mm512_set1_epi16((short)PW_F0402);
  const __m512i pw_mf0228 = _mm512_set1_epi16((short)PW_MF0228);
  const __m512i pw_one = _mm512_set1_epi16((short)PW_ONE);
  const __m512i pd_onehalf = _mm512_set1_epi32(PD_ONEHALF);
  /* Interleaved {-FIX(0.34414), FIX(0.28586)} for vpmaddwd */
  const __m512i pw_mf0344_f0285 = _mm512_set1_epi32(
    (unsigned short)((short)PW_MF0344) |
    ((unsigned int)(unsigned short)((short)PW_F0285) << 16));

  __m512i cb2, cr2;
  __m512i r_y_val, b_y_val;
  __m512i g_lo, g_hi;
  __m512i cb_lo, cr_lo, cb_hi, cr_hi, cbcr_lo, cbcr_hi;

  /* ---- R-Y = round(Cr * FIX(0.40200)) + Cr ---- */
  cr2 = _mm512_add_epi16(cr, cr);                     /* 2 * Cr */
  r_y_val = _mm512_mulhi_epi16(cr2, pw_f0402);        /* (2*Cr * 26345) >> 16 */
  r_y_val = _mm512_add_epi16(r_y_val, pw_one);        /* + rounding */
  r_y_val = _mm512_srai_epi16(r_y_val, 1);            /* >> 1  =>  round(Cr * 26345 >> 16) */
  r_y_val = _mm512_add_epi16(r_y_val, cr);            /* + Cr  =>  Cr * FIX(1.40200) */
  *r_y = r_y_val;

  /* ---- B-Y = round(Cb * -FIX(0.22800)) + Cb + Cb ---- */
  cb2 = _mm512_add_epi16(cb, cb);                     /* 2 * Cb */
  b_y_val = _mm512_mulhi_epi16(cb2, pw_mf0228);       /* (2*Cb * -14942) >> 16 */
  b_y_val = _mm512_add_epi16(b_y_val, pw_one);        /* + rounding */
  b_y_val = _mm512_srai_epi16(b_y_val, 1);            /* >> 1  =>  round(Cb * -14942 >> 16) */
  b_y_val = _mm512_add_epi16(b_y_val, cb);            /* + Cb */
  b_y_val = _mm512_add_epi16(b_y_val, cb);            /* + Cb  =>  Cb * FIX(1.77200) */
  *b_y = b_y_val;

  /* ---- G-Y = madd(Cb, Cr, {-FIX(0.34414), FIX(0.28586)}) - Cr ----
   *
   * vpmaddwd takes pairs of 16-bit values {a0,a1} and computes a0*b0 + a1*b1
   * as a 32-bit result.  We interleave Cb and Cr so each pair is {Cb_i, Cr_i},
   * multiply by {-22554, 18734}, giving:
   *   Cb_i * (-22554) + Cr_i * 18734  (32-bit, at SCALEBITS=16)
   * Then add ONE_HALF, >> SCALEBITS, pack to 16-bit, subtract Cr:
   *   G-Y = (Cb * -0.34414 + Cr * 0.28586) - Cr
   *       = Cb * -0.34414 + Cr * (0.28586 - 1.0)
   *       = Cb * -0.34414 + Cr * -0.71414
   *
   * We must process 32 pixels in two halves of 16, since madd halves the
   * element count (32 x int16 -> 16 x int32).
   */

  /* Low 16 pixels (lanes 0..15 of the ZMM) */
  cb_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(cb));
  cr_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(cr));
  /* Interleave: {Cb0, Cr0, Cb1, Cr1, ...} packed as 16-bit in a ZMM */
  cbcr_lo = _mm512_mask_blend_epi16(0xAAAAAAAA,
              _mm512_castsi256_si512(_mm512_castsi512_si256(cb)),
              _mm512_castsi256_si512(_mm512_castsi512_si256(cr)));
  /* Actually, let's do it properly with unpacklo/unpackhi at 16-bit level */

  /* Re-approach: split into 256-bit halves, interleave with unpack, then madd */
  {
    __m256i cb_lo256 = _mm512_castsi512_si256(cb);
    __m256i cr_lo256 = _mm512_castsi512_si256(cr);
    __m256i cb_hi256 = _mm512_extracti64x4_epi64(cb, 1);
    __m256i cr_hi256 = _mm512_extracti64x4_epi64(cr, 1);

    /* For the low 16 pixels: interleave Cb and Cr words */
    __m256i cbcr_even_lo = _mm256_unpacklo_epi16(cb_lo256, cr_lo256);
    /* = {Cb0,Cr0, Cb1,Cr1, Cb2,Cr2, Cb3,Cr3 | Cb8,Cr8, Cb9,Cr9, ...} */
    __m256i cbcr_odd_lo = _mm256_unpackhi_epi16(cb_lo256, cr_lo256);
    /* = {Cb4,Cr4, Cb5,Cr5, Cb6,Cr6, Cb7,Cr7 | Cb12,Cr12, Cb13,Cr13, ...} */

    __m256i pw_mf0344_f0285_256 = _mm256_set1_epi32(
      (unsigned short)((short)PW_MF0344) |
      ((unsigned int)(unsigned short)((short)PW_F0285) << 16));

    __m256i g_even_lo = _mm256_madd_epi16(cbcr_even_lo, pw_mf0344_f0285_256);
    __m256i g_odd_lo = _mm256_madd_epi16(cbcr_odd_lo, pw_mf0344_f0285_256);

    __m256i pd_onehalf_256 = _mm256_set1_epi32(PD_ONEHALF);
    g_even_lo = _mm256_add_epi32(g_even_lo, pd_onehalf_256);
    g_odd_lo = _mm256_add_epi32(g_odd_lo, pd_onehalf_256);
    g_even_lo = _mm256_srai_epi32(g_even_lo, SCALEBITS);
    g_odd_lo = _mm256_srai_epi32(g_odd_lo, SCALEBITS);

    __m256i g_lo256 = _mm256_packs_epi32(g_even_lo, g_odd_lo);
    /* packs interleaves: result has even-indexed pixels in low 64-bit of each
     * 128-bit lane and odd-indexed pixels in high 64-bit.  This matches the
     * original ordering since unpacklo produced {0,1,2,3|8,9,10,11} and
     * unpackhi produced {4,5,6,7|12,13,14,15}.  After packs_epi32 we get
     * {0,1,2,3,4,5,6,7 | 8,9,10,11,12,13,14,15} which is correct. */

    /* Same for high 16 pixels */
    __m256i cbcr_even_hi = _mm256_unpacklo_epi16(cb_hi256, cr_hi256);
    __m256i cbcr_odd_hi = _mm256_unpackhi_epi16(cb_hi256, cr_hi256);

    __m256i g_even_hi = _mm256_madd_epi16(cbcr_even_hi, pw_mf0344_f0285_256);
    __m256i g_odd_hi = _mm256_madd_epi16(cbcr_odd_hi, pw_mf0344_f0285_256);

    g_even_hi = _mm256_add_epi32(g_even_hi, pd_onehalf_256);
    g_odd_hi = _mm256_add_epi32(g_odd_hi, pd_onehalf_256);
    g_even_hi = _mm256_srai_epi32(g_even_hi, SCALEBITS);
    g_odd_hi = _mm256_srai_epi32(g_odd_hi, SCALEBITS);

    __m256i g_hi256 = _mm256_packs_epi32(g_even_hi, g_odd_hi);

    /* Combine the two 256-bit halves into one 512-bit result */
    __m512i g_val = _mm512_inserti64x4(_mm512_castsi256_si512(g_lo256),
                                       g_hi256, 1);

    /* Subtract Cr to get final G-Y */
    g_val = _mm512_sub_epi16(g_val, cr);
    *g_y = g_val;
  }
}


/*
 * Macro to generate a YCbCr->RGB conversion function for a specific pixel
 * format.  The macro parameters control channel ordering and pixel size.
 *
 * FUNC_NAME    - output function name
 * R_IDX        - byte offset of R within the output pixel (0..3)
 * G_IDX        - byte offset of G within the output pixel (0..3)
 * B_IDX        - byte offset of B within the output pixel (0..3)
 * X_IDX        - byte offset of filler/alpha within the output pixel (-1 for
 *                3-byte formats, 0..3 for 4-byte formats)
 * PIXEL_SIZE   - bytes per output pixel (3 or 4)
 */
#define GENERATE_YCC_RGB_CONVERT_AVX512(FUNC_NAME, R_IDX, G_IDX, B_IDX, \
                                        X_IDX, PIXEL_SIZE)              \
                                                                        \
GLOBAL(void)                                                            \
FUNC_NAME(JDIMENSION out_width, JSAMPIMAGE input_buf,                   \
          JDIMENSION input_row, JSAMPARRAY output_buf, int num_rows)    \
{                                                                       \
  JSAMPROW inptr0, inptr1, inptr2, outptr;                              \
  JDIMENSION col;                                                       \
  int num_cols = (int)out_width;                                        \
                                                                        \
  while (--num_rows >= 0) {                                             \
    inptr0 = input_buf[0][input_row];                                   \
    inptr1 = input_buf[1][input_row];                                   \
    inptr2 = input_buf[2][input_row];                                   \
    input_row++;                                                        \
    outptr = *output_buf++;                                             \
                                                                        \
    for (col = 0; col < (JDIMENSION)num_cols; col += 32) {             \
      JDIMENSION remaining = (JDIMENSION)num_cols - col;                \
      JDIMENSION count = (remaining >= 32) ? 32 : remaining;           \
      __mmask32 load_mask = (count >= 32) ? 0xFFFFFFFF :               \
                            ((__mmask32)1 << count) - 1;                \
                                                                        \
      /* Load 32 (or fewer) bytes of Y, Cb, Cr */                      \
      __m256i y8  = _mm256_maskz_loadu_epi8(load_mask,                  \
                      (__m256i *)(inptr0 + col));                       \
      __m256i cb8 = _mm256_maskz_loadu_epi8(load_mask,                  \
                      (__m256i *)(inptr1 + col));                       \
      __m256i cr8 = _mm256_maskz_loadu_epi8(load_mask,                  \
                      (__m256i *)(inptr2 + col));                       \
                                                                        \
      /* Zero-extend bytes to 16-bit words (32 pixels -> 512 bits) */   \
      __m512i y16  = _mm512_cvtepu8_epi16(y8);                         \
      __m512i cb16 = _mm512_cvtepu8_epi16(cb8);                        \
      __m512i cr16 = _mm512_cvtepu8_epi16(cr8);                        \
                                                                        \
      /* Subtract CENTERJSAMPLE (128) from Cb and Cr */                 \
      __m512i bias = _mm512_set1_epi16(CENTERJSAMPLE);                  \
      cb16 = _mm512_sub_epi16(cb16, bias);                              \
      cr16 = _mm512_sub_epi16(cr16, bias);                              \
                                                                        \
      /* Compute color differences */                                   \
      __m512i r_y, g_y, b_y;                                           \
      ycc_rgb_convert_avx512_compute(cb16, cr16, &r_y, &g_y, &b_y);   \
                                                                        \
      /* Add Y and clamp to [0, 255] */                                 \
      __m512i r16 = _mm512_add_epi16(y16, r_y);                        \
      __m512i g16 = _mm512_add_epi16(y16, g_y);                        \
      __m512i b16 = _mm512_add_epi16(y16, b_y);                        \
                                                                        \
      /* Pack 16-bit to 8-bit with unsigned saturation.                 \
       * _mm512_packus_epi16 operates on 128-bit lanes, so the result  \
       * has an interleaved lane layout:                                \
       *   Input  r16 = {r0..r7 | r8..r15 | r16..r23 | r24..r31}      \
       *   packus(r16, r16) = {r0..r7,r0..r7 | r8..r15,r8..r15 |      \
       *                       r16..r23,r16..r23 | r24..r31,r24..r31}  \
       *                                                                \
       * We pack R with G and B with X (or B with itself for 3-byte).  \
       * Then use permutexvar to produce the final pixel layout.        \
       */                                                               \
      /* Pack R and G together: low 8 from R, high 8 from G per lane */ \
      __m512i rg8 = _mm512_packus_epi16(r16, g16);                     \
      /* rg8 per 128-bit lane: {R0..R7, G0..G7} */                     \
                                                                        \
      if (PIXEL_SIZE == 3) {                                            \
        __m512i bx8 = _mm512_packus_epi16(b16, b16);                   \
        /* bx8 per 128-bit lane: {B0..B7, B0..B7} */                   \
                                                                        \
        /* We now have 32 R, 32 G, 32 B values packed in two ZMMs with \
         * the lane-interleaved layout from packus.  Use permutexvar    \
         * to gather them into the correct RGB triplet order.           \
         *                                                              \
         * rg8 layout (bytes, lane-relative indices):                   \
         *   lane 0 (bytes 0-15):  R0  R1  R2  R3  R4  R5  R6  R7     \
         *                         G0  G1  G2  G3  G4  G5  G6  G7     \
         *   lane 1 (bytes 16-31): R8  R9  R10 R11 R12 R13 R14 R15    \
         *                         G8  G9  G10 G11 G12 G13 G14 G15    \
         *   lane 2 (bytes 32-47): R16 R17 R18 R19 R20 R21 R22 R23    \
         *                         G16 G17 G18 G19 G20 G21 G22 G23    \
         *   lane 3 (bytes 48-63): R24 R25 R26 R27 R28 R29 R30 R31    \
         *                         G24 G25 G26 G27 G28 G29 G30 G31    \
         *                                                              \
         * bx8 layout:                                                  \
         *   lane 0 (bytes 0-15):  B0  B1  B2  B3  B4  B5  B6  B7     \
         *                         B0  B1  B2  B3  B4  B5  B6  B7     \
         *   (similarly for lanes 1-3)                                  \
         *                                                              \
         * Build 3 output ZMMs of 32 bytes each (= 96 bytes total for  \
         * 32 3-byte pixels).  We use permutex2var_epi8 to select      \
         * bytes from rg8 (source 1) and bx8 (source 2, indices 64+).  \
         */                                                             \
        /* Map absolute byte positions in rg8 and bx8:                  \
         * rg8[0..7]  = R0..R7    rg8[8..15]  = G0..G7                \
         * rg8[16..23] = R8..R15  rg8[24..31] = G8..G15               \
         * rg8[32..39] = R16..R23 rg8[40..47] = G16..G23              \
         * rg8[48..55] = R24..R31 rg8[56..63] = G24..G31              \
         *                                                              \
         * bx8[0..7]  = B0..B7   (bx8[8..15]  = duplicate)            \
         * bx8[16..23] = B8..B15 (bx8[24..31] = duplicate)            \
         * bx8[32..39] = B16..B23(bx8[40..47] = duplicate)            \
         * bx8[48..55] = B24..B31(bx8[56..63] = duplicate)            \
         *                                                              \
         * For permutex2var_epi8, index bit 6 selects source:           \
         *   0..63 = from rg8,  64..127 = from bx8                     \
         *                                                              \
         * Output byte[i] for pixel p at position 3*p+channel:         \
         *   R at position R_IDX, G at G_IDX, B at B_IDX               \
         */                                                             \
                                                                        \
        /* Build permutation indices for 3-byte output.                 \
         * Pixel p (0..31):                                             \
         *   R byte in rg8 at: (p/8)*16 + (p%8) = lane_base + p%8     \
         *   G byte in rg8 at: (p/8)*16 + 8 + (p%8)                   \
         *   B byte in bx8 at: (p/8)*16 + (p%8) -> +64 for src2       \
         */                                                             \
        unsigned char perm0[64], perm1[64], perm2[64];                  \
        int p, byte_idx = 0;                                            \
        /* First output ZMM: pixels 0..20 (63 bytes needed, but we     \
         * fill bytes 0..63; pixel 21 starts at byte 63, partial)      \
         * Actually: 64 / 3 = 21.33, so 21 full pixels + 1 byte.      \
         *                                                              \
         * We output all 96 bytes across two stores:                    \
         *   Store 1: bytes 0..63 (pixels 0..20 + 1 byte of pixel 21)  \
         *   Store 2: bytes 64..95 (remaining, with masked store)       \
         *                                                              \
         * Or simpler: build 2 ZMMs, store first fully, second with    \
         * mask.  But building permutation tables at runtime is costly. \
         *                                                              \
         * Better approach: precompute static permutation tables.       \
         */                                                             \
        (void)perm0; (void)perm1; (void)perm2;                         \
        (void)p; (void)byte_idx;                                        \
                                                                        \
        /* Use pre-built static permutation indices */                  \
        ALIGN(64) static const char                                      \
          rgb3_perm0_data[64] = { 0 },                                  \
          rgb3_perm1_data[64] = { 0 };                                  \
        /* ^^ placeholder: actual tables built below */                 \
        (void)rgb3_perm0_data; (void)rgb3_perm1_data;                   \
                                                                        \
        /* Instead of permutex2var which requires complex index tables, \
         * use a simpler scatter approach that's still efficient:        \
         * Extract R, G, B as separate 32-byte vectors and interleave. \
         *                                                              \
         * First, undo the packus lane interleaving to get contiguous   \
         * R[0..31], G[0..31], B[0..31] bytes.                         \
         */                                                             \
        /* Alternative cleaner approach: don't use packus at all.       \
         * Instead, use _mm512_cvtusepi16_epi8 (vpmovuswb) which       \
         * saturates 16-bit to unsigned 8-bit and packs contiguously    \
         * into a 256-bit result with NO lane interleaving.             \
         */                                                             \
        __m256i r8 = _mm512_cvtusepi16_epi8(                            \
                       _mm512_max_epi16(r16, _mm512_setzero_si512()));  \
        __m256i g8 = _mm512_cvtusepi16_epi8(                            \
                       _mm512_max_epi16(g16, _mm512_setzero_si512()));  \
        __m256i b8 = _mm512_cvtusepi16_epi8(                            \
                       _mm512_max_epi16(b16, _mm512_setzero_si512()));  \
                                                                        \
        /* Now r8, g8, b8 are __m256i with R[0..31], G[0..31],         \
         * B[0..31] as contiguous bytes.  Promote to __m512i to use    \
         * permutex2var.                                                \
         */                                                             \
        __m512i rr = _mm512_castsi256_si512(r8);                        \
        __m512i gg = _mm512_castsi256_si512(g8);                        \
        __m512i bb = _mm512_castsi256_si512(b8);                        \
                                                                        \
        /* Build a single combined vector {R0..R31, G0..G31} for the   \
         * first source of permutex2var, and {B0..B31, 0xFF..} for     \
         * the second source.  The permutation index selects bytes     \
         * from either source (bit 6 = source select).                  \
         */                                                             \
        __m512i rg = _mm512_inserti64x4(rr, g8, 1);                    \
        /* rg: bytes 0..31 = R[0..31], bytes 32..63 = G[0..31] */      \
                                                                        \
        /* For 3-byte RGB: output is 96 bytes for 32 pixels.           \
         * We generate two 64-byte ZMMs using permutex2var_epi8:       \
         *   out0 = pixels 0..20 + 1 byte of pixel 21 (bytes 0..63)    \
         *   out1 = remaining pixels 21..31 (bytes 0..31, store 32)    \
         *                                                              \
         * Source a = rg = {R0..R31, G0..G31}                           \
         * Source b = bb = {B0..B31, ...don't care...}                  \
         *                                                              \
         * For pixel p, output bytes at offset 3*p + R_IDX/G_IDX/B_IDX:\
         *   R[p] is at rg byte p       -> index p                     \
         *   G[p] is at rg byte 32+p    -> index 32+p                  \
         *   B[p] is at bb byte p       -> index 64+p (bit6=1)         \
         */                                                             \
        {                                                               \
          ALIGN(64) static const char                                    \
            tbl0[64] = {                                                \
              JDCOL_3BYTE_PERM0(R_IDX, G_IDX, B_IDX)                   \
            },                                                          \
            tbl1[64] = {                                                \
              JDCOL_3BYTE_PERM1(R_IDX, G_IDX, B_IDX)                   \
            };                                                          \
                                                                        \
          __m512i idx0 = _mm512_load_si512((const __m512i *)tbl0);      \
          __m512i idx1 = _mm512_load_si512((const __m512i *)tbl1);      \
                                                                        \
          __m512i out0 = _mm512_permutex2var_epi8(rg, idx0, bb);        \
          __m512i out1 = _mm512_permutex2var_epi8(rg, idx1, bb);        \
                                                                        \
          if (count >= 32) {                                            \
            _mm512_storeu_si512((__m512i *)outptr, out0);               \
            _mm256_storeu_si256((__m256i *)(outptr + 64),               \
                                _mm512_castsi512_si256(out1));          \
          } else {                                                      \
            /* Partial store: count pixels = count*3 bytes */           \
            int total_bytes = (int)count * 3;                           \
            if (total_bytes >= 64) {                                    \
              _mm512_storeu_si512((__m512i *)outptr, out0);             \
              int rem = total_bytes - 64;                               \
              __mmask32 smask = (rem >= 32) ? 0xFFFFFFFF :              \
                                ((__mmask32)1 << rem) - 1;              \
              _mm256_mask_storeu_epi8(outptr + 64, smask,               \
                                     _mm512_castsi512_si256(out1));     \
            } else {                                                    \
              __mmask64 smask64 = (((__mmask64)1 << total_bytes) - 1);  \
              _mm512_mask_storeu_epi8(outptr, smask64, out0);           \
            }                                                           \
          }                                                             \
          outptr += count * 3;                                          \
        }                                                               \
      } else {                                                          \
        /* PIXEL_SIZE == 4 */                                           \
        __m256i r8 = _mm512_cvtusepi16_epi8(                            \
                       _mm512_max_epi16(r16, _mm512_setzero_si512()));  \
        __m256i g8 = _mm512_cvtusepi16_epi8(                            \
                       _mm512_max_epi16(g16, _mm512_setzero_si512()));  \
        __m256i b8 = _mm512_cvtusepi16_epi8(                            \
                       _mm512_max_epi16(b16, _mm512_setzero_si512()));  \
        __m256i x8 = _mm256_set1_epi8((char)0xFF);                     \
                                                                        \
        /* Build two source vectors for permutex2var:                   \
         *   src_rg = {R0..R31, G0..G31}                                \
         *   src_bx = {B0..B31, X0..X31}                                \
         */                                                             \
        __m512i src_rg = _mm512_inserti64x4(                            \
                           _mm512_castsi256_si512(r8), g8, 1);          \
        __m512i src_bx = _mm512_inserti64x4(                            \
                           _mm512_castsi256_si512(b8), x8, 1);          \
                                                                        \
        /* For pixel p (0..31), output 4 bytes at offset 4*p:           \
         *   byte[4*p + R_IDX] = R[p]  -> src_rg byte p    = index p   \
         *   byte[4*p + G_IDX] = G[p]  -> src_rg byte 32+p = index 32+p\
         *   byte[4*p + B_IDX] = B[p]  -> src_bx byte p    = index 64+p\
         *   byte[4*p + X_IDX] = 0xFF  -> src_bx byte 32+p = idx 96+p  \
         *                                                              \
         * 32 pixels * 4 bytes = 128 bytes = 2 ZMMs.                   \
         */                                                             \
        {                                                               \
          ALIGN(64) static const char                                    \
            tbl0[64] = {                                                \
              JDCOL_4BYTE_PERM0(R_IDX, G_IDX, B_IDX, X_IDX)            \
            },                                                          \
            tbl1[64] = {                                                \
              JDCOL_4BYTE_PERM1(R_IDX, G_IDX, B_IDX, X_IDX)            \
            };                                                          \
                                                                        \
          __m512i idx0 = _mm512_load_si512((const __m512i *)tbl0);      \
          __m512i idx1 = _mm512_load_si512((const __m512i *)tbl1);      \
                                                                        \
          __m512i out0 = _mm512_permutex2var_epi8(src_rg, idx0, src_bx);\
          __m512i out1 = _mm512_permutex2var_epi8(src_rg, idx1, src_bx);\
                                                                        \
          if (count >= 32) {                                            \
            _mm512_storeu_si512((__m512i *)outptr, out0);               \
            _mm512_storeu_si512((__m512i *)(outptr + 64), out1);        \
          } else {                                                      \
            /* Partial store: count * 4 bytes total */                  \
            int total_bytes = (int)count * 4;                           \
            if (total_bytes > 64) {                                     \
              _mm512_storeu_si512((__m512i *)outptr, out0);             \
              int rem = total_bytes - 64;                               \
              __mmask64 smask64 = ((__mmask64)1 << rem) - 1;            \
              _mm512_mask_storeu_epi8(outptr + 64, smask64, out1);      \
            } else {                                                    \
              __mmask64 smask64 = total_bytes >= 64 ? ~((__mmask64)0) : \
                                 ((__mmask64)1 << total_bytes) - 1;     \
              _mm512_mask_storeu_epi8(outptr, smask64, out0);           \
            }                                                           \
          }                                                             \
          outptr += count * 4;                                          \
        }                                                               \
      }                                                                 \
    }                                                                   \
  }                                                                     \
}


/*
 * Permutation table macros for 3-byte pixel formats.
 *
 * For 32 pixels at 3 bytes each = 96 bytes, split across:
 *   tbl0: output bytes 0..63  (pixels 0..20, plus byte 0 of pixel 21)
 *   tbl1: output bytes 64..95 (pixels 21..31, pad rest with 0)
 *
 * Source layout for permutex2var_epi8:
 *   indices 0..31  -> rg bytes 0..31  = R[0..31]
 *   indices 32..63 -> rg bytes 32..63 = G[0..31]
 *   indices 64..95 -> bb bytes 0..31  = B[0..31]  (bit 6 set = second source)
 *   indices 96..127 -> bb bytes 32..63 = don't care
 *
 * For pixel p, the three output bytes at positions 3*p+R, 3*p+G, 3*p+B are:
 *   channel at offset R_IDX -> R[p] -> index p
 *   channel at offset G_IDX -> G[p] -> index 32+p
 *   channel at offset B_IDX -> B[p] -> index 64+p
 *
 * The tables encode this mapping byte-by-byte.
 */

/* Helper: for output byte position 'pos' (0..95), which pixel and channel?
 * pixel = pos / 3, channel_offset = pos % 3
 * Then look up the index based on which channel is at that offset.
 */
#define JDCOL_3BYTE_IDX(R, G, B, pos)  \
  ((pos) >= 96 ? 0 :                   \
   ((pos) % 3 == (R)) ? ((pos) / 3) :  \
   ((pos) % 3 == (G)) ? (32 + (pos) / 3) : \
   (64 + (pos) / 3))

#define JDCOL_3BYTE_PERM0(R, G, B)                                    \
  JDCOL_3BYTE_IDX(R,G,B, 0), JDCOL_3BYTE_IDX(R,G,B, 1),             \
  JDCOL_3BYTE_IDX(R,G,B, 2), JDCOL_3BYTE_IDX(R,G,B, 3),             \
  JDCOL_3BYTE_IDX(R,G,B, 4), JDCOL_3BYTE_IDX(R,G,B, 5),             \
  JDCOL_3BYTE_IDX(R,G,B, 6), JDCOL_3BYTE_IDX(R,G,B, 7),             \
  JDCOL_3BYTE_IDX(R,G,B, 8), JDCOL_3BYTE_IDX(R,G,B, 9),             \
  JDCOL_3BYTE_IDX(R,G,B,10), JDCOL_3BYTE_IDX(R,G,B,11),             \
  JDCOL_3BYTE_IDX(R,G,B,12), JDCOL_3BYTE_IDX(R,G,B,13),             \
  JDCOL_3BYTE_IDX(R,G,B,14), JDCOL_3BYTE_IDX(R,G,B,15),             \
  JDCOL_3BYTE_IDX(R,G,B,16), JDCOL_3BYTE_IDX(R,G,B,17),             \
  JDCOL_3BYTE_IDX(R,G,B,18), JDCOL_3BYTE_IDX(R,G,B,19),             \
  JDCOL_3BYTE_IDX(R,G,B,20), JDCOL_3BYTE_IDX(R,G,B,21),             \
  JDCOL_3BYTE_IDX(R,G,B,22), JDCOL_3BYTE_IDX(R,G,B,23),             \
  JDCOL_3BYTE_IDX(R,G,B,24), JDCOL_3BYTE_IDX(R,G,B,25),             \
  JDCOL_3BYTE_IDX(R,G,B,26), JDCOL_3BYTE_IDX(R,G,B,27),             \
  JDCOL_3BYTE_IDX(R,G,B,28), JDCOL_3BYTE_IDX(R,G,B,29),             \
  JDCOL_3BYTE_IDX(R,G,B,30), JDCOL_3BYTE_IDX(R,G,B,31),             \
  JDCOL_3BYTE_IDX(R,G,B,32), JDCOL_3BYTE_IDX(R,G,B,33),             \
  JDCOL_3BYTE_IDX(R,G,B,34), JDCOL_3BYTE_IDX(R,G,B,35),             \
  JDCOL_3BYTE_IDX(R,G,B,36), JDCOL_3BYTE_IDX(R,G,B,37),             \
  JDCOL_3BYTE_IDX(R,G,B,38), JDCOL_3BYTE_IDX(R,G,B,39),             \
  JDCOL_3BYTE_IDX(R,G,B,40), JDCOL_3BYTE_IDX(R,G,B,41),             \
  JDCOL_3BYTE_IDX(R,G,B,42), JDCOL_3BYTE_IDX(R,G,B,43),             \
  JDCOL_3BYTE_IDX(R,G,B,44), JDCOL_3BYTE_IDX(R,G,B,45),             \
  JDCOL_3BYTE_IDX(R,G,B,46), JDCOL_3BYTE_IDX(R,G,B,47),             \
  JDCOL_3BYTE_IDX(R,G,B,48), JDCOL_3BYTE_IDX(R,G,B,49),             \
  JDCOL_3BYTE_IDX(R,G,B,50), JDCOL_3BYTE_IDX(R,G,B,51),             \
  JDCOL_3BYTE_IDX(R,G,B,52), JDCOL_3BYTE_IDX(R,G,B,53),             \
  JDCOL_3BYTE_IDX(R,G,B,54), JDCOL_3BYTE_IDX(R,G,B,55),             \
  JDCOL_3BYTE_IDX(R,G,B,56), JDCOL_3BYTE_IDX(R,G,B,57),             \
  JDCOL_3BYTE_IDX(R,G,B,58), JDCOL_3BYTE_IDX(R,G,B,59),             \
  JDCOL_3BYTE_IDX(R,G,B,60), JDCOL_3BYTE_IDX(R,G,B,61),             \
  JDCOL_3BYTE_IDX(R,G,B,62), JDCOL_3BYTE_IDX(R,G,B,63)

#define JDCOL_3BYTE_PERM1(R, G, B)                                    \
  JDCOL_3BYTE_IDX(R,G,B,64), JDCOL_3BYTE_IDX(R,G,B,65),             \
  JDCOL_3BYTE_IDX(R,G,B,66), JDCOL_3BYTE_IDX(R,G,B,67),             \
  JDCOL_3BYTE_IDX(R,G,B,68), JDCOL_3BYTE_IDX(R,G,B,69),             \
  JDCOL_3BYTE_IDX(R,G,B,70), JDCOL_3BYTE_IDX(R,G,B,71),             \
  JDCOL_3BYTE_IDX(R,G,B,72), JDCOL_3BYTE_IDX(R,G,B,73),             \
  JDCOL_3BYTE_IDX(R,G,B,74), JDCOL_3BYTE_IDX(R,G,B,75),             \
  JDCOL_3BYTE_IDX(R,G,B,76), JDCOL_3BYTE_IDX(R,G,B,77),             \
  JDCOL_3BYTE_IDX(R,G,B,78), JDCOL_3BYTE_IDX(R,G,B,79),             \
  JDCOL_3BYTE_IDX(R,G,B,80), JDCOL_3BYTE_IDX(R,G,B,81),             \
  JDCOL_3BYTE_IDX(R,G,B,82), JDCOL_3BYTE_IDX(R,G,B,83),             \
  JDCOL_3BYTE_IDX(R,G,B,84), JDCOL_3BYTE_IDX(R,G,B,85),             \
  JDCOL_3BYTE_IDX(R,G,B,86), JDCOL_3BYTE_IDX(R,G,B,87),             \
  JDCOL_3BYTE_IDX(R,G,B,88), JDCOL_3BYTE_IDX(R,G,B,89),             \
  JDCOL_3BYTE_IDX(R,G,B,90), JDCOL_3BYTE_IDX(R,G,B,91),             \
  JDCOL_3BYTE_IDX(R,G,B,92), JDCOL_3BYTE_IDX(R,G,B,93),             \
  JDCOL_3BYTE_IDX(R,G,B,94), JDCOL_3BYTE_IDX(R,G,B,95),             \
  0, 0, 0, 0, 0, 0, 0, 0,                                             \
  0, 0, 0, 0, 0, 0, 0, 0,                                             \
  0, 0, 0, 0, 0, 0, 0, 0,                                             \
  0, 0, 0, 0, 0, 0, 0, 0


/*
 * Permutation table macros for 4-byte pixel formats.
 *
 * 32 pixels * 4 bytes = 128 bytes = 2 ZMMs.
 *   tbl0: output bytes 0..63  (pixels 0..15)
 *   tbl1: output bytes 64..127 -> stored as bytes 0..63 (pixels 16..31)
 *
 * Source layout for permutex2var_epi8 (both ZMMs use same sources):
 *   indices 0..31  -> src_rg bytes 0..31  = R[0..31]
 *   indices 32..63 -> src_rg bytes 32..63 = G[0..31]
 *   indices 64..95 -> src_bx bytes 0..31  = B[0..31]
 *   indices 96..127 -> src_bx bytes 32..63 = X[0..31] (0xFF filler)
 */
#define JDCOL_4BYTE_IDX(R, G, B, X, pos)  \
  (((pos) % 4 == (R)) ? ((pos) / 4) :     \
   ((pos) % 4 == (G)) ? (32 + (pos) / 4) : \
   ((pos) % 4 == (B)) ? (64 + (pos) / 4) : \
   (96 + (pos) / 4))

#define JDCOL_4BYTE_PERM0(R, G, B, X)                                 \
  JDCOL_4BYTE_IDX(R,G,B,X, 0), JDCOL_4BYTE_IDX(R,G,B,X, 1),         \
  JDCOL_4BYTE_IDX(R,G,B,X, 2), JDCOL_4BYTE_IDX(R,G,B,X, 3),         \
  JDCOL_4BYTE_IDX(R,G,B,X, 4), JDCOL_4BYTE_IDX(R,G,B,X, 5),         \
  JDCOL_4BYTE_IDX(R,G,B,X, 6), JDCOL_4BYTE_IDX(R,G,B,X, 7),         \
  JDCOL_4BYTE_IDX(R,G,B,X, 8), JDCOL_4BYTE_IDX(R,G,B,X, 9),         \
  JDCOL_4BYTE_IDX(R,G,B,X,10), JDCOL_4BYTE_IDX(R,G,B,X,11),         \
  JDCOL_4BYTE_IDX(R,G,B,X,12), JDCOL_4BYTE_IDX(R,G,B,X,13),         \
  JDCOL_4BYTE_IDX(R,G,B,X,14), JDCOL_4BYTE_IDX(R,G,B,X,15),         \
  JDCOL_4BYTE_IDX(R,G,B,X,16), JDCOL_4BYTE_IDX(R,G,B,X,17),         \
  JDCOL_4BYTE_IDX(R,G,B,X,18), JDCOL_4BYTE_IDX(R,G,B,X,19),         \
  JDCOL_4BYTE_IDX(R,G,B,X,20), JDCOL_4BYTE_IDX(R,G,B,X,21),         \
  JDCOL_4BYTE_IDX(R,G,B,X,22), JDCOL_4BYTE_IDX(R,G,B,X,23),         \
  JDCOL_4BYTE_IDX(R,G,B,X,24), JDCOL_4BYTE_IDX(R,G,B,X,25),         \
  JDCOL_4BYTE_IDX(R,G,B,X,26), JDCOL_4BYTE_IDX(R,G,B,X,27),         \
  JDCOL_4BYTE_IDX(R,G,B,X,28), JDCOL_4BYTE_IDX(R,G,B,X,29),         \
  JDCOL_4BYTE_IDX(R,G,B,X,30), JDCOL_4BYTE_IDX(R,G,B,X,31),         \
  JDCOL_4BYTE_IDX(R,G,B,X,32), JDCOL_4BYTE_IDX(R,G,B,X,33),         \
  JDCOL_4BYTE_IDX(R,G,B,X,34), JDCOL_4BYTE_IDX(R,G,B,X,35),         \
  JDCOL_4BYTE_IDX(R,G,B,X,36), JDCOL_4BYTE_IDX(R,G,B,X,37),         \
  JDCOL_4BYTE_IDX(R,G,B,X,38), JDCOL_4BYTE_IDX(R,G,B,X,39),         \
  JDCOL_4BYTE_IDX(R,G,B,X,40), JDCOL_4BYTE_IDX(R,G,B,X,41),         \
  JDCOL_4BYTE_IDX(R,G,B,X,42), JDCOL_4BYTE_IDX(R,G,B,X,43),         \
  JDCOL_4BYTE_IDX(R,G,B,X,44), JDCOL_4BYTE_IDX(R,G,B,X,45),         \
  JDCOL_4BYTE_IDX(R,G,B,X,46), JDCOL_4BYTE_IDX(R,G,B,X,47),         \
  JDCOL_4BYTE_IDX(R,G,B,X,48), JDCOL_4BYTE_IDX(R,G,B,X,49),         \
  JDCOL_4BYTE_IDX(R,G,B,X,50), JDCOL_4BYTE_IDX(R,G,B,X,51),         \
  JDCOL_4BYTE_IDX(R,G,B,X,52), JDCOL_4BYTE_IDX(R,G,B,X,53),         \
  JDCOL_4BYTE_IDX(R,G,B,X,54), JDCOL_4BYTE_IDX(R,G,B,X,55),         \
  JDCOL_4BYTE_IDX(R,G,B,X,56), JDCOL_4BYTE_IDX(R,G,B,X,57),         \
  JDCOL_4BYTE_IDX(R,G,B,X,58), JDCOL_4BYTE_IDX(R,G,B,X,59),         \
  JDCOL_4BYTE_IDX(R,G,B,X,60), JDCOL_4BYTE_IDX(R,G,B,X,61),         \
  JDCOL_4BYTE_IDX(R,G,B,X,62), JDCOL_4BYTE_IDX(R,G,B,X,63)

#define JDCOL_4BYTE_PERM1(R, G, B, X)                                 \
  JDCOL_4BYTE_IDX(R,G,B,X, 64), JDCOL_4BYTE_IDX(R,G,B,X, 65),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 66), JDCOL_4BYTE_IDX(R,G,B,X, 67),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 68), JDCOL_4BYTE_IDX(R,G,B,X, 69),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 70), JDCOL_4BYTE_IDX(R,G,B,X, 71),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 72), JDCOL_4BYTE_IDX(R,G,B,X, 73),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 74), JDCOL_4BYTE_IDX(R,G,B,X, 75),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 76), JDCOL_4BYTE_IDX(R,G,B,X, 77),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 78), JDCOL_4BYTE_IDX(R,G,B,X, 79),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 80), JDCOL_4BYTE_IDX(R,G,B,X, 81),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 82), JDCOL_4BYTE_IDX(R,G,B,X, 83),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 84), JDCOL_4BYTE_IDX(R,G,B,X, 85),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 86), JDCOL_4BYTE_IDX(R,G,B,X, 87),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 88), JDCOL_4BYTE_IDX(R,G,B,X, 89),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 90), JDCOL_4BYTE_IDX(R,G,B,X, 91),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 92), JDCOL_4BYTE_IDX(R,G,B,X, 93),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 94), JDCOL_4BYTE_IDX(R,G,B,X, 95),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 96), JDCOL_4BYTE_IDX(R,G,B,X, 97),       \
  JDCOL_4BYTE_IDX(R,G,B,X, 98), JDCOL_4BYTE_IDX(R,G,B,X, 99),       \
  JDCOL_4BYTE_IDX(R,G,B,X,100), JDCOL_4BYTE_IDX(R,G,B,X,101),       \
  JDCOL_4BYTE_IDX(R,G,B,X,102), JDCOL_4BYTE_IDX(R,G,B,X,103),       \
  JDCOL_4BYTE_IDX(R,G,B,X,104), JDCOL_4BYTE_IDX(R,G,B,X,105),       \
  JDCOL_4BYTE_IDX(R,G,B,X,106), JDCOL_4BYTE_IDX(R,G,B,X,107),       \
  JDCOL_4BYTE_IDX(R,G,B,X,108), JDCOL_4BYTE_IDX(R,G,B,X,109),       \
  JDCOL_4BYTE_IDX(R,G,B,X,110), JDCOL_4BYTE_IDX(R,G,B,X,111),       \
  JDCOL_4BYTE_IDX(R,G,B,X,112), JDCOL_4BYTE_IDX(R,G,B,X,113),       \
  JDCOL_4BYTE_IDX(R,G,B,X,114), JDCOL_4BYTE_IDX(R,G,B,X,115),       \
  JDCOL_4BYTE_IDX(R,G,B,X,116), JDCOL_4BYTE_IDX(R,G,B,X,117),       \
  JDCOL_4BYTE_IDX(R,G,B,X,118), JDCOL_4BYTE_IDX(R,G,B,X,119),       \
  JDCOL_4BYTE_IDX(R,G,B,X,120), JDCOL_4BYTE_IDX(R,G,B,X,121),       \
  JDCOL_4BYTE_IDX(R,G,B,X,122), JDCOL_4BYTE_IDX(R,G,B,X,123),       \
  JDCOL_4BYTE_IDX(R,G,B,X,124), JDCOL_4BYTE_IDX(R,G,B,X,125),       \
  JDCOL_4BYTE_IDX(R,G,B,X,126), JDCOL_4BYTE_IDX(R,G,B,X,127)


/* ---- Generate all 7 YCbCr->RGB conversion functions ---- */

/* 1. Default RGB (R=0, G=1, B=2, 3-byte) */
GENERATE_YCC_RGB_CONVERT_AVX512(jsimd_ycc_rgb_convert_avx512,
                                RGB_RED, RGB_GREEN, RGB_BLUE, -1,
                                RGB_PIXELSIZE)

/* 2. EXT_RGB (R=0, G=1, B=2, 3-byte) */
GENERATE_YCC_RGB_CONVERT_AVX512(jsimd_ycc_extrgb_convert_avx512,
                                EXT_RGB_RED, EXT_RGB_GREEN, EXT_RGB_BLUE, -1,
                                EXT_RGB_PIXELSIZE)

/* 3. EXT_RGBX (R=0, G=1, B=2, X=3, 4-byte) */
GENERATE_YCC_RGB_CONVERT_AVX512(jsimd_ycc_extrgbx_convert_avx512,
                                EXT_RGBX_RED, EXT_RGBX_GREEN, EXT_RGBX_BLUE,
                                3, EXT_RGBX_PIXELSIZE)

/* 4. EXT_BGR (B=0, G=1, R=2, 3-byte) */
GENERATE_YCC_RGB_CONVERT_AVX512(jsimd_ycc_extbgr_convert_avx512,
                                EXT_BGR_RED, EXT_BGR_GREEN, EXT_BGR_BLUE, -1,
                                EXT_BGR_PIXELSIZE)

/* 5. EXT_BGRX (B=0, G=1, R=2, X=3, 4-byte) */
GENERATE_YCC_RGB_CONVERT_AVX512(jsimd_ycc_extbgrx_convert_avx512,
                                EXT_BGRX_RED, EXT_BGRX_GREEN, EXT_BGRX_BLUE,
                                3, EXT_BGRX_PIXELSIZE)

/* 6. EXT_XBGR (X=0, B=1, G=2, R=3, 4-byte) */
GENERATE_YCC_RGB_CONVERT_AVX512(jsimd_ycc_extxbgr_convert_avx512,
                                EXT_XBGR_RED, EXT_XBGR_GREEN, EXT_XBGR_BLUE,
                                0, EXT_XBGR_PIXELSIZE)

/* 7. EXT_XRGB (X=0, R=1, G=2, B=3, 4-byte) */
GENERATE_YCC_RGB_CONVERT_AVX512(jsimd_ycc_extxrgb_convert_avx512,
                                EXT_XRGB_RED, EXT_XRGB_GREEN, EXT_XRGB_BLUE,
                                0, EXT_XRGB_PIXELSIZE)
