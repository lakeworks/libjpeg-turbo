/*
 * jdcolor-avx512.c - YCbCr -> RGB color conversion (AVX-512BW)
 *
 * Copyright (C) 2025, Lakeworks.
 *
 * AVX-512BW/VBMI2 implementation of YCbCr -> RGB color conversion for
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
 *   R-Y = round(Cr * 0.40200) + Cr       (= Cr * 1.40200)
 *   B-Y = round(Cb * -0.22800) + Cb + Cb (= Cb * 1.77200)
 *   G-Y = round(madd({Cb,Cr}, {-0.34414, 0.28586})) - Cr
 *       = Cb * -0.34414 + Cr * (0.28586 - 1) = Cb * -0.34414 + Cr * -0.71414
 *
 * Key intrinsics:
 *   _mm512_cvtepu8_epi16  (VPMOVZXBW) - zero-extend 32 bytes to 32 words
 *   _mm512_mulhi_epi16    (VPMULHW)   - signed high multiply (x*y >> 16)
 *   _mm512_madd_epi16     (VPMADDWD)  - multiply-add pairs to dwords
 *   _mm512_cvtusepi16_epi8 (VPMOVUSWB) - unsigned saturate 32 words to 32 bytes
 *   _mm512_permutex2var_epi8 (VPERMI2B) - byte-granularity shuffle from 2 srcs
 */

#include "../../src/jinclude.h"
#include "../../src/jpeglib.h"
#include "../../src/jdct.h"
#include "../jsimd.h"

#include <immintrin.h>


/*
 * Fixed-point constants (SCALEBITS = 16):
 *
 * FIX(1.40200) = 91881    -> split: FIX(0.40200) = 26345, add 1x Cr
 * FIX(0.34414) = 22554
 * FIX(0.71414) = 46802    -> split: FIX(0.28586) = 18734, subtract 1x Cr
 * FIX(1.77200) = 116130   -> split: FIX(0.22800) = 14942, add 2x Cb
 */
#define SCALEBITS       16
#define ONE_HALF        ((int)1 << (SCALEBITS - 1))

#define F_0_402         26345   /* FIX(0.40200) = FIX(1.402) - 65536 */
#define MF_0_228      (-14942)  /* -FIX(0.22800) = -(2*65536 - FIX(1.772)) */
#define MF_0_344      (-22554)  /* -FIX(0.34414) */
#define F_0_285         18734   /* FIX(0.28586) = 65536 - FIX(0.71414) */


/* --------------------------------------------------------------------------
 * Permutation index macros for _mm512_permutex2var_epi8.
 *
 * After computing R, G, B as 32 contiguous bytes each (via vpmovuswb), we
 * pack them into two source ZMM vectors:
 *
 *   src1 = { R[0..31], G[0..31] }   (bytes 0..63)
 *   src2 = { B[0..31], X[0..31] }   (bytes 0..63, X = 0xFF filler for 4-byte)
 *
 * The permutex2var index selects a byte from src1 (indices 0..63) or src2
 * (indices 64..127, i.e., bit 6 set selects the second source).
 *
 * For each output byte position 'pos', we determine which pixel it belongs
 * to (pos / pixel_size) and which channel within that pixel (pos % pixel_size),
 * then compute the source index:
 *   R[p] -> index p       (src1, low half)
 *   G[p] -> index 32 + p  (src1, high half)
 *   B[p] -> index 64 + p  (src2, low half)
 *   X[p] -> index 96 + p  (src2, high half)
 * -------------------------------------------------------------------------- */

/*
 * 3-byte format index: output byte 'pos' (0..95) maps to which source byte?
 * pixel = pos/3, channel_offset = pos%3.
 */
#define IDX3(R, G, B, pos)  \
  ((pos) >= 96 ? 0 :        \
   ((pos) % 3 == (R)) ? ((pos) / 3) :          \
   ((pos) % 3 == (G)) ? (32 + (pos) / 3) :     \
   (64 + (pos) / 3))

/* First 64 output bytes (pixels 0..21, partial pixel 21 at byte 63) */
#define PERM3_LO(R, G, B)                                              \
  IDX3(R,G,B, 0), IDX3(R,G,B, 1), IDX3(R,G,B, 2), IDX3(R,G,B, 3),   \
  IDX3(R,G,B, 4), IDX3(R,G,B, 5), IDX3(R,G,B, 6), IDX3(R,G,B, 7),   \
  IDX3(R,G,B, 8), IDX3(R,G,B, 9), IDX3(R,G,B,10), IDX3(R,G,B,11),   \
  IDX3(R,G,B,12), IDX3(R,G,B,13), IDX3(R,G,B,14), IDX3(R,G,B,15),   \
  IDX3(R,G,B,16), IDX3(R,G,B,17), IDX3(R,G,B,18), IDX3(R,G,B,19),   \
  IDX3(R,G,B,20), IDX3(R,G,B,21), IDX3(R,G,B,22), IDX3(R,G,B,23),   \
  IDX3(R,G,B,24), IDX3(R,G,B,25), IDX3(R,G,B,26), IDX3(R,G,B,27),   \
  IDX3(R,G,B,28), IDX3(R,G,B,29), IDX3(R,G,B,30), IDX3(R,G,B,31),   \
  IDX3(R,G,B,32), IDX3(R,G,B,33), IDX3(R,G,B,34), IDX3(R,G,B,35),   \
  IDX3(R,G,B,36), IDX3(R,G,B,37), IDX3(R,G,B,38), IDX3(R,G,B,39),   \
  IDX3(R,G,B,40), IDX3(R,G,B,41), IDX3(R,G,B,42), IDX3(R,G,B,43),   \
  IDX3(R,G,B,44), IDX3(R,G,B,45), IDX3(R,G,B,46), IDX3(R,G,B,47),   \
  IDX3(R,G,B,48), IDX3(R,G,B,49), IDX3(R,G,B,50), IDX3(R,G,B,51),   \
  IDX3(R,G,B,52), IDX3(R,G,B,53), IDX3(R,G,B,54), IDX3(R,G,B,55),   \
  IDX3(R,G,B,56), IDX3(R,G,B,57), IDX3(R,G,B,58), IDX3(R,G,B,59),   \
  IDX3(R,G,B,60), IDX3(R,G,B,61), IDX3(R,G,B,62), IDX3(R,G,B,63)

/* Remaining 32 output bytes (pixels 21..31), padded with zeros */
#define PERM3_HI(R, G, B)                                              \
  IDX3(R,G,B,64), IDX3(R,G,B,65), IDX3(R,G,B,66), IDX3(R,G,B,67),   \
  IDX3(R,G,B,68), IDX3(R,G,B,69), IDX3(R,G,B,70), IDX3(R,G,B,71),   \
  IDX3(R,G,B,72), IDX3(R,G,B,73), IDX3(R,G,B,74), IDX3(R,G,B,75),   \
  IDX3(R,G,B,76), IDX3(R,G,B,77), IDX3(R,G,B,78), IDX3(R,G,B,79),   \
  IDX3(R,G,B,80), IDX3(R,G,B,81), IDX3(R,G,B,82), IDX3(R,G,B,83),   \
  IDX3(R,G,B,84), IDX3(R,G,B,85), IDX3(R,G,B,86), IDX3(R,G,B,87),   \
  IDX3(R,G,B,88), IDX3(R,G,B,89), IDX3(R,G,B,90), IDX3(R,G,B,91),   \
  IDX3(R,G,B,92), IDX3(R,G,B,93), IDX3(R,G,B,94), IDX3(R,G,B,95),   \
  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,                  \
  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0

/*
 * 4-byte format index: output byte 'pos' (0..127) maps to which source byte?
 * pixel = pos/4, channel_offset = pos%4.
 */
#define IDX4(R, G, B, X, pos)   \
  (((pos) % 4 == (R)) ? ((pos) / 4) :          \
   ((pos) % 4 == (G)) ? (32 + (pos) / 4) :     \
   ((pos) % 4 == (B)) ? (64 + (pos) / 4) :     \
   (96 + (pos) / 4))

/* First 64 output bytes (pixels 0..15) */
#define PERM4_LO(R, G, B, X)                                           \
  IDX4(R,G,B,X, 0), IDX4(R,G,B,X, 1), IDX4(R,G,B,X, 2), IDX4(R,G,B,X, 3), \
  IDX4(R,G,B,X, 4), IDX4(R,G,B,X, 5), IDX4(R,G,B,X, 6), IDX4(R,G,B,X, 7), \
  IDX4(R,G,B,X, 8), IDX4(R,G,B,X, 9), IDX4(R,G,B,X,10), IDX4(R,G,B,X,11), \
  IDX4(R,G,B,X,12), IDX4(R,G,B,X,13), IDX4(R,G,B,X,14), IDX4(R,G,B,X,15), \
  IDX4(R,G,B,X,16), IDX4(R,G,B,X,17), IDX4(R,G,B,X,18), IDX4(R,G,B,X,19), \
  IDX4(R,G,B,X,20), IDX4(R,G,B,X,21), IDX4(R,G,B,X,22), IDX4(R,G,B,X,23), \
  IDX4(R,G,B,X,24), IDX4(R,G,B,X,25), IDX4(R,G,B,X,26), IDX4(R,G,B,X,27), \
  IDX4(R,G,B,X,28), IDX4(R,G,B,X,29), IDX4(R,G,B,X,30), IDX4(R,G,B,X,31), \
  IDX4(R,G,B,X,32), IDX4(R,G,B,X,33), IDX4(R,G,B,X,34), IDX4(R,G,B,X,35), \
  IDX4(R,G,B,X,36), IDX4(R,G,B,X,37), IDX4(R,G,B,X,38), IDX4(R,G,B,X,39), \
  IDX4(R,G,B,X,40), IDX4(R,G,B,X,41), IDX4(R,G,B,X,42), IDX4(R,G,B,X,43), \
  IDX4(R,G,B,X,44), IDX4(R,G,B,X,45), IDX4(R,G,B,X,46), IDX4(R,G,B,X,47), \
  IDX4(R,G,B,X,48), IDX4(R,G,B,X,49), IDX4(R,G,B,X,50), IDX4(R,G,B,X,51), \
  IDX4(R,G,B,X,52), IDX4(R,G,B,X,53), IDX4(R,G,B,X,54), IDX4(R,G,B,X,55), \
  IDX4(R,G,B,X,56), IDX4(R,G,B,X,57), IDX4(R,G,B,X,58), IDX4(R,G,B,X,59), \
  IDX4(R,G,B,X,60), IDX4(R,G,B,X,61), IDX4(R,G,B,X,62), IDX4(R,G,B,X,63)

/* Second 64 output bytes (pixels 16..31) */
#define PERM4_HI(R, G, B, X)                                           \
  IDX4(R,G,B,X, 64), IDX4(R,G,B,X, 65), IDX4(R,G,B,X, 66), IDX4(R,G,B,X, 67), \
  IDX4(R,G,B,X, 68), IDX4(R,G,B,X, 69), IDX4(R,G,B,X, 70), IDX4(R,G,B,X, 71), \
  IDX4(R,G,B,X, 72), IDX4(R,G,B,X, 73), IDX4(R,G,B,X, 74), IDX4(R,G,B,X, 75), \
  IDX4(R,G,B,X, 76), IDX4(R,G,B,X, 77), IDX4(R,G,B,X, 78), IDX4(R,G,B,X, 79), \
  IDX4(R,G,B,X, 80), IDX4(R,G,B,X, 81), IDX4(R,G,B,X, 82), IDX4(R,G,B,X, 83), \
  IDX4(R,G,B,X, 84), IDX4(R,G,B,X, 85), IDX4(R,G,B,X, 86), IDX4(R,G,B,X, 87), \
  IDX4(R,G,B,X, 88), IDX4(R,G,B,X, 89), IDX4(R,G,B,X, 90), IDX4(R,G,B,X, 91), \
  IDX4(R,G,B,X, 92), IDX4(R,G,B,X, 93), IDX4(R,G,B,X, 94), IDX4(R,G,B,X, 95), \
  IDX4(R,G,B,X, 96), IDX4(R,G,B,X, 97), IDX4(R,G,B,X, 98), IDX4(R,G,B,X, 99), \
  IDX4(R,G,B,X,100), IDX4(R,G,B,X,101), IDX4(R,G,B,X,102), IDX4(R,G,B,X,103), \
  IDX4(R,G,B,X,104), IDX4(R,G,B,X,105), IDX4(R,G,B,X,106), IDX4(R,G,B,X,107), \
  IDX4(R,G,B,X,108), IDX4(R,G,B,X,109), IDX4(R,G,B,X,110), IDX4(R,G,B,X,111), \
  IDX4(R,G,B,X,112), IDX4(R,G,B,X,113), IDX4(R,G,B,X,114), IDX4(R,G,B,X,115), \
  IDX4(R,G,B,X,116), IDX4(R,G,B,X,117), IDX4(R,G,B,X,118), IDX4(R,G,B,X,119), \
  IDX4(R,G,B,X,120), IDX4(R,G,B,X,121), IDX4(R,G,B,X,122), IDX4(R,G,B,X,123), \
  IDX4(R,G,B,X,124), IDX4(R,G,B,X,125), IDX4(R,G,B,X,126), IDX4(R,G,B,X,127)


/* Portable aligned-array declaration (MSVC uses __declspec, GCC uses attr) */
#ifdef _MSC_VER
#define ALIGN64  __declspec(align(64))
#else
#define ALIGN64  __attribute__((aligned(64)))
#endif


/* --------------------------------------------------------------------------
 * Function-generating macro.
 *
 * FUNC_NAME  - exported function name (7 variants)
 * R_IDX      - byte offset of R within the output pixel (0..3)
 * G_IDX      - byte offset of G within the output pixel (0..3)
 * B_IDX      - byte offset of B within the output pixel (0..3)
 * X_IDX      - byte offset of filler (0xFF) byte (-1 for 3-byte formats)
 * PIXEL_SIZE - 3 or 4
 * -------------------------------------------------------------------------- */

#define GENERATE_YCC_RGB_CONVERT(FUNC_NAME, R_IDX, G_IDX, B_IDX,       \
                                X_IDX, PIXEL_SIZE)                      \
                                                                        \
GLOBAL(void)                                                            \
FUNC_NAME(JDIMENSION out_width, JSAMPIMAGE input_buf,                   \
          JDIMENSION input_row, JSAMPARRAY output_buf, int num_rows)    \
{                                                                       \
  JSAMPROW inptr0, inptr1, inptr2, outptr;                              \
  JDIMENSION col;                                                       \
  int num_cols = (int)out_width;                                        \
                                                                        \
  /* Broadcast constants into ZMM registers (hoisted out of row loop) */\
  const __m512i zero     = _mm512_setzero_si512();                      \
  const __m512i bias     = _mm512_set1_epi16(CENTERJSAMPLE);            \
  const __m512i pw_f0402 = _mm512_set1_epi16((short)F_0_402);          \
  const __m512i pw_mf0228= _mm512_set1_epi16((short)MF_0_228);         \
  const __m512i pw_one   = _mm512_set1_epi16(1);                        \
                                                                        \
  while (--num_rows >= 0) {                                             \
    inptr0 = input_buf[0][input_row];   /* Y  */                        \
    inptr1 = input_buf[1][input_row];   /* Cb */                        \
    inptr2 = input_buf[2][input_row];   /* Cr */                        \
    input_row++;                                                        \
    outptr = *output_buf++;                                             \
                                                                        \
    for (col = 0; col < (JDIMENSION)num_cols; col += 32) {             \
      JDIMENSION remaining = (JDIMENSION)num_cols - col;                \
      JDIMENSION count = (remaining >= 32) ? 32 : remaining;           \
      __mmask32 load_mask = (count >= 32) ? (__mmask32)0xFFFFFFFF :    \
                            ((__mmask32)1 << count) - 1;                \
                                                                        \
      /* ---- Load 32 (or fewer) samples of Y, Cb, Cr ---- */          \
      __m256i y8  = _mm256_maskz_loadu_epi8(load_mask,                  \
                      (const __m256i *)(inptr0 + col));                 \
      __m256i cb8 = _mm256_maskz_loadu_epi8(load_mask,                  \
                      (const __m256i *)(inptr1 + col));                 \
      __m256i cr8 = _mm256_maskz_loadu_epi8(load_mask,                  \
                      (const __m256i *)(inptr2 + col));                 \
                                                                        \
      /* Zero-extend 8-bit to 16-bit (32 pixels -> 512 bits) */         \
      __m512i y16  = _mm512_cvtepu8_epi16(y8);                         \
      __m512i cb16 = _mm512_cvtepu8_epi16(cb8);                        \
      __m512i cr16 = _mm512_cvtepu8_epi16(cr8);                        \
                                                                        \
      /* Center Cb and Cr around zero */                                \
      cb16 = _mm512_sub_epi16(cb16, bias);                              \
      cr16 = _mm512_sub_epi16(cr16, bias);                              \
                                                                        \
      /* ---- R-Y = round(Cr * FIX(0.40200)) + Cr ----                 \
       *                                                                \
       * Trick: vpmulhw gives (a * b) >> 16.  To get (Cr * 26345) >> 16\
       * with rounding, compute vpmulhw(2*Cr, 26345), add 1, sra 1.    \
       */                                                               \
      __m512i cr2  = _mm512_add_epi16(cr16, cr16);                      \
      __m512i r_y  = _mm512_mulhi_epi16(cr2, pw_f0402);                \
      r_y = _mm512_add_epi16(r_y, pw_one);                             \
      r_y = _mm512_srai_epi16(r_y, 1);                                 \
      r_y = _mm512_add_epi16(r_y, cr16);                                \
                                                                        \
      /* ---- B-Y = round(Cb * -FIX(0.22800)) + 2*Cb ---- */           \
      __m512i cb2  = _mm512_add_epi16(cb16, cb16);                      \
      __m512i b_y  = _mm512_mulhi_epi16(cb2, pw_mf0228);               \
      b_y = _mm512_add_epi16(b_y, pw_one);                             \
      b_y = _mm512_srai_epi16(b_y, 1);                                 \
      b_y = _mm512_add_epi16(b_y, cb16);                                \
      b_y = _mm512_add_epi16(b_y, cb16);                                \
                                                                        \
      /* ---- G-Y via vpmaddwd ----                                     \
       *                                                                \
       * Interleave Cb and Cr words into {Cb_i, Cr_i} pairs, then      \
       * madd with {-FIX(0.34414), FIX(0.28586)} gives 32-bit:         \
       *   Cb_i * (-22554) + Cr_i * 18734                              \
       *                                                                \
       * After rounding and >> SCALEBITS, pack back to 16-bit, then    \
       * subtract Cr to get the full -0.71414 coefficient:              \
       *   G-Y = (Cb * -0.34414 + Cr * 0.28586) - Cr                   \
       *       = Cb * -0.34414 + Cr * -0.71414                          \
       *                                                                \
       * Split into two 256-bit halves: unpacklo/hi produce 16 {Cb,Cr} \
       * pairs each, madd produces 16 x 32-bit each, packs_epi32       \
       * recombines to 16 x 16-bit.                                     \
       */                                                               \
      __m512i g_y;                                                      \
      {                                                                 \
        __m256i cb_lo = _mm512_castsi512_si256(cb16);                   \
        __m256i cr_lo = _mm512_castsi512_si256(cr16);                   \
        __m256i cb_hi = _mm512_extracti64x4_epi64(cb16, 1);            \
        __m256i cr_hi = _mm512_extracti64x4_epi64(cr16, 1);            \
                                                                        \
        /* {-FIX(0.34414), FIX(0.28586)} packed as {lo16, hi16} */     \
        __m256i gcoeff = _mm256_set1_epi32(                             \
          ((unsigned int)(unsigned short)((short)F_0_285) << 16) |      \
           (unsigned short)((short)MF_0_344));                          \
        __m256i half = _mm256_set1_epi32(ONE_HALF);                     \
                                                                        \
        /* Low 16 pixels */                                             \
        __m256i cbcr_a = _mm256_unpacklo_epi16(cb_lo, cr_lo);          \
        __m256i cbcr_b = _mm256_unpackhi_epi16(cb_lo, cr_lo);          \
        __m256i g32_a  = _mm256_madd_epi16(cbcr_a, gcoeff);            \
        __m256i g32_b  = _mm256_madd_epi16(cbcr_b, gcoeff);            \
        g32_a = _mm256_srai_epi32(                                      \
                  _mm256_add_epi32(g32_a, half), SCALEBITS);            \
        g32_b = _mm256_srai_epi32(                                      \
                  _mm256_add_epi32(g32_b, half), SCALEBITS);            \
        __m256i g_lo = _mm256_packs_epi32(g32_a, g32_b);               \
                                                                        \
        /* High 16 pixels */                                            \
        cbcr_a = _mm256_unpacklo_epi16(cb_hi, cr_hi);                  \
        cbcr_b = _mm256_unpackhi_epi16(cb_hi, cr_hi);                  \
        g32_a  = _mm256_madd_epi16(cbcr_a, gcoeff);                    \
        g32_b  = _mm256_madd_epi16(cbcr_b, gcoeff);                    \
        g32_a = _mm256_srai_epi32(                                      \
                  _mm256_add_epi32(g32_a, half), SCALEBITS);            \
        g32_b = _mm256_srai_epi32(                                      \
                  _mm256_add_epi32(g32_b, half), SCALEBITS);            \
        __m256i g_hi = _mm256_packs_epi32(g32_a, g32_b);               \
                                                                        \
        /* Combine halves and subtract Cr */                            \
        g_y = _mm512_inserti64x4(                                       \
                _mm512_castsi256_si512(g_lo), g_hi, 1);                 \
        g_y = _mm512_sub_epi16(g_y, cr16);                              \
      }                                                                 \
                                                                        \
      /* ---- Add Y to each color difference ---- */                    \
      __m512i r16 = _mm512_add_epi16(y16, r_y);                        \
      __m512i g16 = _mm512_add_epi16(y16, g_y);                        \
      __m512i b16 = _mm512_add_epi16(y16, b_y);                        \
                                                                        \
      /* ---- Clamp to [0, 255] and pack to bytes ----                  \
       *                                                                \
       * max(val, 0) clamps negatives; vpmovuswb saturates >255 to 255.\
       * Result is 32 contiguous bytes in a __m256i with NO lane        \
       * interleaving (unlike vpackuswb which shuffles within lanes).   \
       */                                                               \
      __m256i r8 = _mm512_cvtusepi16_epi8(                              \
                     _mm512_max_epi16(r16, zero));                      \
      __m256i g8 = _mm512_cvtusepi16_epi8(                              \
                     _mm512_max_epi16(g16, zero));                      \
      __m256i b8 = _mm512_cvtusepi16_epi8(                              \
                     _mm512_max_epi16(b16, zero));                      \
                                                                        \
      /* ---- Interleave R, G, B (,X) into output pixel format ----     \
       *                                                                \
       * Build two ZMM source vectors for permutex2var_epi8:            \
       *   src1 = { R[0..31], G[0..31] }                               \
       *   src2 = { B[0..31], ... }                                     \
       */                                                               \
      __m512i src1 = _mm512_inserti64x4(                                \
                       _mm512_castsi256_si512(r8), g8, 1);              \
                                                                        \
      if (PIXEL_SIZE == 3) {                                            \
        __m512i src2 = _mm512_castsi256_si512(b8);                      \
                                                                        \
        ALIGN64 static const char perm_lo[64] = {                       \
          PERM3_LO(R_IDX, G_IDX, B_IDX) };                             \
        ALIGN64 static const char perm_hi[64] = {                       \
          PERM3_HI(R_IDX, G_IDX, B_IDX) };                             \
                                                                        \
        __m512i idx0 = _mm512_load_si512((const __m512i *)perm_lo);     \
        __m512i idx1 = _mm512_load_si512((const __m512i *)perm_hi);     \
                                                                        \
        __m512i out0 = _mm512_permutex2var_epi8(src1, idx0, src2);      \
        __m512i out1 = _mm512_permutex2var_epi8(src1, idx1, src2);      \
                                                                        \
        /* Store: 32 pixels * 3 bytes = 96 bytes = 64 + 32 */          \
        if (count >= 32) {                                              \
          _mm512_storeu_si512((__m512i *)outptr, out0);                  \
          _mm256_storeu_si256((__m256i *)(outptr + 64),                  \
                              _mm512_castsi512_si256(out1));             \
        } else {                                                        \
          int total = (int)count * 3;                                   \
          if (total >= 64) {                                            \
            _mm512_storeu_si512((__m512i *)outptr, out0);                \
            int rem = total - 64;                                       \
            if (rem > 0) {                                              \
              __mmask32 sm = (rem >= 32) ?                               \
                             (__mmask32)0xFFFFFFFF :                     \
                             ((__mmask32)1 << rem) - 1;                 \
              _mm256_mask_storeu_epi8(outptr + 64, sm,                  \
                                     _mm512_castsi512_si256(out1));      \
            }                                                           \
          } else {                                                      \
            __mmask64 sm64 = ((__mmask64)1 << total) - 1;               \
            _mm512_mask_storeu_epi8(outptr, sm64, out0);                \
          }                                                             \
        }                                                               \
        outptr += (int)count * 3;                                       \
                                                                        \
      } else { /* PIXEL_SIZE == 4 */                                    \
                                                                        \
        __m256i x8 = _mm256_set1_epi8((char)0xFF);                      \
        __m512i src2 = _mm512_inserti64x4(                              \
                         _mm512_castsi256_si512(b8), x8, 1);            \
                                                                        \
        ALIGN64 static const char perm_lo[64] = {                       \
          PERM4_LO(R_IDX, G_IDX, B_IDX, X_IDX) };                      \
        ALIGN64 static const char perm_hi[64] = {                       \
          PERM4_HI(R_IDX, G_IDX, B_IDX, X_IDX) };                      \
                                                                        \
        __m512i idx0 = _mm512_load_si512((const __m512i *)perm_lo);     \
        __m512i idx1 = _mm512_load_si512((const __m512i *)perm_hi);     \
                                                                        \
        __m512i out0 = _mm512_permutex2var_epi8(src1, idx0, src2);      \
        __m512i out1 = _mm512_permutex2var_epi8(src1, idx1, src2);      \
                                                                        \
        /* Store: 32 pixels * 4 bytes = 128 bytes = 2 * 64 */          \
        if (count >= 32) {                                              \
          _mm512_storeu_si512((__m512i *)outptr, out0);                  \
          _mm512_storeu_si512((__m512i *)(outptr + 64), out1);          \
        } else {                                                        \
          int total = (int)count * 4;                                   \
          if (total > 64) {                                             \
            _mm512_storeu_si512((__m512i *)outptr, out0);               \
            __mmask64 sm64 = ((__mmask64)1 << (total - 64)) - 1;       \
            _mm512_mask_storeu_epi8(outptr + 64, sm64, out1);           \
          } else if (total == 64) {                                     \
            _mm512_storeu_si512((__m512i *)outptr, out0);               \
          } else {                                                      \
            __mmask64 sm64 = ((__mmask64)1 << total) - 1;               \
            _mm512_mask_storeu_epi8(outptr, sm64, out0);                \
          }                                                             \
        }                                                               \
        outptr += (int)count * 4;                                       \
      }                                                                 \
    } /* column loop */                                                 \
  } /* row loop */                                                      \
}


/* --------------------------------------------------------------------------
 * Generate all 7 YCbCr -> RGB conversion functions.
 *
 * Each pixel format is defined by the byte offset of R, G, B (and optionally
 * filler X) within each output pixel.  The GENERATE_YCC_RGB_CONVERT macro
 * stamps out a complete function with the correct permutation tables compiled
 * in as static constants.
 * -------------------------------------------------------------------------- */

/* 1. Default RGB (R=0, G=1, B=2, 3-byte) */
GENERATE_YCC_RGB_CONVERT(jsimd_ycc_rgb_convert_avx512,
                         RGB_RED, RGB_GREEN, RGB_BLUE, -1,
                         RGB_PIXELSIZE)

/* 2. EXT_RGB (R=0, G=1, B=2, 3-byte) -- same layout as default */
GENERATE_YCC_RGB_CONVERT(jsimd_ycc_extrgb_convert_avx512,
                         EXT_RGB_RED, EXT_RGB_GREEN, EXT_RGB_BLUE, -1,
                         EXT_RGB_PIXELSIZE)

/* 3. EXT_RGBX (R=0, G=1, B=2, X=3, 4-byte) */
GENERATE_YCC_RGB_CONVERT(jsimd_ycc_extrgbx_convert_avx512,
                         EXT_RGBX_RED, EXT_RGBX_GREEN, EXT_RGBX_BLUE,
                         3, EXT_RGBX_PIXELSIZE)

/* 4. EXT_BGR (B=0, G=1, R=2, 3-byte) */
GENERATE_YCC_RGB_CONVERT(jsimd_ycc_extbgr_convert_avx512,
                         EXT_BGR_RED, EXT_BGR_GREEN, EXT_BGR_BLUE, -1,
                         EXT_BGR_PIXELSIZE)

/* 5. EXT_BGRX (B=0, G=1, R=2, X=3, 4-byte) */
GENERATE_YCC_RGB_CONVERT(jsimd_ycc_extbgrx_convert_avx512,
                         EXT_BGRX_RED, EXT_BGRX_GREEN, EXT_BGRX_BLUE,
                         3, EXT_BGRX_PIXELSIZE)

/* 6. EXT_XBGR (X=0, B=1, G=2, R=3, 4-byte) */
GENERATE_YCC_RGB_CONVERT(jsimd_ycc_extxbgr_convert_avx512,
                         EXT_XBGR_RED, EXT_XBGR_GREEN, EXT_XBGR_BLUE,
                         0, EXT_XBGR_PIXELSIZE)

/* 7. EXT_XRGB (X=0, R=1, G=2, B=3, 4-byte) */
GENERATE_YCC_RGB_CONVERT(jsimd_ycc_extxrgb_convert_avx512,
                         EXT_XRGB_RED, EXT_XRGB_GREEN, EXT_XRGB_BLUE,
                         0, EXT_XRGB_PIXELSIZE)
