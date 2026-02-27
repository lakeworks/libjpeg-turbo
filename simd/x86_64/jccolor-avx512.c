/*
 * jccolor-avx512.c - RGB->YCbCr colorspace conversion (AVX-512BW)
 *
 * Copyright (C) 2025, Lakeworks.
 *
 * AVX-512BW/VBMI implementation of RGB->YCbCr color conversion.
 * Processes 32 pixels per iteration using ZMM registers.
 *
 * Uses _mm512_permutexvar_epi8 (VBMI) for byte-level deinterleaving,
 * which replaces the multi-step unpack/interleave dance in the AVX2
 * NASM version with a single permute instruction per channel.
 *
 * Based on the AVX2 NASM implementation:
 * Copyright (C) 2009, 2016, 2024-2025, D. R. Commander.
 * Copyright (C) 2015, Intel Corporation.
 * Copyright (C) 2018, Matthias Raencker.
 * Copyright (C) 2023, Aliaksiej Kandracienka.
 */

#include "../../src/jinclude.h"
#include "../../src/jpeglib.h"
#include "../../src/jdct.h"
#include "../jsimd.h"

#include <immintrin.h>

/*
 * Fixed-point constants for RGB->YCbCr conversion.
 * SCALEBITS = 16, constants are FIX(x) = (int)(x * 65536 + 0.5)
 *
 * Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
 *     = (0.29900 * R + 0.33700 * G) + (0.11400 * B + 0.25000 * G)
 * Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + 128
 * Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + 128
 */
#define SCALEBITS       16

#define FIX_029900      19595   /* FIX(0.29900) */
#define FIX_033700      22087   /* FIX(0.33700) = FIX(0.58700) - FIX(0.25000) */
#define FIX_011400       7471   /* FIX(0.11400) */
#define FIX_025000      16384   /* FIX(0.25000) */
#define FIX_016874      11059   /* FIX(0.16874) */
#define FIX_033126      21709   /* FIX(0.33126) */
#define FIX_050000      32768   /* FIX(0.50000) */
#define FIX_041869      27439   /* FIX(0.41869) */
#define FIX_008131       5329   /* FIX(0.08131) */

/* Rounding constant: (1 << (SCALEBITS - 1)) = 32768 */
#define ONE_HALF        ((JLONG)1 << (SCALEBITS - 1))

/* Cb/Cr bias: CENTERJSAMPLE << SCALEBITS = 128 << 16 = 8388608 */
#define CBCR_OFFSET     ((JLONG)CENTERJSAMPLE << SCALEBITS)

/*
 * Build permute index vectors for _mm512_permutexvar_epi8 (VBMI).
 *
 * For 3-byte pixels (RGB/BGR), 32 pixels = 96 bytes loaded into
 * a ZMM register (only the low 96 bytes matter; we load 96 bytes).
 * For 4-byte pixels (RGBX/BGRX/XBGR/XRGB), 32 pixels = 128 bytes
 * loaded into two ZMM registers.
 *
 * The permute indices extract the R, G, or B channel from the
 * interleaved pixel data.
 *
 * MAKE_PERM3(ch): build a 64-byte index vector that picks byte at
 *   offset 'ch' from each group of 3 bytes, for positions 0..31.
 *   Bytes 32..63 are don't-care (set to 0).
 *
 * MAKE_PERM4_LO(ch): picks byte at offset 'ch' from each group of 4
 *   in the first ZMM (pixels 0..15).
 * MAKE_PERM4_HI(ch): picks byte at offset 'ch' from each group of 4
 *   in the second ZMM (pixels 16..31).
 */

/*
 * Helper: build a __m512i constant from 64 individual byte values.
 * This is used to create the permute index vectors at compile time.
 */

/* For 3-byte pixels: extract channel 'ch' from 32 groups of 3 bytes.
 * Input is 96 bytes in one ZMM (only first 96 bytes valid).
 * Output is 32 bytes in the low half of ZMM.
 *
 * We split into two halves:
 * - Low 32 bytes of ZMM hold input bytes [0..31]
 * - High 32 bytes of ZMM hold input bytes [32..63]
 * But we load 96 bytes, so we need two ZMM registers.
 *
 * Actually, for 3-byte pixel, 32 pixels = 96 bytes.
 * We load into two registers:
 *   zmm_lo = bytes [0..63]   (from memory)
 *   zmm_hi = bytes [64..95]  (only 32 bytes, zero-padded)
 *
 * Then we use _mm512_permutex2var_epi8 which selects bytes from
 * two source registers based on a 7-bit index (bit 6 selects which src).
 *
 * For pixel i (0..31), channel at byte offset ch, the source byte is:
 *   i * 3 + ch
 * This index is in [0..95]. If < 64, select from zmm_lo (bit6=0).
 * If >= 64, select from zmm_hi (bit6=1, index bits[5:0] = byte-64).
 */

/* For 4-byte pixels: extract channel 'ch' from 32 groups of 4 bytes.
 * Input is 128 bytes in two ZMM registers.
 *   zmm_lo = pixels 0..15  = bytes [0..63]
 *   zmm_hi = pixels 16..31 = bytes [64..127]
 *
 * For the low register (pixels 0..15):
 *   pixel i, byte index = i * 4 + ch, range [0..63]
 *   Use _mm512_permutexvar_epi8(idx, zmm_lo)
 *
 * For the high register (pixels 16..31):
 *   pixel (i+16), byte index = i * 4 + ch, range [0..63]
 *   Use _mm512_permutexvar_epi8(idx, zmm_hi)
 *
 * Then combine the two 128-bit results (each has 16 bytes in low part).
 */


/*
 * Function-generating macro for RGB->YCbCr conversion.
 *
 * Parameters:
 *   FUNC_NAME   - output function name
 *   R_IDX       - byte offset of R within pixel (0-3)
 *   G_IDX       - byte offset of G within pixel (0-3)
 *   B_IDX       - byte offset of B within pixel (0-3)
 *   PIXEL_SIZE  - bytes per pixel (3 or 4)
 */
#define GENERATE_YCC_CONVERT(FUNC_NAME, R_IDX, G_IDX, B_IDX, PIXEL_SIZE) \
                                                                          \
GLOBAL(void)                                                              \
FUNC_NAME(JDIMENSION img_width, JSAMPARRAY input_buf,                     \
          JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)     \
{                                                                         \
  JSAMPROW inptr;                                                         \
  JSAMPROW outptr0, outptr1, outptr2;                                     \
  JDIMENSION col;                                                         \
  JDIMENSION num_cols = img_width;                                        \
  int row;                                                                \
                                                                          \
  /* Fixed-point multiply constants for _mm512_madd_epi16.                \
   * madd_epi16 computes: (a0*b0 + a1*b1) as 32-bit results,             \
   * where a and b are interleaved 16-bit word pairs.                     \
   *                                                                      \
   * Y  = (0.29900*R + 0.33700*G) + (0.11400*B + 0.25000*G)              \
   * We pair (R, G) with coefficients (FIX_029900, FIX_033700) for        \
   * the first term, and (B, G) with (FIX_011400, FIX_025000) for        \
   * the second term. Sum gives Y in 32-bit fixed-point.                  \
   *                                                                      \
   * Cb = (-0.16874*R + -0.33126*G) + 0.50000*B + bias                   \
   * We pair (R, G) with (-FIX_016874, -FIX_033126) for first term.       \
   * B * FIX_050000 done separately (shift left by 15 = *32768).          \
   *                                                                      \
   * Cr = 0.50000*R + (-0.08131*B + -0.41869*G) + bias                   \
   * We pair (B, G) with (-FIX_008131, -FIX_041869) for second term.      \
   * R * FIX_050000 done separately.                                      \
   */                                                                     \
  const __m512i pw_f0299_f0337 = _mm512_set1_epi32(                       \
      (int)((unsigned short)FIX_029900 |                                  \
            ((unsigned int)(unsigned short)FIX_033700 << 16)));            \
  const __m512i pw_f0114_f0250 = _mm512_set1_epi32(                       \
      (int)((unsigned short)FIX_011400 |                                  \
            ((unsigned int)(unsigned short)FIX_025000 << 16)));            \
  const __m512i pw_mf016_mf033 = _mm512_set1_epi32(                       \
      (int)((unsigned short)(-FIX_016874 & 0xFFFF) |                      \
            ((unsigned int)(unsigned short)(-FIX_033126 & 0xFFFF) << 16)));\
  const __m512i pw_mf008_mf041 = _mm512_set1_epi32(                       \
      (int)((unsigned short)(-FIX_008131 & 0xFFFF) |                      \
            ((unsigned int)(unsigned short)(-FIX_041869 & 0xFFFF) << 16)));\
  const __m512i pd_onehalf = _mm512_set1_epi32((int)ONE_HALF);            \
  const __m512i pd_onehalfm1_cj = _mm512_set1_epi32(                      \
      (int)(ONE_HALF - 1 + CBCR_OFFSET));                                 \
  const __m512i zero = _mm512_setzero_si512();                            \
                                                                          \
  /* Build permute index vectors for byte deinterleaving. */              \
  __m512i perm_r, perm_g, perm_b;                                        \
  GENERATE_PERM_INDICES(R_IDX, G_IDX, B_IDX, PIXEL_SIZE,                 \
                        perm_r, perm_g, perm_b)                           \
                                                                          \
  for (row = 0; row < num_rows; row++) {                                  \
    inptr = input_buf[row];                                               \
    outptr0 = output_buf[0][output_row + row];                            \
    outptr1 = output_buf[1][output_row + row];                            \
    outptr2 = output_buf[2][output_row + row];                            \
                                                                          \
    for (col = 0; col < num_cols; ) {                                     \
      JDIMENSION remaining = num_cols - col;                              \
                                                                          \
      if (remaining >= 32) {                                              \
        /* Main loop: process 32 pixels. */                               \
        __m512i r_bytes, g_bytes, b_bytes;                                \
        DEINTERLEAVE_LOAD(inptr, PIXEL_SIZE,                              \
                          perm_r, perm_g, perm_b,                         \
                          r_bytes, g_bytes, b_bytes)                      \
                                                                          \
        /* r_bytes, g_bytes, b_bytes each contain 32 bytes in the low     \
         * 256 bits (for 3-byte pixels) or across both halves (4-byte).   \
         * Zero-extend to 16-bit words: split into low and high halves    \
         * of 16 pixels each.                                             \
         */                                                               \
        __m256i r_lo256 = _mm512_castsi512_si256(r_bytes);                \
        __m256i r_hi256 = _mm512_extracti64x4_epi64(r_bytes, 1);         \
        __m256i g_lo256 = _mm512_castsi512_si256(g_bytes);                \
        __m256i g_hi256 = _mm512_extracti64x4_epi64(g_bytes, 1);         \
        __m256i b_lo256 = _mm512_castsi512_si256(b_bytes);                \
        __m256i b_hi256 = _mm512_extracti64x4_epi64(b_bytes, 1);         \
                                                                          \
        /* Zero-extend low 16 bytes of each 256-bit half to 16-bit words  \
         * in a 512-bit register. This gives us 16 words per ZMM.        \
         */                                                               \
        __m512i rw_lo = _mm512_cvtepu8_epi16(r_lo256);                   \
        __m512i rw_hi = _mm512_cvtepu8_epi16(r_hi256);                   \
        __m512i gw_lo = _mm512_cvtepu8_epi16(g_lo256);                   \
        __m512i gw_hi = _mm512_cvtepu8_epi16(g_hi256);                   \
        __m512i bw_lo = _mm512_cvtepu8_epi16(b_lo256);                   \
        __m512i bw_hi = _mm512_cvtepu8_epi16(b_hi256);                   \
                                                                          \
        /* ---- Compute Y ---- */                                         \
        /* Y = (FIX_029900*R + FIX_033700*G) + (FIX_011400*B + FIX_025000*G) */ \
        /*                                                                  \
         * Interleave R and G as word pairs, then madd with coefficients.   \
         * rg_lo_even = (R0, G0, R1, G1, ..., R15, G15) as words          \
         * Then madd_epi16 gives 32-bit: R0*0.299 + G0*0.337, ...         \
         */                                                               \
        __m512i rg_even, rg_odd;                                          \
        __m512i bg_even, bg_odd;                                          \
        __m512i y_even, y_odd;                                            \
                                                                          \
        /* Low 16 pixels (even set) */                                    \
        rg_even = _mm512_unpacklo_epi16(rw_lo, gw_lo);                   \
        rg_odd  = _mm512_unpackhi_epi16(rw_lo, gw_lo);                   \
        bg_even = _mm512_unpacklo_epi16(bw_lo, gw_lo);                   \
        bg_odd  = _mm512_unpackhi_epi16(bw_lo, gw_lo);                   \
                                                                          \
        __m512i y_rg_even = _mm512_madd_epi16(rg_even, pw_f0299_f0337);  \
        __m512i y_rg_odd  = _mm512_madd_epi16(rg_odd,  pw_f0299_f0337); \
        __m512i y_bg_even = _mm512_madd_epi16(bg_even, pw_f0114_f0250);  \
        __m512i y_bg_odd  = _mm512_madd_epi16(bg_odd,  pw_f0114_f0250); \
                                                                          \
        y_even = _mm512_add_epi32(y_rg_even, y_bg_even);                 \
        y_even = _mm512_add_epi32(y_even, pd_onehalf);                   \
        y_even = _mm512_srli_epi32(y_even, SCALEBITS);                   \
                                                                          \
        y_odd = _mm512_add_epi32(y_rg_odd, y_bg_odd);                    \
        y_odd = _mm512_add_epi32(y_odd, pd_onehalf);                     \
        y_odd = _mm512_srli_epi32(y_odd, SCALEBITS);                     \
                                                                          \
        __m512i y_lo = _mm512_packs_epi32(y_even, y_odd);                \
                                                                          \
        /* High 16 pixels */                                              \
        rg_even = _mm512_unpacklo_epi16(rw_hi, gw_hi);                   \
        rg_odd  = _mm512_unpackhi_epi16(rw_hi, gw_hi);                   \
        bg_even = _mm512_unpacklo_epi16(bw_hi, gw_hi);                   \
        bg_odd  = _mm512_unpackhi_epi16(bw_hi, gw_hi);                   \
                                                                          \
        y_rg_even = _mm512_madd_epi16(rg_even, pw_f0299_f0337);          \
        y_rg_odd  = _mm512_madd_epi16(rg_odd,  pw_f0299_f0337);         \
        y_bg_even = _mm512_madd_epi16(bg_even, pw_f0114_f0250);          \
        y_bg_odd  = _mm512_madd_epi16(bg_odd,  pw_f0114_f0250);         \
                                                                          \
        y_even = _mm512_add_epi32(y_rg_even, y_bg_even);                 \
        y_even = _mm512_add_epi32(y_even, pd_onehalf);                   \
        y_even = _mm512_srli_epi32(y_even, SCALEBITS);                   \
                                                                          \
        y_odd = _mm512_add_epi32(y_rg_odd, y_bg_odd);                    \
        y_odd = _mm512_add_epi32(y_odd, pd_onehalf);                     \
        y_odd = _mm512_srli_epi32(y_odd, SCALEBITS);                     \
                                                                          \
        __m512i y_hi = _mm512_packs_epi32(y_even, y_odd);                \
                                                                          \
        /* ---- Compute Cb ---- */                                        \
        /* Cb = (-FIX_016874*R + -FIX_033126*G) + FIX_050000*B + bias */  \
        __m512i cb_rg_even, cb_rg_odd;                                    \
        __m512i cb_even, cb_odd;                                          \
        __m512i b_shifted;                                                \
                                                                          \
        /* Low 16 pixels */                                               \
        rg_even = _mm512_unpacklo_epi16(rw_lo, gw_lo);                   \
        rg_odd  = _mm512_unpackhi_epi16(rw_lo, gw_lo);                   \
                                                                          \
        cb_rg_even = _mm512_madd_epi16(rg_even, pw_mf016_mf033);         \
        cb_rg_odd  = _mm512_madd_epi16(rg_odd,  pw_mf016_mf033);        \
                                                                          \
        /* B * FIX(0.500) = B * 32768 = B << 15                           \
         * But B is 16-bit unsigned [0..255], so B<<15 fits in 32-bit.    \
         * We need B as 32-bit first. Use unpacklo/hi with zero.          \
         */                                                               \
        __m512i bw_lo_even32 = _mm512_unpacklo_epi16(bw_lo, zero);       \
        __m512i bw_lo_odd32  = _mm512_unpackhi_epi16(bw_lo, zero);       \
        b_shifted = _mm512_slli_epi32(bw_lo_even32, 15);                 \
        cb_even = _mm512_add_epi32(cb_rg_even, b_shifted);               \
        cb_even = _mm512_add_epi32(cb_even, pd_onehalfm1_cj);            \
        cb_even = _mm512_srli_epi32(cb_even, SCALEBITS);                 \
                                                                          \
        b_shifted = _mm512_slli_epi32(bw_lo_odd32, 15);                  \
        cb_odd = _mm512_add_epi32(cb_rg_odd, b_shifted);                 \
        cb_odd = _mm512_add_epi32(cb_odd, pd_onehalfm1_cj);              \
        cb_odd = _mm512_srli_epi32(cb_odd, SCALEBITS);                   \
                                                                          \
        __m512i cb_lo = _mm512_packs_epi32(cb_even, cb_odd);             \
                                                                          \
        /* High 16 pixels */                                              \
        rg_even = _mm512_unpacklo_epi16(rw_hi, gw_hi);                   \
        rg_odd  = _mm512_unpackhi_epi16(rw_hi, gw_hi);                   \
                                                                          \
        cb_rg_even = _mm512_madd_epi16(rg_even, pw_mf016_mf033);         \
        cb_rg_odd  = _mm512_madd_epi16(rg_odd,  pw_mf016_mf033);        \
                                                                          \
        __m512i bw_hi_even32 = _mm512_unpacklo_epi16(bw_hi, zero);       \
        __m512i bw_hi_odd32  = _mm512_unpackhi_epi16(bw_hi, zero);       \
        b_shifted = _mm512_slli_epi32(bw_hi_even32, 15);                 \
        cb_even = _mm512_add_epi32(cb_rg_even, b_shifted);               \
        cb_even = _mm512_add_epi32(cb_even, pd_onehalfm1_cj);            \
        cb_even = _mm512_srli_epi32(cb_even, SCALEBITS);                 \
                                                                          \
        b_shifted = _mm512_slli_epi32(bw_hi_odd32, 15);                  \
        cb_odd = _mm512_add_epi32(cb_rg_odd, b_shifted);                 \
        cb_odd = _mm512_add_epi32(cb_odd, pd_onehalfm1_cj);              \
        cb_odd = _mm512_srli_epi32(cb_odd, SCALEBITS);                   \
                                                                          \
        __m512i cb_hi = _mm512_packs_epi32(cb_even, cb_odd);             \
                                                                          \
        /* ---- Compute Cr ---- */                                        \
        /* Cr = FIX_050000*R + (-FIX_041869*G + -FIX_008131*B) + bias */  \
        __m512i cr_gb_even, cr_gb_odd;                                    \
        __m512i cr_even, cr_odd;                                          \
        __m512i r_shifted;                                                \
                                                                          \
        /* Low 16 pixels */                                               \
        /* Pair (B, G) for the negative coefficients:                     \
         * B*(-FIX_008131) + G*(-FIX_041869) = -0.08131*B - 0.41869*G   \
         */                                                               \
        __m512i bg_lo_even_cr = _mm512_unpacklo_epi16(bw_lo, gw_lo);     \
        __m512i bg_lo_odd_cr  = _mm512_unpackhi_epi16(bw_lo, gw_lo);     \
                                                                          \
        cr_gb_even = _mm512_madd_epi16(bg_lo_even_cr, pw_mf008_mf041);   \
        cr_gb_odd  = _mm512_madd_epi16(bg_lo_odd_cr,  pw_mf008_mf041);  \
                                                                          \
        /* R * FIX(0.500) = R << 15 */                                    \
        __m512i rw_lo_even32 = _mm512_unpacklo_epi16(rw_lo, zero);       \
        __m512i rw_lo_odd32  = _mm512_unpackhi_epi16(rw_lo, zero);       \
        r_shifted = _mm512_slli_epi32(rw_lo_even32, 15);                 \
        cr_even = _mm512_add_epi32(cr_gb_even, r_shifted);               \
        cr_even = _mm512_add_epi32(cr_even, pd_onehalfm1_cj);            \
        cr_even = _mm512_srli_epi32(cr_even, SCALEBITS);                 \
                                                                          \
        r_shifted = _mm512_slli_epi32(rw_lo_odd32, 15);                  \
        cr_odd = _mm512_add_epi32(cr_gb_odd, r_shifted);                 \
        cr_odd = _mm512_add_epi32(cr_odd, pd_onehalfm1_cj);              \
        cr_odd = _mm512_srli_epi32(cr_odd, SCALEBITS);                   \
                                                                          \
        __m512i cr_lo = _mm512_packs_epi32(cr_even, cr_odd);             \
                                                                          \
        /* High 16 pixels */                                              \
        __m512i bg_hi_even_cr = _mm512_unpacklo_epi16(bw_hi, gw_hi);     \
        __m512i bg_hi_odd_cr  = _mm512_unpackhi_epi16(bw_hi, gw_hi);     \
                                                                          \
        cr_gb_even = _mm512_madd_epi16(bg_hi_even_cr, pw_mf008_mf041);   \
        cr_gb_odd  = _mm512_madd_epi16(bg_hi_odd_cr,  pw_mf008_mf041);  \
                                                                          \
        __m512i rw_hi_even32 = _mm512_unpacklo_epi16(rw_hi, zero);       \
        __m512i rw_hi_odd32  = _mm512_unpackhi_epi16(rw_hi, zero);       \
        r_shifted = _mm512_slli_epi32(rw_hi_even32, 15);                 \
        cr_even = _mm512_add_epi32(cr_gb_even, r_shifted);               \
        cr_even = _mm512_add_epi32(cr_even, pd_onehalfm1_cj);            \
        cr_even = _mm512_srli_epi32(cr_even, SCALEBITS);                 \
                                                                          \
        r_shifted = _mm512_slli_epi32(rw_hi_odd32, 15);                  \
        cr_odd = _mm512_add_epi32(cr_gb_odd, r_shifted);                 \
        cr_odd = _mm512_add_epi32(cr_odd, pd_onehalfm1_cj);              \
        cr_odd = _mm512_srli_epi32(cr_odd, SCALEBITS);                   \
                                                                          \
        __m512i cr_hi = _mm512_packs_epi32(cr_even, cr_odd);             \
                                                                          \
        /* ---- Pack 16-bit words down to 8-bit bytes ---- */             \
        /* _mm512_packs_epi32 produced signed 16-bit values in [0..255].  \
         * _mm512_packus_epi16 saturates signed 16-bit to unsigned 8-bit. \
         * y_lo has 16 words (lo pixels), y_hi has 16 words (hi pixels).  \
         * packus gives 32 bytes per output ZMM.                          \
         *                                                                \
         * NOTE: _mm512_packs_epi32 interleaves lanes (128-bit granularity), \
         * so the order within each 128-bit lane is:                      \
         *   lane[i] = {even[0..3], odd[0..3]} as 16-bit                  \
         * And _mm512_packus_epi16 similarly interleaves.                 \
         * We need to fix the lane ordering at the end.                   \
         */                                                               \
        __m512i y_packed  = _mm512_packus_epi16(y_lo,  y_hi);            \
        __m512i cb_packed = _mm512_packus_epi16(cb_lo, cb_hi);           \
        __m512i cr_packed = _mm512_packus_epi16(cr_lo, cr_hi);           \
                                                                          \
        /* Fix the cross-lane interleaving from pack operations.          \
         * After unpacklo/hi + packs_epi32 + packus_epi16, the data is    \
         * shuffled within 128-bit lanes.  We need a final permutation    \
         * to restore pixel order.                                        \
         *                                                                \
         * The sequence unpacklo/hi_epi16 -> packs_epi32 -> packus_epi16  \
         * on 512-bit registers operates on each 128-bit lane             \
         * independently. The result of packus_epi16(lo, hi) has:         \
         *   lane0: lo_lane0_bytes[0..7], hi_lane0_bytes[0..7]            \
         *   lane1: lo_lane1_bytes[0..7], hi_lane1_bytes[0..7]            \
         *   lane2: lo_lane2_bytes[0..7], hi_lane2_bytes[0..7]            \
         *   lane3: lo_lane3_bytes[0..7], hi_lane3_bytes[0..7]            \
         *                                                                \
         * Due to the way unpacklo/hi interleaves within 128-bit lanes,   \
         * the 16 pixels from the "lo" part (pixels 0..15) are at:        \
         *   lane0 low 8: pixels 0,1,2,3                                  \
         *   lane1 low 8: pixels 4,5,6,7                                  \
         *   lane2 low 8: pixels 8,9,10,11                                \
         *   lane3 low 8: pixels 12,13,14,15                              \
         * And the 16 pixels from "hi" part (pixels 16..31) are at:       \
         *   lane0 high 8: pixels 16,17,18,19                             \
         *   lane1 high 8: pixels 20,21,22,23                             \
         *   lane2 high 8: pixels 24,25,26,27                             \
         *   lane3 high 8: pixels 28,29,30,31                             \
         *                                                                \
         * We need sequential order: pixels 0..31.                        \
         * Use vpermq (64-bit permute) to rearrange the 8 qwords:         \
         *   src qwords:  [0,1, 2,3, 4,5, 6,7]                           \
         *   contains:    [lo0-3, hi16-19, lo4-7, hi20-23,                \
         *                 lo8-11, hi24-27, lo12-15, hi28-31]             \
         *   desired:     [lo0-3, lo4-7, lo8-11, lo12-15,                 \
         *                 hi16-19, hi20-23, hi24-27, hi28-31]            \
         *   permute idx: [0, 2, 4, 6, 1, 3, 5, 7]                       \
         */                                                               \
        const __m512i fix_perm = _mm512_setr_epi64(0, 2, 4, 6,           \
                                                   1, 3, 5, 7);          \
        y_packed  = _mm512_permutexvar_epi64(fix_perm, y_packed);         \
        cb_packed = _mm512_permutexvar_epi64(fix_perm, cb_packed);        \
        cr_packed = _mm512_permutexvar_epi64(fix_perm, cr_packed);        \
                                                                          \
        /* Store 32 bytes of Y, Cb, Cr to their respective planes. */     \
        _mm256_storeu_si256((__m256i *)(outptr0 + col),                   \
                            _mm512_castsi512_si256(y_packed));            \
        _mm256_storeu_si256((__m256i *)(outptr1 + col),                   \
                            _mm512_castsi512_si256(cb_packed));           \
        _mm256_storeu_si256((__m256i *)(outptr2 + col),                   \
                            _mm512_castsi512_si256(cr_packed));           \
                                                                          \
        inptr += 32 * PIXEL_SIZE;                                         \
        col += 32;                                                        \
      } else {                                                            \
        /* Tail: process remaining pixels with masked operations. */       \
        JDIMENSION load_bytes = remaining * (JDIMENSION)PIXEL_SIZE;       \
        __mmask64 store_mask = (__mmask64)((1ULL << remaining) - 1);      \
                                                                          \
        __m512i r_bytes, g_bytes, b_bytes;                                \
        DEINTERLEAVE_LOAD_MASKED(inptr, PIXEL_SIZE,                       \
                                 perm_r, perm_g, perm_b,                  \
                                 load_bytes,                              \
                                 r_bytes, g_bytes, b_bytes)               \
                                                                          \
        __m256i r_lo256 = _mm512_castsi512_si256(r_bytes);                \
        __m256i r_hi256 = _mm512_extracti64x4_epi64(r_bytes, 1);         \
        __m256i g_lo256 = _mm512_castsi512_si256(g_bytes);                \
        __m256i g_hi256 = _mm512_extracti64x4_epi64(g_bytes, 1);         \
        __m256i b_lo256 = _mm512_castsi512_si256(b_bytes);                \
        __m256i b_hi256 = _mm512_extracti64x4_epi64(b_bytes, 1);         \
                                                                          \
        __m512i rw_lo = _mm512_cvtepu8_epi16(r_lo256);                   \
        __m512i rw_hi = _mm512_cvtepu8_epi16(r_hi256);                   \
        __m512i gw_lo = _mm512_cvtepu8_epi16(g_lo256);                   \
        __m512i gw_hi = _mm512_cvtepu8_epi16(g_hi256);                   \
        __m512i bw_lo = _mm512_cvtepu8_epi16(b_lo256);                   \
        __m512i bw_hi = _mm512_cvtepu8_epi16(b_hi256);                   \
                                                                          \
        /* Y */                                                           \
        __m512i rg_even, rg_odd, bg_even, bg_odd;                        \
        __m512i y_even, y_odd;                                            \
        __m512i cb_even, cb_odd, cr_even, cr_odd;                         \
                                                                          \
        rg_even = _mm512_unpacklo_epi16(rw_lo, gw_lo);                   \
        rg_odd  = _mm512_unpackhi_epi16(rw_lo, gw_lo);                   \
        bg_even = _mm512_unpacklo_epi16(bw_lo, gw_lo);                   \
        bg_odd  = _mm512_unpackhi_epi16(bw_lo, gw_lo);                   \
                                                                          \
        y_even = _mm512_add_epi32(                                        \
            _mm512_madd_epi16(rg_even, pw_f0299_f0337),                  \
            _mm512_madd_epi16(bg_even, pw_f0114_f0250));                 \
        y_even = _mm512_srli_epi32(                                       \
            _mm512_add_epi32(y_even, pd_onehalf), SCALEBITS);            \
        y_odd = _mm512_add_epi32(                                         \
            _mm512_madd_epi16(rg_odd, pw_f0299_f0337),                   \
            _mm512_madd_epi16(bg_odd, pw_f0114_f0250));                  \
        y_odd = _mm512_srli_epi32(                                        \
            _mm512_add_epi32(y_odd, pd_onehalf), SCALEBITS);             \
        __m512i yt_lo = _mm512_packs_epi32(y_even, y_odd);               \
                                                                          \
        rg_even = _mm512_unpacklo_epi16(rw_hi, gw_hi);                   \
        rg_odd  = _mm512_unpackhi_epi16(rw_hi, gw_hi);                   \
        bg_even = _mm512_unpacklo_epi16(bw_hi, gw_hi);                   \
        bg_odd  = _mm512_unpackhi_epi16(bw_hi, gw_hi);                   \
                                                                          \
        y_even = _mm512_add_epi32(                                        \
            _mm512_madd_epi16(rg_even, pw_f0299_f0337),                  \
            _mm512_madd_epi16(bg_even, pw_f0114_f0250));                 \
        y_even = _mm512_srli_epi32(                                       \
            _mm512_add_epi32(y_even, pd_onehalf), SCALEBITS);            \
        y_odd = _mm512_add_epi32(                                         \
            _mm512_madd_epi16(rg_odd, pw_f0299_f0337),                   \
            _mm512_madd_epi16(bg_odd, pw_f0114_f0250));                  \
        y_odd = _mm512_srli_epi32(                                        \
            _mm512_add_epi32(y_odd, pd_onehalf), SCALEBITS);             \
        __m512i yt_hi = _mm512_packs_epi32(y_even, y_odd);               \
                                                                          \
        /* Cb */                                                          \
        rg_even = _mm512_unpacklo_epi16(rw_lo, gw_lo);                   \
        rg_odd  = _mm512_unpackhi_epi16(rw_lo, gw_lo);                   \
        __m512i bwle32 = _mm512_unpacklo_epi16(bw_lo, zero);             \
        __m512i bwlo32 = _mm512_unpackhi_epi16(bw_lo, zero);             \
                                                                          \
        cb_even = _mm512_add_epi32(                                       \
            _mm512_madd_epi16(rg_even, pw_mf016_mf033),                  \
            _mm512_slli_epi32(bwle32, 15));                              \
        cb_even = _mm512_srli_epi32(                                      \
            _mm512_add_epi32(cb_even, pd_onehalfm1_cj), SCALEBITS);     \
        cb_odd = _mm512_add_epi32(                                        \
            _mm512_madd_epi16(rg_odd, pw_mf016_mf033),                   \
            _mm512_slli_epi32(bwlo32, 15));                              \
        cb_odd = _mm512_srli_epi32(                                       \
            _mm512_add_epi32(cb_odd, pd_onehalfm1_cj), SCALEBITS);      \
        __m512i cbt_lo = _mm512_packs_epi32(cb_even, cb_odd);            \
                                                                          \
        rg_even = _mm512_unpacklo_epi16(rw_hi, gw_hi);                   \
        rg_odd  = _mm512_unpackhi_epi16(rw_hi, gw_hi);                   \
        __m512i bwhe32 = _mm512_unpacklo_epi16(bw_hi, zero);             \
        __m512i bwho32 = _mm512_unpackhi_epi16(bw_hi, zero);             \
                                                                          \
        cb_even = _mm512_add_epi32(                                       \
            _mm512_madd_epi16(rg_even, pw_mf016_mf033),                  \
            _mm512_slli_epi32(bwhe32, 15));                              \
        cb_even = _mm512_srli_epi32(                                      \
            _mm512_add_epi32(cb_even, pd_onehalfm1_cj), SCALEBITS);     \
        cb_odd = _mm512_add_epi32(                                        \
            _mm512_madd_epi16(rg_odd, pw_mf016_mf033),                   \
            _mm512_slli_epi32(bwho32, 15));                              \
        cb_odd = _mm512_srli_epi32(                                       \
            _mm512_add_epi32(cb_odd, pd_onehalfm1_cj), SCALEBITS);      \
        __m512i cbt_hi = _mm512_packs_epi32(cb_even, cb_odd);            \
                                                                          \
        /* Cr -- pair (B, G) to match pw_mf008_mf041 = (-0.081, -0.418) */ \
        __m512i gble = _mm512_unpacklo_epi16(bw_lo, gw_lo);              \
        __m512i gblo = _mm512_unpackhi_epi16(bw_lo, gw_lo);              \
        __m512i rwle32 = _mm512_unpacklo_epi16(rw_lo, zero);             \
        __m512i rwlo32 = _mm512_unpackhi_epi16(rw_lo, zero);             \
                                                                          \
        cr_even = _mm512_add_epi32(                                       \
            _mm512_madd_epi16(gble, pw_mf008_mf041),                     \
            _mm512_slli_epi32(rwle32, 15));                              \
        cr_even = _mm512_srli_epi32(                                      \
            _mm512_add_epi32(cr_even, pd_onehalfm1_cj), SCALEBITS);     \
        cr_odd = _mm512_add_epi32(                                        \
            _mm512_madd_epi16(gblo, pw_mf008_mf041),                     \
            _mm512_slli_epi32(rwlo32, 15));                              \
        cr_odd = _mm512_srli_epi32(                                       \
            _mm512_add_epi32(cr_odd, pd_onehalfm1_cj), SCALEBITS);      \
        __m512i crt_lo = _mm512_packs_epi32(cr_even, cr_odd);            \
                                                                          \
        __m512i gbhe = _mm512_unpacklo_epi16(bw_hi, gw_hi);              \
        __m512i gbho = _mm512_unpackhi_epi16(bw_hi, gw_hi);              \
        __m512i rwhe32 = _mm512_unpacklo_epi16(rw_hi, zero);             \
        __m512i rwho32 = _mm512_unpackhi_epi16(rw_hi, zero);             \
                                                                          \
        cr_even = _mm512_add_epi32(                                       \
            _mm512_madd_epi16(gbhe, pw_mf008_mf041),                     \
            _mm512_slli_epi32(rwhe32, 15));                              \
        cr_even = _mm512_srli_epi32(                                      \
            _mm512_add_epi32(cr_even, pd_onehalfm1_cj), SCALEBITS);     \
        cr_odd = _mm512_add_epi32(                                        \
            _mm512_madd_epi16(gbho, pw_mf008_mf041),                     \
            _mm512_slli_epi32(rwho32, 15));                              \
        cr_odd = _mm512_srli_epi32(                                       \
            _mm512_add_epi32(cr_odd, pd_onehalfm1_cj), SCALEBITS);      \
        __m512i crt_hi = _mm512_packs_epi32(cr_even, cr_odd);            \
                                                                          \
        /* Pack and fix lane ordering */                                  \
        __m512i yt_packed  = _mm512_packus_epi16(yt_lo,  yt_hi);         \
        __m512i cbt_packed = _mm512_packus_epi16(cbt_lo, cbt_hi);        \
        __m512i crt_packed = _mm512_packus_epi16(crt_lo, crt_hi);        \
                                                                          \
        const __m512i tfix_perm = _mm512_setr_epi64(0, 2, 4, 6,          \
                                                    1, 3, 5, 7);         \
        yt_packed  = _mm512_permutexvar_epi64(tfix_perm, yt_packed);      \
        cbt_packed = _mm512_permutexvar_epi64(tfix_perm, cbt_packed);     \
        crt_packed = _mm512_permutexvar_epi64(tfix_perm, crt_packed);     \
                                                                          \
        /* Masked store of remaining bytes */                             \
        _mm512_mask_storeu_epi8(outptr0 + col, store_mask, yt_packed);    \
        _mm512_mask_storeu_epi8(outptr1 + col, store_mask, cbt_packed);   \
        _mm512_mask_storeu_epi8(outptr2 + col, store_mask, crt_packed);   \
                                                                          \
        col += remaining;                                                 \
      }                                                                   \
    }                                                                     \
  }                                                                       \
}


/*
 * Permute index generation macros.
 *
 * For 3-byte pixels, we load 96 bytes into two ZMM registers using
 * masked loads, then use _mm512_permutex2var_epi8 to extract each channel.
 *
 * For 4-byte pixels, we load 128 bytes into two ZMM registers, then
 * use _mm512_permutex2var_epi8 to extract each channel from both.
 */

/* Helper to build a single permute index entry for pixel i, channel offset ch,
 * with pixel_size stride.  Returns the byte index into the concatenated
 * {src1, src2} space used by _mm512_permutex2var_epi8.
 */
#define PERM_IDX(i, ch, ps)  ((unsigned char)((i) * (ps) + (ch)))

/* Generate the 64-byte index vector for extracting channel 'ch' from
 * 32 pixels with stride 'ps'.  Indices for pixels 0..31 go into
 * bytes 0..31 of the vector.  Bytes 32..63 are set to 0 (don't-care,
 * since we only use the first 32 result bytes after pack).
 */
#define MAKE_PERM_VEC(ch, ps) \
  _mm512_set_epi8(                                                        \
    /* Bytes 63..32 (don't-care, set to 0) */                             \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     \
    /* Bytes 31..0: pixel 31 down to pixel 0 */                           \
    PERM_IDX(31, ch, ps), PERM_IDX(30, ch, ps),                           \
    PERM_IDX(29, ch, ps), PERM_IDX(28, ch, ps),                           \
    PERM_IDX(27, ch, ps), PERM_IDX(26, ch, ps),                           \
    PERM_IDX(25, ch, ps), PERM_IDX(24, ch, ps),                           \
    PERM_IDX(23, ch, ps), PERM_IDX(22, ch, ps),                           \
    PERM_IDX(21, ch, ps), PERM_IDX(20, ch, ps),                           \
    PERM_IDX(19, ch, ps), PERM_IDX(18, ch, ps),                           \
    PERM_IDX(17, ch, ps), PERM_IDX(16, ch, ps),                           \
    PERM_IDX(15, ch, ps), PERM_IDX(14, ch, ps),                           \
    PERM_IDX(13, ch, ps), PERM_IDX(12, ch, ps),                           \
    PERM_IDX(11, ch, ps), PERM_IDX(10, ch, ps),                           \
    PERM_IDX( 9, ch, ps), PERM_IDX( 8, ch, ps),                           \
    PERM_IDX( 7, ch, ps), PERM_IDX( 6, ch, ps),                           \
    PERM_IDX( 5, ch, ps), PERM_IDX( 4, ch, ps),                           \
    PERM_IDX( 3, ch, ps), PERM_IDX( 2, ch, ps),                           \
    PERM_IDX( 1, ch, ps), PERM_IDX( 0, ch, ps)                            \
  )


/*
 * GENERATE_PERM_INDICES: build permute index vectors for the three channels.
 *
 * For both 3-byte and 4-byte pixel formats, the approach is the same:
 * we generate a permute index vector that, when applied to the loaded
 * pixel data, extracts 32 bytes of a single channel.
 *
 * For 3-byte pixels (96 bytes total):
 *   Bytes [0..63] in zmm_lo, bytes [64..95] in zmm_hi.
 *   Use _mm512_permutex2var_epi8(zmm_lo, perm, zmm_hi) which selects
 *   from the concatenated 128-byte space (bit 6 of index selects source).
 *
 * For 4-byte pixels (128 bytes total):
 *   Same approach: bytes [0..63] in zmm_lo, bytes [64..127] in zmm_hi.
 *   _mm512_permutex2var_epi8 handles the full 128-byte space.
 */
#define GENERATE_PERM_INDICES(R_IDX, G_IDX, B_IDX, PS, pr, pg, pb) \
  pr = MAKE_PERM_VEC(R_IDX, PS);                                         \
  pg = MAKE_PERM_VEC(G_IDX, PS);                                         \
  pb = MAKE_PERM_VEC(B_IDX, PS);


/*
 * DEINTERLEAVE_LOAD: Load 32 pixels of interleaved RGB data and
 * deinterleave into separate R, G, B byte vectors.
 *
 * For 3-byte pixels: load 96 bytes using two loads (64 + 32).
 * For 4-byte pixels: load 128 bytes using two 64-byte loads.
 *
 * After permutex2var, each output register has 32 bytes of one channel
 * in positions [0..31], with positions [32..63] being don't-care.
 *
 * We then rearrange so that bytes [0..15] are in the low 256-bit half
 * and bytes [16..31] are in the high 256-bit half, suitable for
 * _mm512_cvtepu8_epi16 which operates on the low 256 bits.
 */
#define DEINTERLEAVE_LOAD(inptr, PS, pr, pg, pb, r_out, g_out, b_out)     \
  {                                                                       \
    __m512i src_lo, src_hi;                                               \
    if ((PS) == 3) {                                                      \
      /* Load 96 bytes: 64 + 32 */                                        \
      src_lo = _mm512_loadu_si512((const __m512i *)(inptr));              \
      src_hi = _mm512_castsi256_si512(                                    \
          _mm256_loadu_si256((const __m256i *)((inptr) + 64)));           \
    } else {                                                              \
      /* Load 128 bytes: 64 + 64 */                                       \
      src_lo = _mm512_loadu_si512((const __m512i *)(inptr));              \
      src_hi = _mm512_loadu_si512((const __m512i *)((inptr) + 64));       \
    }                                                                     \
    /* Extract each channel: 32 bytes in positions [0..31] */             \
    __m512i r_raw = _mm512_permutex2var_epi8(src_lo, pr, src_hi);         \
    __m512i g_raw = _mm512_permutex2var_epi8(src_lo, pg, src_hi);         \
    __m512i b_raw = _mm512_permutex2var_epi8(src_lo, pb, src_hi);         \
    /* r_raw has 32 channel bytes in the low 256 bits (bytes 0..31).      \
     * Split into two halves for cvtepu8_epi16:                           \
     *   low 256 bits  = bytes 0..15  (for pixels 0..15)                  \
     *   high 256 bits = bytes 16..31 (for pixels 16..31)                 \
     * Bytes 0..15 are already in the low 128 bits of the low 256-bit     \
     * lane. Bytes 16..31 are in the high 128 bits of the low 256-bit     \
     * lane. We need to move bytes 16..31 into the low 128 bits of        \
     * the high 256-bit lane.                                             \
     *                                                                    \
     * Use vpermq to rearrange: qword layout after permutex2var:          \
     *   q0=[0..7], q1=[8..15], q2=[16..23], q3=[24..31],                \
     *   q4=[32..39], q5=[40..47], q6=[48..55], q7=[56..63]              \
     * We want: q0,q1 in low ymm, q2,q3 in high ymm.                     \
     * That is: rearrange to q0,q1,dc,dc, q2,q3,dc,dc.                   \
     * But _mm512_cvtepu8_epi16 takes a __m256i and zero-extends the      \
     * low 128 bits. So we need:                                          \
     *   low half:  __m256i with bytes[0..15] in low 128 bits             \
     *   high half: __m256i with bytes[16..31] in low 128 bits            \
     * The low __m256i = _mm512_castsi512_si256(r_raw) has q0,q1 which    \
     * is bytes[0..15] -- perfect.                                        \
     * The high __m256i needs bytes[16..31] in its low 128 bits.          \
     * bytes[16..31] = q2,q3 which are at offset 128 bits in r_raw.      \
     * _mm512_extracti64x4_epi64(r_raw, 0) gets the low 256 bits,        \
     * then we need the high 128 of that. Or just shift:                  \
     * __m256i hi = _mm256_extracti128_si256(lo256, 1) cast to __m256i.   \
     *                                                                    \
     * Actually simpler: just store the full 256-bit low half and         \
     * extract bytes [16..31] using _mm512_extracti32x4_epi32.            \
     */                                                                   \
    /* Re-pack: low 256 bits has bytes 0..31. Split at byte 16. */        \
    __m256i r_lo256_raw = _mm512_castsi512_si256(r_raw);                  \
    __m256i g_lo256_raw = _mm512_castsi512_si256(g_raw);                  \
    __m256i b_lo256_raw = _mm512_castsi512_si256(b_raw);                  \
    /* For cvtepu8_epi16, we need __m256i input where low 128 bits        \
     * contain the 16 bytes to extend.                                    \
     * lo256_raw has bytes[0..15] in low 128, bytes[16..31] in high 128.  \
     * For pixels 0..15: just pass lo256_raw (cvt uses low 128 bits).     \
     * For pixels 16..31: need bytes[16..31] in low 128 bits.             \
     */                                                                   \
    __m128i r_hi128 = _mm256_extracti128_si256(r_lo256_raw, 1);           \
    __m128i g_hi128 = _mm256_extracti128_si256(g_lo256_raw, 1);           \
    __m128i b_hi128 = _mm256_extracti128_si256(b_lo256_raw, 1);           \
    r_out = _mm512_inserti64x4(                                           \
        _mm512_castsi256_si512(r_lo256_raw),                              \
        _mm256_castsi128_si256(r_hi128), 1);                              \
    g_out = _mm512_inserti64x4(                                           \
        _mm512_castsi256_si512(g_lo256_raw),                              \
        _mm256_castsi128_si256(g_hi128), 1);                              \
    b_out = _mm512_inserti64x4(                                           \
        _mm512_castsi256_si512(b_lo256_raw),                              \
        _mm256_castsi128_si256(b_hi128), 1);                              \
  }


/*
 * DEINTERLEAVE_LOAD_MASKED: Same as DEINTERLEAVE_LOAD but with masked
 * loads for the tail case (fewer than 32 pixels remaining).
 */
#define DEINTERLEAVE_LOAD_MASKED(inptr, PS, pr, pg, pb,                   \
                                 load_bytes, r_out, g_out, b_out)         \
  {                                                                       \
    __m512i src_lo, src_hi;                                               \
    if ((PS) == 3) {                                                      \
      /* Up to 93 bytes (31 pixels * 3). Load up to 64 from first reg,    \
       * remainder from second. */                                        \
      if (load_bytes <= 64) {                                             \
        __mmask64 mask1 = (__mmask64)((load_bytes >= 64) ?                \
            0xFFFFFFFFFFFFFFFFULL : ((1ULL << load_bytes) - 1));          \
        src_lo = _mm512_maskz_loadu_epi8(mask1,                           \
                                         (const __m512i *)(inptr));       \
        src_hi = _mm512_setzero_si512();                                  \
      } else {                                                            \
        src_lo = _mm512_loadu_si512((const __m512i *)(inptr));            \
        JDIMENSION rem = load_bytes - 64;                                 \
        __mmask64 mask2 = (__mmask64)((1ULL << rem) - 1);                 \
        src_hi = _mm512_maskz_loadu_epi8(mask2,                           \
                                         (const __m512i *)((inptr) + 64));\
      }                                                                   \
    } else {                                                              \
      /* Up to 124 bytes (31 pixels * 4). */                              \
      if (load_bytes <= 64) {                                             \
        __mmask64 mask1 = (__mmask64)((load_bytes >= 64) ?                \
            0xFFFFFFFFFFFFFFFFULL : ((1ULL << load_bytes) - 1));          \
        src_lo = _mm512_maskz_loadu_epi8(mask1,                           \
                                         (const __m512i *)(inptr));       \
        src_hi = _mm512_setzero_si512();                                  \
      } else {                                                            \
        src_lo = _mm512_loadu_si512((const __m512i *)(inptr));            \
        JDIMENSION rem = load_bytes - 64;                                 \
        __mmask64 mask2 = (__mmask64)((1ULL << rem) - 1);                 \
        src_hi = _mm512_maskz_loadu_epi8(mask2,                           \
                                         (const __m512i *)((inptr) + 64));\
      }                                                                   \
    }                                                                     \
    __m512i r_raw = _mm512_permutex2var_epi8(src_lo, pr, src_hi);         \
    __m512i g_raw = _mm512_permutex2var_epi8(src_lo, pg, src_hi);         \
    __m512i b_raw = _mm512_permutex2var_epi8(src_lo, pb, src_hi);         \
    __m256i r_lo256_raw = _mm512_castsi512_si256(r_raw);                  \
    __m256i g_lo256_raw = _mm512_castsi512_si256(g_raw);                  \
    __m256i b_lo256_raw = _mm512_castsi512_si256(b_raw);                  \
    __m128i r_hi128 = _mm256_extracti128_si256(r_lo256_raw, 1);           \
    __m128i g_hi128 = _mm256_extracti128_si256(g_lo256_raw, 1);           \
    __m128i b_hi128 = _mm256_extracti128_si256(b_lo256_raw, 1);           \
    r_out = _mm512_inserti64x4(                                           \
        _mm512_castsi256_si512(r_lo256_raw),                              \
        _mm256_castsi128_si256(r_hi128), 1);                              \
    g_out = _mm512_inserti64x4(                                           \
        _mm512_castsi256_si512(g_lo256_raw),                              \
        _mm256_castsi128_si256(g_hi128), 1);                              \
    b_out = _mm512_inserti64x4(                                           \
        _mm512_castsi256_si512(b_lo256_raw),                              \
        _mm256_castsi128_si256(b_hi128), 1);                              \
  }


/* -------------------------------------------------------------------- */
/* Generate the 7 RGB->YCbCr conversion functions.                      */
/*                                                                      */
/* The default RGB format uses the same indices as EXT_RGB.              */
/* -------------------------------------------------------------------- */

/* RGB (default, same as EXT_RGB): R=0, G=1, B=2, pixelsize=3 */
GENERATE_YCC_CONVERT(jsimd_rgb_ycc_convert_avx512,
                     EXT_RGB_RED, EXT_RGB_GREEN, EXT_RGB_BLUE,
                     EXT_RGB_PIXELSIZE)

/* EXT_RGB: R=0, G=1, B=2, pixelsize=3 */
GENERATE_YCC_CONVERT(jsimd_extrgb_ycc_convert_avx512,
                     EXT_RGB_RED, EXT_RGB_GREEN, EXT_RGB_BLUE,
                     EXT_RGB_PIXELSIZE)

/* EXT_RGBX: R=0, G=1, B=2, pixelsize=4 */
GENERATE_YCC_CONVERT(jsimd_extrgbx_ycc_convert_avx512,
                     EXT_RGBX_RED, EXT_RGBX_GREEN, EXT_RGBX_BLUE,
                     EXT_RGBX_PIXELSIZE)

/* EXT_BGR: R=2, G=1, B=0, pixelsize=3 */
GENERATE_YCC_CONVERT(jsimd_extbgr_ycc_convert_avx512,
                     EXT_BGR_RED, EXT_BGR_GREEN, EXT_BGR_BLUE,
                     EXT_BGR_PIXELSIZE)

/* EXT_BGRX: R=2, G=1, B=0, pixelsize=4 */
GENERATE_YCC_CONVERT(jsimd_extbgrx_ycc_convert_avx512,
                     EXT_BGRX_RED, EXT_BGRX_GREEN, EXT_BGRX_BLUE,
                     EXT_BGRX_PIXELSIZE)

/* EXT_XBGR: R=3, G=2, B=1, pixelsize=4 */
GENERATE_YCC_CONVERT(jsimd_extxbgr_ycc_convert_avx512,
                     EXT_XBGR_RED, EXT_XBGR_GREEN, EXT_XBGR_BLUE,
                     EXT_XBGR_PIXELSIZE)

/* EXT_XRGB: R=1, G=2, B=3, pixelsize=4 */
GENERATE_YCC_CONVERT(jsimd_extxrgb_ycc_convert_avx512,
                     EXT_XRGB_RED, EXT_XRGB_GREEN, EXT_XRGB_BLUE,
                     EXT_XRGB_PIXELSIZE)
