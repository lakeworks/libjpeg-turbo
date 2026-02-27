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

#define JPEG_INTERNALS
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
 * Byte deinterleaving via _mm512_permutex2var_epi8 (AVX-512 VBMI).
 *
 * Pixel data is loaded into two ZMM registers (src_lo, src_hi):
 *   - 3-byte pixels: 96 bytes total -> src_lo[0..63], src_hi[0..31]
 *   - 4-byte pixels: 128 bytes total -> src_lo[0..63], src_hi[0..63]
 *
 * _mm512_permutex2var_epi8(src_lo, idx, src_hi) selects from the
 * concatenated 128-byte space using a 7-bit index per output byte
 * (bit 6 selects src_lo vs src_hi, bits 5:0 select byte within).
 *
 * For pixel i (0..31), the R/G/B byte is at byte offset (i * pixelsize + ch).
 * The MAKE_PERM_VEC macro generates the 64-byte index vector at compile time.
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
      int is_tail = (remaining < 32);                                     \
      __mmask64 store_mask = 0;                                           \
      __m512i r_bytes, g_bytes, b_bytes;                                  \
                                                                          \
      if (!is_tail) {                                                     \
        DEINTERLEAVE_LOAD(inptr, PIXEL_SIZE,                              \
                          perm_r, perm_g, perm_b,                         \
                          r_bytes, g_bytes, b_bytes)                      \
      } else {                                                            \
        JDIMENSION load_bytes = remaining * (JDIMENSION)PIXEL_SIZE;       \
        store_mask = (__mmask64)((1ULL << remaining) - 1);                \
        DEINTERLEAVE_LOAD_MASKED(inptr, PIXEL_SIZE,                       \
                                 perm_r, perm_g, perm_b,                  \
                                 load_bytes,                              \
                                 r_bytes, g_bytes, b_bytes)               \
      }                                                                   \
                                                                          \
      /* After deinterleave, r/g/b_bytes each have 32 channel bytes in    \
       * the low 256 bits. Zero-extend all 32 to 16-bit words in a ZMM.  \
       * _mm512_cvtepu8_epi16 takes a __m256i (32 bytes) and produces    \
       * 32 words across 4 x 128-bit lanes.                              \
       */                                                                 \
      __m256i r_256 = _mm512_castsi512_si256(r_bytes);                    \
      __m256i g_256 = _mm512_castsi512_si256(g_bytes);                    \
      __m256i b_256 = _mm512_castsi512_si256(b_bytes);                    \
                                                                          \
      __m512i rw = _mm512_cvtepu8_epi16(r_256);                          \
      __m512i gw = _mm512_cvtepu8_epi16(g_256);                          \
      __m512i bw = _mm512_cvtepu8_epi16(b_256);                          \
                                                                          \
      /* ---- Compute Y ---- */                                           \
      /* Y = (R*FIX_029900 + G*FIX_033700) + (B*FIX_011400 + G*FIX_025000) \
       *                                                                  \
       * unpacklo/hi_epi16 interleaves within each 128-bit lane:          \
       *   unpacklo(rw, gw) lane_i = (R0,G0, R1,G1, R2,G2, R3,G3)       \
       *   unpackhi(rw, gw) lane_i = (R4,G4, R5,G5, R6,G6, R7,G7)       \
       * Each madd produces 4 dwords per lane = 16 total.                 \
       * packs_epi32(even, odd) per lane = 8 words per lane = 32 total.   \
       */                                                                 \
      __m512i rg_even = _mm512_unpacklo_epi16(rw, gw);                   \
      __m512i rg_odd  = _mm512_unpackhi_epi16(rw, gw);                   \
      __m512i bg_even = _mm512_unpacklo_epi16(bw, gw);                   \
      __m512i bg_odd  = _mm512_unpackhi_epi16(bw, gw);                   \
                                                                          \
      __m512i y_rg_even = _mm512_madd_epi16(rg_even, pw_f0299_f0337);    \
      __m512i y_rg_odd  = _mm512_madd_epi16(rg_odd,  pw_f0299_f0337);   \
      __m512i y_bg_even = _mm512_madd_epi16(bg_even, pw_f0114_f0250);    \
      __m512i y_bg_odd  = _mm512_madd_epi16(bg_odd,  pw_f0114_f0250);   \
                                                                          \
      __m512i y_even = _mm512_add_epi32(y_rg_even, y_bg_even);           \
      y_even = _mm512_add_epi32(y_even, pd_onehalf);                     \
      y_even = _mm512_srli_epi32(y_even, SCALEBITS);                     \
                                                                          \
      __m512i y_odd = _mm512_add_epi32(y_rg_odd, y_bg_odd);              \
      y_odd = _mm512_add_epi32(y_odd, pd_onehalf);                       \
      y_odd = _mm512_srli_epi32(y_odd, SCALEBITS);                       \
                                                                          \
      /* packs_epi32 produces 32 signed 16-bit words in [0..255]. */      \
      __m512i y_words = _mm512_packs_epi32(y_even, y_odd);               \
                                                                          \
      /* ---- Compute Cb ---- */                                          \
      /* Cb = (-FIX_016874*R + -FIX_033126*G) + FIX_050000*B + bias */    \
      __m512i cb_rg_even = _mm512_madd_epi16(rg_even, pw_mf016_mf033);   \
      __m512i cb_rg_odd  = _mm512_madd_epi16(rg_odd,  pw_mf016_mf033);  \
                                                                          \
      /* B * FIX(0.500) = B << 15.  Need B as 32-bit first. */            \
      __m512i bw_even32 = _mm512_unpacklo_epi16(bw, zero);               \
      __m512i bw_odd32  = _mm512_unpackhi_epi16(bw, zero);               \
                                                                          \
      __m512i cb_even = _mm512_add_epi32(cb_rg_even,                      \
                            _mm512_slli_epi32(bw_even32, 15));            \
      cb_even = _mm512_add_epi32(cb_even, pd_onehalfm1_cj);              \
      cb_even = _mm512_srli_epi32(cb_even, SCALEBITS);                   \
                                                                          \
      __m512i cb_odd = _mm512_add_epi32(cb_rg_odd,                        \
                           _mm512_slli_epi32(bw_odd32, 15));              \
      cb_odd = _mm512_add_epi32(cb_odd, pd_onehalfm1_cj);                \
      cb_odd = _mm512_srli_epi32(cb_odd, SCALEBITS);                     \
                                                                          \
      __m512i cb_words = _mm512_packs_epi32(cb_even, cb_odd);             \
                                                                          \
      /* ---- Compute Cr ---- */                                          \
      /* Cr = FIX_050000*R + (-FIX_008131*B + -FIX_041869*G) + bias */    \
      /* Pair (B, G) with (-FIX_008131, -FIX_041869). */                  \
      __m512i bg_even_cr = _mm512_unpacklo_epi16(bw, gw);                \
      __m512i bg_odd_cr  = _mm512_unpackhi_epi16(bw, gw);                \
                                                                          \
      __m512i cr_bg_even = _mm512_madd_epi16(bg_even_cr, pw_mf008_mf041);\
      __m512i cr_bg_odd  = _mm512_madd_epi16(bg_odd_cr,  pw_mf008_mf041);\
                                                                          \
      /* R * FIX(0.500) = R << 15 */                                      \
      __m512i rw_even32 = _mm512_unpacklo_epi16(rw, zero);               \
      __m512i rw_odd32  = _mm512_unpackhi_epi16(rw, zero);               \
                                                                          \
      __m512i cr_even = _mm512_add_epi32(cr_bg_even,                      \
                            _mm512_slli_epi32(rw_even32, 15));            \
      cr_even = _mm512_add_epi32(cr_even, pd_onehalfm1_cj);              \
      cr_even = _mm512_srli_epi32(cr_even, SCALEBITS);                   \
                                                                          \
      __m512i cr_odd = _mm512_add_epi32(cr_bg_odd,                        \
                           _mm512_slli_epi32(rw_odd32, 15));              \
      cr_odd = _mm512_add_epi32(cr_odd, pd_onehalfm1_cj);                \
      cr_odd = _mm512_srli_epi32(cr_odd, SCALEBITS);                     \
                                                                          \
      __m512i cr_words = _mm512_packs_epi32(cr_even, cr_odd);             \
                                                                          \
      /* ---- Pack 16-bit words to 8-bit bytes ---- */                    \
      /* packs_epi32 interleaves within 128-bit lanes:                    \
       *   lane_i = {even[0..3], odd[0..3]} as 16-bit words              \
       * which is correct pixel order (0..7) within each lane.            \
       *                                                                  \
       * Use VPMOVUSWB (_mm512_cvtusepi16_epi8) to narrow 32 words to    \
       * 32 bytes in a __m256i with unsigned saturation to [0..255].      \
       * This safely handles edge-case rounding (e.g. Cb/Cr = 256).      \
       * The output preserves the per-lane ordering of the input.         \
       */                                                                 \
      __m256i y_out  = _mm512_cvtusepi16_epi8(y_words);                   \
      __m256i cb_out = _mm512_cvtusepi16_epi8(cb_words);                 \
      __m256i cr_out = _mm512_cvtusepi16_epi8(cr_words);                   \
                                                                          \
      /* Store Y, Cb, Cr to their respective output planes. */            \
      if (!is_tail) {                                                     \
        _mm256_storeu_si256((__m256i *)(outptr0 + col), y_out);           \
        _mm256_storeu_si256((__m256i *)(outptr1 + col), cb_out);          \
        _mm256_storeu_si256((__m256i *)(outptr2 + col), cr_out);          \
        inptr += 32 * PIXEL_SIZE;                                         \
        col += 32;                                                        \
      } else {                                                            \
        /* Masked store for tail pixels (< 32 remaining). */              \
        /* Widen __m256i to __m512i for _mm512_mask_storeu_epi8. */       \
        _mm512_mask_storeu_epi8(outptr0 + col, store_mask,                \
                                _mm512_castsi256_si512(y_out));           \
        _mm512_mask_storeu_epi8(outptr1 + col, store_mask,                \
                                _mm512_castsi256_si512(cb_out));          \
        _mm512_mask_storeu_epi8(outptr2 + col, store_mask,                \
                                _mm512_castsi256_si512(cr_out));          \
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
 * After permutex2var, each output register has 32 channel bytes in
 * positions [0..31] (the low 256 bits). The caller extracts the low
 * __m256i and passes it to _mm512_cvtepu8_epi16 to get 32 words.
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
    /* Extract each channel: 32 bytes in positions [0..31] of the         \
     * output ZMM register. Bytes [32..63] are don't-care.                \
     * The low 256 bits contain all 32 channel bytes.                     \
     */                                                                   \
    r_out = _mm512_permutex2var_epi8(src_lo, pr, src_hi);                 \
    g_out = _mm512_permutex2var_epi8(src_lo, pg, src_hi);                 \
    b_out = _mm512_permutex2var_epi8(src_lo, pb, src_hi);                 \
  }


/*
 * DEINTERLEAVE_LOAD_MASKED: Same as DEINTERLEAVE_LOAD but with masked
 * loads for the tail case (fewer than 32 pixels remaining).
 */
#define DEINTERLEAVE_LOAD_MASKED(inptr, PS, pr, pg, pb,                   \
                                 load_bytes, r_out, g_out, b_out)         \
  {                                                                       \
    __m512i src_lo, src_hi;                                               \
    if (load_bytes <= 64) {                                               \
      __mmask64 mask1 = (load_bytes >= 64) ?                              \
          (__mmask64)0xFFFFFFFFFFFFFFFFULL :                               \
          (__mmask64)((1ULL << load_bytes) - 1);                          \
      src_lo = _mm512_maskz_loadu_epi8(mask1,                             \
                                       (const __m512i *)(inptr));         \
      src_hi = _mm512_setzero_si512();                                    \
    } else {                                                              \
      src_lo = _mm512_loadu_si512((const __m512i *)(inptr));              \
      JDIMENSION rem = load_bytes - 64;                                   \
      __mmask64 mask2 = (__mmask64)((1ULL << rem) - 1);                   \
      src_hi = _mm512_maskz_loadu_epi8(mask2,                             \
                                       (const __m512i *)((inptr) + 64));  \
    }                                                                     \
    r_out = _mm512_permutex2var_epi8(src_lo, pr, src_hi);                 \
    g_out = _mm512_permutex2var_epi8(src_lo, pg, src_hi);                 \
    b_out = _mm512_permutex2var_epi8(src_lo, pb, src_hi);                 \
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
