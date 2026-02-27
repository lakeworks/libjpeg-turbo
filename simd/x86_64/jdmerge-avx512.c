/*
 * Merged upsampling/color conversion (64-bit AVX-512BW)
 *
 * Copyright (C) 2025, Lakeworks.
 *
 * AVX-512BW/VBMI C intrinsics implementation of merged chroma upsampling +
 * YCbCr -> RGB color conversion.  Processes 32 output pixels (= 16 chroma
 * samples) per iteration using 512-bit ZMM registers.
 *
 * Based on the AVX2 NASM implementation:
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009, 2012, 2016, 2024-2025, D. R. Commander.
 * Copyright (C) 2015, Intel Corporation.
 * Copyright (C) 2018, Matthias Raencker.
 * Copyright (C) 2023, Aliaksiej Kandracienka.
 *
 * And the Arm Neon C intrinsics implementation:
 * Copyright (C) 2020, Arm Limited.
 * Copyright (C) 2024, D. R. Commander.
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

#define JPEG_INTERNALS
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
 * YCbCr -> RGB conversion using the same algorithm as the AVX2 NASM
 * implementation (jdmrgext-avx2.asm), scaled up to 512-bit vectors.
 *
 * The AVX2 code uses an alternate fixed-point form:
 *   R = Y + 0.40200 * Cr' + Cr'          [= Y + 1.40200 * Cr']
 *   G = Y - 0.34414 * Cb' + 0.28586 * Cr' - Cr'
 *                                         [= Y - 0.34414 * Cb' - 0.71414 * Cr']
 *   B = Y - 0.22800 * Cb' + Cb' + Cb'    [= Y + 1.77200 * Cb']
 *
 * where Cb' = Cb - 128, Cr' = Cr - 128.
 *
 * Constants (16-bit signed, for use with vpmulhw which computes
 * (a * b) >> 16 for signed operands):
 *   F_0_402 =  26345 = round(0.40200 * 65536)
 *   F_0_228 =  14942 = round(0.22800 * 65536)
 *   F_0_344 =  22554 = round(0.34414 * 65536)
 *   F_0_285 =  18734 = round(0.28586 * 65536)
 */

#define SCALEBITS  16
#define ONE_HALF   ((JLONG)1 << (SCALEBITS - 1))

#define F_0_344  ((short)22554)   /* FIX(0.34414) */
#define F_0_402  ((short)26345)   /* FIX(0.40200) */
#define F_0_285  ((short)18734)   /* FIX(0.28586) = FIX(1 - 0.71414) */
#define F_0_228  ((short)14942)   /* FIX(0.22800) = FIX(2 - 1.77200) */

#define RGBX_FILLER_0XFF  1


/* Include inline routines for colorspace extensions. */

#include "jdmrgext-avx512.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE

#define RGB_RED  EXT_RGB_RED
#define RGB_GREEN  EXT_RGB_GREEN
#define RGB_BLUE  EXT_RGB_BLUE
#define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
#define jsimd_h2v1_merged_upsample_avx512 \
        jsimd_h2v1_extrgb_merged_upsample_avx512
#define jsimd_h2v2_merged_upsample_avx512 \
        jsimd_h2v2_extrgb_merged_upsample_avx512
#include "jdmrgext-avx512.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_h2v1_merged_upsample_avx512
#undef jsimd_h2v2_merged_upsample_avx512

#define RGB_RED  EXT_RGBX_RED
#define RGB_GREEN  EXT_RGBX_GREEN
#define RGB_BLUE  EXT_RGBX_BLUE
#define RGB_ALPHA  3
#define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
#define jsimd_h2v1_merged_upsample_avx512 \
        jsimd_h2v1_extrgbx_merged_upsample_avx512
#define jsimd_h2v2_merged_upsample_avx512 \
        jsimd_h2v2_extrgbx_merged_upsample_avx512
#include "jdmrgext-avx512.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef jsimd_h2v1_merged_upsample_avx512
#undef jsimd_h2v2_merged_upsample_avx512

#define RGB_RED  EXT_BGR_RED
#define RGB_GREEN  EXT_BGR_GREEN
#define RGB_BLUE  EXT_BGR_BLUE
#define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
#define jsimd_h2v1_merged_upsample_avx512 \
        jsimd_h2v1_extbgr_merged_upsample_avx512
#define jsimd_h2v2_merged_upsample_avx512 \
        jsimd_h2v2_extbgr_merged_upsample_avx512
#include "jdmrgext-avx512.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_h2v1_merged_upsample_avx512
#undef jsimd_h2v2_merged_upsample_avx512

#define RGB_RED  EXT_BGRX_RED
#define RGB_GREEN  EXT_BGRX_GREEN
#define RGB_BLUE  EXT_BGRX_BLUE
#define RGB_ALPHA  3
#define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
#define jsimd_h2v1_merged_upsample_avx512 \
        jsimd_h2v1_extbgrx_merged_upsample_avx512
#define jsimd_h2v2_merged_upsample_avx512 \
        jsimd_h2v2_extbgrx_merged_upsample_avx512
#include "jdmrgext-avx512.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef jsimd_h2v1_merged_upsample_avx512
#undef jsimd_h2v2_merged_upsample_avx512

#define RGB_RED  EXT_XBGR_RED
#define RGB_GREEN  EXT_XBGR_GREEN
#define RGB_BLUE  EXT_XBGR_BLUE
#define RGB_ALPHA  0
#define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
#define jsimd_h2v1_merged_upsample_avx512 \
        jsimd_h2v1_extxbgr_merged_upsample_avx512
#define jsimd_h2v2_merged_upsample_avx512 \
        jsimd_h2v2_extxbgr_merged_upsample_avx512
#include "jdmrgext-avx512.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef jsimd_h2v1_merged_upsample_avx512
#undef jsimd_h2v2_merged_upsample_avx512

#define RGB_RED  EXT_XRGB_RED
#define RGB_GREEN  EXT_XRGB_GREEN
#define RGB_BLUE  EXT_XRGB_BLUE
#define RGB_ALPHA  0
#define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
#define jsimd_h2v1_merged_upsample_avx512 \
        jsimd_h2v1_extxrgb_merged_upsample_avx512
#define jsimd_h2v2_merged_upsample_avx512 \
        jsimd_h2v2_extxrgb_merged_upsample_avx512
#include "jdmrgext-avx512.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef jsimd_h2v1_merged_upsample_avx512
#undef jsimd_h2v2_merged_upsample_avx512
