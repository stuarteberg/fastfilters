// fastfilters
// Copyright (c) 2016 Sven Peter
// sven.peter@iwr.uni-heidelberg.de or mail@svenpeter.me
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// based on vigra's include/gaussian.gxx with the following license:
/************************************************************************/
/*                                                                      */
/*               Copyright 1998-2004 by Ullrich Koethe                  */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#include "fastfilters.h"
#include "common.h"

static convolve_fn_t g_convolve_inner = NULL;
static convolve_fn_t g_convolve_outer = NULL;

void fastfilters_fir_init(void)
{
    if (fastfilters_cpu_check(FASTFILTERS_CPU_FMA)) {
        g_convolve_outer = &fastfilters_fir_convolve_fir_outer_avxfma;
        g_convolve_inner = &fastfilters_fir_convolve_fir_inner_avxfma;
    } else if (fastfilters_cpu_check(FASTFILTERS_CPU_AVX)) {
        g_convolve_outer = &fastfilters_fir_convolve_fir_outer_avx;
        g_convolve_inner = &fastfilters_fir_convolve_fir_inner_avx;
    } else {
        g_convolve_outer = &fastfilters_fir_convolve_fir_outer;
        g_convolve_inner = &fastfilters_fir_convolve_fir_inner;
    }
}

fastfilters_kernel_t DLL_PUBLIC fastfilters_kernel_fir_gaussian(unsigned int order, double sigma, float window_ratio)
{
    double norm;
    double sigma2 = -0.5 / sigma / sigma;

    if (order > 2)
        return NULL;

    if (sigma < 0)
        return NULL;

    fastfilters_kernel_t kernel = fastfilters_memory_alloc(sizeof(struct _fastfilters_kernel_t));
    if (!kernel)
        goto out;

    kernel->iir = NULL;
    kernel->fir = fastfilters_memory_alloc(sizeof(struct _fastfilters_kernel_fir_t));
    if (!kernel->fir)
        goto out;

    kernel->convolve_inner = g_convolve_inner;
    kernel->convolve_outer = g_convolve_outer;

    if (window_ratio > 0)
        kernel->fir->len = floor(window_ratio * sigma + 0.5);
    else
        kernel->fir->len = floor(3.0 * sigma + 0.5 * order + 0.5); // FIXME: this is the formula currently used by vigra
    // kernel->fir->len = ceil((3.0 + 0.5 * (double)order) * sigma);

    if (fabs(sigma) < 1e-6)
        kernel->fir->len = 0;

    kernel->fir->coefs = fastfilters_memory_alloc(sizeof(float) * (kernel->fir->len + 1));
    if (!kernel->fir->coefs)
        goto out;

    if (order == 1)
        kernel->fir->is_symmetric = false;
    else
        kernel->fir->is_symmetric = true;

    switch (order) {
    case 1:
    case 2:
        norm = -1.0 / (sqrt(2.0 * M_PI) * pow(sigma, 3));
        break;

    default:
        norm = 1.0 / (sqrt(2.0 * M_PI) * sigma);
        break;
    }

    for (unsigned int x = 0; x <= kernel->fir->len; ++x) {
        double x2 = x * x;
        double g = norm * exp(x2 * sigma2);
        switch (order) {
        case 0:
            kernel->fir->coefs[x] = g;
            break;
        case 1:
            kernel->fir->coefs[x] = x * g;
            break;
        case 2:
            kernel->fir->coefs[x] = (1.0 - (x / sigma) * (x / sigma)) * g;
            break;
        }
    }

    if (order == 2) {
        double dc = kernel->fir->coefs[0];
        for (unsigned int x = 1; x <= kernel->fir->len; ++x)
            dc += 2 * kernel->fir->coefs[x];
        dc /= (2.0 * (double)kernel->fir->len + 1.0);

        for (unsigned int x = 0; x <= kernel->fir->len; ++x)
            kernel->fir->coefs[x] -= dc;
    }

    double sum = 0.0;
    if (order == 0) {
        sum = kernel->fir->coefs[0];
        for (unsigned int x = 1; x <= kernel->fir->len; ++x)
            sum += 2 * kernel->fir->coefs[x];
    } else {
        unsigned int faculty = 1;

        int sign;

        if (kernel->fir->is_symmetric)
            sign = 1;
        else
            sign = -1;

        for (unsigned int i = 2; i <= order; ++i)
            faculty *= i;

        sum = 0.0;
        for (unsigned int x = 1; x <= kernel->fir->len; ++x) {
            sum += kernel->fir->coefs[x] * pow(-(double)x, (int)order) / (double)faculty;
            sum += sign * kernel->fir->coefs[x] * pow((double)x, (int)order) / (double)faculty;
        }
    }

    for (unsigned int x = 0; x <= kernel->fir->len; ++x)
        kernel->fir->coefs[x] /= sum;

    if (!kernel->fir->is_symmetric)
        for (unsigned int x = 0; x <= kernel->fir->len; ++x)
            kernel->fir->coefs[x] *= -1;

    kernel->fir->fn_inner_mirror = NULL;
    kernel->fir->fn_inner_ptr = NULL;
    kernel->fir->fn_inner_optimistic = NULL;
    kernel->fir->fn_outer_mirror = NULL;
    kernel->fir->fn_outer_ptr = NULL;
    kernel->fir->fn_outer_optimistic = NULL;

    return kernel;

out:
    if (kernel->fir)
        fastfilters_memory_free(kernel->fir->coefs);
    if (kernel)
        fastfilters_memory_free(kernel->fir);
    fastfilters_memory_free(kernel);

    return NULL;
}

void DLL_PUBLIC fastfilters_kernel_fir_free(fastfilters_kernel_t kernel)
{
    if (kernel->fir)
        fastfilters_memory_free(kernel->fir->coefs);
    if (kernel)
        fastfilters_memory_free(kernel->fir);
    fastfilters_memory_free(kernel);
}

unsigned int DLL_PUBLIC fastfilters_kernel_fir_get_length(fastfilters_kernel_t kernel)
{
    return kernel->fir->len;
}