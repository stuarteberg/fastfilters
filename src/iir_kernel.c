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

#include "fastfilters.h"
#include "common.h"
#include "iir_kernel.h"
#include "boost/preprocessor/library.hpp"

#include <complex.h>
#include <math.h>

#ifndef M_SQRT2PI
#define M_SQRT2PI 2.50662827463100050241576528481104525
#endif

typedef struct {
    unsigned iir_order;
    unsigned deriv_order;
    double complex alpha[4];
    double complex lambda[4];
} deriche_precomputed_t;

// Rachid Deriche. Recursively implementating the Gaussian and its derivatives.
// [Research Report] RR-1893, 1993, pp.24. <inria-00074778>
// eq. 35 - 46
// 4th order coefficients from The Mines Java Toolkit
// https://github.com/dhale/jtk/blob/master/core/src/main/java/edu/mines/jtk/dsp/RecursiveGaussianFilter.java

static const double g_deriche_2[3][4] = {
    {0.9629, 1.942, 1.26, 0.8448}, {0.1051, -1.898, 0.9338, 0.9459}, {-1.228, 0.2976, 0.6134, 1.332}};

static const double g_deriche_3[3][6] = {{-0.8929, 1.021, 1.898, 1.512, 1.475, 1.556},
                                         {1.666, -0.4557, -1.682, 1.266, 1.558, 1.244},
                                         {-1.773, -1.129, 0.8309, 1.032, 1.664, 0.8459}};

static const double g_deriche_4[3][8] = {
    {1.6797292232361107, 3.7348298269103580, -0.6802783501806897, -0.2598300478959625, 1.7831906544515104,
     0.6318113174569493, 1.7228297663338028, 1.9969276832487770},
    {0.6494024008440620, 0.9557370760729773, -0.6472105276644291, -4.5306923044570760, 1.5159726670750566,
     2.0718953658782650, 1.5267608734791140, 0.6719055957689513},
    {0.3224570510072559, -1.7382843963561239, -1.3312275593739595, 3.6607035671974897, 1.3138054926516880,
     2.1656041357418863, 1.2402181393295362, 0.7479888745408682}};

static inline bool deriche_get_precomputed(deriche_precomputed_t *coefs, unsigned iir_order, unsigned deriv_order)
{
    if (deriv_order > 2)
        return false;

    switch (iir_order) {
    default:
        return false;
        break;

    case 2:
        coefs->alpha[0] = g_deriche_2[deriv_order][0] / 2.0 + g_deriche_2[deriv_order][1] / 2.0 * I;
        coefs->alpha[1] = g_deriche_2[deriv_order][0] / 2.0 - g_deriche_2[deriv_order][1] / 2.0 * I;
        coefs->lambda[0] = g_deriche_2[deriv_order][2] + g_deriche_2[deriv_order][3] * I;
        coefs->lambda[1] = g_deriche_2[deriv_order][2] - g_deriche_2[deriv_order][3] * I;
        break;

    case 3:
        coefs->alpha[0] = g_deriche_3[deriv_order][0] / 2.0 + g_deriche_3[deriv_order][1] / 2.0 * I;
        coefs->alpha[1] = g_deriche_3[deriv_order][0] / 2.0 - g_deriche_3[deriv_order][1] / 2.0 * I;
        coefs->alpha[2] = g_deriche_3[deriv_order][2];
        coefs->alpha[3] = g_deriche_3[deriv_order][2];
        coefs->lambda[0] = g_deriche_3[deriv_order][3] + g_deriche_3[deriv_order][4] * I;
        coefs->lambda[1] = g_deriche_3[deriv_order][3] - g_deriche_3[deriv_order][4] * I;
        coefs->lambda[2] = g_deriche_3[deriv_order][5];
        coefs->lambda[2] = g_deriche_3[deriv_order][5];
        break;

    case 4:
        coefs->alpha[0] = g_deriche_4[deriv_order][0] / 2.0 + g_deriche_4[deriv_order][1] / 2.0 * I;
        coefs->alpha[1] = g_deriche_4[deriv_order][0] / 2.0 - g_deriche_4[deriv_order][1] / 2.0 * I;
        coefs->alpha[2] = g_deriche_4[deriv_order][2] / 2.0 - g_deriche_4[deriv_order][3] / 2.0 * I;
        coefs->alpha[3] = g_deriche_4[deriv_order][2] / 2.0 - g_deriche_4[deriv_order][3] / 2.0 * I;
        coefs->lambda[0] = g_deriche_4[deriv_order][4] + g_deriche_4[deriv_order][5] * I;
        coefs->lambda[1] = g_deriche_4[deriv_order][4] - g_deriche_4[deriv_order][5] * I;
        coefs->lambda[2] = g_deriche_4[deriv_order][6] + g_deriche_4[deriv_order][7] * I;
        coefs->lambda[2] = g_deriche_4[deriv_order][6] - g_deriche_4[deriv_order][7] * I;
        break;
    }

    coefs->iir_order = iir_order;
    coefs->deriv_order = deriv_order;

    return true;
}

static inline bool deriche_scale_coefs(deriche_precomputed_t *precoefs, double sigma, fastfilters_kernel_iir_t kernel)
{
    for (unsigned int i = 0; i < precoefs->iir_order; ++i) {
        precoefs->lambda[i] /= sigma;
        precoefs->lambda[i] = -exp(-creal(precoefs->lambda[i])) * cos(cimag(precoefs->lambda[i])) +
                              exp(-creal(precoefs->lambda[i])) * sin(cimag(precoefs->lambda[i])) * I;
    }

    double complex a[5], b[4];

    b[0] = precoefs->alpha[0];
    a[0] = 1.0;
    a[1] = precoefs->lambda[0];

    for (unsigned int k = 1; k < precoefs->iir_order; ++k) {
        b[k] = precoefs->lambda[k] * b[k - 1];

        for (unsigned int j = k - 1; j > 0; --j)
            b[j] += precoefs->lambda[k] * b[j - 1];

        for (unsigned int j = 0; j <= k; ++j)
            b[j] += precoefs->alpha[k] * a[j];

        a[k + 1] = precoefs->lambda[k] * a[k];

        for (unsigned int j = k; j > 0; --j)
            a[j] += precoefs->lambda[k] * a[j - 1];
    }

    for (unsigned int i = 0; i < precoefs->iir_order; ++i) {
        kernel->n_causal[i] = creal(b[i]) / (M_SQRT2PI * pow(sigma, precoefs->deriv_order + 1));
        kernel->d[i] = creal(a[i + 1]);
    }

    int sign = 1;
    if (precoefs->deriv_order == 1)
        sign = -1;

    for (unsigned int i = 0; i < precoefs->iir_order - 1; ++i)
        kernel->n_anticausal[i] = sign * (kernel->n_causal[i + 1] - kernel->n_causal[0] * kernel->d[i]);

    kernel->n_anticausal[precoefs->iir_order - 1] =
        sign * (-1.0) * kernel->n_causal[0] * kernel->d[precoefs->iir_order - 1];

    return true;
}

static bool deriche_compute_coefs(fastfilters_kernel_iir_t kernel, unsigned int iir_order, unsigned int deriv_order,
                                  double sigma)
{
    deriche_precomputed_t coefs;

    if (!deriche_get_precomputed(&coefs, iir_order, deriv_order))
        return false;

    if (!deriche_scale_coefs(&coefs, sigma, kernel))
        return false;

    return true;
}

fastfilters_kernel_t DLL_PUBLIC fastfilters_kernel_iir_gaussian(unsigned int order, double sigma, float window_ratio)
{
    if (order > 2)
        return NULL;

    if (sigma < 0)
        return NULL;

    fastfilters_kernel_t kernel = fastfilters_memory_alloc(sizeof(struct _fastfilters_kernel_t));
    if (!kernel)
        goto out;

    kernel->fir = NULL;
    kernel->iir = fastfilters_memory_alloc(sizeof(struct _fastfilters_kernel_iir_t));
    if (!kernel->iir)
        goto out;

    if (window_ratio > 0)
        kernel->iir->radius = floor(window_ratio * sigma + 0.5);
    else
        kernel->iir->radius =
            floor(3.0 * sigma + 0.5 * order + 0.5); // FIXME: this is the formula currently used by vigra

    // memcpy only
    if (fabs(sigma) < 1e-6) {
        kernel->iir->radius = 0;
        kernel->iir->order = 0;
        return kernel;
    }

    if (!deriche_compute_coefs(kernel->iir, 4, order, sigma))
        goto out;

    kernel->iir->order = 4;

    return kernel;

out:
    if (kernel && kernel->iir)
        fastfilters_memory_free(kernel->iir);
    fastfilters_memory_free(kernel);
    return NULL;
}
