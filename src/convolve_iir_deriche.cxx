#include "fastfilters.hxx"
#include <complex>
#include <cassert>

#ifndef M_SQRT2PI
#define M_SQRT2PI       2.50662827463100050241576528481104525
#endif

namespace fastfilters
{

namespace deriche
{
    // Rachid Deriche. Recursively implementating the Gaussian and its derivatives.
    // [Research Report] RR-1893, 1993, pp.24. <inria-00074778>

    struct DerichePrecomputed
    {
        const std::complex<double> alpha0, alpha1, alpha2, alpha3;
        const std::complex<double> lambda0, lambda1, lambda2, lambda3;

        DerichePrecomputed(const double a0, const double a1, const double a2, const double a3, const double l0, const double l1, const double l2, const double l3) :
        alpha0(std::complex<double>(a0 / 2.0, a1 / 2.0)), alpha1(std::complex<double>(a0 / 2.0, -a1 / 2.0)), alpha2(std::complex<double>(a2 / 2.0, a3 / 2.0)), alpha3(std::complex<double>(a2 / 2.0, -a3 / 2.0)),
        lambda0(std::complex<double>(l0, l1)), lambda1(std::complex<double>(l0, -l1)), lambda2(std::complex<double>(l2, l3)), lambda3(std::complex<double>(l2, -l3))
        {
        }

        DerichePrecomputed(const double a0, const double a1, const double a2, const double l0, const double l1, const double l2) :
        alpha0(std::complex<double>(a0 / 2.0, a1 / 2.0)), alpha1(std::complex<double>(a0 / 2.0, -a1 / 2.0)), alpha2(std::complex<double>(a2, 0)), alpha3(std::complex<double>(a2, 0)),
        lambda0(std::complex<double>(l0, l1)), lambda1(std::complex<double>(l0, -l1)), lambda2(std::complex<double>(l2, 0)), lambda3(std::complex<double>(l2, 0))
        {
        }

        DerichePrecomputed(const double a0, const double a1, const double l0, const double l1) :
        alpha0(std::complex<double>(a0 / 2.0, a1 / 2.0)), alpha1(std::complex<double>(a0 / 2.0, -a1 / 2.0)), alpha2(std::complex<double>(0, 0)), alpha3(std::complex<double>(0, 0)),
        lambda0(std::complex<double>(l0, l1)), lambda1(std::complex<double>(l0, -l1)), lambda2(std::complex<double>(0, 0)), lambda3(std::complex<double>(0, 0))
        {
        }

        const std::complex<double> get_alpha(const unsigned int idx) const {
            switch (idx) {
                case 0: return alpha0;
                case 1: return alpha1;
                case 2: return alpha2;
                case 3: return alpha3;
            }

            assert(false && "DerichePrecomputed::get_alpha: index out of bounds.");
            return 0;
        }

        const std::complex<double> get_lambda(const unsigned int idx) const {
            switch (idx) {
                case 0: return lambda0;
                case 1: return lambda1;
                case 2: return lambda2;
                case 3: return lambda3;
            }

            assert(false && "DerichePrecomputed::get_lambda: index out of bounds.");
            return 0;
        }
    };

    // eq. 35 - 46
    // 4th order coefficients from The Mines Java Toolkit
    // https://github.com/dhale/jtk/blob/master/core/src/main/java/edu/mines/jtk/dsp/RecursiveGaussianFilter.java
    static const DerichePrecomputed deriche_precomputed_coefs[3] = {
        // smoothing
        DerichePrecomputed(1.6797292232361107, 3.7348298269103580, -0.6802783501806897, -0.2598300478959625, 1.7831906544515104, 0.6318113174569493, 1.7228297663338028, 1.9969276832487770),
        // first derivative
        DerichePrecomputed(0.6494024008440620, 0.9557370760729773, -0.6472105276644291, -4.5306923044570760, 1.5159726670750566, 2.0718953658782650, 1.5267608734791140, 0.6719055957689513),
        // second derivative
        DerichePrecomputed(0.3224570510072559, -1.7382843963561239, -1.3312275593739595, 3.6607035671974897, 1.3138054926516880, 2.1656041357418863, 1.2402181393295362, 0.7479888745408682)
    };

    // based on code Copyright (c) 2012-2013, Pascal Getreuer
    // <getreuer@cmla.ens-cachan.fr>
    // licensed under the terms of the simplified BSD license.
    void compute_coefs(const double sigma, const unsigned order, float *n_causal, float *n_anticausal, float *d)
    {
        std::complex<double> alpha[4];
        std::complex<double> lambda[4];
        std::complex<double> beta[4];

        for (unsigned int i = 0; i < 4; ++i) {
            alpha[i] = deriche_precomputed_coefs[order].get_alpha(i);
            lambda[i] = deriche_precomputed_coefs[order].get_lambda(i) / sigma;
            beta[i] = std::complex<double>(
                -exp(-lambda[i].real()) * cos(lambda[i].imag()),
                exp(-lambda[i].real()) * sin(lambda[i].imag()));
        }

        std::complex<double> a[4 + 1];
        std::complex<double> b[4];

        b[0] = alpha[0];
        a[0] = std::complex<double>(1, 0);
        a[1] = beta[0];

        for (unsigned int k = 1; k < 4; ++k) {
            b[k] = beta[k] * b[k - 1];

            for (unsigned int j = k - 1; j > 0; --j)
               b[j] += beta[k] * b[j - 1];

            for (unsigned int j = 0; j <= k; ++j)
                b[j] += alpha[k] * a[j];

            a[k + 1] = beta[k] * a[k];

            for (unsigned int j = k; j > 0; --j)
                a[j] += beta[k] * a[j - 1];
        }

        for (unsigned int i = 0; i < 4; ++i) {
            n_causal[i] = b[i].real() / (M_SQRT2PI * pow(sigma, order + 1));
            d[i] = a[i + 1].real();
        }

        // eq. 31 for symmetrical filters and eq. 32 for antisymmetrical
        int sign = 1;
        if (order == 1)
            sign = -1;

        for (unsigned int i = 0; i < 3; ++i)
            n_anticausal[i] = sign * (n_causal[i + 1] - n_causal[0] * d[i]);
        n_anticausal[3] = sign * (-1.0) * n_causal[0] * d[3];
    }


} // namespace deriche

} // namespace fastfilters