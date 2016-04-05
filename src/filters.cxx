#include "fastfilters.hxx"

using fastfilters::fir::Kernel;
using fastfilters::fir::convolve_fir;

using fastfilters::iir::Coefficients;
using fastfilters::iir::convolve_iir;

namespace
{
template <class KernelType> class ConvolveFunctor
{
};

template <> class ConvolveFunctor<fastfilters::iir::Coefficients>
{
  public:
    void operator()(const float *input, const unsigned int pixel_stride, const unsigned int pixel_n,
                    const unsigned int dim_stride, const unsigned int n_dim, float *output,
                    const fastfilters::iir::Coefficients &coefs)
    {
        fastfilters::iir::convolve_iir(input, pixel_stride, pixel_n, dim_stride, n_dim, output, coefs);
    }
};

template <> class ConvolveFunctor<fastfilters::fir::Kernel>
{
  public:
    void operator()(const float *input, const unsigned int pixel_stride, const unsigned int pixel_n,
                    const unsigned int dim_stride, const unsigned int n_dim, float *output,
                    fastfilters::fir::Kernel &kernel)
    {
        fastfilters::fir::convolve_fir(input, pixel_stride, pixel_n, dim_stride, n_dim, output, kernel);
    }
};

} // anonymous namespace

namespace fastfilters
{

void gaussian2d(const float *input, const std::size_t n_x, const std::size_t n_y, const double sigma,
                const unsigned int order, float *output)
{
    if (sigma < 3.0) {
        Kernel gauss(sigma, order);

        convolve_fir(input, n_y, n_x, n_x, 1, output, gauss);
        convolve_fir(output, n_x, 1, n_y, n_x, output, gauss);
    } else {
        Coefficients gauss(sigma, order);

        convolve_iir(input, n_y, n_x, n_x, 1, output, gauss);
        convolve_iir(output, n_x, 1, n_y, n_x, output, gauss);
    }
}
}