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
#include "fastfilters.hxx"
#include "gaussians.hxx"

namespace fastfilters
{
namespace fir
{

bool make_gaussian(double stddev, unsigned order, unsigned n_coefs, std::array<float, 13> &v)
{
    if (stddev < 0)
        return false;
    if (n_coefs > 12)
        return false;

    Gaussian<float> gauss(stddev, order);

    float norm = 0.0;
    if(order == 0) {
        norm = gauss(0);
        for(unsigned int k = 1; k < n_coefs; ++k)
            norm += 2*gauss(k);
    } else {
        unsigned int faculty = 1;
        for(unsigned int i = 2; i <= order; ++i)
            faculty *= i;

        int sign = 1;
        if (order == 1)
            sign = -1;

        norm = 0.0;
        for (unsigned int k = 1; k < n_coefs; ++k) {
            norm += gauss(k) * std::pow((double)k, int(order)) / faculty;
            norm += sign * gauss(k) * std::pow((-1) * (double)k, int(order)) / faculty;
        }
    }

    for (unsigned int i = 0; i < n_coefs; ++i)
        v[i] = gauss(i)/norm;

    return true;
}
}
}