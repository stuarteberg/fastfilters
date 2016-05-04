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

#ifndef FASTFILTERS_KERNEL_H
#define FASTFILTERS_KERNEL_H

#include "common.h"

void DLL_LOCAL fastfilters_fir_init(void);
void DLL_LOCAL fastfilters_iir_init(void);

struct _fastfilters_kernel_iir_t {
    size_t order;
    float n_causal[4];
    float n_anticausal[4];
    float d[4];
    double radius;
};

struct _fastfilters_kernel_fir_t {
    size_t len;
    bool is_symmetric;
    float *coefs;

    impl_fn_t fn_inner_mirror;
    impl_fn_t fn_inner_ptr;
    impl_fn_t fn_inner_optimistic;

    impl_fn_t fn_outer_mirror;
    impl_fn_t fn_outer_ptr;
    impl_fn_t fn_outer_optimistic;
};

struct _fastfilters_kernel_t {
    fastfilters_kernel_fir_t fir;
    fastfilters_kernel_iir_t iir;

    convolve_fn_t convolve_outer;
    convolve_fn_t convolve_inner;
};

#endif