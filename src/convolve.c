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

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "fastfilters.h"
#include "common.h"
#include "kernel.h"

bool DLL_PUBLIC fastfilters_convolve2d(const fastfilters_array2d_t *inarray, const fastfilters_kernel_t kernelx,
                                       const fastfilters_kernel_t kernely, const fastfilters_array2d_t *outarray,
                                       const fastfilters_options_t *options)
{
    (void)options;
    if (!kernelx->convolve_inner(inarray->ptr, inarray->n_x, inarray->stride_x, inarray->n_y, inarray->stride_y,
                                 outarray->ptr, outarray->stride_y, kernelx, FASTFILTERS_BORDER_MIRROR,
                                 FASTFILTERS_BORDER_MIRROR, NULL, NULL, 0))
        return false;

    return kernely->convolve_outer(outarray->ptr, inarray->n_y, outarray->stride_y, inarray->n_x * inarray->n_channels,
                                   inarray->stride_x / inarray->n_channels, outarray->ptr, outarray->stride_y, kernely,
                                   FASTFILTERS_BORDER_MIRROR, FASTFILTERS_BORDER_MIRROR, NULL, NULL, 0);
}

bool DLL_PUBLIC fastfilters_convolve3d(const fastfilters_array3d_t *inarray, const fastfilters_kernel_t kernelx,
                                       const fastfilters_kernel_t kernely, const fastfilters_kernel_t kernelz,
                                       const fastfilters_array3d_t *outarray, const fastfilters_options_t *options)
{
    (void)options;

    for (size_t z = 0; z < inarray->n_z; ++z) {
        const float *planeptr = inarray->ptr + z * inarray->stride_z;
        float *planeptr_out = outarray->ptr + z * outarray->stride_z;

        if (!kernelx->convolve_inner(planeptr, inarray->n_x, inarray->stride_x, inarray->n_y, inarray->stride_y,
                                     planeptr_out, outarray->stride_y, kernelx, FASTFILTERS_BORDER_MIRROR,
                                     FASTFILTERS_BORDER_MIRROR, NULL, NULL, 0))
            return false;

        if (!kernely->convolve_outer(planeptr_out, inarray->n_y, outarray->stride_y, inarray->n_x * inarray->n_channels,
                                     inarray->stride_x / inarray->n_channels, planeptr_out, outarray->stride_y, kernely,
                                     FASTFILTERS_BORDER_MIRROR, FASTFILTERS_BORDER_MIRROR, NULL, NULL, 0))
            return false;
    }

    return kernelz->convolve_outer(outarray->ptr, outarray->n_z, outarray->stride_z,
                                   inarray->n_y * inarray->n_x * inarray->n_channels, 1, outarray->ptr,
                                   outarray->stride_z, kernelz, FASTFILTERS_BORDER_MIRROR, FASTFILTERS_BORDER_MIRROR,
                                   NULL, NULL, 0);
}