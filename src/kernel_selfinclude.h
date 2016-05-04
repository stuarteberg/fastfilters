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

#ifndef FF_KERNEL_IMPL_IS_INCLUDED

#if !defined(FF_BOUNDARY_OPTIMISTIC_LEFT) && !defined(FF_BOUNDARY_MIRROR_LEFT) && !defined(FF_BOUNDARY_PTR_LEFT)

#define FF_BOUNDARY_OPTIMISTIC_LEFT
#include "kernel_selfinclude.h"
#undef FF_BOUNDARY_OPTIMISTIC_LEFT

#define FF_BOUNDARY_MIRROR_LEFT
#include "kernel_selfinclude.h"
#undef FF_BOUNDARY_MIRROR_LEFT

#define FF_BOUNDARY_PTR_LEFT
#include "kernel_selfinclude.h"
#undef FF_BOUNDARY_PTR_LEFT

#elif !defined(FF_BOUNDARY_OPTIMISTIC_RIGHT) && !defined(FF_BOUNDARY_MIRROR_RIGHT) && !defined(FF_BOUNDARY_PTR_RIGHT)

#define FF_BOUNDARY_OPTIMISTIC_RIGHT
#include "kernel_selfinclude.h"
#undef FF_BOUNDARY_OPTIMISTIC_RIGHT

#define FF_BOUNDARY_MIRROR_RIGHT
#include "kernel_selfinclude.h"
#undef FF_BOUNDARY_MIRROR_RIGHT

#define FF_BOUNDARY_PTR_RIGHT
#include "kernel_selfinclude.h"
#undef FF_BOUNDARY_PTR_RIGHT

#elif !defined(FF_KERNEL_SYMMETRIC) && !defined(FF_KERNEL_ANTISYMMETRIC)

#define FF_KERNEL_SYMMETRIC
#include "kernel_selfinclude.h"
#undef FF_KERNEL_SYMMETRIC

#define FF_KERNEL_ANTISYMMETRIC
#include "kernel_selfinclude.h"
#undef FF_KERNEL_ANTISYMMETRIC

#else

#define FF_KERNEL_IMPL_IS_INCLUDED
#include FF_KERNEL_IMPL_INCLUDE()
#undef FF_KERNEL_IMPL_IS_INCLUDED

#endif

#endif