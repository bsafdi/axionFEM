// <omp.h>, or serial replacements for the three OpenMP functions axionFEM calls.
//
// CMake defines AXF_HAVE_OPENMP when it found an OpenMP runtime.  Without one
// (Apple clang has no libomp unless it is installed, so the macOS wheels are
// built this way) the `#pragma omp` directives are ignored by the compiler and
// these stubs make the surrounding code describe a single thread.  Results are
// identical; only the speed of assembly changes.  See docs/install.md.
#pragma once

#ifdef AXF_HAVE_OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_in_parallel() { return 0; }
inline void omp_set_num_threads(int) {}
#endif
