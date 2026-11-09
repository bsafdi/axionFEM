#!/usr/bin/env bash
# Build STRUMPACK 8.0.0 (BSD-3; multifrontal sparse direct solver, serial + OpenMP, METIS ordering) into the
# dependency prefix, for the large axisymmetric problems (CMake option AXF_USE_STRUMPACK). Optional.
# Run after scripts/build_deps.sh (it needs METIS from the same prefix).
#
# Needs C, C++ and Fortran compilers (CC, CXX, FC; STRUMPACK's CMake project requires Fortran even for this
# serial build), OpenMP, and BLAS/LAPACK:
#   AXF_BLAS_LIBS, AXF_LAPACK_LIBS   explicit libraries (;-separated); otherwise OpenBLAS through pkg-config or
#                                    Homebrew, otherwise STRUMPACK's own FindBLAS/FindLAPACK search (reference
#                                    LAPACK, FlexiBLAS, MKL, Accelerate, ...; AXF_BLA_VENDOR steers it)
#   AXIONFEM_DEPS                    prefix (default third_party/install); NPROC parallel jobs
# macOS: Apple clang has no OpenMP; use Homebrew's libomp (OpenMP_ROOT=$(brew --prefix libomp)) and gfortran
# (brew install gcc), or a GCC toolchain.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DEPS="${AXIONFEM_DEPS:-$AXF_ROOT/third_party/install}"
SRC="${AXF_DEPS_SRC:-$AXF_ROOT/third_party/src}"
BLD="${AXF_DEPS_BUILD:-$AXF_ROOT/third_party/build}"
NPROC="$(axf_nproc)"
VER=8.0.0

METIS_LIB="$(axf_lib_path metis "$DEPS")"
[ -n "$METIS_LIB" ] || axf_die "METIS not found in $DEPS/lib: run scripts/build_deps.sh first (or set AXIONFEM_DEPS)"
axf_require "${FC:-gfortran}" "STRUMPACK needs a Fortran compiler: install gfortran (Debian/Ubuntu: gfortran; \
Fedora/RHEL: gcc-gfortran; macOS: brew install gcc) or set FC."
axf_require_tar
mkdir -p "$SRC" "$BLD"
axf_fetch_verified "$SRC" "STRUMPACK-$VER.tar.gz" "https://github.com/pghysels/STRUMPACK/archive/refs/tags/v$VER.tar.gz"

{ read -r BLAS; read -r LAPACK; } < <(axf_blas_lapack)
TPL=()
if [ -n "$BLAS" ]; then
  TPL+=("-DTPL_BLAS_LIBRARIES=$BLAS" "-DTPL_LAPACK_LIBRARIES=$LAPACK")
  echo "== BLAS $BLAS, LAPACK $LAPACK"
else
  echo "== BLAS/LAPACK: STRUMPACK's own search (AXF_BLA_VENDOR=${AXF_BLA_VENDOR:-any})"
fi
v="$(axf_bla_vendor_args)"; [ -z "$v" ] || TPL+=("$v")
OMP_ARGS=()
if [ -n "${OpenMP_ROOT:-}" ]; then OMP_ARGS+=("-DOpenMP_ROOT=$OpenMP_ROOT"); fi
RPATH_ARGS=()
while IFS= read -r a; do RPATH_ARGS+=("$a"); done < <(axf_cmake_rpath_args "$DEPS")

rm -rf "$BLD/STRUMPACK-$VER" "$BLD/strumpack-build" && tar -xzf "$SRC/STRUMPACK-$VER.tar.gz" -C "$BLD"
cmake -S "$BLD/STRUMPACK-$VER" -B "$BLD/strumpack-build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DEPS" "${RPATH_ARGS[@]}" -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_Fortran_COMPILER="${FC:-gfortran}" ${OMP_ARGS[@]+"${OMP_ARGS[@]}"} \
  -DSTRUMPACK_USE_MPI=OFF -DSTRUMPACK_USE_OPENMP=ON \
  -DSTRUMPACK_USE_CUDA=OFF -DSTRUMPACK_USE_HIP=OFF -DSTRUMPACK_USE_SYCL=OFF \
  -DTPL_ENABLE_SCOTCH=OFF -DTPL_ENABLE_PTSCOTCH=OFF -DTPL_ENABLE_PARMETIS=OFF -DTPL_ENABLE_ZFP=OFF \
  -DTPL_ENABLE_BPACK=OFF -DTPL_ENABLE_SLATE=OFF -DTPL_ENABLE_COMBBLAS=OFF -DTPL_ENABLE_MAGMA=OFF \
  -DTPL_ENABLE_KBLAS=OFF -DTPL_ENABLE_SZ=OFF \
  -DTPL_METIS_INCLUDE_DIRS="$DEPS/include" -DTPL_METIS_LIBRARIES="$METIS_LIB" \
  ${TPL[@]+"${TPL[@]}"} \
  > "$BLD/strumpack-cmake.log" 2>&1 || {
    tail -40 "$BLD/strumpack-cmake.log"
    axf_die "configuring STRUMPACK failed (log: $BLD/strumpack-cmake.log). Missing BLAS/LAPACK: install OpenBLAS
  (Debian/Ubuntu: libopenblas-dev; Fedora/RHEL: openblas-devel; macOS: brew install openblas) or set AXF_BLAS_LIBS."
  }
cmake --build "$BLD/strumpack-build" -j "$NPROC" > "$BLD/strumpack-build.log" 2>&1 ||
  { tail -40 "$BLD/strumpack-build.log"; axf_die "building STRUMPACK failed (log: $BLD/strumpack-build.log)"; }
cmake --install "$BLD/strumpack-build" > /dev/null
echo "STRUMPACK $VER installed into $DEPS"
