#!/usr/bin/env bash
# Build the pinned third-party libraries of the MPI configuration (the 3D engine) into third_party/install:
# METIS, hypre and MFEM (with MPI, METIS and LAPACK), the Eigen and nlohmann/json headers, and the gmsh SDK.
# Run scripts/fetch_deps.sh first (it downloads and verifies the sources).
#
# Needs: C and C++ compilers, MPI with its compiler wrappers, BLAS/LAPACK, CMake >= 3.20, make, tar.
#   CC                     compiler for METIS (default cc)
#   MPICC, MPICXX          MPI wrappers for hypre and MFEM (default mpicc, mpicxx)
#   AXF_BLA_VENDOR         BLA_VENDOR for MFEM's LAPACK search (default: CMake's own search order)
#   AXF_OPENMP=1           hypre and MFEM with OpenMP threads
#   NPROC                  parallel jobs (default: all usable cores)
#   AXF_DEPS_PREFIX        install prefix (default $AXIONFEM_DEPS, else third_party/install)
#   AXF_DEPS_SRC, AXF_DEPS_BUILD   source and build directories (default third_party/src, third_party/build)
# Each library is skipped when it is already installed in the prefix; delete the prefix to rebuild.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SRC="${AXF_DEPS_SRC:-$AXF_ROOT/third_party/src}"
BUILD="${AXF_DEPS_BUILD:-$AXF_ROOT/third_party/build}"
PREFIX="${AXF_DEPS_PREFIX:-${AXIONFEM_DEPS:-$AXF_ROOT/third_party/install}}"
OMP="${AXF_OPENMP:-0}"
HYPRE_OMP=$([ "$OMP" = 1 ] && echo ON || echo OFF)
MFEM_OMP=$([ "$OMP" = 1 ] && echo YES || echo NO)
NPROC="$(axf_nproc)"
CC="${CC:-cc}"
MPICC="${MPICC:-mpicc}"
MPICXX="${MPICXX:-mpicxx}"

MFEM_VER=4.10
HYPRE_VER=2.33.0
METIS_VER=5.1.0
EIGEN_VER=3.4.0
JSON_VER=3.12.0
GMSH_VER=4.15.2

axf_require cmake "Install CMake >= 3.20 (a distribution package, 'pip install cmake', or 'brew install cmake')."
axf_require make "Install make (Debian/Ubuntu: build-essential; Fedora/RHEL: make; macOS: xcode-select --install)."
axf_require "$CC" "Install a C compiler or set CC."
axf_require "$MPICC" "$AXF_MPI_HINT"
axf_require "$MPICXX" "$AXF_MPI_HINT"
axf_require_tar
for f in "mfem-${MFEM_VER}.tar.gz" "hypre-${HYPRE_VER}.tar.gz" "metis-${METIS_VER}.tar.gz" \
         "eigen-${EIGEN_VER}.tar.gz" "json-${JSON_VER}.hpp"; do
  [ -s "$SRC/$f" ] || axf_die "$SRC/$f is missing: run 'bash scripts/fetch_deps.sh' first (or set AXF_DEPS_SRC)"
done
mkdir -p "$BUILD" "$PREFIX"
RPATH_ARGS=()
while IFS= read -r a; do RPATH_ARGS+=("$a"); done < <(axf_cmake_rpath_args "$PREFIX")
BLA_ARGS=()  # expanded as ${BLA_ARGS[@]+...}: bash 3.2 (macOS) treats an empty array as unset under set -u
v="$(axf_bla_vendor_args)"; [ -z "$v" ] || BLA_ARGS+=("$v")

echo "== toolchain ($AXF_OS $AXF_ARCH, $NPROC jobs, prefix $PREFIX)"
echo "   CC=$(command -v "$CC")  MPICC=$(command -v "$MPICC")  MPICXX=$(command -v "$MPICXX")"
"$MPICXX" --version 2>/dev/null | head -1 || true
cmake --version | head -1

# ---------------------------------------------------------------- header-only
mkdir -p "$PREFIX/include/nlohmann"
cp "$SRC/json-${JSON_VER}.hpp" "$PREFIX/include/nlohmann/json.hpp"
if [ ! -d "$PREFIX/include/eigen3/Eigen" ]; then
  tar -C "$BUILD" -xzf "$SRC/eigen-${EIGEN_VER}.tar.gz"
  mkdir -p "$PREFIX/include/eigen3"
  cp -R "$BUILD/eigen-${EIGEN_VER}/Eigen" "$BUILD/eigen-${EIGEN_VER}/unsupported" "$PREFIX/include/eigen3/"
fi

# ---------------------------------------------------------------- gmsh SDK (axf-mesher only; a separate program)
GMSH="$(axf_gmsh_sdk "$GMSH_VER")"
if [ -z "$GMSH" ]; then
  echo "== gmsh: no SDK for this platform; axf-mesher will use the PyPI wheel (pip install 'gmsh>=4.15,<4.16')"
elif [ ! -x "$PREFIX/gmsh-sdk/bin/gmsh" ]; then
  name="${GMSH%% *}"
  [ -s "$SRC/$name" ] || axf_die "$SRC/$name is missing: run 'bash scripts/fetch_deps.sh' first"
  tar -C "$BUILD" -xzf "$SRC/$name"
  rm -rf "$PREFIX/gmsh-sdk"
  cp -R "$BUILD/${name%.tgz}" "$PREFIX/gmsh-sdk"
fi
[ ! -x "$PREFIX/gmsh-sdk/bin/gmsh" ] || echo "== gmsh SDK $("$PREFIX/gmsh-sdk/bin/gmsh" --version 2>&1 | tail -1)"

# ---------------------------------------------------------------- METIS 5.1.0 (shared, 32-bit indices as MFEM expects)
if ! axf_have_lib metis "$PREFIX"; then
  rm -rf "$BUILD/metis-${METIS_VER}"
  tar -C "$BUILD" -xzf "$SRC/metis-${METIS_VER}.tar.gz"
  (
    cd "$BUILD/metis-${METIS_VER}"
    # Newer CMake rejects the ancient minimum version in METIS 5.1.0.
    axf_sed_inplace 's/cmake_minimum_required(VERSION 2.8)/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt
    make config shared=1 cc="$CC" prefix="$PREFIX"
    make -j"$NPROC" install
  )
  # macOS: an absolute install name, so that MFEM and axionFEM find it without DYLD_LIBRARY_PATH
  if [ "$AXF_OS" = Darwin ]; then install_name_tool -id "$PREFIX/lib/libmetis.dylib" "$PREFIX/lib/libmetis.dylib"; fi
fi

# ---------------------------------------------------------------- hypre
if ! axf_have_lib HYPRE "$PREFIX"; then
  rm -rf "$BUILD/hypre-${HYPRE_VER}" "$BUILD/hypre-build"
  tar -C "$BUILD" -xzf "$SRC/hypre-${HYPRE_VER}.tar.gz"
  cmake -S "$BUILD/hypre-${HYPRE_VER}/src" -B "$BUILD/hypre-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" "${RPATH_ARGS[@]}" \
    -DCMAKE_C_COMPILER="$MPICC" -DCMAKE_CXX_COMPILER="$MPICXX" \
    -DHYPRE_ENABLE_SHARED=ON -DBUILD_SHARED_LIBS=ON \
    -DHYPRE_ENABLE_MPI=ON -DHYPRE_WITH_MPI=ON -DHYPRE_ENABLE_OPENMP=$HYPRE_OMP -DHYPRE_WITH_OPENMP=$HYPRE_OMP
  cmake --build "$BUILD/hypre-build" -j"$NPROC"
  cmake --install "$BUILD/hypre-build"
fi
HYPRE_LIB="$(axf_lib_path HYPRE "$PREFIX")"
[ -n "$HYPRE_LIB" ] || axf_die "hypre did not install a shared library into $PREFIX/lib"

# ---------------------------------------------------------------- MFEM (MPI + METIS + hypre + LAPACK, shared)
if ! axf_have_lib mfem "$PREFIX"; then
  rm -rf "$BUILD/mfem-${MFEM_VER}" "$BUILD/mfem-build"
  tar -C "$BUILD" -xzf "$SRC/mfem-${MFEM_VER}.tar.gz"
  if ! cmake -S "$BUILD/mfem-${MFEM_VER}" -B "$BUILD/mfem-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" "${RPATH_ARGS[@]}" ${BLA_ARGS[@]+"${BLA_ARGS[@]}"} \
    -DCMAKE_C_COMPILER="$MPICC" -DCMAKE_CXX_COMPILER="$MPICXX" \
    -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DMFEM_USE_MPI=YES -DMFEM_USE_METIS_5=YES -DMFEM_USE_LAPACK=YES -DMFEM_USE_OPENMP=$MFEM_OMP \
    -DMETIS_DIR="$PREFIX" \
    -DHYPRE_DIR="$PREFIX" -DHYPRE_INCLUDE_DIRS="$PREFIX/include" -DHYPRE_LIBRARIES="$HYPRE_LIB" \
    -DMFEM_ENABLE_EXAMPLES=NO -DMFEM_ENABLE_MINIAPPS=NO -DMFEM_ENABLE_TESTING=NO; then
    axf_die "configuring MFEM failed (see above). If it could not find LAPACK/BLAS, install them (Debian/Ubuntu:
  libopenblas-dev or liblapack-dev; Fedora/RHEL: openblas-devel or lapack-devel; macOS: Accelerate is found
  automatically) or choose one with AXF_BLA_VENDOR (CMake's BLA_VENDOR, e.g. OpenBLAS, FlexiBLAS, Generic)."
  fi
  cmake --build "$BUILD/mfem-build" -j"$NPROC"
  cmake --install "$BUILD/mfem-build"
fi

echo "== installed in $PREFIX"
ls "$PREFIX"/lib*/ | head -40
grep -E "^set\(MFEM_USE_(MPI|METIS|LAPACK) " "$PREFIX"/lib*/cmake/mfem/MFEMConfig.cmake 2>/dev/null || true
echo "BUILD_DEPS_OK"
