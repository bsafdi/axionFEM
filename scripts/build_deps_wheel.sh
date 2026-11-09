#!/usr/bin/env bash
# Sequential (no-MPI) third-party stack: the sequential source build and the planned wheels.
#
# A wheel on PyPI cannot require a system MPI, so this build uses a *serial* MFEM.  Without MFEM_USE_MPI there is no
# ParMesh and no HypreParMatrix, so the files that use MPI or hypre (the 3D engine's model3d* sources, the fluids
# module, the chemistry 3D current distribution; tests/test_packaging.py lists them) are left out: CMake builds the
# stub cpp/src/model3d_nompi.cpp instead, and every 3D entry point raises a clear error.  See docs/install.md for the
# build matrix.
#
# What this installs into $AXF_WHEEL_DEPS (default third_party/install-seq):
#   * MFEM 4.10, shared, serial: no MPI, no hypre, no METIS, no LAPACK, no zlib;
#   * METIS 5.1, shared, for axionFEM's own nested-dissection ordering of large axisymmetric systems;
#   * Eigen 3.4 and nlohmann/json headers.
# gmsh is NOT installed here: axf-mesher is built and run against the PyPI
# `gmsh` wheel (same upstream 4.15.2 binaries as the SDK), which keeps the GPL
# library out of our wheels.  See docs/install.md, "Licensing of each wheel".
#
# Usage:
#   bash scripts/build_deps_wheel.sh                  # uses third_party/src, fetches what is missing
#   AXF_DEPS_SRC=/path/to/src AXF_WHEEL_DEPS=/opt/axf bash scripts/build_deps_wheel.sh
#
# cibuildwheel calls this from before-all (see [tool.cibuildwheel] in pyproject.toml).
# Needs a C and a C++ compiler (CC, CXX; default cc, c++), CMake >= 3.20, make, tar and curl or wget.
# NPROC sets the parallel jobs (default: all usable cores). Archives are verified against
# third_party/SHA256SUMS.txt.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ROOT="$AXF_ROOT"
SRC="${AXF_DEPS_SRC:-$ROOT/third_party/src}"
PREFIX="${AXF_WHEEL_DEPS:-$ROOT/third_party/install-seq}"
BUILD="${AXF_WHEEL_BUILD:-$ROOT/third_party/build-seq}"
NPROC="$(axf_nproc)"

# Pinned versions.  Keep in step with scripts/fetch_deps.sh; tests/test_packaging.py
# checks that the two agree.
MFEM_VER=4.10
METIS_VER=5.1.0
EIGEN_VER=3.4.0
JSON_VER=3.12.0

mkdir -p "$SRC" "$BUILD" "$PREFIX"

fetch() {  # url output   (download only when the file is not already in $SRC; always verify)
  axf_fetch_verified "$SRC" "$2" "$1"
}

axf_require cmake "Install CMake >= 3.20 (a distribution package, 'pip install cmake', or 'brew install cmake')."
axf_require make "Install make (Debian/Ubuntu: build-essential; Fedora/RHEL: make; macOS: xcode-select --install)."
axf_require "${CC:-cc}" "Install a C compiler or set CC."
axf_require_tar
echo "== toolchain ($AXF_OS $AXF_ARCH, $NPROC jobs)"
"${CC:-cc}" --version | head -1 || true
cmake --version | head -1

fetch "https://github.com/mfem/mfem/archive/refs/tags/v${MFEM_VER}.tar.gz" "mfem-${MFEM_VER}.tar.gz"
fetch "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VER}/eigen-${EIGEN_VER}.tar.gz" "eigen-${EIGEN_VER}.tar.gz"
fetch "https://github.com/nlohmann/json/releases/download/v${JSON_VER}/json.hpp" "json-${JSON_VER}.hpp"
fetch "https://github.com/mfem/tpls/raw/gh-pages/metis-${METIS_VER}.tar.gz" "metis-${METIS_VER}.tar.gz"

# ---------------------------------------------------------------- header-only
mkdir -p "$PREFIX/include/nlohmann"
cp "$SRC/json-${JSON_VER}.hpp" "$PREFIX/include/nlohmann/json.hpp"
if [ ! -d "$PREFIX/include/eigen3/Eigen" ]; then
  tar -C "$BUILD" -xzf "$SRC/eigen-${EIGEN_VER}.tar.gz"
  mkdir -p "$PREFIX/include/eigen3"
  cp -r "$BUILD/eigen-${EIGEN_VER}/Eigen" "$BUILD/eigen-${EIGEN_VER}/unsupported" "$PREFIX/include/eigen3/"
fi

# ---------------------------------------------------------------- METIS (Apache-2.0), for axionFEM itself
# Not for MFEM (serial, no METIS): the axisymmetric engine's nested-dissection ordering, without which a
# sequential build refuses systems above 4e5 unknowns (Eigen's AMD ordering fails there; docs/install.md).
if ! ls "$PREFIX"/lib*/libmetis.* >/dev/null 2>&1; then
  rm -rf "$BUILD/metis-${METIS_VER}"
  tar -C "$BUILD" -xzf "$SRC/metis-${METIS_VER}.tar.gz"
  (cd "$BUILD/metis-${METIS_VER}" &&
   axf_sed_inplace 's/cmake_minimum_required(VERSION 2.8)/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt &&
   make config shared=1 cc="${CC:-cc}" prefix="$PREFIX" && make -j"$NPROC" install)
fi

# ---------------------------------------------------------------- MFEM, serial
if ! ls "$PREFIX"/lib*/libmfem.* >/dev/null 2>&1; then
  rm -rf "$BUILD/mfem-${MFEM_VER}" "$BUILD/mfem-build"
  tar -C "$BUILD" -xzf "$SRC/mfem-${MFEM_VER}.tar.gz"
  # @loader_path/$ORIGIN so that the copy bundled in the wheel resolves beside itself.
  # The macOS install name is @rpath/libmfem...: what links MFEM then finds it through its own rpaths
  # (@loader_path/_libs, @loader_path/../_libs), as for METIS.  An @loader_path install name is looked
  # for beside each consumer instead, where the wheel does not have it (delocate: "not found").
  if [ "$(uname -s)" = "Darwin" ]; then RP='@loader_path'; else RP='$ORIGIN'; fi
  cmake -S "$BUILD/mfem-${MFEM_VER}" -B "$BUILD/mfem-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_RPATH="$RP" -DCMAKE_INSTALL_NAME_DIR='@rpath' \
    -DMFEM_USE_MPI=NO -DMFEM_USE_METIS=NO -DMFEM_USE_METIS_5=NO \
    -DMFEM_USE_LAPACK=NO -DMFEM_USE_ZLIB=NO -DMFEM_USE_OPENMP=NO \
    -DMFEM_ENABLE_EXAMPLES=NO -DMFEM_ENABLE_MINIAPPS=NO -DMFEM_ENABLE_TESTING=NO
  cmake --build "$BUILD/mfem-build" -j"$NPROC"
  cmake --install "$BUILD/mfem-build"
fi

echo "== installed in $PREFIX"
ls "$PREFIX"/lib*/ 2>/dev/null | head -20
grep -E "^set\(MFEM_USE_(MPI|METIS|LAPACK) " "$PREFIX"/lib*/cmake/mfem/MFEMConfig.cmake 2>/dev/null || true
echo "BUILD_DEPS_WHEEL_OK"
