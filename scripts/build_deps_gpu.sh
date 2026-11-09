#!/usr/bin/env bash
# Build a CUDA third-party stack so that axionFEM can be configured with -DAXF_GPU=CUDA.
# See docs/install.md, "GPU builds".
#
# MPI + hypre + METIS + MFEM, all with the CUDA backend, because the only GPU-relevant solver in axionFEM is the
# 3D engine's hypre AMS/AME/BoomerAMG (cpp/src/model3d.cpp), which needs MPI. There is no GPU path for the
# axisymmetric engine, whose linear algebra is a sparse direct factorisation on the CPU. STRUMPACK (host only, as
# in the CPU stack) completes the stack, so that the CUDA build has the CPU build's direct paths.
#
#   AXF_CUDA_ARCH   compute capabilities without the dot, ;-separated, for hypre and MFEM (default "80;90":
#                   A100 and H100/H200). Set it to the GPUs you will run on, e.g. "70" (V100), "75" (T4,
#                   RTX 20xx), "86" (A40, RTX 30xx), "89" (L4, L40, RTX 40xx); "native" (CMake >= 3.24) builds
#                   for the GPUs of the build machine only.
#   AXF_GPU_DEPS    install prefix (default third_party/install-cuda)
#   MPICC, MPICXX, CC, NPROC as in scripts/build_deps.sh; nvcc must be on PATH (or set CUDACXX).
#
# Run scripts/fetch_deps.sh first. Compiling needs nvcc and the system C headers, not a GPU, so the stack can be
# built on a machine without one. hypre keeps HYPRE_ENABLE_UNIFIED_MEMORY=ON: axionFEM reads hypre's matrices on
# the host in places (docs/validation/gpu.md).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ROOT="$AXF_ROOT"
SRC="${AXF_DEPS_SRC:-$ROOT/third_party/src}"
PREFIX="${AXF_GPU_DEPS:-$ROOT/third_party/install-cuda}"
BUILD="${AXF_GPU_BUILD:-$ROOT/third_party/build-cuda}"
ARCH="${AXF_CUDA_ARCH:-80;90}"
NPROC="$(axf_nproc)"
CC="${CC:-cc}"
MPICC="${MPICC:-mpicc}"
MPICXX="${MPICXX:-mpicxx}"

MFEM_VER=4.10
HYPRE_VER=2.33.0
METIS_VER=5.1.0
EIGEN_VER=3.4.0
JSON_VER=3.12.0

axf_require "${CUDACXX:-nvcc}" "Install the CUDA toolkit and put nvcc on PATH (or set CUDACXX)."
axf_require "$MPICC" "$AXF_MPI_HINT"
axf_require "$MPICXX" "$AXF_MPI_HINT"
axf_require cmake "Install CMake >= 3.20."
axf_require_tar
for f in "mfem-${MFEM_VER}.tar.gz" "hypre-${HYPRE_VER}.tar.gz" "metis-${METIS_VER}.tar.gz" \
         "eigen-${EIGEN_VER}.tar.gz" "json-${JSON_VER}.hpp"; do
  [ -s "$SRC/$f" ] || axf_die "$SRC/$f is missing: run 'bash scripts/fetch_deps.sh' first (or set AXF_DEPS_SRC)"
done
mkdir -p "$BUILD" "$PREFIX"
RPATH_ARGS=()
while IFS= read -r a; do RPATH_ARGS+=("$a"); done < <(axf_cmake_rpath_args "$PREFIX")
echo "== toolchain (CUDA architectures $ARCH, $NPROC jobs)"
command -v "${CUDACXX:-nvcc}" "$MPICC" "$MPICXX" cmake; "${CUDACXX:-nvcc}" --version | tail -2

# ---------------------------------------------------------------- header-only
mkdir -p "$PREFIX/include/nlohmann"
cp "$SRC/json-${JSON_VER}.hpp" "$PREFIX/include/nlohmann/json.hpp"
if [ ! -d "$PREFIX/include/eigen3/Eigen" ]; then
  tar -C "$BUILD" -xzf "$SRC/eigen-${EIGEN_VER}.tar.gz"
  mkdir -p "$PREFIX/include/eigen3"
  cp -R "$BUILD/eigen-${EIGEN_VER}/Eigen" "$BUILD/eigen-${EIGEN_VER}/unsupported" "$PREFIX/include/eigen3/"
fi

# ---------------------------------------------------------------- METIS
if ! axf_have_lib metis "$PREFIX"; then
  rm -rf "$BUILD/metis-${METIS_VER}"
  tar -C "$BUILD" -xzf "$SRC/metis-${METIS_VER}.tar.gz"
  (
    cd "$BUILD/metis-${METIS_VER}"
    axf_sed_inplace 's/cmake_minimum_required(VERSION 2.8)/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt
    make config shared=1 cc="$CC" prefix="$PREFIX"
    make -j"$NPROC" install
  )
fi

# ---------------------------------------------------------------- hypre with CUDA
if ! axf_have_lib HYPRE "$PREFIX"; then
  rm -rf "$BUILD/hypre-${HYPRE_VER}" "$BUILD/hypre-build"
  tar -C "$BUILD" -xzf "$SRC/hypre-${HYPRE_VER}.tar.gz"
  # hypre 2.33.0 joins CMAKE_CUDA_ARCHITECTURES with commas ("80,90") and puts that string on the HYPRE target,
  # which CMake reads as one architecture: nvcc then fails with "'90' is not in 'keyword=value' format". A single
  # architecture is unaffected. Give the target the semicolon list instead.
  # (Rewritten with awk rather than GNU sed's \n in the replacement, which BSD sed does not support.)
  f="$BUILD/hypre-${HYPRE_VER}/src/config/cmake/HYPRE_SetupCUDAToolkit.cmake"
  awk '$0 == "set_property(TARGET HYPRE PROPERTY CUDA_ARCHITECTURES \"${CMAKE_CUDA_ARCHITECTURES}\")" {
         print "string(REPLACE \",\" \";\" HYPRE_CUDA_ARCH_LIST \"${CMAKE_CUDA_ARCHITECTURES}\")"
         print "set_property(TARGET HYPRE PROPERTY CUDA_ARCHITECTURES \"${HYPRE_CUDA_ARCH_LIST}\")"; next } { print }' \
    "$f" > "$f.axf" && mv "$f.axf" "$f"
  if ! grep -q "HYPRE_CUDA_ARCH_LIST" "$BUILD/hypre-${HYPRE_VER}/src/config/cmake/HYPRE_SetupCUDAToolkit.cmake"; then
    echo "build_deps_gpu.sh: the hypre CUDA-architecture patch did not apply (hypre changed?); see the comment above" >&2
    exit 1
  fi
  cmake -S "$BUILD/hypre-${HYPRE_VER}/src" -B "$BUILD/hypre-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" "${RPATH_ARGS[@]}" \
    -DCMAKE_C_COMPILER="$MPICC" -DCMAKE_CXX_COMPILER="$MPICXX" \
    -DHYPRE_ENABLE_SHARED=ON -DBUILD_SHARED_LIBS=ON \
    -DHYPRE_ENABLE_MPI=ON -DHYPRE_ENABLE_OPENMP=OFF \
    -DHYPRE_ENABLE_CUDA=ON -DHYPRE_ENABLE_UNIFIED_MEMORY=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
  cmake --build "$BUILD/hypre-build" -j"$NPROC"
  cmake --install "$BUILD/hypre-build"
fi

# ---------------------------------------------------------------- MFEM with CUDA
if ! axf_have_lib mfem "$PREFIX"; then
  rm -rf "$BUILD/mfem-${MFEM_VER}" "$BUILD/mfem-build"
  tar -C "$BUILD" -xzf "$SRC/mfem-${MFEM_VER}.tar.gz"
  HYPRE_LIB="$(axf_lib_path HYPRE "$PREFIX")"
  cmake -S "$BUILD/mfem-${MFEM_VER}" -B "$BUILD/mfem-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" "${RPATH_ARGS[@]}" \
    -DCMAKE_C_COMPILER="$MPICC" -DCMAKE_CXX_COMPILER="$MPICXX" \
    -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DMFEM_USE_MPI=YES -DMFEM_USE_METIS_5=YES -DMFEM_USE_LAPACK=YES \
    -DMFEM_USE_CUDA=YES -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    -DMETIS_DIR="$PREFIX" \
    -DHYPRE_DIR="$PREFIX" -DHYPRE_INCLUDE_DIRS="$PREFIX/include" -DHYPRE_LIBRARIES="$HYPRE_LIB" \
    -DMFEM_ENABLE_EXAMPLES=NO -DMFEM_ENABLE_MINIAPPS=NO -DMFEM_ENABLE_TESTING=NO
  cmake --build "$BUILD/mfem-build" -j"$NPROC"
  cmake --install "$BUILD/mfem-build"
fi

# ---------------------------------------------------------------- STRUMPACK (serial + OpenMP, on the host)
# The axisymmetric engine above AXF_STRUMPACK_MIN_DOFS, the 3D shift-invert eigensolver and the STRUMPACK / tree-cotree
# full-wave paths need it; without it the CUDA build refuses those paths and differs from the CPU build on large 2D
# solves. It never runs on the GPU.
if ! axf_have_lib strumpack "$PREFIX"; then
  AXIONFEM_DEPS="$PREFIX" AXF_DEPS_SRC="$SRC" AXF_DEPS_BUILD="$BUILD" NPROC="$NPROC" bash "$ROOT/scripts/build_strumpack.sh"
fi

echo "== installed in $PREFIX"
grep -E "^set\(MFEM_USE_(MPI|CUDA|HIP) " "$PREFIX"/lib*/cmake/mfem/MFEMConfig.cmake || true
echo "BUILD_DEPS_GPU_OK"
