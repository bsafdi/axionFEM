#!/usr/bin/env bash
# Build the distributed-memory STRUMPACK (MPI + ScaLAPACK) for the 3D direct solves into its own prefix
# $AXF_STRUMPACK_MPI_PREFIX (default third_party/install_mpi), leaving $AXIONFEM_DEPS (the serial STRUMPACK, MFEM,
# hypre, METIS) untouched. Optional; run after scripts/build_deps.sh.
#   1. ScaLAPACK 2.2.0 (netlib tarball) against the BLAS/LAPACK chosen as in build_strumpack.sh. It is built here
#      rather than taken from the system so that it uses the same BLAS as the rest of the process.
#   2. STRUMPACK 8.0.0 (the tarball of build_strumpack.sh) with STRUMPACK_USE_MPI=ON, METIS from $AXIONFEM_DEPS
#      (ordering on the root; ParMETIS off).
# Both are static, position-independent libraries: the serial build in $AXIONFEM_DEPS is a shared libstrumpack,
# and libaxionfem's RPATH lists that prefix first, so a second shared copy would be shadowed.
#
# Needs MPI with C, C++ and Fortran wrappers (MPICC, MPICXX, MPIFC; default mpicc, mpicxx, mpif90; Cray: cc, CC,
# ftn), BLAS/LAPACK (AXF_BLAS_LIBS/AXF_LAPACK_LIBS or as in build_strumpack.sh), NPROC parallel jobs. MPI's Fortran
# bindings library is read from the Fortran wrapper; AXF_MPI_FORTRAN_LIBS names it where that fails.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DEPS="${AXIONFEM_DEPS:-$AXF_ROOT/third_party/install}"
PREFIX="${AXF_STRUMPACK_MPI_PREFIX:-$AXF_ROOT/third_party/install_mpi}"
SRC="${AXF_DEPS_SRC:-$AXF_ROOT/third_party/src}"
BLD="${AXF_DEPS_BUILD:-$AXF_ROOT/third_party/build_mpi}"
NPROC="$(axf_nproc)"
MPICC="${MPICC:-mpicc}"
MPICXX="${MPICXX:-mpicxx}"
MPIFC="${MPIFC:-mpif90}"
SPK_VER=8.0.0
SCA_VER=2.2.0

axf_require "$MPICC" "$AXF_MPI_HINT"
axf_require "$MPICXX" "$AXF_MPI_HINT"
axf_require "$MPIFC" "The distributed STRUMPACK needs the MPI Fortran wrapper (and gfortran); set MPIFC to it."
axf_require_tar
METIS_LIB="$(axf_lib_path metis "$DEPS")"
[ -n "$METIS_LIB" ] || axf_die "METIS not found in $DEPS/lib: run scripts/build_deps.sh first (or set AXIONFEM_DEPS)"
mkdir -p "$SRC" "$BLD" "$PREFIX"
axf_fetch_verified "$SRC" "scalapack-$SCA_VER.tgz" "https://www.netlib.org/scalapack/scalapack-$SCA_VER.tgz"
axf_fetch_verified "$SRC" "STRUMPACK-$SPK_VER.tar.gz" "https://github.com/pghysels/STRUMPACK/archive/refs/tags/v$SPK_VER.tar.gz"

{ read -r BLAS; read -r LAPACK; } < <(axf_blas_lapack)
BLAS_ARGS=()
SPK_BLAS=()
if [ -n "$BLAS" ]; then
  BLAS_ARGS+=("-DLAPACK_FOUND=true" "-DLAPACK_LIBRARIES=$LAPACK" "-DBLAS_LIBRARIES=$BLAS")
  SPK_BLAS+=("-DTPL_BLAS_LIBRARIES=$BLAS" "-DTPL_LAPACK_LIBRARIES=$LAPACK")
  echo "== BLAS $BLAS, LAPACK $LAPACK"
else
  echo "== BLAS/LAPACK: the projects' own search (AXF_BLA_VENDOR=${AXF_BLA_VENDOR:-any})"
fi
v="$(axf_bla_vendor_args)"; [ -z "$v" ] || { BLAS_ARGS+=("$v"); SPK_BLAS+=("$v"); }

# gfortran >= 10 rejects ScaLAPACK 2.2.0's mismatched MPI argument types unless told to allow them.
FFLAGS="-O2"
probe="$BLD/fflags_probe"
mkdir -p "$probe" && printf 'end\n' > "$probe/t.f90"
if "$MPIFC" -fallow-argument-mismatch -c "$probe/t.f90" -o "$probe/t.o" >/dev/null 2>&1; then
  FFLAGS="$FFLAGS -fallow-argument-mismatch"
fi

# ---- ScaLAPACK
rm -rf "$BLD/scalapack-$SCA_VER" "$BLD/scalapack-build" && tar -xzf "$SRC/scalapack-$SCA_VER.tgz" -C "$BLD"
cmake -S "$BLD/scalapack-$SCA_VER" -B "$BLD/scalapack-build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_C_COMPILER="$MPICC" -DCMAKE_Fortran_COMPILER="$MPIFC" \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DSCALAPACK_BUILD_TESTS=OFF \
  -DCMAKE_Fortran_FLAGS="$FFLAGS" -DCMAKE_C_FLAGS="-O2 -Wno-implicit-function-declaration" \
  ${BLAS_ARGS[@]+"${BLAS_ARGS[@]}"} \
  > "$BLD/scalapack-cmake.log" 2>&1 ||
  { tail -40 "$BLD/scalapack-cmake.log"; axf_die "configuring ScaLAPACK failed (log: $BLD/scalapack-cmake.log)"; }
cmake --build "$BLD/scalapack-build" -j "$NPROC" > "$BLD/scalapack-build.log" 2>&1 ||
  { tail -40 "$BLD/scalapack-build.log"; axf_die "building ScaLAPACK failed (log: $BLD/scalapack-build.log)"; }
cmake --install "$BLD/scalapack-build" > /dev/null
[ -f "$PREFIX/lib/libscalapack.a" ] || axf_die "libscalapack.a missing in $PREFIX/lib"
echo "ScaLAPACK $SCA_VER installed into $PREFIX"

# ---- STRUMPACK (MPI)
# The static ScaLAPACK needs the Fortran runtime and MPI's Fortran bindings (BLACS calls mpi_abort_ ...):
# libmpi_mpifh (OpenMPI), libmpifort or libmpichfort (MPICH), libmpifort_<compiler> (Cray MPICH), named on the
# wrapper's link line and found in its -L directories or, failing that, in the compiler's search path.
# AXF_MPI_FORTRAN_LIBS (;-separated paths) overrides the search.
mpi_fortran_lib() {
  local flags d f n ext
  if [ -n "${AXF_MPI_FORTRAN_LIBS:-}" ]; then echo "$AXF_MPI_FORTRAN_LIBS"; return 0; fi
  # OpenMPI, Intel MPI, MPICH, Cray compiler wrappers; Cray lists libraries as -Wl,--as-needed,-lmpifort_gnu,...
  flags="$("$MPIFC" --showme:link 2>/dev/null || "$MPIFC" -link_info 2>/dev/null || "$MPIFC" -show 2>/dev/null ||
           "$MPIFC" --cray-print-opts=libs 2>/dev/null || true)"
  flags="$(printf '%s\n' "$flags" | tr ',' ' ')"
  for n in $(printf '%s\n' $flags | sed -n 's/^-l//p' | grep -E '^(mpi_mpifh|mpifort[A-Za-z0-9_]*|mpichfort)$' || true); do
    for d in $(printf '%s\n' $flags | sed -n 's/^-L//p'); do
      for f in "$d/lib$n.so" "$d/lib$n.dylib"; do
        if [ -e "$f" ]; then echo "$f"; return 0; fi
      done
    done
    for ext in so dylib; do
      f="$("$MPIFC" -print-file-name="lib$n.$ext" 2>/dev/null || true)"
      case "$f" in /*) if [ -e "$f" ]; then echo "$f"; return 0; fi ;; esac
    done
  done
  return 0
}
SCA_LIBS="$PREFIX/lib/libscalapack.a"
MPIFH="$(mpi_fortran_lib)"
if [ -n "$MPIFH" ]; then
  SCA_LIBS="$SCA_LIBS;$MPIFH"
else
  echo "warning: no MPI Fortran bindings library (libmpi_mpifh, libmpifort*, libmpichfort) found on the link line of" \
       "$MPIFC. If linking axionFEM against this STRUMPACK reports undefined mpi_* symbols (mpi_abort_, ...), run" \
       "this script again with AXF_MPI_FORTRAN_LIBS=/path/to/libmpifort.so (or your MPI's equivalent)." >&2
fi
[ -z "$LAPACK" ] || SCA_LIBS="$SCA_LIBS;$LAPACK"
for ext in so dylib; do
  GF="$("$MPIFC" -print-file-name=libgfortran.$ext 2>/dev/null || true)"
  case "$GF" in /*) if [ -e "$GF" ]; then SCA_LIBS="$SCA_LIBS;$GF"; break; fi ;; esac
done
echo "== ScaLAPACK link: $SCA_LIBS"
RPATH_ARGS=()
while IFS= read -r a; do RPATH_ARGS+=("$a"); done < <(axf_cmake_rpath_args "$PREFIX")
rm -rf "$BLD/STRUMPACK-$SPK_VER" "$BLD/strumpack-build" && tar -xzf "$SRC/STRUMPACK-$SPK_VER.tar.gz" -C "$BLD"
cmake -S "$BLD/STRUMPACK-$SPK_VER" -B "$BLD/strumpack-build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" "${RPATH_ARGS[@]}" \
  -DCMAKE_C_COMPILER="$MPICC" -DCMAKE_CXX_COMPILER="$MPICXX" -DCMAKE_Fortran_COMPILER="$MPIFC" \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DSTRUMPACK_USE_MPI=ON -DSTRUMPACK_USE_OPENMP=ON \
  -DSTRUMPACK_USE_CUDA=OFF -DSTRUMPACK_USE_HIP=OFF -DSTRUMPACK_USE_SYCL=OFF \
  -DTPL_ENABLE_SCOTCH=OFF -DTPL_ENABLE_PTSCOTCH=OFF -DTPL_ENABLE_PARMETIS=OFF -DTPL_ENABLE_ZFP=OFF \
  -DTPL_ENABLE_BPACK=OFF -DTPL_ENABLE_SLATE=OFF -DTPL_ENABLE_COMBBLAS=OFF -DTPL_ENABLE_MAGMA=OFF \
  -DTPL_ENABLE_KBLAS=OFF -DTPL_ENABLE_SZ=OFF \
  -DTPL_METIS_INCLUDE_DIRS="$DEPS/include" -DTPL_METIS_LIBRARIES="$METIS_LIB" \
  ${SPK_BLAS[@]+"${SPK_BLAS[@]}"} -DTPL_SCALAPACK_LIBRARIES="$SCA_LIBS" \
  > "$BLD/strumpack-cmake.log" 2>&1 ||
  { tail -40 "$BLD/strumpack-cmake.log"; axf_die "configuring STRUMPACK failed (log: $BLD/strumpack-cmake.log)"; }
cmake --build "$BLD/strumpack-build" -j "$NPROC" > "$BLD/strumpack-build.log" 2>&1 ||
  { tail -40 "$BLD/strumpack-build.log"; axf_die "building STRUMPACK failed (log: $BLD/strumpack-build.log)"; }
cmake --install "$BLD/strumpack-build" > /dev/null
grep -q "define STRUMPACK_USE_MPI" "$PREFIX/include/StrumpackConfig.h" ||
  axf_die "STRUMPACK in $PREFIX was not built with MPI"
echo "STRUMPACK $SPK_VER (MPI) installed into $PREFIX"
