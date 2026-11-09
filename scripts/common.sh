# Helpers shared by the dependency scripts (fetch_deps.sh, build_deps*.sh, build_strumpack*.sh).
# Sourced, not executed. Written for bash on Linux (GNU tools) and macOS (BSD tools).
#
# Environment understood by the scripts that source this file:
#   CC, CXX, FC            serial compilers (default: cc, c++, gfortran)
#   MPICC, MPICXX, MPIFC   MPI compiler wrappers (default: mpicc, mpicxx, mpif90); on a Cray system,
#                          for example, MPICC=cc MPICXX=CC MPIFC=ftn
#   NPROC                  parallel build jobs (default: the number of usable cores)
#   AXF_DEPS_SRC           where the source archives are (default: third_party/src)
#   AXF_BLA_VENDOR         passed to CMake as BLA_VENDOR (e.g. OpenBLAS, FlexiBLAS, Generic, Apple, Intel10_64lp)
#   AXF_BLAS_LIBS, AXF_LAPACK_LIBS
#                          explicit BLAS / LAPACK libraries (;-separated) for STRUMPACK and ScaLAPACK
#   AXF_GMSH_SDK           which gmsh SDK to fetch and install: auto (default), linux64-nox, linux64,
#                          macos-x86_64, macos-arm64, or none

AXF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AXF_SUMS="$AXF_ROOT/third_party/SHA256SUMS.txt"
AXF_OS="$(uname -s)"
AXF_ARCH="$(uname -m)"

axf_die() { echo "error: $*" >&2; exit 1; }

# Parallel jobs: $NPROC, else the cores this process may use. GNU nproc also obeys OMP_NUM_THREADS and
# OMP_THREAD_LIMIT, which batch scripts often set to 1 for the run, so both are removed for the query.
axf_nproc() {
  if [ -n "${NPROC:-}" ]; then echo "$NPROC"; return; fi
  if command -v nproc >/dev/null 2>&1; then env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc; return; fi
  if [ "$AXF_OS" = Darwin ]; then sysctl -n hw.ncpu; return; fi
  getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4
}

# sha256 of a file (GNU coreutils or the macOS/BSD shasum).
axf_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else axf_die "neither sha256sum nor shasum is available"
  fi
}

# In-place sed that works with GNU and BSD sed.
axf_sed_inplace() {  # axf_sed_inplace EXPR FILE
  sed -i.axfbak "$1" "$2" && rm -f "$2.axfbak"
}

# Download URL to FILE (via a .part file, so that an interrupted download is not mistaken for a finished one).
axf_download() {  # axf_download URL FILE
  echo "== fetching $(basename "$2")"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$2.part" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$2.part" "$1"
  else
    axf_die "neither curl nor wget is available to download $1"
  fi
  mv "$2.part" "$2"
}

# The checksum recorded for an archive in third_party/SHA256SUMS.txt (entries "sha  name" or "sha  ./name").
axf_recorded_sha() {  # axf_recorded_sha NAME
  [ -f "$AXF_SUMS" ] || return 0
  awk -v n="$1" '{f=$2; sub(/^\.\//, "", f); if (f == n) {print $1; exit}}' "$AXF_SUMS"
}

# Verify DIR/NAME against third_party/SHA256SUMS.txt; exit with an explanation if it does not match.
axf_verify() {  # axf_verify DIR NAME
  local want got
  want="$(axf_recorded_sha "$2")"
  [ -n "$want" ] || axf_die "no checksum recorded for $2 in $AXF_SUMS.
  After a version change, check the new archive against upstream and record it with
  bash scripts/fetch_deps.sh --update-checksums"
  got="$(axf_sha256 "$1/$2")"
  if [ "$got" != "$want" ]; then
    axf_die "$1/$2 does not match its recorded sha256.
  recorded: $want
  file:     $got
  Delete the file and run the script again to download it afresh. If upstream really changed the archive,
  check it and record the new checksum with: bash scripts/fetch_deps.sh --update-checksums"
  fi
  echo "   ok  $2"
}

# Download DIR/NAME from URL if it is missing, then verify it.
axf_fetch_verified() {  # axf_fetch_verified DIR NAME URL
  [ -s "$1/$2" ] || axf_download "$3" "$1/$2"
  axf_verify "$1" "$2"
}

# Record (or replace) the checksum of DIR/NAME in third_party/SHA256SUMS.txt.
axf_record_sha() {  # axf_record_sha DIR NAME
  local sha tmp
  sha="$(axf_sha256 "$1/$2")"
  tmp="$AXF_SUMS.tmp"
  touch "$AXF_SUMS"
  awk -v n="$2" '{f=$2; sub(/^\.\//, "", f); if (f != n) print}' "$AXF_SUMS" > "$tmp"
  echo "$sha  $2" >> "$tmp"
  LC_ALL=C sort -k2 "$tmp" > "$AXF_SUMS" && rm -f "$tmp"
  echo "   recorded $sha  $2"
}

# A command that must exist, with a hint on how to get it.
axf_require() {  # axf_require CMD HINT
  command -v "$1" >/dev/null 2>&1 || axf_die "'$1' not found. $2"
}

# tar and gzip unpack every archive (minimal container images and RHEL 9 installs may lack them).
axf_require_tar() {
  axf_require tar "Install tar (Debian/Ubuntu: tar; Fedora/RHEL: tar)."
  axf_require gzip "Install gzip (Debian/Ubuntu: gzip; Fedora/RHEL: gzip)."
}

AXF_MPI_HINT="Install an MPI implementation with its compiler wrappers (Debian/Ubuntu: libopenmpi-dev openmpi-bin; \
Fedora/RHEL: openmpi-devel or mpich-devel, then 'module load mpi/openmpi-x86_64' or put /usr/lib64/openmpi/bin on \
PATH; macOS: brew install open-mpi), or name the wrappers with MPICC/MPICXX (e.g. MPICC=cc MPICXX=CC on Cray systems)."

# Is the shared library libNAME (.so or .dylib) installed under PREFIX/lib or PREFIX/lib64? (The scripts build
# shared libraries and link them by these names; a static-only libNAME.a does not count.)
axf_have_lib() {  # axf_have_lib NAME PREFIX
  local f
  for f in "$2"/lib/lib"$1".so "$2"/lib64/lib"$1".so "$2"/lib/lib"$1".dylib "$2"/lib64/lib"$1".dylib; do
    [ -e "$f" ] && return 0
  done
  return 1
}

# Path of the shared library libNAME under PREFIX (lib or lib64; .so or .dylib), empty if absent.
# (Status 0 either way: the scripts call it as VAR=$(...) under set -e.)
axf_lib_path() {  # axf_lib_path NAME PREFIX
  local f
  for f in "$2"/lib/lib"$1".so "$2"/lib64/lib"$1".so "$2"/lib/lib"$1".dylib "$2"/lib64/lib"$1".dylib; do
    if [ -e "$f" ]; then echo "$f"; return 0; fi
  done
  return 0
}

# CMake arguments that make installed shared libraries find their dependencies in PREFIX/lib without
# LD_LIBRARY_PATH: an RPATH on Linux, absolute install names on macOS.
axf_cmake_rpath_args() {  # axf_cmake_rpath_args PREFIX
  printf '%s\n' "-DCMAKE_INSTALL_LIBDIR=lib" "-DCMAKE_INSTALL_RPATH=$1/lib" "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON"
  if [ "$AXF_OS" = Darwin ]; then printf '%s\n' "-DCMAKE_INSTALL_NAME_DIR=$1/lib"; fi
}

# Optional -DBLA_VENDOR for CMake's FindBLAS/FindLAPACK.
axf_bla_vendor_args() {
  if [ -n "${AXF_BLA_VENDOR:-}" ]; then echo "-DBLA_VENDOR=$AXF_BLA_VENDOR"; fi
}

# BLAS/LAPACK libraries for STRUMPACK and ScaLAPACK, as two lines "BLAS" and "LAPACK" (;-separated lists).
# Order: AXF_BLAS_LIBS/AXF_LAPACK_LIBS; OpenBLAS through pkg-config; Homebrew's OpenBLAS; otherwise empty, and
# the CMake projects search themselves (FindBLAS/FindLAPACK: reference LAPACK, FlexiBLAS, MKL, Accelerate, ...,
# steered by AXF_BLA_VENDOR).
axf_blas_lapack() {
  local blas="${AXF_BLAS_LIBS:-}" lapack="${AXF_LAPACK_LIBS:-}" d f
  if [ -z "$blas" ] && command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openblas 2>/dev/null; then
    d="$(pkg-config --variable=libdir openblas)"
    for f in "$d/libopenblas.so" "$d/libopenblas.dylib"; do if [ -e "$f" ]; then blas="$f"; break; fi; done
  fi
  if [ -z "$blas" ] && [ "$AXF_OS" = Darwin ] && command -v brew >/dev/null 2>&1; then
    d="$(brew --prefix openblas 2>/dev/null || true)"
    if [ -n "$d" ] && [ -e "$d/lib/libopenblas.dylib" ]; then blas="$d/lib/libopenblas.dylib"; fi
  fi
  [ -n "$lapack" ] || lapack="$blas"
  echo "$blas"
  echo "$lapack"
}

# The gmsh SDK archive for this platform ("" when there is none: then use the PyPI gmsh wheel).
# On Linux x86_64 the "nox" SDK (no FLTK/OpenGL) is the default, so that no X11/OpenGL libraries are needed.
axf_gmsh_sdk() {  # axf_gmsh_sdk VERSION -> "<archive> <url>" or ""
  local v="$1" kind="${AXF_GMSH_SDK:-auto}"
  if [ "$kind" = auto ]; then
    case "$AXF_OS-$AXF_ARCH" in
      Linux-x86_64) kind=linux64-nox ;;
      Darwin-x86_64) kind=macos-x86_64 ;;
      Darwin-arm64) kind=macos-arm64 ;;
      *) kind=none ;;
    esac
  fi
  case "$kind" in
    linux64-nox) echo "gmsh-$v-nox-Linux64-sdk.tgz https://gmsh.info/bin/Linux/gmsh-$v-nox-Linux64-sdk.tgz" ;;
    linux64) echo "gmsh-$v-Linux64-sdk.tgz https://gmsh.info/bin/Linux/gmsh-$v-Linux64-sdk.tgz" ;;
    macos-x86_64) echo "gmsh-$v-MacOSX-sdk.tgz https://gmsh.info/bin/macOS/gmsh-$v-MacOSX-sdk.tgz" ;;
    macos-arm64) echo "gmsh-$v-MacOSARM-sdk.tgz https://gmsh.info/bin/macOS/gmsh-$v-MacOSARM-sdk.tgz" ;;
    none) echo "" ;;
    *) axf_die "AXF_GMSH_SDK=$kind is not one of auto, linux64-nox, linux64, macos-x86_64, macos-arm64, none" ;;
  esac
}
