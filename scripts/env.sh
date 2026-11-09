# Source this file (do not execute it) to put the axionFEM dependency prefix on the search paths of the build and
# of the programs that use it:
#
#   source scripts/env.sh
#
# It loads nothing and changes no MPI settings. Provide a C++17 compiler, MPI, BLAS/LAPACK, CMake and
# Python yourself (distribution packages, Homebrew, conda, environment modules, ...); docs/guides/install.md lists
# what each platform needs. For site-specific setup that should run whenever this file is sourced (for example
# `module load ...` lines), put it in a file of your own and point AXF_ENV_HOOK at it:
#
#   AXF_ENV_HOOK=$HOME/axf-site.sh source scripts/env.sh
#
# Variables set: AXIONFEM_ROOT (this checkout), AXIONFEM_DEPS (default third_party/install; set it beforehand to
# use another prefix), CMAKE_PREFIX_PATH, PATH, LD_LIBRARY_PATH (Linux; macOS relies on the install names and
# rpaths that scripts/build_deps.sh writes), AXIONFEM_GMSH when the gmsh SDK is installed, and
# OPENBLAS_NUM_THREADS=1 unless it is set (BLAS is called from inside OpenMP regions, e.g. by STRUMPACK).

if [ -n "${AXF_ENV_HOOK:-}" ]; then
  if [ -f "$AXF_ENV_HOOK" ]; then
    # shellcheck disable=SC1090
    . "$AXF_ENV_HOOK"
  else
    echo "scripts/env.sh: AXF_ENV_HOOK=$AXF_ENV_HOOK is not a file; ignored" >&2
  fi
fi

_axf_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export AXIONFEM_ROOT="${AXIONFEM_ROOT:-$_axf_here}"
export AXIONFEM_DEPS="${AXIONFEM_DEPS:-$AXIONFEM_ROOT/third_party/install}"

export CMAKE_PREFIX_PATH="$AXIONFEM_DEPS${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PATH="$AXIONFEM_DEPS/bin${PATH:+:$PATH}"
if [ "$(uname -s)" != Darwin ]; then
  export LD_LIBRARY_PATH="$AXIONFEM_DEPS/lib:$AXIONFEM_DEPS/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# gmsh is used only by the separate axf-mesher program (it is GPL; see docs/install.md, "Licensing of each wheel").
if [ -x "$AXIONFEM_DEPS/gmsh-sdk/bin/gmsh" ]; then
  export AXIONFEM_GMSH="${AXIONFEM_GMSH:-$AXIONFEM_DEPS/gmsh-sdk/bin/gmsh}"
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
unset _axf_here
