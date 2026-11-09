#!/usr/bin/env bash
# Download the pinned third-party sources into third_party/src and verify each archive against
# third_party/SHA256SUMS.txt. Network access only, no compilation. Safe to re-run: archives already present are
# verified and reused.
#
#   bash scripts/fetch_deps.sh                      # download what is missing, verify everything
#   bash scripts/fetch_deps.sh --update-checksums   # after changing a version below: record the checksums of
#                                                   # the downloaded archives (check them against upstream first)
#
# AXF_DEPS_SRC sets the download directory; AXF_GMSH_SDK picks the gmsh SDK (see scripts/common.sh): by default
# the one for this platform (Linux x86_64: the headless "nox" build; macOS: x86_64 or arm64) and none elsewhere,
# where the PyPI gmsh wheel provides libgmsh instead.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODE=verify
case "${1:-}" in
  "") ;;
  --update-checksums) MODE=update ;;
  -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
  *) axf_die "unknown argument '$1' (use --update-checksums or nothing)" ;;
esac

SRC="${AXF_DEPS_SRC:-$AXF_ROOT/third_party/src}"
mkdir -p "$SRC"

MFEM_VER=4.10
HYPRE_VER=2.33.0
METIS_VER=5.1.0
EIGEN_VER=3.4.0
JSON_VER=3.12.0
GMSH_VER=4.15.2

ARCHIVES=(
  "mfem-${MFEM_VER}.tar.gz https://github.com/mfem/mfem/archive/refs/tags/v${MFEM_VER}.tar.gz"
  "hypre-${HYPRE_VER}.tar.gz https://github.com/hypre-space/hypre/archive/refs/tags/v${HYPRE_VER}.tar.gz"
  "metis-${METIS_VER}.tar.gz https://github.com/mfem/tpls/raw/gh-pages/metis-${METIS_VER}.tar.gz"
  "eigen-${EIGEN_VER}.tar.gz https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VER}/eigen-${EIGEN_VER}.tar.gz"
  "json-${JSON_VER}.hpp https://github.com/nlohmann/json/releases/download/v${JSON_VER}/json.hpp"
)
GMSH="$(axf_gmsh_sdk "$GMSH_VER")"
if [ -n "$GMSH" ]; then
  ARCHIVES+=("$GMSH")
else
  echo "== no gmsh SDK for $AXF_OS $AXF_ARCH (AXF_GMSH_SDK=${AXF_GMSH_SDK:-auto}): axf-mesher will be built against"
  echo "   the PyPI gmsh wheel; install it into the Python environment used for CMake: pip install 'gmsh>=4.15,<4.16'"
fi

echo "== sources in $SRC"
for entry in "${ARCHIVES[@]}"; do
  name="${entry%% *}"; url="${entry#* }"
  [ -s "$SRC/$name" ] || axf_download "$url" "$SRC/$name"
  if [ "$MODE" = update ]; then axf_record_sha "$SRC" "$name"; else axf_verify "$SRC" "$name"; fi
done
if [ "$MODE" = update ]; then
  echo "done; checksums recorded in third_party/SHA256SUMS.txt (review the diff before committing)"
else
  echo "FETCH_DEPS_OK"
fi
