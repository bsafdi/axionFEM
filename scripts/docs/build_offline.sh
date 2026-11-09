#!/usr/bin/env bash
# Build the documentation site so that it browses from disk (open site/index.html,
# no web server): plain .html links instead of directory URLs, and the Material
# "offline" plugin (search index inlined, no fetch()).  MathJax and the notebook
# renderer still load from a CDN, so formulas need an internet connection.
# The site goes to ./site unless -d is given (relative paths: from the caller's cwd).
#
#   bash scripts/docs/build_offline.sh [extra mkdocs build args, e.g. -d /path/to/site]
set -euo pipefail
# make a relative -d/--site-dir absolute before changing to the repository root
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -d|--site-dir) args+=("$1" "$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"); shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
cd "$(dirname "$0")/../.."
export AXF_DOCS_OFFLINE=true AXF_DOCS_DIRECTORY_URLS=false
export NO_MKDOCS_2_WARNING=1 DISABLE_MKDOCS_2_WARNING=true
exec mkdocs build --strict "${args[@]}"
