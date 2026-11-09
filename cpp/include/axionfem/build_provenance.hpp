// Build provenance of libaxionfem itself.
#pragma once
#include <string>

namespace axf {
// JSON {"git_sha", "git_dirty", "git_dirty_files", "build_dir"}: the first three compiled into THIS shared library
// from the header that cmake/axf_gitinfo.cmake regenerates on every build (the Python module compiles the same header
// and reports both, so a library rebuilt from other sources disagrees); "build_dir" is the directory the loaded
// libaxionfem was found in, determined at run time (no build path is compiled in; "" where the platform cannot say).
// Comparing the two shas detects a Python module built from other sources than the library it loaded.
std::string library_build_info();
}  // namespace axf
