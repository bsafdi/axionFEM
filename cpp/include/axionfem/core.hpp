// axionFEM core utilities: errors, logging, small helpers.
#pragma once

#include <chrono>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace axf {

inline constexpr const char* kVersion = "0.1.0";

// Default visibility for types thrown in libaxionfem and caught elsewhere.  The
// Python module is built with -fvisibility=hidden, so it would otherwise get its
// own hidden type_info for axf::Error; libc++ (macOS) compares type_info by
// address, the module's `catch (const axf::Error&)` then misses, and Python sees
// RuntimeError instead of AxionFEMError.
#if defined(__GNUC__) || defined(__clang__)
#define AXF_EXPORT __attribute__((visibility("default")))
#else
#define AXF_EXPORT
#endif

// An error that tells an agent or a person what went wrong, where, and how to
// fix it.  `where` is a JSON-pointer-like path into the design spec when the
// error concerns the input ("/regions/3/shape").
class AXF_EXPORT Error : public std::runtime_error {
 public:
  Error(std::string what, std::string where = "", std::string hint = "")
      : std::runtime_error(compose(what, where, hint)),
        what_(std::move(what)), where_(std::move(where)), hint_(std::move(hint)) {}
  const std::string& message() const { return what_; }
  const std::string& where() const { return where_; }
  const std::string& hint() const { return hint_; }

 private:
  static std::string compose(const std::string& w, const std::string& p, const std::string& h) {
    std::string s = w;
    if (!p.empty()) s += " [at " + p + "]";
    if (!h.empty()) s += " Hint: " + h;
    return s;
  }
  std::string what_, where_, hint_;
};

// A linear solve or factorization that broke down (singular, indefinite, or a residual check that failed) deep in an
// engine that does not know the design's region names. The model that owns the mesh catches it and rethrows an Error
// with the mesh check appended (the worst elements and how to fix them; axionfem/mesh_quality.hpp).
class AXF_EXPORT SolverFailure : public Error {
 public:
  using Error::Error;
};

#define AXF_REQUIRE(cond, ...)                         \
  do {                                                 \
    if (!(cond)) throw ::axf::Error(__VA_ARGS__);      \
  } while (0)

// Verbosity: 0 quiet, 1 progress, 2 debug.  Messages go to stderr so that
// stdout can carry machine-readable output.
int& log_level();
void log(int level, const std::string& msg);

class Timer {
 public:
  Timer() : t0_(std::chrono::steady_clock::now()) {}
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }

 private:
  std::chrono::steady_clock::time_point t0_;
};

template <class... Args>
std::string strcat(const Args&... args) {
  std::ostringstream os;
  (os << ... << args);
  return os.str();
}

}  // namespace axf
