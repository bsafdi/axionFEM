// The design expression grammar (cpp/include/axionfem/expr.hpp): precedence and associativity
// (unary minus binds looser than '^', '^' is right-associative and its exponent may be signed),
// strict number literals, NaN arguments of min/max, and the nesting-depth cap.
#include "axionfem/expr.hpp"

#include <cmath>
#include <cstdio>
#include <string>

#include "axionfem/core.hpp"

namespace {

int failures = 0;

void check(const std::string& text, const axf::ParamMap& p, double want) {
  double got = axf::eval_expr(text, p, "test_expr");
  double tol = 1e-14 * std::fmax(1.0, std::fabs(want));
  if (!(std::fabs(got - want) <= tol)) {
    std::printf("FAIL: '%s' = %.17g, want %.17g\n", text.c_str(), got, want);
    ++failures;
  }
}

// `text` must be rejected with an axf::Error whose message contains `needle` and names `where`.
void reject(const std::string& text, const axf::ParamMap& p, const std::string& needle) {
  try {
    double v = axf::eval_expr(text, p, "/parameters/q");
    std::printf("FAIL: '%.60s' accepted (= %.17g), want an error containing '%s'\n", text.c_str(), v,
                needle.c_str());
    ++failures;
  } catch (const axf::Error& e) {
    std::string msg = e.what();
    if (msg.find(needle) == std::string::npos || e.where() != "/parameters/q" || e.hint().empty()) {
      std::printf("FAIL: '%.60s' error lacks '%s', the path or a hint: %s\n", text.c_str(), needle.c_str(),
                  msg.c_str());
      ++failures;
    }
  } catch (const std::exception& e) {
    std::printf("FAIL: '%.60s' threw a bare %s instead of axf::Error\n", text.c_str(), e.what());
    ++failures;
  }
}

}  // namespace

int main() {
  const axf::ParamMap p = {{"x", 1.5}, {"z", 0.3}, {"s", 0.7}};
  const double x = 1.5;
  const double z = 0.3;
  const double s = 0.7;
  check("-2^2", p, -4.0);
  check("(-2)^2", p, 4.0);
  check("-(2^2)", p, -4.0);
  check("3*-2^2", p, -12.0);
  check("-x^2", p, -(x * x));
  check("(-x)^2", p, x * x);
  check("-(x^2)", p, -(x * x));
  check("x^-2", p, 1.0 / (x * x));
  check("-x^-2", p, -1.0 / (x * x));
  check("2^-2", p, 0.25);
  check("-2^-2", p, -0.25);
  check("2^3^2", p, 512.0);
  check("(2^3)^2", p, 64.0);
  check("-2^3^2", p, -512.0);
  check("2^-3^2", p, std::pow(2.0, -9.0));
  check("e^-x^2", p, std::exp(-(x * x)));
  check("exp(-z^2/(2*s^2))", p, std::exp(-(z * z) / (2 * s * s)));
  check("1 - -x^2", p, 1.0 + x * x);
  check("2*x^2/4", p, x * x / 2);
  check("--2^2", p, 4.0);
  check("+-2^2", p, -4.0);
  check("2^+2", p, 4.0);
  check("1e-2^2", p, 1e-4);
  check("pow(-2, 2)", p, 4.0);
  // numbers: strict literals (no silent truncation), zero or normal doubles only
  check("2.", p, 2.0);
  check(".5", p, 0.5);
  check("1.5E+2", p, 150.0);
  check("0e-400", p, 0.0);
  check("2.2250738585072014e-308", p, 2.2250738585072014e-308);
  reject("1.2.5", p, "malformed number '1.2.5'");
  reject("1..5", p, "malformed number '1..5'");
  reject(".", p, "malformed number '.'");
  reject("2*.", p, "malformed number '.'");
  reject("1e400", p, "number '1e400' is out of the range");
  reject("-1e400", p, "number '1e400' is out of the range");
  reject("1e-310", p, "number '1e-310' is out of the range");
  reject("1e-400", p, "number '1e-400' is out of the range");
  check("0.0e+999", p, 0.0);
  // min/max: a NaN argument is an error in any position
  check("min(1, 2, -3)", p, -3.0);
  check("max(7)", p, 7.0);
  reject("min(1, sqrt(-1))", p, "min() argument 2 is not a number");
  reject("min(sqrt(-1), 1)", p, "min() argument 1 is not a number");
  reject("max(0/0, 1)", p, "max() argument 1 is not a number");
  // nesting depth: capped at 1000 levels with an error instead of a stack overflow
  std::string deep = std::string(999, '(') + "2" + std::string(999, ')');
  check(deep, p, 2.0);
  check(std::string(998, '-') + "2", p, 2.0);
  reject(std::string(1000, '(') + "2" + std::string(1000, ')'), p, "nesting deeper than 1000 levels");
  reject(std::string(100000, '(') + "2" + std::string(100000, ')'), p, "nesting deeper than 1000 levels");
  std::string tower = "2";
  for (int k = 0; k < 2000; ++k) tower += "^1";
  reject(tower, p, "nesting deeper than 1000 levels");
  reject(std::string(5000, '-') + "2", p, "nesting deeper than 1000 levels");
  if (failures != 0) {
    std::printf("%d expression precedence check(s) failed\n", failures);
    return 1;
  }
  std::puts("expression grammar (precedence, numbers, min/max NaN, depth cap): passed");
  return 0;
}
