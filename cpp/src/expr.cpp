#include "axionfem/expr.hpp"

#include <cctype>
#include <cmath>
#include <functional>
#include <set>
#include <stdexcept>
#include <vector>

#include "axionfem/core.hpp"
#include "axionfem/units.hpp"

namespace axf {
namespace {

constexpr int kMaxExprDepth = 1000;

struct Parser {
  const std::string& s;
  const ParamMap& p;
  const std::string& where;
  size_t i = 0;
  std::set<std::string>* missing = nullptr;  // collect unknown identifiers instead of failing
  int depth = 0;                              // nesting of signs, '^' exponents and parentheses

  [[noreturn]] void fail(const std::string& msg,
                         const std::string& hint = "Use numbers, parameter names, + - * / ^, parentheses and "
                                                   "functions such as sqrt, exp, log, min, max.") const {
    throw Error(strcat("cannot evaluate expression '", s, "': ", msg, " (near position ", i, ")"), where, hint);
  }
  void ws() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
  }
  bool eat(char c) {
    ws();
    if (i < s.size() && s[i] == c) {
      ++i;
      return true;
    }
    return false;
  }
  double expr() {
    double v = term();
    for (;;) {
      if (eat('+')) v += term();
      else if (eat('-')) v -= term();
      else return v;
    }
  }
  double term() {
    double v = unary();
    for (;;) {
      if (eat('*')) v *= unary();
      else if (eat('/')) v /= unary();
      else return v;
    }
  }
  // Standard precedence: unary minus binds looser than '^' (-2^2 = -4), '^' is
  // right-associative (2^3^2 = 2^9) and its exponent may carry a sign (2^-2 = 0.25).
  // Every level of nesting (a sign, a '^' exponent, a parenthesis or a function argument) passes
  // through unary(), so the cap here bounds the recursion and turns a stack overflow into an error.
  double unary() {
    if (++depth > kMaxExprDepth) {
      fail(strcat("nesting deeper than ", kMaxExprDepth, " levels"),
           "Simplify the expression, or split it into several parameters.");
    }
    double v = 0.0;
    if (eat('-')) v = -unary();
    else if (eat('+')) v = unary();
    else v = power();
    --depth;
    return v;
  }
  double power() {
    double b = primary();
    if (eat('^')) return std::pow(b, unary());
    return b;
  }
  double number() {
    ws();
    size_t j = i;
    while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '.')) ++j;
    if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
      size_t k = j + 1;
      if (k < s.size() && (s[k] == '+' || s[k] == '-')) ++k;
      if (k < s.size() && std::isdigit(static_cast<unsigned char>(s[k]))) {
        j = k;
        while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
      }
    }
    // Strict: the whole token must be one number (no silent '1.2.5' -> 1.2), and it must be zero or a
    // normal double; std::stod reports overflow (1e400) and underflow (the subnormal 1e-310, 1e-400 -> 0)
    // as out_of_range, and such literals are rejected rather than rounded.  An exact 0e-400 is fine.
    const std::string tok = s.substr(i, j - i);
    size_t pos = 0;
    double v = 0.0;
    try {
      v = std::stod(tok, &pos);
    } catch (const std::out_of_range&) {
      fail(strcat("number '", tok, "' is out of the range of double precision"),
           "Numbers must be 0 or have a magnitude between 2.2e-308 and 1.7e308; check the exponent.");
    } catch (const std::invalid_argument&) {
      fail(strcat("malformed number '", tok, "'"),
           "Write numbers as 12, 1.25, .5, 2. or 1.5e-3 (one decimal point, digits in the exponent).");
    }
    if (pos != tok.size()) {
      fail(strcat("malformed number '", tok, "'"),
           "Write numbers as 12, 1.25, .5, 2. or 1.5e-3 (one decimal point, digits in the exponent).");
    }
    i = j;
    return v;
  }
  double primary() {
    ws();
    if (i >= s.size()) fail("unexpected end");
    char c = s[i];
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') return number();
    if (eat('(')) {
      double v = expr();
      if (!eat(')')) fail("missing ')'");
      return v;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t j = i;
      while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
      std::string id = s.substr(i, j - i);
      i = j;
      if (eat('(')) {
        std::vector<double> a;
        if (!eat(')')) {
          do a.push_back(expr());
          while (eat(','));
          if (!eat(')')) fail("missing ')' after function arguments");
        }
        return call(id, a);
      }
      auto it = p.find(id);
      if (it != p.end()) return it->second;
      if (id == "pi") return units::pi;
      if (id == "e") return std::exp(1.0);
      if (id == "mu0") return units::mu0;
      if (missing) {
        missing->insert(id);
        return 1.0;
      }
      fail("unknown name '" + id + "'");
    }
    fail(std::string("unexpected character '") + c + "'");
  }
  double call(const std::string& f, const std::vector<double>& a) {
    auto need = [&](size_t n) {
      if (a.size() != n) fail(strcat(f, "() takes ", n, " argument(s)"));
    };
    if (f == "sqrt") { need(1); return std::sqrt(a[0]); }
    if (f == "exp") { need(1); return std::exp(a[0]); }
    if (f == "log") { need(1); return std::log(a[0]); }
    if (f == "log10") { need(1); return std::log10(a[0]); }
    if (f == "sin") { need(1); return std::sin(a[0]); }
    if (f == "cos") { need(1); return std::cos(a[0]); }
    if (f == "tan") { need(1); return std::tan(a[0]); }
    if (f == "asin") { need(1); return std::asin(a[0]); }
    if (f == "acos") { need(1); return std::acos(a[0]); }
    if (f == "atan") { need(1); return std::atan(a[0]); }
    if (f == "atan2") { need(2); return std::atan2(a[0], a[1]); }
    if (f == "abs") { need(1); return std::fabs(a[0]); }
    if (f == "pow") { need(2); return std::pow(a[0], a[1]); }
    if (f == "min" || f == "max") {
      if (a.empty()) fail(f + "() needs arguments");
      // a NaN argument is an error whatever its position (std::min/max would keep or drop it by order);
      // skipped in the dependency dry run, where unknown names stand in as 1
      for (size_t k = 0; k < a.size() && !missing; ++k) {
        if (std::isnan(a[k])) {
          fail(strcat(f, "() argument ", k + 1, " is not a number (NaN)"),
               "Check for sqrt, log, asin or acos of an out-of-range value, or 0/0, inside the arguments.");
        }
      }
      double v = a[0];
      for (double x : a) v = (f == "min") ? std::min(v, x) : std::max(v, x);
      return v;
    }
    fail("unknown function '" + f + "'");
  }
};

}  // namespace

double eval_expr(const std::string& text, const ParamMap& params, const std::string& where) {
  Parser ps{text, params, where};
  double v = ps.expr();
  ps.ws();
  if (ps.i != text.size()) ps.fail("trailing characters");
  if (!std::isfinite(v)) {
    throw Error(strcat("expression '", text, "' evaluated to a non-finite value"), where,
                "Check for division by zero or log of a non-positive number.");
  }
  return v;
}

ParamMap resolve_params(const std::map<std::string, std::string>& defs, const std::string& where) {
  // Determine dependencies by dry-running each expression.
  std::map<std::string, std::set<std::string>> deps;
  for (const auto& [name, txt] : defs) {
    ParamMap empty;
    std::set<std::string> miss;
    Parser ps{txt, empty, where + "/" + name};
    ps.missing = &miss;
    ps.expr();
    for (const auto& m : miss) {
      if (!defs.count(m)) {
        throw Error(strcat("parameter '", name, "' refers to unknown name '", m, "'"), where + "/" + name,
                    "Define it under \"parameters\" or use a number.");
      }
    }
    deps[name] = miss;
  }
  ParamMap out;
  std::set<std::string> visiting;
  std::function<void(const std::string&)> visit = [&](const std::string& n) {
    if (out.count(n)) return;
    if (visiting.count(n)) {
      throw Error(strcat("parameters form a dependency cycle through '", n, "'"), where + "/" + n,
                  "Make sure no parameter depends on itself, directly or indirectly.");
    }
    visiting.insert(n);
    for (const auto& d : deps[n]) visit(d);
    out[n] = eval_expr(defs.at(n), out, where + "/" + n);
    visiting.erase(n);
  };
  for (const auto& kv : defs) visit(kv.first);
  return out;
}

}  // namespace axf
