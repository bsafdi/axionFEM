// Tiny arithmetic expression evaluator for design parameters.
//
// Grammar (standard math precedence, loosest first):
//   expr    := term (('+'|'-') term)*
//   term    := unary (('*'|'/') unary)*
//   unary   := ('+'|'-') unary | power
//   power   := primary ('^' unary)?
//   primary := number | identifier | identifier '(' args ')' | '(' expr ')'
// So unary minus binds looser than '^' (-2^2 = -4, exp(-z^2) = exp(-(z^2))), '^' is
// right-associative (2^3^2 = 2^9 = 512) and an exponent may be signed (2^-2 = 0.25,
// -x^-2 = -(x^-2)).
// Identifiers resolve to parameters or the constants pi, e, mu0.
// Functions: sqrt exp log log10 sin cos tan asin acos atan atan2 abs min max pow.
// Numbers are strict: the whole token ([0-9.]* with an optional exponent) must be one literal
// ('1.2.5', '1..5' and '.' are errors), and it must be 0 or a normal double (1e400 and the subnormal
// 1e-310 are errors).  min/max reject NaN arguments.  Nesting (signs, '^' exponents, parentheses,
// function arguments) is capped at 1000 levels.  Errors name the expression, the JSON path and a fix.
#pragma once

#include <map>
#include <string>

namespace axf {

using ParamMap = std::map<std::string, double>;

// Evaluate `text`; throws axf::Error naming `where` on failure.
double eval_expr(const std::string& text, const ParamMap& params, const std::string& where = "");

// Resolve a set of possibly inter-dependent parameter definitions (number or
// expression strings) in dependency order.  Detects cycles and unknown names.
ParamMap resolve_params(const std::map<std::string, std::string>& defs, const std::string& where);

}  // namespace axf
