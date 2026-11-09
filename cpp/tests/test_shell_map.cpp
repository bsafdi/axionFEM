#include "../src/shell_map.hpp"
#include <cmath>
#include <limits>
#include <cstdio>
#include <initializer_list>
int main() {
  for (double R : {0.01, 1., 100.}) for (double F : {1.1, 3., 5., 20.}) {
    const double d=0.25*R, delta=axf::detail::shell_delta(d,F);
    const double c=R*d/(d+delta)-R;
    auto map=[&](double rho) {return R*d/(R+d+delta-rho)-c;};
    if (std::abs(map(R)/R-1)>1e-13 || std::abs(map(R+d)/R-F)>1e-11) return 1;
    // Manufactured static TE multipoles: radial solution vanishes at the mapped cap.
    // Its derivative at the inner sphere must be the coefficient subtracted by DtN.
    for (int n=1;n<=8;++n) {
      double cap=map(R+d)/R, q=std::pow(cap,-2*n-1);
      auto radial = [&](double r) { return std::pow(r,-n-1)-q*std::pow(r,n); };
      const double eps=1e-5;
      double deriv=(radial(1+eps)-radial(1-eps))/(2*eps);
      double st=-1-deriv/radial(1);
      double nominal=(n+(n+1)*std::pow(F,-2*n-1))/(1-std::pow(F,-2*n-1));
      if (std::abs(st-nominal)>1e-6) return 2;
    }
  }
  if (!axf::detail::valid_element_width(0.001,0.1)) return 3;
  for (double h : {0., -1., 1e-300, 7.6e-16, std::numeric_limits<double>::infinity()})
    if (axf::detail::valid_element_width(h,0.02)) return 4;
  std::puts("capped shell endpoint/multipoles and degenerate width checks: passed");
}
