#pragma once
#include <cmath>
namespace axf::detail {
// Stable positive root of delta*(width+delta)=width^2/(cap-1).
inline double shell_delta(double width, double cap) {
  return 2*width/((cap-1)*(std::sqrt(1+4/(cap-1))+1));
}
inline bool valid_element_width(double width, double maximum_width) {
  return std::isfinite(width) && std::isfinite(maximum_width) &&
         maximum_width > 0 && width >= 1e-6*maximum_width &&
         width > 1e-140;
}
}
