#include "axionfem/core.hpp"

#include <iostream>

namespace axf {

int& log_level() {
  static int level = 1;
  return level;
}

void log(int level, const std::string& msg) {
  if (level <= log_level()) std::cerr << "[axf] " << msg << std::endl;
}

}  // namespace axf
