#pragma once
#include <sstream>
#include <stdexcept>
#include <string>
namespace irspack {

inline void check_arg(bool condition, const std::string &failure_message) {
  if (!condition) {
    throw std::invalid_argument(failure_message);
  }
}
template <typename Field>
void check_arg_lower_bounded(Field x, Field low, const std::string &varname) {
  if ((x < low)) {
    std::stringstream ss;
    ss << varname << " must be greater than or equal to  " << low;
    throw std::invalid_argument(ss.str());
  }
}

template <typename Field>
void check_arg_upper_bounded(Field x, Field up, const std::string &varname) {
  if ((x > up)) {
    std::stringstream ss;
    ss << varname << " must be less than or equal to  " << up;
    throw std::invalid_argument(ss.str());
  }
}

template <typename Field>
void check_arg_doubly_bounded(Field x, Field low, Field up,
                              const std::string &varname) {
  check_arg_lower_bounded(x, low, varname);
  check_arg_upper_bounded(x, up, varname);
}

} // namespace irspack
