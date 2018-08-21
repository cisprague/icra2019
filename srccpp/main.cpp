// Christopher Iliffe Sprague
// christopher.iliffe.sprague@gmail.com

#include <iostream>
#include <vector>
#include "dynamics.hpp"
#include "direct.hpp"
#include "utils.hpp"

int main() {

  // state
  const std::vector<double> x{0.5, 0.1, 0.1, 0.1};

  // control
  const double u(0.5);

  // state transition
  const std::vector<double> dxdt = dynamics::eom_state(x, u);

  // state jacobian
  const std::vector<std::vector<double>> ddxddt = dynamics::eom_state_jac(x, u);

  // instantiate direct segment
  direct seg(4, {0,0,0,0}, {1,0,0,0});

  // decision vector
  const std::vector<double> dv{10, 0, 0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5};

  // fitness
  std::vector<double> fit = seg.fitness(dv);

  //utils::print(dxdt);




  return 0;
};
