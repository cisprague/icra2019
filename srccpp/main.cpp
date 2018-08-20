// Christopher Iliffe Sprague
// christopher.iliffe.sprague@gmail.com

#include <iostream>
#include <vector>
#include "dynamics.hpp"

int main() {

  // state
  std::vector<double> x{0.1, 0.2, 0.3, 0.4};

  // control
  double u(0.5);

  // state transition array
  std::vector<double> dxdt(dynamics::xdim);

  // state Jacobian array
  std::vector<std::vector<double>> ddxddt(dynamics::xdim, std::vector<double>(dynamics::xdim));

  // compute state transition
  dynamics::eom_state(x, u, dxdt);

  // compute Jacobian
  dynamics::eom_state_jac(x, u, ddxddt);

  for (int i=0; i<dynamics::xdim; i++) {
    std::cout << dxdt[i];
  };
  std::cout << std::endl;

  for (int i=0; i<dynamics::xdim; i++) {
    for (int j=0; j<dynamics::xdim; j++) {
      std::cout << ddxddt[i][j] << std::endl;
    };
  };


}
