// Christopher Iliffe Sprague
// christopher.iliffe.sprageu@gmail.com

#ifndef dynamics_hpp
#define dynamics_hpp
#include <cmath>
#include <vector>

namespace dynamics {

  // state and control dimensions
  const unsigned short int xdim(4);
  const unsigned short int udim(1);

  // state bounds
  const std::vector<double> xlb{-5, -3, 0, -3};
  const std::vector<double> xub{5, 3, 2*M_PI, 3};

  // control bounds
  const double ulb(-1);
  const double uub(1);

  // state equations of motion
  std::vector<double> eom_state(const std::vector<double> &x, const double &u) {

    std::vector<double> dxdt(xdim);

    dxdt[0] = x[1];
    dxdt[1] = u;
    dxdt[2] = x[3];
    dxdt[3] = std::sin(x[2]) - u*std::cos(x[2]);
    
    return dxdt;

  };

  // state equations of motion Jacobian
  std::vector<std::vector<double>> eom_state_jac(const std::vector<double> &x, const double &u) {

    // Jacobian matrix
    std::vector<std::vector<double>> ddxddt(xdim, std::vector<double>(xdim));

    ddxddt[0][0] = 0;
    ddxddt[0][1] = 1;
    ddxddt[0][2] = 0;
    ddxddt[0][3] = 0;

    ddxddt[1][0] = 0;
    ddxddt[1][1] = 0;
    ddxddt[1][2] = 0;
    ddxddt[1][3] = 0;

    ddxddt[2][0] = 0;
    ddxddt[2][1] = 0;
    ddxddt[2][2] = 0;
    ddxddt[2][3] = 1;

    ddxddt[3][0] = 0;
    ddxddt[3][1] = 0;
    ddxddt[3][2] = u*std::sin(x[2]) + std::cos(x[2]);
    ddxddt[3][3] = 0;

    return ddxddt;

  };

};

#endif
