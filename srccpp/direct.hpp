// Christopher Iliffe Sprague
// christopher.iliffe.sprague@gmail.com

#ifndef direct_hpp
#define direct_hpp
#include <vector>
#include <cmath>
#include "dynamics.hpp"
#include "utils.hpp"

struct direct {

  // number of segments
  unsigned short int N;

  // boundary conditions
  std::vector<double> state_0;
  std::vector<double> state_f;

  // construct with number of nodes and boundary condtions
  direct(const unsigned short int &N_, const std::vector<double> &s0, const std::vector<double> &sf): N(N_), state_0(s0), state_f(sf) {};

  // ftiness vector
  std::vector<double> fitness(const std::vector<double> &dv) const {

    // trajectory duration
    const double T = dv[0];

    // time step size
    const double dt = T/(N-1);

    // time grid
    std::vector<double> times_(N);
    for (int i=0; i<N; i++) {times_[i] = i*dt;}

    // states and controls
    std::vector<std::vector<double>> states_(N, std::vector<double>(dynamics::xdim));
    std::vector<double> controls_(N);
    for (int i=1, j=0; i<N*(dynamics::xdim+dynamics::udim); i+=dynamics::xdim+dynamics::udim, j++) {

      // state
      states_[j][0] = dv[i];
      states_[j][1] = dv[i+1];
      states_[j][2] = dv[i+2];
      states_[j][3] = dv[i+3];

      // controls
      controls_[j] = dv[i+4];

    }
    utils::print(states_);

    // compute objective and equality constraints
    double J(0);
    std::vector<double> ec;
    for (int k=0; k<N-1; k++) {

      // intial, final, and mid controls
      double u0  = controls_[k];
      double u1  = controls_[k+1];
      double u05 = (u0 + u1)/2;

      // initial and final states
      std::vector<double> x0 = states_[k];
      std::vector<double> x1 = states_[k+1];

      // initial and final dynamics
      std::vector<double> f0 = dynamics::eom_state(x0, u0);
      std::vector<double> f1 = dynamics::eom_state(x1, u1);

      // mid state
      std::vector<double> x05(dynamics::xdim);
      for (int i=0; i<dynamics::xdim; i++) {
        x05[i] = (x0[i] + x1[i])/2 + dt*(f0[i] - f1[i])/8;
      }

      // mid dynamics
      std::vector<double> f05 = dynamics::eom_state(x05, u05);

      // compute mismatch
      for (int i=0; i<dynamics::xdim; i++) {
        ec.push_back(dt*(f0[i] + 4*f05[i] + f1[i])/6 + x0[i] - x1[i]);
      }

      // compute objective
      J += dt*(std::pow(u0,2) + 4*std::pow(u05,2) + std::pow(u1,2))/6;
    }

    // boundary conditions
    for (int i=0; i<dynamics::xdim; i++) {
      ec.push_back(state_0[i] - states_[0][i]);
      ec.push_back(state_f[i] - states_[dynamics::xdim][i]);
    }

    // return fitness vector
    std::vector<double> fit(ec);
    fit.insert(fit.begin(), J);
    std::cout << J << std::endl;
    return fit;

  };

  // number of ojectives
  std::vector<double> get_nobj(void) const {
    return {1};
  };



};

#endif
