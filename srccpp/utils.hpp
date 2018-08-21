// Christopher Iliffe Sprague
// christopher.iliffe.sprague@gmail.com

#ifndef utils_hpp
#define utils_hpp
#include <iostream>
#include <vector>

namespace utils {

  void print(const std::vector<double> &v) {

    // print every element
    std::cout << "Vector:" << std::endl;
    std::cout << "[ ";
    for (int i=0; i<v.size(); i++) {std::cout << v[i] << " ";};
    std::cout << "]" << std::endl;

  };

  void print(const std::vector<std::vector<double>> &m) {

    std::cout << "Matrix:" << std::endl;

    // each row
    for (int i=0; i<m.size(); i++) {

      // each element of row
      std::cout << "[";
      for (int j=0; j<m[i].size(); j++) {std::cout << m[i][j] << " ";};
      std::cout << " ]" << std::endl;

    };
  };

  std::vector<double> linspace(const double &l, const double &u, const double &N) {

    // size
    const double T = u - l;

    // step size
    const double dt = T/(N-1);

    // linspace
    std::vector<double> v(N);
    for (int i=0; i<N; i++) {v[i] = l + i*dt;}
    return v;

  };

};

#endif
