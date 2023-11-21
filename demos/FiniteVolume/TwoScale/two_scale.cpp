// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "two_scale.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  constexpr std::size_t dim = 2;

  // Mesh parameters
  xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0.0, 0.0};
  xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {2.0, 1.0};
  std::size_t min_level = 6;
  std::size_t max_level = 6;

  // Simulation parameters
  double Tf  = 0.09;
  double cfl = 0.5;

  bool apply_relaxation = true;

  // Output parameters
  std::size_t nfiles = 100;

  // Create the instance of the class to perform the simulation
  auto TwoScaleSim = TwoScale(min_corner, max_corner, min_level, max_level,
                              Tf, cfl, nfiles, apply_relaxation);


  TwoScaleSim.run();

  return 0;
}
