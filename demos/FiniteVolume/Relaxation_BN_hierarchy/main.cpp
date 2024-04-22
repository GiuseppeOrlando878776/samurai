// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include "relaxation.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  // Mesh parameters
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {0.0};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {1.0};
  std::size_t min_level = 12;
  std::size_t max_level = 12;

  // Simulation parameters
  double Tf  = 3.2e-3;
  double cfl = 0.4;

  // Output parameters
  std::size_t nfiles = 100;

  bool apply_velocity_relax  = true;
  bool apply_pressure_relax  = true;
  bool apply_pressure_reinit = false;
  bool energy_update_phase_1 = true;
  bool preserve_energy       = false;

  // Create the instance of the class to perform the simulation
  auto Relaxation_Suliciu_Sim = Relaxation(min_corner, max_corner, min_level, max_level,
                                           Tf, cfl, nfiles,
                                           apply_velocity_relax, apply_pressure_relax,
                                           apply_pressure_reinit, energy_update_phase_1,
                                           preserve_energy);

  Relaxation_Suliciu_Sim.run();

  return 0;
}
