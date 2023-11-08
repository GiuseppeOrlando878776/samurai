// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <samurai/algorithm/update.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include "two_scale_FV.hpp"

#include <filesystem>
namespace fs = std::filesystem;

// Declare some parameters related to EOS.
// TODO: create a class for EOS
static constexpr double p0          = 1e5;

static constexpr double rho0_phase1 = 1.0;
static constexpr double rho0_phase2 = 1e3;

static constexpr double c0_phase1   = 3.0;
static constexpr double c0_phase2   = 15.0;

// Specifiy the use of this namespace where we just store the indices
using namespace EquationData;

// Create the velocity field
//
template<class Mesh, int dim>
auto init_velocity(Mesh& mesh) {
  // Create the variable for the velocity
  auto vel = samurai::make_field<double, dim>("vel", mesh);

  // Initialize the velocity field
  using mesh_id_t = typename Mesh::mesh_id_t;
  const double PI = xt::numeric_constants<double>::PI;

  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];

                           vel[cell][0] = -std::sin(PI*x)*std::sin(PI*x)*std::sin(2.0*PI*y);
                           vel[cell][1] = std::sin(PI*y)*std::sin(PI*y)*std::sin(2.0*PI*x);
                         });

  // Specify the boundary condition
  samurai::make_bc<samurai::Neumann>(vel, 0.0, 0.0);

  return vel;
}


// Create conserved variables
//
template<class Mesh, class Field>
auto init_conserved_variables(Mesh& mesh, const Field& vel) {
  // Create the variable to fill the volume fraction values
  auto conserved_variables = samurai::make_field<double, 7>("conserved", mesh);

  using mesh_id_t = typename Mesh::mesh_id_t;

  // Initialize the field
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           conserved_variables[cell][M1_INDEX]             = 1.0;
                           conserved_variables[cell][M2_INDEX]             = 2.0;
                           conserved_variables[cell][M1_D_INDEX]           = 0.5;
                           conserved_variables[cell][ALPHA1_D_INDEX]       = 0.1;
                           conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = 1.75;

                           const double rho = conserved_variables[cell][M1_INDEX]
                                            + conserved_variables[cell][M2_INDEX]
                                            + conserved_variables[cell][M1_D_INDEX];

                            conserved_variables[cell][RHO_U_INDEX] = rho*vel[cell][0];
                            conserved_variables[cell][RHO_V_INDEX] = rho*vel[cell][1];
                         });

  return conserved_variables;
}


// Implement the EOS for phase 1
//
template<class Field>
auto EOS_phase1(const Field& rho1) {
  return rho1 + 0.0*(p0 + c0_phase1*c0_phase1*(rho1 - rho0_phase1));
}


// Implement the EOS for phase 2
//
template<class Field>
auto EOS_phase2(const Field& rho2) {
  return rho2 + 0.0*(p0 + c0_phase2*c0_phase2*(rho2 - rho0_phase2));
}


// Auxiliary routine to save the results
//
template<class Mesh, class... Variables>
void save(const fs::path& path, const std::string& filename, const std::string& suffix, const Mesh& mesh, const Variables&... fields) {
  auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

  if(!fs::exists(path)) {
    fs::create_directory(path);
  }

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           level_[cell] = cell.level;
                         });

  samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, fields..., level_);
}


// Auxiliary routine to compute maximum velocity along horizontal direction
//
template<class Mesh, class Field>
double get_max_velocity_horizontal(const Mesh& mesh, const Field& vel) {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           if(std::abs(vel[cell][0]) > res) {
                             res = std::abs(vel[cell][0]);
                           }
                         });

  return res;
}


// Auxiliary routine to compute maximum velocity along vertical direction
//
template<class Mesh, class Field>
double get_max_velocity_vertical(const Mesh& mesh, const Field& vel) {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           if(std::abs(vel[cell][1]) > res) {
                             res = std::abs(vel[cell][1]);
                           }
                         });

  return res;
}


// Auxiliary routine to compute maximum celerity
//
template<class Mesh, class Field>
double get_max_celerity(const Mesh& mesh, const Field& c) {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           if(c[cell] > res) {
                             res = c[cell];
                           }
                         });

  return res;
}


// Main function to run the program
//
int main(int argc, char* argv[]) {
  constexpr std::size_t dim = 2;
  using Config              = samurai::amr::Config<dim, 2>;

  // Mesh parameters
  xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0.0, 0.0};
  xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1.0, 1.0};
  std::size_t start_level = 8;
  std::size_t min_level   = 8;
  std::size_t max_level   = 8;

  // Simulation parameters
  double Tf  = 0.2;
  double cfl = 0.25;
  double t   = 0.0;

  // Output parameters
  fs::path path        = fs::current_path();
  std::string filename = "FV_two_scale";
  std::size_t nfiles   = 5;
  const double dt_save = Tf / static_cast<double>(nfiles);

  // Parse command line parameters
  CLI::App app{"Finite volume example for two-scale model in 2D"};
  app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Mesh parameters");
  app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Mesh parameters");
  app.add_option("--start-level", start_level, "Start level of AMR")->capture_default_str()->group("Mesh parameters");
  app.add_option("--min-level", min_level, "Minimum level of AMR")->capture_default_str()->group("Mesh parameters");
  app.add_option("--max-level", max_level, "Maximum level of AMR")->capture_default_str()->group("Mesh parameters");
  app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
  app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
  app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
  app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
  app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
  CLI11_PARSE(app, argc, argv);

  // Create the mesh
  const samurai::Box<double, dim> box(min_corner, max_corner);
  samurai::amr::Mesh<Config> mesh(box, start_level, min_level, max_level);

  // Create the fields
  auto vel                 = init_velocity<samurai::amr::Mesh<Config>, dim>(mesh);
  auto conserved_variables = init_conserved_variables(mesh, vel);

  // Set initial time step
  double dx = samurai::cell_length(start_level);
  double dt = cfl*dx/(get_max_velocity_horizontal(mesh, vel) + get_max_velocity_vertical(mesh, vel));

  // Create auxiliary useful fields. In principle, I could compute it
  // inside the flux function, but, since I want a no flux bc and I cannot specify
  // it directly through 'make_two_scale', I do it somehow manually, specificying
  // Neumman boundary conditions for velocity and pressure (neither this is optimal
  // for the velocity, where I would need a free slip condition)
  using mesh_id_t = typename samurai::amr::Mesh<Config>::mesh_id_t;

  auto rho = samurai::make_field<double, 1>("rho", mesh);
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           rho[cell] = conserved_variables[cell][M1_INDEX]
                                     + conserved_variables[cell][M2_INDEX]
                                     + conserved_variables[cell][M1_D_INDEX];
                         });

  auto alpha1_bar = samurai::make_field<double, 1>("alpha1_bar", mesh);
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           alpha1_bar[cell] = conserved_variables[cell][RHO_ALPHA1_BAR_INDEX]/
                                              rho[cell];
                         });

  auto alpha2_bar = samurai::make_field<double, 1>("alpha2_bar", mesh);
  alpha2_bar      = 1.0 - alpha1_bar;

  auto alpha1     = samurai::make_field<double, 1>("alpha1", mesh);
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           alpha1[cell] = alpha1_bar[cell]*
                                          (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                         });

  auto rho1 = samurai::make_field<double, 1>("rho1", mesh);
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           rho1[cell] = conserved_variables[cell][M1_INDEX]/
                                        alpha1[cell];
                         });

  auto alpha2 = samurai::make_field<double, 1>("alpha2", mesh);
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           alpha2[cell] = 1.0
                                        - alpha1[cell]
                                        - conserved_variables[cell][ALPHA1_D_INDEX];
                         });

  auto rho2 = samurai::make_field<double, 1>("rho2", mesh);
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           rho2[cell] = conserved_variables[cell][M2_INDEX]/alpha2[cell];
                         });

  auto p_bar = samurai::make_field<double, 1>("p_bar", mesh);
  p_bar      = alpha1_bar*EOS_phase1(rho1) + alpha2_bar*EOS_phase2(rho2);
  samurai::make_bc<samurai::Neumann>(p_bar, 0.0);

  auto c     = samurai::make_field<double, 1>("c", mesh);
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           const double c_squared = conserved_variables[cell][M1_INDEX]*c0_phase1
                                                  + conserved_variables[cell][M2_INDEX]*c0_phase2;

                           c[cell] = std::sqrt(c_squared/rho[cell])/
                                     (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                         });

  // Create the flux variable
  auto flux = samurai::make_two_scale<decltype(conserved_variables)>(vel, p_bar, c);

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, mesh, conserved_variables, vel, alpha1_bar, alpha1, p_bar, c);

  // Start the loop
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  while(t != Tf) {
    t += dt;
    if(t > Tf) {
      dt += Tf - t;
      t = Tf;
    }

    std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

    // Apply the numerical scheme without relaxation
    samurai::update_ghost_mr(conserved_variables, vel, p_bar, c);
    samurai::update_bc(vel, p_bar);
    conserved_variables = conserved_variables - dt*flux(conserved_variables);

    // Update auxiliary useful fields which are not modified by relaxation
    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             rho[cell] = conserved_variables[cell][M1_INDEX]
                                       + conserved_variables[cell][M2_INDEX]
                                       + conserved_variables[cell][M1_D_INDEX];
                           });

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             vel[cell][0] = conserved_variables[cell][RHO_U_INDEX]/
                                            rho[cell];
                             vel[cell][1] = conserved_variables[cell][RHO_V_INDEX]/
                                            rho[cell];
                           });

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             const double c_squared = conserved_variables[cell][M1_INDEX]*c0_phase1
                                                    + conserved_variables[cell][M2_INDEX]*c0_phase2;

                             c[cell] = std::sqrt(c_squared/rho[cell])/
                                       (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                           });

    // Compute updated time step
    dx = samurai::cell_length(max_level);
    dt = std::min(dt, cfl*dx/(get_max_velocity_horizontal(mesh, vel) + get_max_velocity_vertical(mesh, vel) + get_max_celerity(mesh, c)));

    // Apply relaxation, which will modify alpha1_bar and, consequently, for what
    // concerns next time step, rho_alpha1_bar and p_bar
    auto alpha1_bar_rho1 = samurai::make_field<double, 1>("alpha1_bar_rho1", mesh);
    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             alpha1_bar_rho1[cell] = conserved_variables[cell][M1_INDEX]/
                                                     (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                           });

    auto alpha2_bar_rho2 = samurai::make_field<double, 1>("alpha2_bar_rho2", mesh);
    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             alpha2_bar_rho2[cell] = conserved_variables[cell][M2_INDEX]/
                                                     (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                           });

    const auto q      = rho0_phase2*c0_phase2*c0_phase2 - rho0_phase1*c0_phase1*c0_phase1;
    const auto qtilde = alpha2_bar_rho2*c0_phase2*c0_phase2
                      - alpha1_bar_rho1*c0_phase1*c0_phase1;

    const auto betaPos = (q - qtilde +
                          xt::sqrt((q - qtilde)*(q - qtilde) +
                                   4.0*alpha1_bar_rho1*c0_phase1*c0_phase1*alpha2_bar_rho2*c0_phase2*c0_phase2))/
                         (2.0*alpha2_bar_rho2*c0_phase2*c0_phase2);

    alpha1_bar         = betaPos/(1.0 + betaPos);

    // Update auxiliary useful fields
    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = alpha1_bar[cell]*rho[cell];
                           });

    alpha2_bar = 1.0 - alpha1_bar;

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             alpha1[cell] = alpha1_bar[cell]*
                                            (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                           });

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             rho1[cell] = conserved_variables[cell][M1_INDEX]/
                                          alpha1[cell];
                           });

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             alpha2[cell] = 1.0
                                          - alpha1[cell]
                                          - conserved_variables[cell][ALPHA1_D_INDEX];
                           });

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             rho2[cell] = conserved_variables[cell][M2_INDEX]/
                                          alpha2[cell];
                           });

    p_bar = alpha1_bar*EOS_phase1(rho1) + alpha2_bar*EOS_phase2(rho2);

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             const double c_squared = conserved_variables[cell][M1_INDEX]*c0_phase1
                                                    + conserved_variables[cell][M2_INDEX]*c0_phase2;

                             c[cell] = std::sqrt(c_squared/rho[cell])/
                                       (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                           });

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix, mesh, conserved_variables, vel, alpha1_bar, alpha1, p_bar, c);
    }
  }

  return 0;
}
