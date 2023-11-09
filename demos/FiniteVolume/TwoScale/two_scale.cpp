// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include "two_scale_FV.hpp"

#include <filesystem>
namespace fs = std::filesystem;

// Declare some parameters related to EOS.
// TODO: create a class for EOS
static constexpr double p0_phase1   = 1e5;
static constexpr double p0_phase2   = 1e5;

static constexpr double rho0_phase1 = 1.0;
static constexpr double rho0_phase2 = 1e3;

static constexpr double c0_phase1   = 3.0;
static constexpr double c0_phase2   = 15.0;

// Specify the use of this namespace where we just store the indices
using namespace EquationData;

// Create conserved variables
//
template<class Mesh>
auto init_conserved_variables(Mesh& mesh) {
  using mesh_id_t = typename Mesh::mesh_id_t;

  // Create the variable to fill the conserved variables
  auto conserved_variables = samurai::make_field<double, 7>("conserved", mesh);

  // Declare some constant parameters associated to the grid and to the
  // initial state
  const double x0 = 1.0;
  const double y0 = 0.5;

  const double xd = 0.3;

  const double eps = 1e-7;

  // Initialize the field
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];

                           if(x <= xd) {
                             const double alpha1_bar = 1.0 - eps;
                             const double alpha1_d   = 0.0;
                             const double rho1       = 100.0;
                             const double rho2       = 1e4;
                             const double u          = 0.0;
                             const double v          = 0.0;

                             conserved_variables[cell][M1_INDEX]       = alpha1_bar*rho1*(1.0 - alpha1_d);
                             conserved_variables[cell][M2_INDEX]       = (1.0 - alpha1_bar)*rho2*(1.0 - alpha1_d);
                             conserved_variables[cell][M1_D_INDEX]     = rho0_phase1*alpha1_d;
                             conserved_variables[cell][ALPHA1_D_INDEX] = alpha1_d;

                             const double rho = conserved_variables[cell][M1_INDEX]
                                              + conserved_variables[cell][M2_INDEX]
                                              + conserved_variables[cell][M1_D_INDEX];

                             conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = rho*alpha1_bar;
                             conserved_variables[cell][RHO_U_INDEX]          = rho*u;
                             conserved_variables[cell][RHO_V_INDEX]          = rho*v;
                           }
                           else {
                             const double alpha1_bar = eps;
                             const double alpha1_d   = 0.0;
                             const double rho1       = 1.0;
                             const double rho2       = 1e3;
                             const double u          = 0.0;
                             const double v          = 0.0;

                             conserved_variables[cell][M1_INDEX]       = alpha1_bar*rho1*(1.0 - alpha1_d);
                             conserved_variables[cell][M2_INDEX]       = (1.0 - alpha1_bar)*rho2*(1.0 - alpha1_d);
                             conserved_variables[cell][M1_D_INDEX]     = rho0_phase1*alpha1_d;
                             conserved_variables[cell][ALPHA1_D_INDEX] = alpha1_d;

                             const double rho = conserved_variables[cell][M1_INDEX]
                                              + conserved_variables[cell][M2_INDEX]
                                              + conserved_variables[cell][M1_D_INDEX];

                             conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = rho*alpha1_bar;
                             conserved_variables[cell][RHO_U_INDEX]          = rho*u;
                             conserved_variables[cell][RHO_V_INDEX]          = rho*v;
                           }

                           // Identify the beam
                           if(std::abs(y - y0) < 0.1 && std::abs(x - x0) < 0.5) {
                             const double alpha1_bar = eps;
                             const double alpha1_d   = 0.4;
                             const double rho1       = 1.0;
                             const double rho2       = 1e3;
                             const double u          = 0.0;
                             const double v          = 0.0;

                             conserved_variables[cell][M1_INDEX]       = alpha1_bar*rho1*(1.0 - alpha1_d);
                             conserved_variables[cell][M2_INDEX]       = (1.0 - alpha1_bar)*rho2*(1.0 - alpha1_d);
                             conserved_variables[cell][M1_D_INDEX]     = rho0_phase1*alpha1_d;
                             conserved_variables[cell][ALPHA1_D_INDEX] = alpha1_d;

                             const double rho = conserved_variables[cell][M1_INDEX]
                                              + conserved_variables[cell][M2_INDEX]
                                              + conserved_variables[cell][M1_D_INDEX];

                             conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = rho*alpha1_bar;
                             conserved_variables[cell][RHO_U_INDEX]          = rho*u;
                             conserved_variables[cell][RHO_V_INDEX]          = rho*v;
                           }
                         });

  return conserved_variables;
}


// Implement the EOS for phase 1
//
template<class Field>
auto EOS_phase1(const Field& rho1) {
  return p0_phase1 + c0_phase1*c0_phase1*(rho1 - rho0_phase1);
}


// Implement the EOS for phase 2
//
template<class Field>
auto EOS_phase2(const Field& rho2) {
  return p0_phase2 + c0_phase2*c0_phase2*(rho2 - rho0_phase2);
}


// Create an auxiliary routine to impose left Dirichlet boundary condition
template<std::size_t dim, class Field, class Field_Vel, class Field_Scalar>
void impose_left_dirichet_BC(Field& q, Field_Vel& velocity, Field_Scalar& pressure, Field_Scalar& speed_of_sound) {
  // Create the state
  const double eps = 1e-7;

  const double alpha1_bar = 1.0 - eps;
  const double alpha1_d   = 0.0;
  const double rho1       = 100.0;
  const double rho2       = 1e4;
  const double u          = 0.0;
  const double v          = 0.0;

  const double m1   = alpha1_bar*rho1*(1.0 - alpha1_d);
  const double m2   = (1.0 - alpha1_bar)*rho2*(1.0 - alpha1_d);
  const double m1_d = rho0_phase1*alpha1_d;

  const double rho = m1 + m2 + m1_d;

  const double rho_alpha1_bar = rho*alpha1_bar;
  const double rho_u          = rho*u;
  const double rho_v          = rho*v;

  // Impose BC for the conserved field
  samurai::DirectionVector<dim> left = {-1, 0};

  samurai::make_bc<samurai::Dirichlet>(q, m1, m2, m1_d, alpha1_d, rho_alpha1_bar, rho_u, rho_v)->on(left);

  // Impose BC for the velocity
  samurai::make_bc<samurai::Dirichlet>(velocity, u, v)->on(left);

  // Impose BC for the pressure
  const double p_bar = alpha1_bar*EOS_phase1(rho1) + (1.0 - alpha1_bar)*EOS_phase2(rho2);

  samurai::make_bc<samurai::Dirichlet>(pressure, p_bar)->on(left);

  // Impose BC for the speed of sound
  const double c_squared = m1*c0_phase1 + m2*c0_phase2;
  const double c         = std::sqrt(c_squared/rho)/(1.0 - alpha1_d);

  samurai::make_bc<samurai::Dirichlet>(speed_of_sound, c)->on(left);
}


// Create an auxiliary routine to impose left Dirichlet boundary condition
template<std::size_t dim, class Field, class Field_Vel, class Field_Scalar>
void impose_right_dirichet_BC(Field& q, Field_Vel& velocity, Field_Scalar& pressure, Field_Scalar& speed_of_sound) {
  // Create the state
  const double eps = 1e-7;

  const double alpha1_bar = eps;
  const double alpha1_d   = 0.0;
  const double rho1       = 1.0;
  const double rho2       = 1e3;
  const double u          = 0.0;
  const double v          = 0.0;

  const double m1   = alpha1_bar*rho1*(1.0 - alpha1_d);
  const double m2   = (1.0 - alpha1_bar)*rho2*(1.0 - alpha1_d);
  const double m1_d = rho0_phase1*alpha1_d;

  const double rho = m1 + m2 + m1_d;

  const double rho_alpha1_bar = rho*alpha1_bar;
  const double rho_u          = rho*u;
  const double rho_v          = rho*v;

  // Impose BC for the conserved field
  samurai::DirectionVector<dim> right = {1, 0};

  samurai::make_bc<samurai::Dirichlet>(q, m1, m2, m1_d, alpha1_d, rho_alpha1_bar, rho_u, rho_v)->on(right);

  // Impose BC for the velocity
  samurai::make_bc<samurai::Dirichlet>(velocity, u, v)->on(right);

  // Impose BC for the pressure
  const double p_bar = alpha1_bar*EOS_phase1(rho1) + (1.0 - alpha1_bar)*EOS_phase2(rho2);

  samurai::make_bc<samurai::Dirichlet>(pressure, p_bar)->on(right);

  // Impose BC for the speed of sound
  const double c_squared = m1*c0_phase1 + m2*c0_phase2;
  const double c         = std::sqrt(c_squared/rho)/(1.0 - alpha1_d);

  samurai::make_bc<samurai::Dirichlet>(speed_of_sound, c)->on(right);
}


// Auxiliary routine to save the results
//
template<class Mesh, class... Variables>
void save(const fs::path& path, const std::string& filename, const std::string& suffix, const Mesh& mesh, const Variables&... fields) {
  auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

  if(!fs::exists(path)) {
    fs::create_directory(path);
  }

  using mesh_id_t = typename Mesh::mesh_id_t;

  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
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

  using mesh_id_t = typename Mesh::mesh_id_t;

  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
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

  using mesh_id_t = typename Mesh::mesh_id_t;

  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
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

  using mesh_id_t = typename Mesh::mesh_id_t;

  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
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
  using Config              = samurai::MRConfig<dim>;

  // Mesh parameters
  xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0.0, 0.0};
  xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {2.0, 1.0};
  std::size_t min_level = 6;
  std::size_t max_level = 6;

  // Simulation parameters
  double Tf  = 0.09;
  double cfl = 0.5;
  double t   = 0.0;

  bool apply_relaxation = false;

  // Output parameters
  fs::path path        = fs::current_path();
  std::string filename = "FV_two_scale";
  std::size_t nfiles   = 25;
  const double dt_save = Tf / static_cast<double>(nfiles);

  // Parse command line parameters
  CLI::App app{"Finite volume example for two-scale model in 2D"};
  app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Mesh parameters");
  app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Mesh parameters");
  app.add_option("--min-level", min_level, "Minimum level of AMR")->capture_default_str()->group("Mesh parameters");
  app.add_option("--max-level", max_level, "Maximum level of AMR")->capture_default_str()->group("Mesh parameters");
  app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
  app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
  app.add_option("--apply_relaxation", apply_relaxation, "Choose whether apply relaxation or not")->capture_default_str()->group("Simulation parameters");
  app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
  app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
  app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
  CLI11_PARSE(app, argc, argv);

  // Create the mesh
  const samurai::Box<double, dim> box(min_corner, max_corner);
  samurai::MRMesh<Config> mesh(box, min_level, max_level, {false, true});

  // Create the initial fields and the auxliary field for next step
  auto conserved_variables = init_conserved_variables(mesh);

  auto conserved_variables_np1 = samurai::make_field<double, 7>("conserved_np1", mesh);

  // Create auxiliary useful fields
  using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;
  auto rho = samurai::make_field<double, 1>("rho", mesh);
  samurai::update_ghost_mr(conserved_variables);
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           rho[cell] = conserved_variables[cell][M1_INDEX]
                                     + conserved_variables[cell][M2_INDEX]
                                     + conserved_variables[cell][M1_D_INDEX];
                         });

  samurai::update_ghost_mr(rho);
  auto vel = samurai::make_field<double, dim>("vel", mesh);
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           vel[cell][0] = conserved_variables[cell][RHO_U_INDEX]/
                                          rho[cell];
                           vel[cell][1] = conserved_variables[cell][RHO_V_INDEX]/
                                          rho[cell];
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

  auto alpha1 = samurai::make_field<double, 1>("alpha1", mesh);
  samurai::update_ghost_mr(alpha1_bar);
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           alpha1[cell] = alpha1_bar[cell]*
                                          (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                         });

  auto rho1 = samurai::make_field<double, 1>("rho1", mesh);
  samurai::update_ghost_mr(alpha1);
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
  samurai::update_ghost_mr(alpha2);
  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](const auto& cell)
                         {
                           rho2[cell] = conserved_variables[cell][M2_INDEX]/alpha2[cell];
                         });

  auto p_bar = samurai::make_field<double, 1>("p_bar", mesh);
  p_bar      = alpha1_bar*EOS_phase1(rho1) + alpha2_bar*EOS_phase2(rho2);

  auto c = samurai::make_field<double, 1>("c", mesh);
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const double c_squared = conserved_variables[cell][M1_INDEX]*c0_phase1
                                                  + conserved_variables[cell][M2_INDEX]*c0_phase2;

                           c[cell] = std::sqrt(c_squared/rho[cell])/
                                     (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                         });

  // Impose Dirichlet boundary conditions
  impose_left_dirichet_BC<dim>(conserved_variables, vel, p_bar, c);
  impose_right_dirichet_BC<dim>(conserved_variables, vel, p_bar, c);

  // Create the flux variable
  auto flux = samurai::make_two_scale<decltype(conserved_variables)>(vel, p_bar, c);

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, mesh, conserved_variables, vel, alpha1_bar, alpha1, p_bar, c);

  // Set initial time step
  double dx = samurai::cell_length(min_level);
  double dt = cfl*dx/(get_max_velocity_horizontal(mesh, vel) + get_max_velocity_vertical(mesh, vel) + get_max_celerity(mesh, c));

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
    auto flux_conserved = flux(conserved_variables);
    conserved_variables_np1 = conserved_variables - dt*flux_conserved;

    std::swap(conserved_variables.array(), conserved_variables_np1.array());

    // Update auxiliary useful fields which are not modified by relaxation
    samurai::update_ghost_mr(conserved_variables);
    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](const auto& cell)
                           {
                             rho[cell] = conserved_variables[cell][M1_INDEX]
                                       + conserved_variables[cell][M2_INDEX]
                                       + conserved_variables[cell][M1_D_INDEX];
                           });

    samurai::update_ghost_mr(rho);
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

    // Apply relaxation if desired, which will modify alpha1_bar and, consequently, for what
    // concerns next time step, rho_alpha1_bar and p_bar
    if(apply_relaxation) {
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
    }
    else {
      samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                             [&](const auto& cell)
                             {
                               alpha1_bar[cell] = conserved_variables[cell][RHO_ALPHA1_BAR_INDEX]/
                                                  rho[cell];
                             });
    }

    // Update auxiliary useful fields
    samurai::update_ghost_mr(alpha1_bar);
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

    samurai::update_ghost_mr(alpha1);
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

    samurai::update_ghost_mr(alpha2);
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
