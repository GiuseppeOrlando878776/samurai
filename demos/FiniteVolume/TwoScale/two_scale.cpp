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

#include "stencil_field.hpp"

#include <filesystem>
namespace fs = std::filesystem;

static const double p0          = 1e5;

static const double rho0_phase1 = 1.0;
static const double rho0_phase2 = 1e3;

static const double c0_phase1   = 3.0;
static const double c0_phase2   = 15.0;

// Create the variable for "large-scale" mass of phase 1
//
template<class Mesh>
auto init_m1(Mesh& mesh) {
  // Create the variable to fill the volume fraction values
  auto m1 = samurai::make_field<double, 1>("m1", mesh);
  m1.fill(1.0);

  return m1;
}


// Create the variable for mass of phase 2
//
template<class Mesh>
auto init_m2(Mesh& mesh) {
  // Create the variable to fill the volume fraction values
  auto m2 = samurai::make_field<double, 1>("m2", mesh);
  m2.fill(2.0);

  return m2;
}


// Create the variable for "small-scale" mass of phase 1
//
template<class Mesh>
auto init_m1_d(Mesh& mesh) {
  // Create the variable to fill the volume fraction values
  auto m1_d = samurai::make_field<double, 1>("m1_d", mesh);
  m1_d.fill(0.5);

  return m1_d;
}


// Create the variable for "small-scale" volume fraction of phase 1
//
template<class Mesh>
auto init_alpha1_d(Mesh& mesh) {
  // Create the variable to fill the volume fraction values
  auto alpha1_d = samurai::make_field<double, 1>("alpha1_d", mesh);
  alpha1_d.fill(0.1);

  return alpha1_d;
}


// Create the variable for \bar{alpha}\rho
//
template<class Mesh>
auto init_rho_alpha1_bar(Mesh& mesh) {
  // Create the variable to fill the volume fraction values
  auto rho_alpha1_bar = samurai::make_field<double, 1>("rho_alpha1_bar", mesh);
  rho_alpha1_bar.fill(1.75);

  return rho_alpha1_bar;
}


// Create the velocity
//
template<class Mesh, int dim>
auto init_velocity(Mesh& mesh) {
  // Create the variable for the velocity
  auto u = samurai::make_field<double, dim>("u", mesh);

  // Initialize the velocity field
  using mesh_id_t = typename Mesh::mesh_id_t;
  const double PI = xt::numeric_constants<double>::PI;

  samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                         [&](auto& cell)
                         {
                           auto center    = cell.center();
                           const double x = center[0];
                           const double y = center[1];

                           u[cell][0] = -std::sin(PI*x)*std::sin(PI*x)*std::sin(2.0*PI*y);
                           u[cell][1] = std::sin(PI*y)*std::sin(PI*y)*std::sin(2.0*PI*x);
                         });

  samurai::make_bc<samurai::Neumann>(u, 0.0, 0.0);

  return u;
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
  double Tf  = 3.14;
  double cfl = 5.0/8.0;
  double dt  = cfl/(1 << max_level);
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
  auto m1             = init_m1(mesh);
  auto m2             = init_m2(mesh);
  auto m1_d           = init_m1_d(mesh);
  auto alpha1_d       = init_alpha1_d(mesh);
  auto rho_alpha1_bar = init_rho_alpha1_bar(mesh);
  auto u              = init_velocity<samurai::amr::Mesh<Config>, dim>(mesh);

  // Create dependent unknown and other auxiliary useful fields
  auto rho        = samurai::make_field<double, 1>("rho", mesh);
  rho             = m1 + m2 + m1_d;
  auto rho_u      = samurai::make_field<double, 1>("rho_u", mesh);
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           rho_u[cell] = rho[cell]*u[cell][0];
                         });
  auto rho_v      = samurai::make_field<double, 1>("rho_v", mesh);
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           rho_v[cell] = rho[cell]*u[cell][1];
                         });
  auto alpha1_bar = samurai::make_field<double, 1>("alpha1_bar", mesh);
  alpha1_bar      = rho_alpha1_bar/rho;
  auto alpha2_bar = samurai::make_field<double, 1>("alpha2_bar", mesh);
  alpha2_bar      = 1.0 - alpha1_bar;
  auto alpha1     = samurai::make_field<double, 1>("alpha1", mesh);
  alpha1          = alpha1_bar*(1.0 - alpha1_d);
  auto rho1       = samurai::make_field<double, 1>("rho1", mesh);
  rho1            = m1/alpha1;
  auto alpha2     = samurai::make_field<double, 1>("alpha2", mesh);
  alpha2          = 1.0 - alpha1 - alpha1_d;
  auto rho2       = samurai::make_field<double, 1>("rho2", mesh);
  rho2            = m2/alpha2;
  auto p_bar      = samurai::make_field<double, 1>("p_bar", mesh);
  p_bar           = alpha1_bar*EOS_phase1(rho1) + alpha2_bar*EOS_phase2(rho2);
  samurai::make_bc<samurai::Neumann>(p_bar, 0.0);

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, mesh, m1, m2, m1_d, alpha1_d, rho_alpha1_bar, rho_u, rho_v, u, alpha1_bar, alpha1, p_bar);

  // Start the loop
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  while(t != Tf) {
    t += dt;
    if(t > Tf) {
      dt += Tf - t;
      t = Tf;
    }

    std::cout << fmt::format("iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

    // TODO: Compute speed of sound and proper eigenvalue for Rusanov flux

    // Apply the numerical scheme without relaxation
    samurai::update_ghost(m1, m2, m1_d, alpha1_d, rho_alpha1_bar, rho_u, rho_v, u, p_bar);
    m1             = m1 - dt*samurai::upwind_conserved_variable(m1, u);
    m2             = m2 - dt*samurai::upwind_conserved_variable(m2, u);
    m1_d           = m1_d - dt*samurai::upwind_conserved_variable(m1_d, u);
    alpha1_d       = alpha1_d - dt*samurai::upwind_conserved_variable(alpha1_d, u);
    rho_alpha1_bar = rho_alpha1_bar - dt*samurai::upwind_conserved_variable(rho_alpha1_bar, u);
    rho_u          = rho_u - dt*samurai::upwind_horizontal_momentum(rho_u, p_bar, u);
    rho_v          = rho_v - dt*samurai::upwind_vertical_momentum(rho_v, p_bar, u);

    // Apply relaxation, which will modify alpha1_bar and, consequently, for what
    // concerns next time step, rho_alpha1_bar and p_bar
    const auto alpha1_bar_rho1 = m1/(1.0 - alpha1_d);
    const auto alpha2_bar_rho2 = m2/(1.0 - alpha1_d);

    const auto q      = rho0_phase2*c0_phase2*c0_phase2 - rho0_phase1*c0_phase1*c0_phase1;
    const auto qtilde = alpha2_bar_rho2*c0_phase2*c0_phase2
                      - alpha1_bar_rho1*c0_phase1*c0_phase1;

    const auto betaPos = (q - qtilde +
                          xt::sqrt((q - qtilde)*(q - qtilde) +
                                   4.0*alpha1_bar_rho1*c0_phase1*c0_phase1*alpha2_bar_rho2*c0_phase2*c0_phase2))/
                         (2.0*alpha2_bar_rho2*c0_phase2*c0_phase2);

    alpha1_bar         = betaPos/(1.0 + betaPos);

    // Update auxiliary useful fields
    rho            = m1 + m2 + m1_d;
    rho_alpha1_bar = alpha1_bar*rho;
    alpha2_bar     = 1.0 - alpha1_bar;
    alpha1         = alpha1_bar*(1.0 - alpha1_d);
    rho1           = m1/alpha1;
    alpha2         = 1.0 - alpha1 - alpha1_d;
    rho2           = m2/alpha2;
    p_bar          = alpha1_bar*EOS_phase1(rho1) + alpha2_bar*EOS_phase2(rho2);

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix, mesh, m1, m2, m1_d, alpha1_d, rho_alpha1_bar, rho_u, rho_v, u, alpha1_bar, alpha1, p_bar);
    }
  }

  return 0;
}
