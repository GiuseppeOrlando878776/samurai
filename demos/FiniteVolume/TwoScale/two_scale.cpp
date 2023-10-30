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

// Create the volume fraction
//
template<class Mesh>
auto init_alpha(Mesh& mesh) {
  // Create the variable to fill the volume fraction values
  auto alpha = samurai::make_field<double, 1>("alpha1", mesh);

  // Declare parameters that identify our bubble
  constexpr double radius   = 0.15;
  constexpr double x_center = 0.5;
  constexpr double y_center = 0.75;

  // Initialize the field
  using mesh_id_t = typename Mesh::mesh_id_t;

  samurai::for_each_cell(mesh[mesh_id_t::cells],
                         [&](const auto& cell)
                         {
                             const auto center = cell.center();
                             const double x    = center[0];
                             const double y    = center[1];

                             alpha[cell] = std::sqrt((x - x_center)*(x - x_center) + (y - y_center)*(y - y_center)) - radius;
                         });

  // Impose Neumann boundary conditions
  samurai::make_bc<samurai::Neumann>(alpha, 0.0);

  return alpha;
}


// Create the velocity
//
template<class Mesh, int dim>
auto init_velocity(Mesh& mesh) {
  // Create the variable for the velocity
  auto u = samurai::make_field<double, dim>("u", mesh);
  u.fill(0.0);

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

  return u;
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

  // Create the two fields
  auto alpha = init_alpha(mesh);
  auto u     = init_velocity<samurai::amr::Mesh<Config>, dim>(mesh);

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, mesh, alpha, u);

  // Start the loop
  using mesh_id_t   = typename samurai::amr::Mesh<Config>::mesh_id_t;
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  while(t != Tf) {
    t += dt;
    if(t > Tf) {
      dt += Tf - t;
      t = Tf;
    }

    std::cout << fmt::format("iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

    // Compute the flux
    auto alpha_u = samurai::make_field<double, dim>("alpha_u", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](auto& cell)
                           {
                             alpha_u[cell][0] = u[cell][0]*alpha[cell];
                             alpha_u[cell][1] = u[cell][1]*alpha[cell];
                           });

    // Apply the numerical scheme
    samurai::update_ghost(alpha, u, alpha_u);
    alpha = alpha - dt*samurai::upwind_variable(alpha_u, alpha, u);

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix, mesh, alpha, u);
    }
  }

  return 0;
}
