// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

#include "flux.hpp"

#include <samurai/mr/adapt.hpp>

// Specify the use of this namespace where we just store the indices
// and some parameters related to the equations of state
using namespace EquationData;

// This is the class for the simulation of a two-scale model
//
template<std::size_t dim>
class Relaxation {
public:
  using Config = samurai::MRConfig<dim>;

  Relaxation() = default; // Default constructor. This will do nothing
                          // and basically will never be used

  Relaxation(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
             const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
             std::size_t min_level, std::size_t max_level,
             double Tf_, double cfl_, std::size_t nfiles_ = 100);  // Class constrcutor with the arguments related
                                                                   // to the grid, to the physics and to the relaxation.
                                                                   // Maybe in the future,
                                                                   // we could think to add parameters related to EOS

  void run(); // Function which actually executes the temporal loop

  template<class... Variables>
  void save(const fs::path& path,
            const std::string& filename,
            const std::string& suffix,
            const Variables&... fields); // Routine to save the results

private:
  /*--- Now we declare some relevant variables ---*/
  const samurai::Box<double, dim> box;

  samurai::MRMesh<Config> mesh; // Variable to store the mesh

  using Field        = samurai::Field<decltype(mesh), double, EquationData::NVARS, false>;
  using Field_Scalar = samurai::Field<decltype(mesh), double, 1, false>;
  using Field_Vect   = samurai::Field<decltype(mesh), double, dim, false>;

  double Tf;  // Final time of the simulation
  double cfl; // Courant number of the simulation so as to compute the time step

  std::size_t nfiles; // Number of files desired for output

  Field conserved_variables; // The variable which stores the conserved variables,
                             // namely the varialbes for which we solve a PDE system

  const SG_EOS<> EOS_phase1; // Equation of state of phase 1
  const SG_EOS<> EOS_phase2; // Equation of state of phase 2

  samurai::RusanovFlux<Field> numerical_flux_cons; // function to compute the numerical flux for the conservative part
                                                   // (this is necessary to call 'make_flux')

  samurai::NonConservativeFlux<Field> numerical_flux_non_cons; // function to compute the numerical flux for the non-conservative part
                                                               // (this is necessary to call 'make_flux')

  // Now we declare a bunch of fields which depend from the state, but it is useful
  // to have it for the output
  Field_Scalar rho,
               p,
               rho1,
               p1,
               c1,
               rho2,
               p2,
               c2;

  Field_Vect vel1,
             vel2;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void init_variables(); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  void update_auxiliary_fields(); // Routine to update auxilairy fields for output and time step update

  double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue

  void apply_instantaneous_velocity_relxation(); // Apply an instantaneous velocity relaxtion

  void update_pressure_before_relaxation(); // Update pressure fields before relaxation

  void apply_instantaneous_pressure_relaxation(); // Apply an instantaneous pressure relaxation
};


// Implement class constructor
//
template<std::size_t dim>
Relaxation<dim>::Relaxation(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                            const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                            std::size_t min_level, std::size_t max_level,
                            double Tf_, double cfl_, std::size_t nfiles_):
  box(min_corner, max_corner), mesh(box, min_level, max_level, {false}),
  Tf(Tf_), cfl(cfl_), nfiles(nfiles_),
  EOS_phase1(EquationData::gamma_1, EquationData::pi_infty_1, EquationData::q_infty_1),
  EOS_phase2(EquationData::gamma_2, EquationData::pi_infty_2, EquationData::q_infty_2),
  numerical_flux_cons(EOS_phase1, EOS_phase2),
  numerical_flux_non_cons(EOS_phase1, EOS_phase2) {
    init_variables();
}


// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void Relaxation<dim>::init_variables() {
  // Create conserved and auxiliary fields
  conserved_variables = samurai::make_field<double, EquationData::NVARS>("conserved", mesh);

  rho = samurai::make_field<double, 1>("rho", mesh);
  p   = samurai::make_field<double, 1>("p", mesh);

  rho1 = samurai::make_field<double, 1>("rho1", mesh);
  p1   = samurai::make_field<double, 1>("p1", mesh);
  c1   = samurai::make_field<double, 1>("c1", mesh);

  rho2 = samurai::make_field<double, 1>("rho2", mesh);
  p2   = samurai::make_field<double, 1>("p2", mesh);
  c2   = samurai::make_field<double, 1>("c2", mesh);

  vel1 = samurai::make_field<double, dim>("vel1", mesh);
  vel2 = samurai::make_field<double, dim>("vel2", mesh);

  const double xd = 0.5;

  // Initialize the fields with a loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];

                           if(x <= xd) {
                             conserved_variables[cell][ALPHA1_INDEX] = 0.5;

                             rho1[cell] = 1.0;
                             vel1[cell] = 0.0;
                             p1[cell]   = 1.0;

                             rho2[cell] = 1.0;
                             vel2[cell] = 0.0;
                             p2[cell]   = 1.0;
                           }
                           else {
                             conserved_variables[cell][ALPHA1_INDEX] = 0.5;

                             rho1[cell] = 0.125;
                             vel1[cell] = 0.0;
                             p1[cell]   = 0.1;

                             rho2[cell] = 0.125;
                             vel2[cell] = 0.0;
                             p2[cell]   = 0.1;
                           }

                           conserved_variables[cell][ALPHA1_RHO1_INDEX]    = conserved_variables[cell][ALPHA1_INDEX]*rho1[cell];
                           conserved_variables[cell][ALPHA1_RHO1_U1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*vel1[cell];
                           const auto e1 = EOS_phase1.e_value(rho1[cell], p1[cell]);
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*(e1 + 0.5*vel1[cell]*vel1[cell]);

                           conserved_variables[cell][ALPHA2_RHO2_INDEX]    = (1.0 - conserved_variables[cell][ALPHA1_INDEX])*rho2[cell];
                           conserved_variables[cell][ALPHA2_RHO2_U2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*vel2[cell];
                           const auto e2 = EOS_phase2.e_value(rho2[cell], p2[cell]);
                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*(e2 + 0.5*vel2[cell]*vel2[cell]);

                           c1[cell] = EOS_phase1.c_value(rho1[cell], p1[cell]);

                           c2[cell] = EOS_phase2.c_value(rho2[cell], p2[cell]);
                         });

  samurai::make_bc<samurai::Neumann>(conserved_variables, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}


// Apply the instantaneous relaxation for the velocity
//
template<std::size_t dim>
void Relaxation<dim>::apply_instantaneous_velocity_relxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const double rho_eq = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                               + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           // Update the momentum and the energy
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             // Save the velocity obtained after the hyperbolic step to update the energy
                             const auto vel1_d_0 = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d]/conserved_variables[cell][ALPHA1_RHO1_INDEX];
                             const auto vel2_d_0 = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d]/conserved_variables[cell][ALPHA2_RHO2_INDEX];

                             // Compute the equilibirum velocity
                             const auto vel_star_d = (conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d] +
                                                      conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d])/rho_eq;

                             // Update the momentum
                             conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*vel_star_d;
                             conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*vel_star_d;

                             // Update the total energy
                             conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] += conserved_variables[cell][ALPHA1_RHO1_INDEX]*(vel_star_d*(vel_star_d - vel1_d_0));
                             conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] += conserved_variables[cell][ALPHA2_RHO2_INDEX]*(vel_star_d*(vel_star_d - vel2_d_0));
                           }
                         });
}


// Update pressure fields before relaxation
//
template<std::size_t dim>
void Relaxation<dim>::update_pressure_before_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           auto e1 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/conserved_variables[cell][ALPHA1_RHO1_INDEX];
                           auto e2 = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/conserved_variables[cell][ALPHA2_RHO2_INDEX];
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             const double vel1_d = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d]/conserved_variables[cell][ALPHA1_RHO1_INDEX];
                             e1 -= 0.5*(vel1_d*vel1_d);

                             const double vel2_d = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d]/conserved_variables[cell][ALPHA2_RHO2_INDEX];
                             e2 -= 0.5*(vel2_d*vel2_d);
                           }
                           p1[cell] = EOS_phase1.pres_value(conserved_variables[cell][ALPHA1_RHO1_INDEX]/conserved_variables[cell][ALPHA1_INDEX], e1);
                           p2[cell] = EOS_phase2.pres_value(conserved_variables[cell][ALPHA2_RHO2_INDEX]/(1.0 - conserved_variables[cell][ALPHA1_INDEX]), e2);
                         });
}


// Apply the instantaneous relaxation for the pressure
//
template<std::size_t dim>
void Relaxation<dim>::apply_instantaneous_pressure_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Save mixture total energy for later update
                           const auto rhoE_0 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]
                                             + conserved_variables[cell][ALPHA2_RHO2_E2_INDEX];

                           // Compute the pressure equilibirum with the polynomial method (Saurel)
                           const auto b = (conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2*(EquationData::pi_infty_2 - p1[cell]) +
                                           (1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1*(EquationData::pi_infty_1 - p2[cell]))/
                                          (conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2 + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1);
                           const auto c = -(p1[cell]*conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2*EquationData::pi_infty_2 +
                                            p2[cell]*(1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1*EquationData::pi_infty_1)/
                                           (conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2 + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1);

                           const auto p_star = 0.5*(-b + std::sqrt(b*b - 4.0*c));

                           // Update the volume fraction using the compute pressure
                           conserved_variables[cell][ALPHA1_INDEX] *= (p1[cell] + EquationData::gamma_1*EquationData::pi_infty_1 + p_star*(EquationData::gamma_1 - 1.0))/
                                                                      (p_star + EquationData::gamma_1*EquationData::pi_infty_1 + p_star*(EquationData::gamma_1 - 1.0));

                           // Update the total energy of phase 1
                           auto E1 = EOS_phase1.e_value(conserved_variables[cell][ALPHA1_RHO1_INDEX]/conserved_variables[cell][ALPHA1_INDEX], p_star);
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             const auto vel1_d = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d]/conserved_variables[cell][ALPHA1_RHO1_INDEX];
                             E1 += 0.5*(vel1_d*vel1_d);
                           }
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*E1;

                           // Update the total energy of phase 2
                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = rhoE_0 - conserved_variables[cell][ALPHA1_RHO1_E1_INDEX];
                         });
}



// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double Relaxation<dim>::get_max_lambda() const {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           res = std::max(std::max(std::abs(vel1[cell]) + c1[cell],
                                                   std::abs(vel2[cell]) + c2[cell]),
                                          res);
                         });

  return res;
}


// Update auxiliary fields after solution of the system
//
template<std::size_t dim>
void Relaxation<dim>::update_auxiliary_fields() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           rho[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           rho1[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]/conserved_variables[cell][ALPHA1_INDEX];
                           vel1[cell] = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX]/conserved_variables[cell][ALPHA1_RHO1_INDEX];
                           auto e1 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/conserved_variables[cell][ALPHA1_RHO1_INDEX];
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             e1 -= 0.5*vel1[cell]*vel1[cell];
                           }
                           p1[cell] = EOS_phase1.pres_value(rho1[cell], e1);
                           c1[cell] = EOS_phase1.c_value(rho1[cell], p1[cell]);

                           rho2[cell] = conserved_variables[cell][ALPHA2_RHO2_INDEX]/(1.0 - conserved_variables[cell][ALPHA1_INDEX]);
                           vel2[cell] = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX]/conserved_variables[cell][ALPHA2_RHO2_INDEX];
                           auto e2 = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/conserved_variables[cell][ALPHA2_RHO2_INDEX];
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             e2 -= 0.5*vel2[cell]*vel2[cell];
                           }
                           p2[cell] = EOS_phase2.pres_value(rho2[cell], e2);
                           c2[cell] = EOS_phase2.c_value(rho2[cell], p2[cell]);

                           p[cell] = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                   + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*p2[cell];
                         });
}


// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void Relaxation<dim>::save(const fs::path& path,
                         const std::string& filename,
                         const std::string& suffix,
                         const Variables&... fields) {
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


// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void Relaxation<dim>::run() {
  // Default output arguemnts
  fs::path path        = fs::current_path();
  std::string filename = "Relaxation_Rusanov";
  const double dt_save = Tf / static_cast<double>(nfiles);

  // Auxiliary variables to save updated fields
  auto conserved_variables_np1 = samurai::make_field<double, EquationData::NVARS>("conserved_np1", mesh);

  // Create the flux variables
  auto Rusanov_flux         = numerical_flux_cons.make_flux();
  auto NonConservative_flux = numerical_flux_non_cons.make_flux();

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, conserved_variables, rho, p, vel1, rho1, p1, c1, vel2, rho2, p2, c2);

  // Set initial time step
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  double dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());
  double dt = cfl*dx/get_max_lambda();

  // Start the loop
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  double t          = 0.0;
  while(t != Tf) {
    t += dt;
    if(t > Tf) {
      dt += Tf - t;
      t = Tf;
    }

    std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

    // Apply the numerical scheme
    samurai::update_ghost_mr(conserved_variables);
    samurai::update_bc(conserved_variables);
    auto Cons_Flux    = Rusanov_flux(conserved_variables);
    auto NonCons_Flux = NonConservative_flux(conserved_variables);
    conserved_variables_np1 = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;

    std::swap(conserved_variables.array(), conserved_variables_np1.array());

    // Apply the relaxation for the velocity
    apply_instantaneous_velocity_relxation();

    // Apply the relaxation for the pressure
    update_pressure_before_relaxation();
    apply_instantaneous_pressure_relaxation();

    // Compute updated time step
    dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());
    dt = std::min(dt, cfl*dx/get_max_lambda());

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      // Update auxiliary fields which are used only for output purposes
      update_auxiliary_fields();

      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix, conserved_variables, rho, p, vel1, rho1, p1, c1, vel2, rho2, p2, c2);
    }
  }
}
