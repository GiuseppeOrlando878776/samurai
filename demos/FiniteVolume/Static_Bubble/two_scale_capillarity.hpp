// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

#include "barotropic_eos.hpp"

#include "two_scale_capillarity_FV.hpp"

#include <samurai/mr/adapt.hpp>

// Specify the use of this namespace where we just store the indices
// and, in this case, some parameters related to EOS
using namespace EquationData;

// This is the class for the simulation of a two-scale model
//
template<std::size_t dim>
class StaticBubble {
public:
  using Config = samurai::MRConfig<dim>;

  StaticBubble() = default; // Default constructor. This will do nothing
                            // and basically will never be used

  StaticBubble(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
               const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
               std::size_t min_level, std::size_t max_level,
               double Tf_, double cfl_, std::size_t nfiles_ = 100,
               bool apply_relax_ = true);                              // Class constrcutor with the arguments related
                                                                       // to the grid, to the physics and to the relaxation.
                                                                       // Maybe in the future, we could think to add parameters related to EOS

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
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;

  using Field        = samurai::Field<decltype(mesh), double, EquationData::NVARS, false>;
  using Field_Scalar = samurai::Field<decltype(mesh), double, 1, false>;
  using Field_Vect   = samurai::Field<decltype(mesh), double, dim, false>;

  LinearizedBarotropicEOS EOS_phase1,
                          EOS_phase2; // The two variables which take care of the
                                      // barotropic EOS to compute the speed of sound

  bool apply_relax; // Choose whether to apply or not the relaxation

  double Tf;  // Final time of the simulation
  double cfl; // Courant number of the simulation so as to compute the time step

  std::size_t nfiles; // Number of files desired for output

  Field conserved_variables; // The variable which stores the conserved variables,
                             // namely the varialbes for which we solve a PDE system

  // Now we declare a bunch of fields which depend from the state, but it is useful
  // to have it so as to avoid recomputation
  Field_Scalar rho,
               alpha1_bar,
               alpha2_bar,
               alpha1,
               rho1,
               p1,
               alpha2,
               rho2,
               p2,
               p_bar,
               c,
               mod_grad_alpha1_bar,
               H,
               rho1d;

  Field_Vect vel,
             normal,
             grad_alpha1_bar;

  using gradient_type = decltype(samurai::make_gradient<decltype(alpha1_bar)>());
  gradient_type gradient;

  using divergence_type = decltype(samurai::make_divergence<decltype(normal)>());
  divergence_type divergence;

  double eps;                     // Tolerance when we want to avoid division by zero
  double mod_grad_alpha1_bar_min; // Minimum threshold for which not computing anymore the unit normal

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void update_geometry(); // Auxiliary routine to compute normals and curvature

  void init_variables(); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue

  void apply_relaxation(); // Apply the relaxation

  void update_auxiliary_fields_pre_relaxation(); // Update auxiliary fields which are not touched by relaxation

  void update_auxiliary_fields_post_relaxation(); // Update auxiliary fields after relaxation are not touched by relaxation
};


// Implement class constructor
//
template<std::size_t dim>
StaticBubble<dim>::StaticBubble(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                std::size_t min_level, std::size_t max_level,
                                double Tf_, double cfl_, std::size_t nfiles_,
                                bool apply_relax_):
  box(min_corner, max_corner), mesh(box, min_level, max_level, {true, true}),
  apply_relax(apply_relax_), Tf(Tf_), cfl(cfl_), nfiles(nfiles_),
  gradient(samurai::make_gradient<decltype(alpha1_bar)>()),
  divergence(samurai::make_divergence<decltype(normal)>()),
  eps(1e-9), mod_grad_alpha1_bar_min(0.0) {
    EOS_phase1 = LinearizedBarotropicEOS(EquationData::p0_phase1, EquationData::rho0_phase1, EquationData::c0_phase1);
    EOS_phase2 = LinearizedBarotropicEOS(EquationData::p0_phase2, EquationData::rho0_phase2, EquationData::c0_phase2);

    std::cout << "Initializing variables " << std::endl;
    init_variables();
}


// Auxiliary routine to compute normals and curvature
//
template<std::size_t dim>
void StaticBubble<dim>::update_geometry() {
  samurai::update_ghost_mr(alpha1_bar);
  grad_alpha1_bar = gradient(alpha1_bar);
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           mod_grad_alpha1_bar[cell] = std::sqrt(xt::sum(grad_alpha1_bar[cell]*grad_alpha1_bar[cell])());

                           if(mod_grad_alpha1_bar[cell] > mod_grad_alpha1_bar_min) {
                             normal[cell] = grad_alpha1_bar[cell]/mod_grad_alpha1_bar[cell];
                           }
                           else {
                             for(std::size_t d = 0; d < dim; ++d) {
                               normal[cell][d] = nan("");
                             }
                           }
                         });
  samurai::update_ghost_mr(normal);
  H = -divergence(normal);
}


// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void StaticBubble<dim>::init_variables() {
  // Create conserved and auxiliary fields
  conserved_variables = samurai::make_field<double, EquationData::NVARS>("conserved", mesh);

  rho        = samurai::make_field<double, 1>("rho", mesh);
  vel        = samurai::make_field<double, dim>("vel", mesh);
  alpha1_bar = samurai::make_field<double, 1>("alpha1_bar", mesh);
  alpha2_bar = samurai::make_field<double, 1>("alpha2_bar", mesh);
  alpha1     = samurai::make_field<double, 1>("alpha1", mesh);
  rho1       = samurai::make_field<double, 1>("rho1", mesh);
  p1         = samurai::make_field<double, 1>("p1", mesh);
  alpha2     = samurai::make_field<double, 1>("alpha2", mesh);
  rho2       = samurai::make_field<double, 1>("rho2", mesh);
  p2         = samurai::make_field<double, 1>("p2", mesh);
  p_bar      = samurai::make_field<double, 1>("p_bar", mesh);
  c          = samurai::make_field<double, 1>("c", mesh);

  mod_grad_alpha1_bar = samurai::make_field<double, 1>("mod_grad_alpha1_bar", mesh);
  H                   = samurai::make_field<double, 1>("H", mesh);
  normal              = samurai::make_field<double, dim>("normal", mesh);

  grad_alpha1_bar     = samurai::make_field<double, dim>("grad_alpha1_bar", mesh);

  rho1d = samurai::make_field<double, 1>("rho1d", mesh);

  // Declare some constant parameters associated to the grid and to the
  // initial state
  const double L     = 0.75;
  const double x0    = 0.5*L;
  const double y0    = 0.5*L;
  const double R     = 0.2;
  const double eps_R = 0.2*R;

  const double U_0 = 0.0;
  const double U_1 = 0.0;
  const double V   = 0.0;

  // Initialize some fields to define the bubble with a loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];

                           const double r = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                           const double w = (r >= R && r < R + eps_R) ?
                                            std::max(std::exp(2.0*(r - R)*(r - R)/(eps_R*eps_R)*((r - R)*(r - R)/(eps_R*eps_R) - 3.0)/
                                                              (((r - R)*(r - R)/(eps_R*eps_R) - 1.0)*((r - R)*(r - R)/(eps_R*eps_R) - 1.0))), 0.0) :
                                            ((r < R) ? 1.0 : 0.0);

                           conserved_variables[cell][ALPHA1_D_INDEX] = 0.0;
                           conserved_variables[cell][SIGMA_D_INDEX]  = 0.0;

                           alpha1_bar[cell] = w;

                           alpha1[cell] = alpha1_bar[cell]*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);

                           alpha2_bar[cell] = 1.0 - alpha1_bar[cell];

                           alpha2[cell] = alpha2_bar[cell]*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);

                           vel[cell][0] = w*U_1 + (1.0 - w)*U_0;
                           vel[cell][1] = V;
                         });

  // Compute the geometrical quantities
  update_geometry();

  // Loop over a cell to complete the remaining variables
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           p1[cell] = EOS_phase2.get_p0();
                           p1[cell] += (alpha1_bar[cell] > 1.0 - eps) ? EquationData::sigma/R : EquationData::sigma*H[cell];
                           rho1[cell] = EOS_phase1.rho_value(p1[cell]);

                           conserved_variables[cell][M1_INDEX] = (!std::isnan(rho1[cell])) ? alpha1[cell]*rho1[cell] : 0.0;

                           p2[cell]   = (alpha1_bar[cell] < 1.0 - eps) ? EOS_phase2.get_p0() : nan("");
                           rho2[cell] = EOS_phase2.rho_value(p2[cell]);

                           conserved_variables[cell][M2_INDEX] = (!std::isnan(rho2[cell])) ? alpha2[cell]*rho2[cell] : 0.0;

                           conserved_variables[cell][M1_D_INDEX] = conserved_variables[cell][ALPHA1_D_INDEX]*EOS_phase1.get_rho0();

                           rho[cell] = conserved_variables[cell][M1_INDEX]
                                     + conserved_variables[cell][M2_INDEX]
                                     + conserved_variables[cell][M1_D_INDEX];

                           conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = alpha1_bar[cell]*rho[cell];

                           conserved_variables[cell][RHO_U_INDEX] = rho[cell]*vel[cell][0];
                           conserved_variables[cell][RHO_V_INDEX] = rho[cell]*vel[cell][1];

                           p_bar[cell] = (alpha1_bar[cell] > eps && alpha2_bar[cell] > eps) ?
                                         alpha1_bar[cell]*p1[cell] + alpha2_bar[cell]*p2[cell] :
                                         ((alpha1_bar[cell] < eps) ? p2[cell] : p1[cell]);

                           const double c_squared = conserved_variables[cell][M1_INDEX]*EOS_phase1.c_value(rho1[cell])*EOS_phase1.c_value(rho1[cell])
                                                  + conserved_variables[cell][M2_INDEX]*EOS_phase2.c_value(rho2[cell])*EOS_phase2.c_value(rho2[cell]);
                           c[cell] = std::sqrt(c_squared/rho[cell])/
                                     (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                         });
}


// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double StaticBubble<dim>::get_max_lambda() const {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const double r = EquationData::sigma*mod_grad_alpha1_bar[cell]/(rho[cell]*c[cell]*c[cell]);

                           res = std::max(std::max(std::abs(vel[cell][0]) + c[cell]*(1.0 + 0.125*r),
                                                   std::abs(vel[cell][1]) + c[cell]*(1.0 + 0.125*r)),
                                          res);
                         });

  return res;
}


// Update auxiliary fields which are not modified by the relaxation
// (i.e density and vecloity, since no mass trasnfer occur)
//
template<std::size_t dim>
void StaticBubble<dim>::update_auxiliary_fields_pre_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           rho[cell] = conserved_variables[cell][M1_INDEX]
                                     + conserved_variables[cell][M2_INDEX]
                                     + conserved_variables[cell][M1_D_INDEX];

                           vel[cell][0] = conserved_variables[cell][RHO_U_INDEX]/
                                          rho[cell];
                           vel[cell][1] = conserved_variables[cell][RHO_V_INDEX]/
                                          rho[cell];
                         });
}


// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void StaticBubble<dim>::apply_relaxation() {
  // Apply relaxation with Newton method
  const double tol    = 1e-8; /*--- Tolerance of the Newton method ---*/
  const double lambda = 0.9;  /*--- Parameter for bound preserving strategy ---*/

  std::size_t Newton_iter = 0;
  bool relaxation_applied = true;
  while(relaxation_applied == true) {
    relaxation_applied = false;
    Newton_iter++;

    // Update fields affected by the nonlinear function for which we seek a zero
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             // Compute partial densities since alpha1_bar is potentially changed
                             // in the relaxation during the Newton loop
                             alpha1[cell] = alpha1_bar[cell]*
                                            (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                             rho1[cell]   = (alpha1[cell] > eps) ?
                                            conserved_variables[cell][M1_INDEX]/alpha1[cell] :
                                            nan("");
                             p1[cell]     = EOS_phase1.pres_value(rho1[cell]);

                             alpha2[cell] = 1.0
                                          - alpha1[cell]
                                          - conserved_variables[cell][ALPHA1_D_INDEX];
                             rho2[cell]   = (alpha2[cell] > eps) ?
                                            conserved_variables[cell][M2_INDEX]/alpha2[cell] :
                                            nan("");
                             p2[cell]     = EOS_phase2.pres_value(rho2[cell]);

                             rho1d[cell]  = (conserved_variables[cell][M1_D_INDEX] > eps && conserved_variables[cell][ALPHA1_D_INDEX] > eps) ?
                                            conserved_variables[cell][M1_D_INDEX]/conserved_variables[cell][ALPHA1_D_INDEX] : EquationData::rho0_phase1;
                           });

    // Recompute geometric quantities (curvature potentially changed in the Newton loop)
    update_geometry();

    // Perform the Newton step
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             // Compute the nonlinear function for which we seek the zero (basically the Laplace law)
                             const auto F = (1.0 - conserved_variables[cell][ALPHA1_D_INDEX])*(p1[cell] - p2[cell])
                                          - EquationData::sigma*H[cell];

                             // Perform the relaxation only when really needed
                             if(std::abs(F) > tol*EOS_phase1.get_p0() && alpha1_bar[cell] > eps && 1.0 - alpha1_bar[cell] > eps) {
                               relaxation_applied = true;

                               // Compute the derivative w.r.t large scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
                               const auto dF_dalpha1_bar = -conserved_variables[cell][M1_INDEX]/(alpha1_bar[cell]*alpha1_bar[cell])*
                                                            EOS_phase1.c_value(rho1[cell])*EOS_phase1.c_value(rho1[cell])
                                                           -conserved_variables[cell][M2_INDEX]/((1.0 - alpha1_bar[cell])*(1.0 - alpha1_bar[cell]))*
                                                            EOS_phase2.c_value(rho2[cell])*EOS_phase2.c_value(rho2[cell]);

                               /*--- Compute the pseudo time step starting as initial guess from the ideal unmodified Newton method ---*/
                               double dtau_ov_epsilon = std::numeric_limits<double>::infinity();

                               // Upper bound of the pseudo time to preserve the bounds for the volume fraction
                               const auto upper_denominator = 1.0/(1.0 - conserved_variables[cell][ALPHA1_D_INDEX])*
                                                              (F + lambda*(1.0 - alpha1_bar[cell])*dF_dalpha1_bar);
                               if(upper_denominator > 0.0) {
                                 dtau_ov_epsilon = lambda*(1.0 - alpha1_bar[cell])/upper_denominator;
                               }

                               // Lower bound of the pseudo time to preserve the bounds for the volume fraction
                               const auto lower_denominator = 1.0/(1.0 - conserved_variables[cell][ALPHA1_D_INDEX])*
                                                              (F - lambda*alpha1_bar[cell]*dF_dalpha1_bar);
                               if(lower_denominator < 0.0) {
                                 dtau_ov_epsilon = std::min(dtau_ov_epsilon, -lambda*alpha1_bar[cell]/lower_denominator);
                               }

                               // Compute the large scale volume fraction update
                               double dalpha1_bar;
                               if(std::isinf(dtau_ov_epsilon)) {
                                 dalpha1_bar = -F/dF_dalpha1_bar;
                               }
                               else {
                                 dalpha1_bar = dtau_ov_epsilon/(1.0 - conserved_variables[cell][ALPHA1_D_INDEX])*F/
                                               (1.0 - dtau_ov_epsilon*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX])*dF_dalpha1_bar);
                               }
                               alpha1_bar[cell] += dalpha1_bar;
                             }
                           });

    // Newton cycle diverged
    if(Newton_iter > 100) {
      std::cout << "Netwon method not converged" << std::endl;
      save(fs::current_path(), "static_bubble", "_diverged",
           conserved_variables, vel, alpha1_bar, alpha1, p_bar, c, rho1, rho2, p1, p2, mod_grad_alpha1_bar, normal, grad_alpha1_bar, H);
      exit(1);
    }
  }
}


// Update auxiliary fields after relaxation
//
template<std::size_t dim>
void StaticBubble<dim>::update_auxiliary_fields_post_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = alpha1_bar[cell]*rho[cell];

                           alpha2_bar[cell] = 1.0 - alpha1_bar[cell];

                           alpha1[cell] = alpha1_bar[cell]*
                                          (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);

                           rho1[cell] = (alpha1[cell] > eps) ?
                                        conserved_variables[cell][M1_INDEX]/alpha1[cell] :
                                        nan("");
                           p1[cell]   = EOS_phase1.pres_value(rho1[cell]);

                           alpha2[cell] = 1.0
                                        - alpha1[cell]
                                        - conserved_variables[cell][ALPHA1_D_INDEX];

                           rho2[cell] = (alpha2[cell] > eps) ?
                                        conserved_variables[cell][M2_INDEX]/alpha2[cell] :
                                        nan("");
                           p2[cell]   = EOS_phase2.pres_value(rho2[cell]);

                           p_bar[cell] = (alpha1_bar[cell] > eps && alpha2_bar[cell] > eps) ?
                                         alpha1_bar[cell]*p1[cell] + alpha2_bar[cell]*p2[cell] :
                                         ((alpha1_bar[cell] < eps) ? p2[cell] : p1[cell]);

                           const auto c_squared = conserved_variables[cell][M1_INDEX]*EOS_phase1.c_value(rho1[cell])*EOS_phase1.c_value(rho1[cell])
                                                + conserved_variables[cell][M2_INDEX]*EOS_phase2.c_value(rho2[cell])*EOS_phase2.c_value(rho2[cell]);
                           c[cell] = std::sqrt(c_squared/rho[cell])/
                                     (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                         });
}


// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void StaticBubble<dim>::save(const fs::path& path,
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
template<std::size_t dim>
void StaticBubble<dim>::run() {
  // Default output arguemnts
  fs::path path        = fs::current_path();
  std::string filename = "static_bubble";
  const double dt_save = Tf / static_cast<double>(nfiles);

  // Auxiliary variables to save updated fields
  auto conserved_variables_np1 = samurai::make_field<double, EquationData::NVARS>("conserved_np1", mesh);

  // Create the flux variable
  auto flux = samurai::make_two_scale_capillarity<decltype(conserved_variables)>(vel, p_bar, c, normal, mod_grad_alpha1_bar);

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, conserved_variables, vel, alpha1_bar, alpha1, p_bar, c, rho1, rho2, p1, p2, mod_grad_alpha1_bar, normal, grad_alpha1_bar, H);

  // Set initial time step
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

    // Apply the numerical scheme without relaxation
    samurai::update_ghost_mr(conserved_variables, vel, p_bar, c, mod_grad_alpha1_bar, normal);
    samurai::update_bc(conserved_variables, vel, p_bar, c, mod_grad_alpha1_bar, normal);
    auto flux_conserved     = flux(conserved_variables);
    conserved_variables_np1 = conserved_variables - dt*flux_conserved;

    std::swap(conserved_variables.array(), conserved_variables_np1.array());

    // Update auxiliary useful fields which are not modified by relaxation (i.e density and velocity since no mass transfer occurs)
    update_auxiliary_fields_pre_relaxation();

    // Apply relaxation if desired, which will modify alpha1_bar and, consequently, for what
    // concerns next time step, rho_alpha1_bar and p_bar
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             alpha1_bar[cell] = conserved_variables[cell][RHO_ALPHA1_BAR_INDEX]/rho[cell];
                           });
    if(apply_relax) {
      apply_relaxation();
    }

    // Update auxiliary useful fields (after relaxation)
    update_auxiliary_fields_post_relaxation();
    update_geometry();

    // Compute updated time step
    dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());
    dt = std::min(dt, cfl*dx/get_max_lambda());

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix, conserved_variables, vel, alpha1_bar, alpha1, p_bar, c, rho1, rho2, p1, p2, mod_grad_alpha1_bar, normal, grad_alpha1_bar, H);
    }
  }
}
