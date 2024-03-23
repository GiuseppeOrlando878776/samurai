#pragma once
#include <samurai/schemes/fv.hpp>

namespace EquationData {
  // Declare parameters related to surface tension coefficient
  static constexpr double sigma = 1.0;

  // Declare some parameters related to EOS.
  static constexpr double p0_phase1 = 1e5;
  static constexpr double p0_phase2 = 1e5;

  static constexpr double rho0_phase1 = 1e3;
  static constexpr double rho0_phase2 = 1.0;

  static constexpr double c0_phase1 = 1.5e3;
  static constexpr double c0_phase2 = 1e2;

  // Use auxiliary variables for the indices for the sake of generality
  static constexpr std::size_t M1_INDEX             = 0;
  static constexpr std::size_t M2_INDEX             = 1;
  static constexpr std::size_t M1_D_INDEX           = 2;
  static constexpr std::size_t ALPHA1_D_INDEX       = 3;
  static constexpr std::size_t SIGMA_D_INDEX        = 4;
  static constexpr std::size_t RHO_ALPHA1_BAR_INDEX = 5;
  static constexpr std::size_t RHO_U_INDEX          = 6;
  static constexpr std::size_t RHO_V_INDEX          = 7;

  // Save also the total number of (scalar) vairables
  static constexpr std::size_t NVARS = 8;
}


namespace samurai {
  using namespace EquationData;

  /**
   * Implementation of discretization of a conservation law with upwind/Rusanov flux
   */
  template<class Field, class Field_Vect, class Field_Scalar>
  auto make_two_scale_capillarity(const Field_Vect& vel, const Field_Scalar& pres, const Field_Scalar& c,
                                  const Field_Vect& normal, const Field_Scalar& norm_grad_alpha1_bar) {
    static constexpr std::size_t dim = Field::dim;
    static_assert(Field::dim == Field_Vect::size, "No mathcing spactial dimension in make_two_scale_capillarity");

    static constexpr std::size_t field_size        = Field::size;
    static constexpr std::size_t output_field_size = field_size;
    static constexpr std::size_t stencil_size      = 2;

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    FluxDefinition<cfg> Rusanov_f;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, dim>::apply(
      // First, we need a function to compute the "continuous" flux
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        auto f = [&](const auto& q, const auto& velocity, const auto& pressure,
                     const auto& mod_grad_alpha1_bar, const auto& n)
        {
          FluxValue<cfg> res = q;

          res(M1_INDEX) *= velocity(d);
          res(M2_INDEX) *= velocity(d);
          res(M1_D_INDEX) *= velocity(d);
          res(ALPHA1_D_INDEX) *= velocity(d);
          res(SIGMA_D_INDEX) *= velocity(d);
          res(RHO_ALPHA1_BAR_INDEX) *= velocity(d);
          res(RHO_U_INDEX) *= velocity(d);
          res(RHO_V_INDEX) *= velocity(d);

          if constexpr(d == 0) {
            res(RHO_U_INDEX) += pressure;

            // Add constribution due to surface tension
            if(!std::isnan(n(0))) {
              res(RHO_U_INDEX) += sigma*(n(0)*n(0) - 1.0)*mod_grad_alpha1_bar;
            }
            if(!std::isnan(n(0)) && !std::isnan(n(1))) {
              res(RHO_V_INDEX) += sigma*n(0)*n(1)*mod_grad_alpha1_bar;
            }
          }

          if constexpr(d == 1) {
            res(RHO_V_INDEX) += pressure;

            // Add contribution due to surface tension
            if(!std::isnan(n(0)) && !std::isnan(n(1))) {
              res(RHO_U_INDEX) += sigma*n(0)*n(1)*mod_grad_alpha1_bar;
            }
            if(!std::isnan(n(1))) {
              res(RHO_V_INDEX) += sigma*(n(1)*n(1) - 1.0)*mod_grad_alpha1_bar;
            }
          }

          return res;
        };

        // Compute now the "discrete" flux function, in this case a Rusanov flux
        Rusanov_f[d].cons_flux_function = [&](auto& cells, const Field& field)
                                          {
                                            FluxValue<cfg> res;

                                            const auto& left  = cells[0];
                                            const auto& right = cells[1];

                                            const auto rho_L = field[left](M1_INDEX)
                                                             + field[left](M2_INDEX)
                                                             + field[left](M1_D_INDEX);
                                            const auto r_L   = sigma*norm_grad_alpha1_bar[left]/(rho_L*c[left]*c[left]);

                                            const auto rho_R = field[right](M1_INDEX)
                                                             + field[right](M2_INDEX)
                                                             + field[right](M1_D_INDEX);
                                            const auto r_R   = sigma*norm_grad_alpha1_bar[right]/(rho_R*c[right]*c[right]);

                                            // Compute the estimate of the eigenvalue considering also the surface tension contribution
                                            const auto lambda = std::max(std::max(std::abs(vel[left](d) + c[left]*(1.0 + 0.125*r_L)),
                                                                             std::abs(vel[left](d) - c[left]*(1.0 + 0.125*r_L))),
                                                                    std::max(std::abs(vel[right](d) + c[right]*(1.0 + 0.125*r_R)),
                                                                             std::abs(vel[right](d) + c[right]*(1.0 + 0.125*r_R))));

                                            res = 0.5*(f(field[left], vel[left], pres[left], norm_grad_alpha1_bar[left], normal[left]) +
                                                       f(field[right], vel[right], pres[right], norm_grad_alpha1_bar[right], normal[right])) -
                                                  0.5*lambda*(field[right] - field[left]);

                                            return res;
                                          };
    });

    return make_flux_based_scheme(Rusanov_f);
  }

} // end namespace samurai
