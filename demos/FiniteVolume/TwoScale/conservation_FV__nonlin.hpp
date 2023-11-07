#pragma once
#include <samurai/schemes/fv.hpp>

namespace EquationData {
  static constexpr std::size_t M1_INDEX             = 0;
  static constexpr std::size_t M2_INDEX             = 1;
  static constexpr std::size_t M1_D_INDEX           = 2;
  static constexpr std::size_t ALPHA1_D_INDEX       = 3;
  static constexpr std::size_t RHO_ALPHA1_BAR_INDEX = 4;
  static constexpr std::size_t RHO_U_INDEX          = 5;
  static constexpr std::size_t RHO_V_INDEX          = 6;
}

namespace samurai {
  using namespace EquationData;

  /**
   * Implementation of discretization of a conservation law with upwind/Rusanov flux
   * along the horizontal direction
   */
  template<class Field, class Aux, class Aux_b>
  auto make_conservation(const Aux& vel, const Aux_b& pres) {
    static constexpr std::size_t dim = Field::dim;
    static_assert(Field::dim == Aux::size, "No mathcing spactial dimension in make_conservation");

    static constexpr std::size_t field_size        = Field::size;
    static constexpr std::size_t output_field_size = field_size;
    static constexpr std::size_t stencil_size      = 2;

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    FluxDefinition<cfg> Rusanov_f;

    static_for<0, dim>::apply(
      // For each variable
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        auto f = [&](const auto& q, const auto& velocity, const auto& pressure)
        {
          FluxValue<cfg> res = q;

          res(M1_INDEX) *= velocity(d);
          res(M2_INDEX) *= velocity(d);
          res(M1_D_INDEX) *= velocity(d);
          res(ALPHA1_D_INDEX) *= velocity(d);
          res(RHO_ALPHA1_BAR_INDEX) *= velocity(d);
          res(RHO_U_INDEX) *= velocity(d);
          res(RHO_V_INDEX) *= velocity(d);

          if constexpr(d == 0) {
            res(RHO_U_INDEX) += pressure;
          }
          if constexpr(d == 1) {
            res(RHO_V_INDEX) += pressure;
          }

          return res;
        };

        Rusanov_f[d].flux_function = [&](auto& cells, Field& field)
                                     {
                                       const auto& left  = cells[0];
                                       const auto& right = cells[1];

                                       const auto lambda = std::max(std::abs(vel[left](d)),
                                                                    std::abs(vel[right](d)));

                                       return 0.5*(f(field[left], vel[left], pres[left]) + f(field[right], vel[right], pres[right])) +
                                              0.5*lambda*(field[left] - field[right]);
                                     };
      }
    );

    return make_flux_based_scheme(Rusanov_f);
  }

} // end namespace samurai
