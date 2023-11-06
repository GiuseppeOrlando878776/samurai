#pragma once
#include <samurai/schemes/fv.hpp>

namespace samurai {
  /**
   * Implementation of discretization of a conservation law with upwind/Rusanov flux
   * along the horizontal direction
   */
  template<class Field, class Aux>
  auto make_conservation(const Aux& vel) {
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

        auto f = [&](const auto& q, const auto& velocity) -> FluxValue<cfg>
        {
          return velocity(d)*q;
        };

        Rusanov_f[d].flux_function = [&](auto& cells, Field& field)
                                     {
                                       const auto& left  = cells[0];
                                       const auto& right = cells[1];

                                       const auto lambda = std::max(std::abs(vel[left](d)),
                                                                    std::abs(vel[right](d)));

                                       return 0.5*(f(field[left], vel[left]) + f(field[right], vel[right])) +
                                              0.5*lambda*(field[left] - field[right]);
                                     };
      }
    );

    return make_flux_based_scheme(Rusanov_f);
  }

} // end namespace samurai
