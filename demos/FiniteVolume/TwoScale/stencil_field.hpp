// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <samurai/stencil_field.hpp>

namespace samurai {
  /*******************
   * upwind operator for the scalar advection equation with variable velocity
   *******************/

  template<class TInterval>
  class upwind_variable_op : public field_operator_base<TInterval>,
                             public finite_volume<upwind_variable_op<TInterval>> {
    public:

    INIT_OPERATOR(upwind_variable_op)

    // Compute the flux along one direction
    template<class Flux, class Field, class EigenValue>
    inline auto flux(const Flux& Fl, const Flux& Fr, const Field& ql, const Field& qr, const EigenValue& lambda) const {
      // Upwind flux
      return xt::eval(0.5*(Fl + Fr) + 0.5*lambda*(ql - qr));
    }

    // 2D configurations: left flux
    template<class Flux, class Field>
    inline auto left_flux(const Field& q, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(0, level, i - 1, j)),
                                       xt::abs(vel(0, level, i, j)));

      return flux(vel(0, level, i - 1, j)*q(level, i - 1, j), vel(0, level, i, j)*q(level, i, j), q(level, i - 1, j), q(level, i, j), lambda);
    }

    // 2D configurations: right flux
    template<class Flux, class Field>
    inline auto right_flux(const Field& q, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(0, level, i, j)),
                                       xt::abs(vel(0, level, i + 1, j)));

      return flux(vel(0, level, i, j)*q(level, i, j), vel(0, level, i + 1, j)*q(level, i + 1, j), q(level, i, j), q(level, i + 1, j), lambda);
    }

    // 2D configurations: bottom flux
    template<class Flux, class Field>
    inline auto down_flux(const Field& q, const Flux& vel) const  {
      const auto& lambda = xt::maximum(xt::abs(vel(1, level, i, j - 1)),
                                       xt::abs(vel(1, level, i, j)));

      return flux(vel(1, level, i, j - 1)*q(level, i, j - 1), vel(1, level, i, j)*q(level, i, j), q(level, i, j - 1), q(level, i, j), lambda);
    }

    // 2D configurations: up flux
    template<class Flux, class Field>
    inline auto up_flux(const Field& q, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(1, level, i, j)),
                                       xt::abs(vel(1, level, i, j + 1)));

      return flux(vel(1, level, i, j)*q(level, i, j), vel(1, level, i, j + 1)*q(level, i, j + 1), q(level, i, j), q(level, i, j + 1), lambda);
    }
  };

  template<class... CT>
  inline auto upwind_variable(CT&&... e) {
    return make_field_operator_function<upwind_variable_op>(std::forward<CT>(e)...);
  }

} // End of samurai namespace
