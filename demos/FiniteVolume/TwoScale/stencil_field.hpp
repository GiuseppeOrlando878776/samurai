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
    template<class T0, class T1>
    inline auto flux(const T0& vel, const T1& ql, const T1& qr) const {
      // Upwind flux
      return xt::eval(0.5*vel*(ql + qr) + 0.5*xt::abs(vel)*(ql - qr));
    }

    // 2D configurations: left flux
    template <class T0, class T1>
    inline auto left_flux(const T0& vel, const T1& q) const {
      const auto& vel_at_interface = xt::eval(0.5*(vel(0, level, i - 1, j) +
                                                   vel(0, level, i, j)));

      return flux(vel_at_interface, q(level, i - 1, j), q(level, i, j));
    }

    // 2D configurations: right flux
    template <class T0, class T1>
    inline auto right_flux(const T0& vel, const T1& u) const {
      const auto& vel_at_interface = xt::eval(0.5*(vel(0, level, i, j) +
                                                   vel(0, level, i + 1, j)));

      return flux(vel_at_interface, u(level, i, j), u(level, i + 1, j));
    }

    // 2D configurations: bottom flux
    template <class T0, class T1>
    inline auto down_flux(const T0& vel, const T1& u) const  {
      const auto& vel_at_interface = xt::eval(0.5*(vel(1, level, i, j - 1) +
                                                   vel(1, level, i, j)));

      return flux(vel_at_interface, u(level, i, j - 1), u(level, i, j));
    }

    // 2D configurations: up flux
    template <class T0, class T1>
    inline auto up_flux(const T0& vel, const T1& u) const {
      const auto& vel_at_interface = xt::eval(0.5*(vel(1, level, i, j) +
                                                   vel(1, level, i, j + 1)));

      return flux(vel_at_interface, u(level, i, j), u(level, i, j + 1));
    }
  };

  template <class... CT>
  inline auto upwind_variable(CT&&... e) {
    return make_field_operator_function<upwind_variable_op>(std::forward<CT>(e)...);
  }

} // End of samurai namespace
