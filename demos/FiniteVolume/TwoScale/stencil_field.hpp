// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <samurai/stencil_field.hpp>

namespace samurai {
  /*******************
   * upwind operator for a conservation equation
   *******************/

  template<class TInterval>
  class upwind_conserved_variable_op : public field_operator_base<TInterval>,
                                       public finite_volume<upwind_conserved_variable_op<TInterval>> {
    public:

    INIT_OPERATOR(upwind_conserved_variable_op)

    // Compute the flux along one direction
    template<class Field, class Flux, class EigenValue>
    inline auto flux(const Flux& Fl, const Flux& Fr, const Field& ql, const Field& qr, const EigenValue& lambda) const {
      // Upwind (or in principle Rusanov) flux
      return xt::eval(0.5*(Fl + Fr) + 0.5*lambda*(ql - qr));
    }

    // 2D configurations: left flux
    template<class Field, class Flux>
    inline auto left_flux(const Field& q, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(0, level, i - 1, j)),
                                       xt::abs(vel(0, level, i, j)));

      return flux(vel(0, level, i - 1, j)*q(level, i - 1, j), vel(0, level, i, j)*q(level, i, j), q(level, i - 1, j), q(level, i, j), lambda);
    }

    // 2D configurations: right flux
    template<class Field, class Flux>
    inline auto right_flux(const Field& q, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(0, level, i, j)),
                                       xt::abs(vel(0, level, i + 1, j)));

      return flux(vel(0, level, i, j)*q(level, i, j), vel(0, level, i + 1, j)*q(level, i + 1, j), q(level, i, j), q(level, i + 1, j), lambda);
    }

    // 2D configurations: bottom flux
    template<class Field, class Flux>
    inline auto down_flux(const Field& q, const Flux& vel) const  {
      const auto& lambda = xt::maximum(xt::abs(vel(1, level, i, j - 1)),
                                       xt::abs(vel(1, level, i, j)));

      return flux(vel(1, level, i, j - 1)*q(level, i, j - 1), vel(1, level, i, j)*q(level, i, j), q(level, i, j - 1), q(level, i, j), lambda);
    }

    // 2D configurations: up flux
    template<class Field, class Flux>
    inline auto up_flux(const Field& q, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(1, level, i, j)),
                                       xt::abs(vel(1, level, i, j + 1)));

      return flux(vel(1, level, i, j)*q(level, i, j), vel(1, level, i, j + 1)*q(level, i, j + 1), q(level, i, j), q(level, i, j + 1), lambda);
    }
  };

  template<class... CT>
  inline auto upwind_conserved_variable(CT&&... e) {
    return make_field_operator_function<upwind_conserved_variable_op>(std::forward<CT>(e)...);
  }


  /*******************
   * upwind operator for horizontal momentum equation
   *******************/

  template<class TInterval>
  class upwind_horizontal_momentum_op : public field_operator_base<TInterval>,
                                        public finite_volume<upwind_horizontal_momentum_op<TInterval>> {
    public:

    INIT_OPERATOR(upwind_horizontal_momentum_op)

    // Compute the flux along one direction
    template<class Field, class Flux, class EigenValue>
    inline auto flux(const Flux& Fl, const Flux& Fr, const Field& ql, const Field& qr, const EigenValue& lambda) const {
      // Upwind (or in principle Rusanov) flux
      return xt::eval(0.5*(Fl + Fr) + 0.5*lambda*(ql - qr));
    }

    // 2D configurations: left flux
    template<class Field, class Flux>
    inline auto left_flux(const Field& q, const Field& pres, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(0, level, i - 1, j)),
                                       xt::abs(vel(0, level, i, j)));

      return flux(q(level, i - 1, j)*vel(0, level, i - 1, j) + pres(level, i - 1, j),
                  q(level, i, j)*vel(0, level, i, j) + pres(level, i, j),
                  q(level, i - 1, j), q(level, i, j), lambda);
    }

    // 2D configurations: right flux
    template<class Field, class Flux>
    inline auto right_flux(const Field& q, const Field& pres, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(0, level, i, j)),
                                       xt::abs(vel(0, level, i + 1, j)));

      return flux(q(level, i, j)*vel(0, level, i, j) + pres(level, i, j),
                  q(level, i + 1, j)*vel(0, level, i + 1, j) + pres(level, i + 1, j),
                  q(level, i, j), q(level, i + 1, j), lambda);
    }

    // 2D configurations: bottom flux
    template<class Field, class Flux>
    inline auto down_flux(const Field& q, const Field& pres, const Flux& vel) const  {
      const auto& lambda = xt::maximum(xt::abs(vel(1, level, i, j - 1)),
                                       xt::abs(vel(1, level, i, j)));

      return flux(q(level, i, j - 1)*vel(1, level, i, j - 1),
                  q(level, i, j)*vel(1, level, i, j),
                  q(level, i, j - 1), q(level, i, j), lambda);
    }

    // 2D configurations: up flux
    template<class Field, class Flux>
    inline auto up_flux(const Field& q, const Field& pres, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(1, level, i, j)),
                                       xt::abs(vel(1, level, i, j + 1)));

      return flux(q(level, i, j)*vel(1, level, i, j),
                  q(level, i, j + 1)*vel(1, level, i, j + 1),
                  q(level, i, j), q(level, i, j + 1), lambda);
    }
  };

  template<class... CT>
  inline auto upwind_horizontal_momentum(CT&&... e) {
    return make_field_operator_function<upwind_horizontal_momentum_op>(std::forward<CT>(e)...);
  }


  /*******************
   * upwind operator for vertical momentum equation
   *******************/

  template<class TInterval>
  class upwind_vertical_momentum_op : public field_operator_base<TInterval>,
                                      public finite_volume<upwind_vertical_momentum_op<TInterval>> {
    public:

    INIT_OPERATOR(upwind_vertical_momentum_op)

    // Compute the flux along one direction
    template<class Field, class Flux, class EigenValue>
    inline auto flux(const Flux& Fl, const Flux& Fr, const Field& ql, const Field& qr, const EigenValue& lambda) const {
      // Upwind (or in principle Rusanov) flux
      return xt::eval(0.5*(Fl + Fr) + 0.5*lambda*(ql - qr));
    }

    // 2D configurations: left flux
    template<class Field, class Flux>
    inline auto left_flux(const Field& q, const Field& pres, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(0, level, i - 1, j)),
                                       xt::abs(vel(0, level, i, j)));

      return flux(q(level, i - 1, j)*vel(0, level, i - 1, j),
                  q(level, i, j)*vel(0, level, i, j),
                  q(level, i - 1, j), q(level, i, j), lambda);
    }

    // 2D configurations: right flux
    template<class Field, class Flux>
    inline auto right_flux(const Field& q, const Field& pres, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(0, level, i, j)),
                                       xt::abs(vel(0, level, i + 1, j)));

      return flux(q(level, i, j)*vel(0, level, i, j),
                  q(level, i + 1, j)*vel(0, level, i + 1, j),
                  q(level, i, j), q(level, i + 1, j), lambda);
    }

    // 2D configurations: bottom flux
    template<class Field, class Flux>
    inline auto down_flux(const Field& q, const Field& pres, const Flux& vel) const  {
      const auto& lambda = xt::maximum(xt::abs(vel(1, level, i, j - 1)),
                                       xt::abs(vel(1, level, i, j)));

      return flux(q(level, i, j - 1)*vel(1, level, i, j - 1) + pres(level, i, j - 1),
                  q(level, i, j)*vel(1, level, i, j) + pres(level, i, j),
                  q(level, i, j - 1), q(level, i, j), lambda);
    }

    // 2D configurations: up flux
    template<class Field, class Flux>
    inline auto up_flux(const Field& q, const Field& pres, const Flux& vel) const {
      const auto& lambda = xt::maximum(xt::abs(vel(1, level, i, j)),
                                       xt::abs(vel(1, level, i, j + 1)));

      return flux(q(level, i, j)*vel(1, level, i, j) + pres(level, i, j),
                  q(level, i, j + 1)*vel(1, level, i, j + 1) + pres(level, i, j + 1),
                  q(level, i, j), q(level, i, j + 1), lambda);
    }
  };

  template<class... CT>
  inline auto upwind_vertical_momentum(CT&&... e) {
    return make_field_operator_function<upwind_vertical_momentum_op>(std::forward<CT>(e)...);
  }

} // End of samurai namespace
