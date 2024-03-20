#ifndef flux_hpp
#define flux_hpp

#pragma once
#include <samurai/schemes/fv.hpp>

#include "eos.hpp"

namespace EquationData {
  static constexpr std::size_t dim = 1; /*--- Spatial dimension. It would be ideal to be able to get it
                                              direclty from Field, but I need to move the definition of these indices ---*/

  /*--- Declare suitable static variables for the sake of generalities in the indices ---*/
  static constexpr std::size_t ALPHA1_INDEX         = 0;
  static constexpr std::size_t ALPHA1_RHO1_INDEX    = 1;
  static constexpr std::size_t ALPHA1_RHO1_U1_INDEX = 2;
  static constexpr std::size_t ALPHA1_RHO1_E1_INDEX = 2 + dim;
  static constexpr std::size_t ALPHA2_RHO2_INDEX    = ALPHA1_RHO1_E1_INDEX + 1;
  static constexpr std::size_t ALPHA2_RHO2_U2_INDEX = ALPHA2_RHO2_INDEX + 1;
  static constexpr std::size_t ALPHA2_RHO2_E2_INDEX = ALPHA2_RHO2_U2_INDEX + dim;

  static constexpr std::size_t NVARS = ALPHA2_RHO2_E2_INDEX + 1;

  // Parameters related to the EOS for the two phases
  static constexpr double gamma_1    = 1.4;
  static constexpr double pi_infty_1 = 0.0;
  static constexpr double q_infty_1  = 0.0;

  static constexpr double gamma_2    = 1.4;
  static constexpr double pi_infty_2 = 0.0;
  static constexpr double q_infty_2  = 0.0;
}

namespace samurai {
  using namespace EquationData;

  /**
    * Generic class to compute the flux between a left and right state
    */
  template<class Field>
  class Flux {
  public:
    // Definitions and sanity checks
    static constexpr std::size_t field_size        = Field::size;
    static_assert(field_size == EquationData::NVARS, "The number of elements in the state does not correpsond to the number of equations");
    static_assert(Field::dim == EquationData::dim, "The spatial dimesions do not match");
    static constexpr std::size_t output_field_size = field_size;
    static constexpr std::size_t stencil_size      = 2;

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    Flux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2); // Construction which accepts in inputs the equations of stae of the two phases

    FluxValue<cfg> evaluate_continuos_flux(const FluxValue<cfg>& q, const std::size_t curr_d); // Evaluate the 'continuous' flux for the state q along direction curr_d

  protected:
    const EOS<>& phase1; // Pass it by reference because pure virtual (not so nice, maybe moving to pointers)
    const EOS<>& phase2; // Pass it by reference because pure virtual (not so nice, maybe moving to pointers)
  };


  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2): phase1(EOS_phase1), phase2(EOS_phase2) {}


  // Evaluate the 'continuous flux'
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_continuos_flux(const FluxValue<cfg>& q, const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < EquationData::dim);

    FluxValue<cfg> res = q;

    // Compute density, velocity (along the dimension) and internal energy of phase 1
    const auto alpha1 = q(ALPHA1_INDEX);
    const auto rho1   = q(ALPHA1_RHO1_INDEX)/alpha1;
    auto e1           = q(ALPHA1_RHO1_E1_INDEX)/q(ALPHA1_RHO1_INDEX);
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1 -= 0.5*(q(ALPHA1_RHO1_U1_INDEX + d)/q(ALPHA1_RHO1_INDEX))*(q(ALPHA1_RHO1_U1_INDEX + d)/q(ALPHA1_RHO1_INDEX));
    }
    const auto pres1  = this->phase1.pres_value(rho1, e1);
    const auto vel1_d = q(ALPHA1_RHO1_U1_INDEX + curr_d)/q(ALPHA1_RHO1_INDEX);

    // Compute the flux for the equations "associated" to phase 1
    res(ALPHA1_INDEX) = 0.0;
    res(ALPHA1_RHO1_INDEX) *= vel1_d;
    res(ALPHA1_RHO1_U1_INDEX) *= vel1_d;
    if(EquationData::dim > 1) {
      for(std::size_t d = 1; d < EquationData::dim; ++d) {
        res(ALPHA1_RHO1_U1_INDEX + d) *= vel1_d;
      }
    }
    res(ALPHA1_RHO1_U1_INDEX + curr_d) += 0.0*alpha1*pres1;
    res(ALPHA1_RHO1_E1_INDEX) *= vel1_d;
    res(ALPHA1_RHO1_E1_INDEX) += 0.0*alpha1*pres1*vel1_d;

    // Compute density, velocity (along the dimension) and internal energy of phase 2
    const auto alpha2 = 1.0 - alpha1;
    const auto rho2   = q(ALPHA2_RHO2_INDEX)/alpha2;
    auto e2           = q(ALPHA2_RHO2_E2_INDEX)/q(ALPHA2_RHO2_INDEX);
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2 -= 0.5*(q(ALPHA2_RHO2_U2_INDEX + d)/q(ALPHA2_RHO2_INDEX))*(q(ALPHA2_RHO2_U2_INDEX + d)/q(ALPHA2_RHO2_INDEX));
    }
    const auto pres2  = this->phase2.pres_value(rho2, e2);
    const auto vel2_d = q(ALPHA2_RHO2_U2_INDEX + curr_d)/q(ALPHA2_RHO2_INDEX);

    // Compute the flux for the equations "associated" to phase 2
    res(ALPHA2_RHO2_INDEX) *= vel2_d;
    res(ALPHA2_RHO2_U2_INDEX) *= vel2_d;
    if(EquationData::dim > 1) {
      for(std::size_t d = 1; d < EquationData::dim; ++d) {
        res(ALPHA2_RHO2_U2_INDEX + d) *= vel2_d;
      }
    }
    res(ALPHA2_RHO2_U2_INDEX + curr_d) += 0.0*alpha2*pres2;
    res(ALPHA2_RHO2_E2_INDEX) *= vel2_d;
    res(ALPHA2_RHO2_E2_INDEX) += 0.0*alpha2*pres2*vel2_d;

    return res;
  }


  /**
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RusanovFlux: public Flux<Field> {
  public:
    RusanovFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2); // Construction which accepts in inputs the equations of stae of the two phases

    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); // Rusanov flux along direction d

    auto make_flux(); // Compute the flux over all cells
  };


  // Constructor derived from base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2): Flux<Field>(EOS_phase1, EOS_phase2) {}


  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> RusanovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                 std::size_t curr_d) {
    // Left state phase 1
    const auto vel1L_d = qL(ALPHA1_RHO1_U1_INDEX + curr_d)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho1L   = qL(ALPHA1_RHO1_INDEX)/qL(ALPHA1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1L           = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1L -= 0.5*(qL(ALPHA1_RHO1_U1_INDEX + d)/qL(ALPHA1_RHO1_INDEX))*(qL(ALPHA1_RHO1_U1_INDEX + d)/qL(ALPHA1_RHO1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres1L  = this->phase1.pres_value(rho1L, e1L);
    const auto c1L     = this->phase1.c_value(rho1L, pres1L);

    // Left state phase 2
    const auto vel2L_d = qL(ALPHA2_RHO2_U2_INDEX + curr_d)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2L   = qL(ALPHA2_RHO2_INDEX)/(1.0 - qL(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L           = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres2L  = this->phase2.pres_value(rho2L, e2L);
    const auto c2L     = this->phase2.c_value(rho2L, pres2L);

    // Right state phase 1
    const auto vel1R_d = qR(ALPHA1_RHO1_U1_INDEX + curr_d)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/qR(ALPHA1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1R           = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1R -= 0.5*(qR(ALPHA1_RHO1_U1_INDEX + d)/qR(ALPHA1_RHO1_INDEX))*(qR(ALPHA1_RHO1_U1_INDEX + d)/qR(ALPHA1_RHO1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres1R  = this->phase1.pres_value(rho1R, e1R);
    const auto c1R     = this->phase1.c_value(rho1R, pres1R);

    // Right state phase 2
    const auto vel2R_d = qR(ALPHA2_RHO2_U2_INDEX + curr_d)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2R   = qR(ALPHA2_RHO2_INDEX)/(1.0 - qR(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R           = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres2R  = this->phase2.pres_value(rho2R, e2R);
    const auto c2R     = this->phase2.c_value(rho2R, pres2R);

    const auto lambda = std::max(std::max(std::max(std::abs(vel1L_d + 0.0*c1L), std::abs(vel1L_d - 0.0*c1L)),
                                          std::max(std::abs(vel1R_d + 0.0*c1R), std::abs(vel1R_d - 0.0*c1R))),
                                 std::max(std::max(std::abs(vel2L_d + 0.0*c2L), std::abs(vel2L_d - 0.0*c2L)),
                                          std::max(std::abs(vel2R_d + 0.0*c2R), std::abs(vel2R_d - 0.0*c2R))));

    // Auxiliary states to discard contribution for the volume fraction
    auto qL_aux = qL;
    auto qR_aux = qR;
    qL_aux(ALPHA1_INDEX) = 0.0;
    qR_aux(ALPHA1_INDEX) = 0.0;

    return 0.5*(this->evaluate_continuos_flux(qL, curr_d) + this->evaluate_continuos_flux(qR, curr_d)) - // centered contribution
           0.5*lambda*(qR_aux - qL_aux); // upwinding contribution
  }


  // Implement the contribution of the discrete flux for all the cells in the mesh.
  //
  template<class Field>
  auto RusanovFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function
        discrete_flux[d].cons_flux_function = [&](auto& cells, const Field& field)
                                              {
                                                const auto& left  = cells[0];
                                                const auto& right = cells[1];

                                                const auto& qL = field[left];
                                                const auto& qR = field[right];

                                                return compute_discrete_flux(qL, qR, d);
                                              };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }


  /**
    * Implementation of the non-conservative flux
    */
  template<class Field>
  class NonConservativeFlux: public Flux<Field> {
  public:
    NonConservativeFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2); // Construction which accepts in inputs the equations of stae of the two phases

    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux_left_right(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                          const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                          const std::size_t curr_d); // Non-conservative flux from left to right

    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux_right_left(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                          const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                          const std::size_t curr_d); // Non-conservative flux from right to left

    auto make_flux(); // Compute the flux over all cells
  };


  // Constructor derived from base class
  //
  template<class Field>
  NonConservativeFlux<Field>::NonConservativeFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2): Flux<Field>(EOS_phase1, EOS_phase2) {}


  // Implementation of a non-conservative flux from left to right
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> NonConservativeFlux<Field>::compute_discrete_flux_left_right(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                                    const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                                    std::size_t curr_d) {
    FluxValue<typename Flux<Field>::cfg> res;

    // Zero contribution from continuity equations
    res(ALPHA1_RHO1_INDEX) = 0.0;
    res(ALPHA2_RHO2_INDEX) = 0.0;

    // Contribution for the volume fraction
    const auto velIL = qL(ALPHA1_RHO1_U1_INDEX + curr_d)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto velIR = qR(ALPHA1_RHO1_U1_INDEX + curr_d)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto lambda = std::max(std::abs(velIL), std::abs(velIR));

    res(ALPHA1_INDEX) = -(0.5*(velIL*qL(ALPHA1_INDEX) + velIR*qR(ALPHA1_INDEX)) - 0.5*(velIL + velIR)*qR(ALPHA1_INDEX) -
                          0.5*lambda*(qR(ALPHA1_INDEX) - qL(ALPHA1_INDEX))); /*--- Stabilizing term ---*/

    // Contribution for the momentum balance
    const auto rho2L = qL(ALPHA2_RHO2_INDEX)/(1.0 - qL(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L         = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pIL   = this->phase2.pres_value(rho2L, e2L);

    const auto rho2R = qR(ALPHA2_RHO2_INDEX)/(1.0 - qR(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R         = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pIR   = this->phase2.pres_value(rho2R, e2R);

    res(ALPHA1_RHO1_U1_INDEX + curr_d) = 0.0*(0.5*(pIL*qL(ALPHA1_INDEX) + pIR*qR(ALPHA1_INDEX)) - 0.5*(pIL + pIR)*qR(ALPHA1_INDEX));
    res(ALPHA2_RHO2_U2_INDEX + curr_d) = -res(ALPHA1_RHO1_U1_INDEX + curr_d);

    // Contribution for the total energy
    res(ALPHA1_RHO1_E1_INDEX) = 0.0*(0.5*(velIL*pIL*qL(ALPHA1_INDEX) + velIR*pIR*qR(ALPHA1_INDEX)) - 0.5*(pIL + pIR)*qR(ALPHA1_INDEX));
    res(ALPHA2_RHO2_E2_INDEX) = -res(ALPHA1_RHO1_E1_INDEX);

    return res;
  }


  // Implementation of a non-conservative flux from right to left
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> NonConservativeFlux<Field>::compute_discrete_flux_right_left(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                                    const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                                    std::size_t curr_d) {
    FluxValue<typename Flux<Field>::cfg> res;

    // Zero contribution from continuity equations
    res(ALPHA1_RHO1_INDEX) = 0.0;
    res(ALPHA2_RHO2_INDEX) = 0.0;

    // Build the non conservative flux (a lot of approximations to be checked here)

    // Contribution for the volume fraction
    const auto velIL = qL(ALPHA1_RHO1_U1_INDEX + curr_d)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto velIR = qR(ALPHA1_RHO1_U1_INDEX + curr_d)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto lambda = std::max(std::abs(velIL), std::abs(velIR));

    res(ALPHA1_INDEX) = (0.5*(velIL*qL(ALPHA1_INDEX) + velIR*qR(ALPHA1_INDEX)) - 0.5*(velIL + velIR)*qL(ALPHA1_INDEX) -
                         0.5*lambda*(qR(ALPHA1_INDEX) - qL(ALPHA1_INDEX))); /*--- Stabilizing term ---*/

    // Contribution for the momentum balance
    const auto rho2L = qL(ALPHA2_RHO2_INDEX)/(1.0 - qL(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L         = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pIL   = this->phase2.pres_value(rho2L, e2L);

    const auto rho2R = qR(ALPHA2_RHO2_INDEX)/(1.0 - qR(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R         = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pIR   = this->phase2.pres_value(rho2R, e2R);

    res(ALPHA1_RHO1_U1_INDEX + curr_d) = -0.0*(0.5*(pIL*qL(ALPHA1_INDEX) + pIR*qR(ALPHA1_INDEX)) - 0.5*(pIL + pIR)*qL(ALPHA1_INDEX));
    res(ALPHA2_RHO2_U2_INDEX + curr_d) = -res(ALPHA1_RHO1_U1_INDEX + curr_d);

    // Contribution for the total energy
    res(ALPHA1_RHO1_E1_INDEX) = -0.0*(0.5*(velIL*pIL*qL(ALPHA1_INDEX) + velIR*pIR*qR(ALPHA1_INDEX)) - 0.5*(pIL + pIR)*qL(ALPHA1_INDEX));
    res(ALPHA2_RHO2_E2_INDEX) = -res(ALPHA1_RHO1_E1_INDEX);

    return res;
  }


  // Implement the contribution of the discrete flux for all the cells in the mesh.
  //
  template<class Field>
  auto NonConservativeFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" non-conservative flux function
        discrete_flux[d].flux_function = [&](auto& cells, const Field& field)
                                            {
                                              const auto& left  = cells[0];
                                              const auto& right = cells[1];

                                              const auto& qL = field[left];
                                              const auto& qR = field[right];

                                              samurai::FluxValuePair<typename Flux<Field>::cfg> flux;
                                              flux[0] = compute_discrete_flux_right_left(qL, qR, d);
                                              flux[1] = compute_discrete_flux_left_right(qL, qR, d);

                                              return flux;
                                            };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }

} // end namespace samurai

#endif
