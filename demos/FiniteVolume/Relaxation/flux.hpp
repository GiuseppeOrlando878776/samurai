#ifndef flux_hpp
#define flux_hpp

#pragma once
#include <samurai/schemes/fv.hpp>

#include "eos.hpp"

namespace EquationData {
  static std::size_t dim = 1; /*--- Spatial dimension. It would be ideal to be able to get it
                                    direclty from Field, but I need to move the definition of these indices

  /*--- Declare suitable static variables for the sake of generalities in the indices ---*/
  static std::size_t ALPHA1_INDEX         = 0;
  static std::size_t ALPHA1_RHO1_INDEX    = 1;
  static std:.size_t ALPHA1_RHO1_U1_INDEX = 2;
  static std::size_t ALPHA1_RHO1_E1_INDEX = 2 + dim;
  static std::size_t ALPHA2_RHO2_INDEX    = ALPHA1_RHO1_E1_INDEX + 1;
  static std::size_t ALPHA2_RHO2_U2_INDEX = ALPHA2_RHO2_INDEX + 1;
  static std::size_t ALPHA1_RHO2_E2_INDEX = ALPHA2_RHO2_U2_INDEX + dim;

  static std::size_t NVARS = ALPHA1_RHO2_E2_INDEX + 1;
}

namespace samurai {
  /**
    * Generic class to compute the flux between a left and right state
    */
  template<class Field>
  class Flux {
  public:
    // Definitions and sanity checks
    static constexpr std::size_t field_size        = Field::size;
    static_assert(field_size == EquationData::NVARS, "The number of elements in the state does not correpsond to the number of equations")
    static_assert(Field::dim == EquationData::dim, "The spatial dimesions do not match")
    static constexpr std::size_t output_field_size = field_size;
    static constexpr std::size_t stencil_size      = 2;

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    // Routines declaration
    Flux() = default; // Default constructor

    Flux(const EOS& EOS_phase1, const EOS& EOS_phase2); // Construction which accepts in inputs the equations of stae of the two phases

    template<class State>
    FluxValue<cfg> evaluate_continuos_flux(const State& q, std::size_t curr_d = 0); // Evaluate the 'continuous' flux for the state q along direction curr_d

    template<class State>
    virtual FluxValue<cfg> compute_discrete_flux(const State& qL, const State& qR, std::size_t curr_d = 0) = 0; // Compute the flux along direction curr_d (curr_d < dim)
                                                                                                                // This is a pure virtual function and it is the one
                                                                                                                // has to implement ot have its own flux.

    decltype(make_flux_based_scheme(FluxDefinition<cfg>)) make_flux(); // Compute the flux over all cells

  protected:
    EOS& phase1;
    EOS& phase2;
  }


  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const EOS& EOS_phase1, const EOS& EOS_phase2): phase1(EOS_phase1), phase2(EOS_phase2) {}


  // Evaluate the 'continuous flux'
  //
  template<class Field>
  template<class State>
  FluxValue<cfg> Flux<Field>::evaluate_continuos_flux(const State& q, std::size_t curr_d = 0) {
    // Sanity check in terms of dimensions
    static_assert(curr_d < EquationData::dim, "The spatial dimesion of the problem is lower than the direction required for the flux")

    FluxValue<cfg> res = q;

    // Compute density, velocity (along the dimension) and internal energy of phase 1
    const auto alpha1 = q(ALPHA1_INDEX);
    const auto rho1   = q(ALPHA1_RHO1_INDEX)/alpha1;
    auto e1           = q(ALPHA1_RHO1_E1_INDEX)/q(ALPHA1_RHO1_INDEX);
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1 -= 0.5*q(ALPHA1_RHO1_U1_INDEX + d)/q(ALPHA1_RHO1_INDEX);
    }
    const auto pres1  = phase1.pres_value(rho1, e1);
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
    res(ALPHA1_RHO1_U1_INDEX + curr_d) += alpha1*pres1;
    res(ALPHA1_RHO1_E1_INDEX) *= vel1_d;
    res(ALPHA1_RHO1_E1_INDEX) += alpha1*pres1*vel1_d;

    // Compute density, velocity (along the dimension) and internal energy of phase 2
    const auto alpha2 = 1.0 - alpha1;
    const auto rho2   = q(ALPHA2_RHO2_INDEX)/alpha2;
    auto e2           = q(ALPHA2_RHO2_E2_INDEX)/q(ALPHA2_RHO2_INDEX);
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1 -= 0.5*q(ALPHA2_RHO2_U2_INDEX + d)/q(ALPHA2_RHO2_INDEX);
    }
    const auto pres2  = phase2.pres_value(rho2, e2);
    const auto vel2_d = q(ALPHA2_RHO1_U2_INDEX + curr_d)/q(ALPHA2_RHO2_INDEX);

    // Compute the flux for the equations "associated" to phase 2
    res(ALPHA2_RHO2_INDEX) *= vel2_d;
    res(ALPHA2_RHO2_U2_INDEX) *= vel2_d;
    if(EquationData::dim > 1) {
      for(std::size_t d = 1; d < EquationData::dim; ++d) {
        res(ALPHA2_RHO2_U2_INDEX + d) *= vel2_d;
      }
    }
    res(ALPHA1_RHO2_U2_INDEX + curr_d) += alpha2*pres2;
    res(ALPHA1_RHO2_E2_INDEX) *= vel2_d;
    res(ALPHA1_RHO2_E2_INDEX) += alpha2*pres2*vel2_d;

    return res;
  }


  // Implement the contribution of the discrete flux for all the cells in the mesh.
  // It is evident that 'compute_discrete_flux' has to be overriden...
  //
  template<class Field>
  auto Flux<Field>::make_flux() {
    FluxDefinition<cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function
        discrete_flux[d].flux_function = [&](auto& cells, Field& field)
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
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RusanovFlux: public Flux {
    RusanovFlux() = default; // Default constructor

    RusanovFlux(const RusanovFlux&) = default; // Default copy-constructor

    RusanovFlux(const EOS& EOS_phase1, const EOS& EOS_phase2); // Construction which accepts in inputs the equations of stae of the two phases

    template<class State>
    virtual Field compute_discrete_flux(const State& qL, const State& qR, std::size_t curr_d = 0) override; // Rusanov flux along direction d
  };


  // Constructor derived from base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const EOS& EOS_phase1, const EOS& EOS_phase2): Flux(EOS_phase1, EOS_phase2) {}


  // Implementation of a Rusanov flux
  //
  template<class Field>
  template<class State>
  FluxValue<cfg> RusanovFlux<Field>::compute_discrete_flux(const State& qL, const State& qR, std::size_t curr_d) {
    // Left state phase 1
    const auto vel1L_d = qL(ALPHA1_RHO1_U1_INDEX + curr_d)/qL(ALPHA1_RHO1_INDEX);
    const auto rho1L   = qL(ALPHA1_RHO1_INDEX)/qL(ALPHA1_INDEX);
    auto e1L           = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX);
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1L -= 0.5*qL(ALPHA1_RHO1_U1_INDEX + d)/qL(ALPHA1_RHO1_INDEX);
    }
    const auto pres1L  = phase1.pres_value(rho1L, e1L);
    const auto c1L     = phase1.c_value(rho1L, pres1L);

    // Left state phase 2
    const auto vel2L_d = qL(ALPHA2_RHO2_U2_INDEX + curr_d)/qL(ALPHA2_RHO2_INDEX);
    const auto rho2L   = qL(ALPHA2_RHO2_INDEX)/(1.0 - qL(ALPHA1_INDEX));
    auto e2L           = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX);
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX);
    }
    const auto pres2L  = phase2.pres_value(rho2L, e2L);
    const auto c2L     = phase2.c_value(rho2L, pres2L);

    // Right state phase 1
    const auto vel1R_d = qR(ALPHA1_RHO1_U1_INDEX + curr_d)/qR(ALPHA1_RHO1_INDEX);
    const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/qR(ALPHA1_INDEX);
    auto e1R           = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX);
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1R -= 0.5*qR(ALPHA1_RHO1_U1_INDEX + d)/qR(ALPHA1_RHO1_INDEX);
    }
    const auto pres1R  = phase1.pres_value(rho1L, e1L);
    const auto c1R     = phase1.c_value(rho1L, pres1L);

    // Right state phase 2
    const auto vel2R_d = qR(ALPHA2_RHO2_U2_INDEX + curr_d)/qR(ALPHA2_RHO2_INDEX);
    const auto rho2R   = qR(ALPHA2_RHO2_INDEX)/(1.0 - qR(ALPHA1_INDEX));
    auto e2R           = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX);
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX);
    }
    const auto pres2R  = phase2.pres_value(rho2R, e2R);
    const auto c2R     = phase2.c_value(rho2R, pres2R);


    const auto lambda = std::max(std::max(std::abs(vel1L_d + c1L), std::abs(vel1L_d - c1L)),
                                 std::max(std::abs(vel1R_d + c1R), std::abs(vel1R_d - c1R))
                                 std::max(std::abs(vel2L_d + c2L), std::abs(vel2L_d - c2L)),
                                 std::max(std::abs(vel2R_d + c2R), std::abs(vel2R_d - c2R)));

    return 0.5*(evaluate_continuos_flux(qL) + evaluate_continuos_flux(qR)) - // centered contribution
           0.5*lambda*(qR - qL); // upwinding contribution
  }

} // end namespace samurai
