#ifndef eos_hpp
#define eos_hpp


/**
  * Implementation of a generic lcass to handle the EOS. It has several
    pure virtual functions to be implementede for the specific EOS
  */
class EOS {
public:
  EOS() = default; // Default constructor

  EOS(const EOS&) = default // Default copy-constructor

  template<typename Field>
  virtual Field pres_value(const Field& rho, const Field& e) const = 0; // Function to compute the pressure from the density and the internal energy

  template<typename Field>
  virtual Field rho_value(const Field& pres, const Field& e) const = 0; // Function to compute the density from the pressure and the internal energy

  template<typename Field>
  virtual Field e_value(const Field& rho, const Field& pres) const = 0; // Function to compute the internal energy from density and pressure

  template<typename Field>
  virtual Field c_value(const Field& rho, const Field& pres) const = 0; // Function to compute the speed of sound from density and pressure
};


/**
 * Implementation of the stiffened gas equation of state (SG-EOS)
 */
class SG_EOS: public EOS {
public:
  SG_EOS() = default; // Default constructor

  SG_EOS(const SG_EOS&) = default; // Default copy-constructor

  SG_EOS(const double gamma_, const double pi_infty_, const double q_infty_ = 0.0); // Constructor which accepts as arguments
                                                                                    // the isentropic exponent and the two parameters
                                                                                    // that characterize the fluid

  template<typename Field>
  virtual Field pres_value(const Field& rho, const Field& e) const override; // Function to compute the pressure from the density and the internal energy

  template<typename Field>
  virtual Field rho_value(const Field& pres, const Field& e) const override; // Function to compute the density from the pressure and the internal energy

  template<typename Field>
  virtual Field e_value(const Field& rho, const Field& pres) const override; // Function to compute the internal energy from density and pressure

  template<typename Field>
  virtual Field c_value(const Field& rho, const Field& pres) const override; // Function to compute the speed of sound from density and pressure

protected:
  const double gamma;    // Isentropic exponent
  const double pi_infty; // Pressure at 'infinite'
  const double q_infty;  // Internal energy at 'infinite'
};


// Implement the constructor
//
SG_EOS::SG_EOS(const double gamma_, const double pi_infty_, const double q_infty_):
  gamma(gamma_), pi_infty(pi_infty_), q_infty(q_infty_) {}


// Compute the pressure value from the density and the internal energy
//
template<typename Field>
Field SG_EOS::pres_value(const Field& rho, const Field& e) const {
  return (gamma - 1.0)*rho*(e - q_infty) - gamma*pi_infty;
}


// Compute the density from the pressure and the internal energy
//
template<typename Field>
Field SG_EOS::rho_value(const Field& pres, const Field& e) const {
  return (pres + gamma*p_infty)/((gamma - 1.0)*(e - q_infty));
}


// Compute the internal energy from density and pressure
//
template<typename Field>
Field SG_EOS::e_value(const Field& rho, const Field& pres) const {
  return (pres + gamma*p_infty)/((gamma - 1.0)**rho) + q_infty;
}


// Compute the speed of sound from density and pressure
//
template<typename Field>
Field SG_EOS::c_value(const Field& rho, const Field& pres) const {
  return std::sqrt(gamma*(pres + pi_infty)/rho);
}


#endif
